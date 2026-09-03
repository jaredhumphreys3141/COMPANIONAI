"""Hands-free microphone loop.

Captures 16 kHz mono from the local sound card, segments it with WebRTC voice
activity detection (energy-based fallback if ``webrtcvad`` is not installed),
transcribes each utterance and hands the text to a callback.  While the
companion is talking the loop keeps listening so the user can interrupt -
"barge-in" - which is what makes the Jetson feel like a conversation rather
than a walkie-talkie.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from . import asr, config

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

EventFn = Callable[[dict], None]


class _EnergyVAD:
    """Fallback detector: adaptive noise floor, no extra dependency."""

    def __init__(self, aggressiveness: int = 2) -> None:
        self.floor = 0.008
        self.margin = 2.0 + aggressiveness  # stricter as aggressiveness rises

    def is_speech(self, frame: np.ndarray, _rate: int = SAMPLE_RATE) -> bool:
        rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32) / 32768.0))) + 1e-9)
        speech = rms > self.floor * self.margin
        if not speech:  # track the room's noise floor while nobody is talking
            self.floor = 0.98 * self.floor + 0.02 * rms
        return speech


def _make_vad(aggressiveness: int):
    try:
        import webrtcvad  # noqa: PLC0415

        vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))

        class _WebRTC:
            @staticmethod
            def is_speech(frame: np.ndarray, rate: int = SAMPLE_RATE) -> bool:
                return vad.is_speech(frame.astype(np.int16).tobytes(), rate)

        return _WebRTC()
    except ImportError:
        return _EnergyVAD(aggressiveness)


class MicrophoneUnavailable(RuntimeError):
    pass


class VoiceLoop:
    """Background listener.  ``on_utterance(text)`` runs on the loop's thread."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.speaking = threading.Event()      # set while the companion talks
        self.interrupted = threading.Event()   # set when the user barges in
        self.events: queue.Queue[dict] = queue.Queue()
        self.last_error = ""

    # ------------------------------------------------------------- lifecycle
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, character, on_utterance: Callable[[str], None]) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(character, on_utterance), daemon=True, name="companion-voice"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def emit(self, kind: str, **payload) -> None:
        self.events.put({"type": kind, "time": time.time(), **payload})

    def drain(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    # ------------------------------------------------------------------ loop
    def _run(self, character, on_utterance: Callable[[str], None]) -> None:
        try:
            import sounddevice as sd  # noqa: PLC0415
        except ImportError:
            self.last_error = (
                "sounddevice is not installed, so the microphone loop cannot start.  "
                "Use the push-to-talk recorder instead, or `pip install sounddevice`."
            )
            self.emit("error", message=self.last_error)
            return

        settings = config.get()
        vad = _make_vad(settings.vad_aggressiveness)
        silence_frames = max(1, settings.silence_ms // FRAME_MS)
        min_speech_frames = max(1, settings.min_speech_ms // FRAME_MS)

        buffer: list[np.ndarray] = []
        speech_frames = 0
        quiet_frames = 0
        gain = settings.mic_gain

        def callback(indata, _frames, _time_info, status):  # pragma: no cover - realtime
            if status:
                pass  # over/underruns are normal on a busy Pi; keep going
            frames.put(indata.copy().reshape(-1))

        frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

        try:
            device = settings.input_device or None
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=FRAME_SAMPLES, device=device if device else None,
                callback=callback,
            )
        except Exception as exc:  # pragma: no cover - no mic in CI
            self.last_error = f"Could not open the microphone: {exc}"
            self.emit("error", message=self.last_error)
            return

        self.emit("listening")
        with stream:
            while not self._stop.is_set():
                try:
                    frame = frames.get(timeout=0.5)
                except queue.Empty:
                    continue
                if gain != 1.0:
                    frame = np.clip(frame.astype(np.float32) * gain, -32768, 32767).astype(np.int16)

                voiced = vad.is_speech(frame)

                if self.speaking.is_set():
                    # Companion is talking: only look for a barge-in.
                    if voiced and config.get().barge_in:
                        speech_frames += 1
                        if speech_frames >= min_speech_frames:
                            self.interrupted.set()
                            self.emit("interrupt")
                            speech_frames = 0
                    else:
                        speech_frames = max(0, speech_frames - 1)
                    buffer.clear()
                    continue

                if voiced:
                    buffer.append(frame)
                    speech_frames += 1
                    quiet_frames = 0
                    if speech_frames == min_speech_frames:
                        self.emit("speech")
                elif buffer:
                    buffer.append(frame)
                    quiet_frames += 1
                    if quiet_frames >= silence_frames:
                        utterance = np.concatenate(buffer) if buffer else np.zeros(0, np.int16)
                        captured = speech_frames
                        buffer, speech_frames, quiet_frames = [], 0, 0
                        if captured >= min_speech_frames:
                            self._handle(utterance, character, on_utterance)
                        self.emit("listening")

        self.emit("stopped")

    def _handle(self, utterance: np.ndarray, character, on_utterance) -> None:
        audio = utterance.astype(np.float32) / 32768.0
        self.emit("transcribing", seconds=round(audio.size / SAMPLE_RATE, 1))
        try:
            text = asr.transcriber().transcribe(audio, character)
        except Exception as exc:
            self.last_error = str(exc)
            self.emit("error", message=str(exc))
            return
        text = text.strip()
        if not text or len(text) < 2:
            return
        self.emit("user", text=text)
        try:
            on_utterance(text)
        except Exception as exc:  # keep the loop alive through a bad turn
            self.last_error = str(exc)
            self.emit("error", message=str(exc))


_loop: VoiceLoop | None = None


def loop() -> VoiceLoop:
    global _loop
    if _loop is None:
        _loop = VoiceLoop()
    return _loop


def list_devices() -> tuple[list[str], list[str]]:
    """``(input_names, output_names)`` for the Settings tab."""
    try:
        import sounddevice as sd  # noqa: PLC0415

        devices = sd.query_devices()
    except Exception:  # pragma: no cover - no audio stack in CI
        return [], []
    inputs = [d["name"] for d in devices if d.get("max_input_channels", 0) > 0]
    outputs = [d["name"] for d in devices if d.get("max_output_channels", 0) > 0]
    return inputs, outputs
