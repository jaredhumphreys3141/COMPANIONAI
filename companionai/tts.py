"""Speech synthesis with Piper.

Piper is an ONNX VITS runtime: a medium voice synthesises far faster than
real time even on a Raspberry Pi 5 CPU, which is what keeps the companion's
replies conversational on every target board.  If Piper is unavailable we fall
back to ``espeak-ng`` so the app still speaks (robotically) out of the box.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from . import catalog, config, paths


class TTSUnavailable(RuntimeError):
    pass


def voice_path(voice_id: str) -> Path:
    entry = catalog.get("tts", voice_id)
    if entry is not None and entry.installed:
        return entry.path
    for candidate in (paths.TTS_DIR / voice_id, paths.TTS_DIR / f"{voice_id}.onnx"):
        if candidate.exists():
            return candidate
    raise TTSUnavailable(
        f"Voice '{voice_id}' is not installed.  Install one from the Models tab, or copy a "
        f"Piper .onnx voice and its .onnx.json into {paths.TTS_DIR}."
    )


class PiperSpeaker:
    """One loaded Piper voice."""

    def __init__(self) -> None:
        self._voice = None
        self._key: str | None = None
        self._rate = 22050
        self._lock = threading.Lock()

    def load(self, character) -> None:
        path = voice_path(character.voice)
        with self._lock:
            if self._key == str(path) and self._voice is not None:
                return
            try:
                from piper import PiperVoice  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise TTSUnavailable(
                    "piper-tts is not installed.  Run scripts/install.sh, or "
                    "`pip install piper-tts`."
                ) from exc
            config_path = path.with_suffix(path.suffix + ".json")
            self._voice = PiperVoice.load(
                str(path), config_path=str(config_path) if config_path.exists() else None
            )
            self._rate = getattr(getattr(self._voice, "config", None), "sample_rate", 22050)
            self._key = str(path)

    @property
    def ready(self) -> bool:
        return self._voice is not None

    @property
    def sample_rate(self) -> int:
        return self._rate

    def unload(self) -> None:
        with self._lock:
            self._voice = None
            self._key = None

    def _synth_kwargs(self, character) -> dict:
        return {
            "length_scale": max(0.3, character.speech_rate),
            "noise_scale": character.voice_variation,
            "noise_w": character.voice_cadence,
            "speaker_id": character.speaker_index or None,
        }

    def stream(self, text: str, character) -> Iterator[np.ndarray]:
        """Yield int16 mono chunks as they are synthesised.

        Chunks come out at the voice's own level; volume is applied by the
        caller so there is exactly one place that scales the signal.
        """
        self.load(character)
        text = text.strip()
        if not text:
            return
        params = self._synth_kwargs(character)

        # piper-tts >= 1.3 exposes synthesize() yielding AudioChunk objects;
        # 1.2.x exposes synthesize_stream_raw() yielding raw bytes.
        if hasattr(self._voice, "synthesize"):
            try:
                from piper import SynthesisConfig  # noqa: PLC0415

                synthesis_config = SynthesisConfig(
                    length_scale=params["length_scale"],
                    noise_scale=params["noise_scale"],
                    noise_w_scale=params["noise_w"],
                    speaker_id=params["speaker_id"],
                )
                for chunk in self._voice.synthesize(text, syn_config=synthesis_config):
                    self._rate = getattr(chunk, "sample_rate", self._rate)
                    yield np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                return
            except (ImportError, TypeError, AttributeError):
                pass  # fall through to the older API

        raw = self._voice.synthesize_stream_raw(text, **params)
        for block in raw:
            yield np.frombuffer(block, dtype=np.int16)

    def synth(self, text: str, character) -> tuple[int, np.ndarray]:
        chunks = list(self.stream(text, character))
        if not chunks:
            return self.sample_rate, np.zeros(0, dtype=np.int16)
        audio = np.concatenate(chunks)
        if character.volume != 1.0:
            audio = np.clip(audio.astype(np.float32) * character.volume, -32768, 32767).astype(np.int16)
        return self.sample_rate, audio


class EspeakSpeaker:
    """Last-resort fallback so a fresh install can still talk."""

    sample_rate = 22050

    @property
    def ready(self) -> bool:
        return bool(shutil.which("espeak-ng") or shutil.which("espeak"))

    def load(self, character) -> None:
        if not self.ready:
            raise TTSUnavailable("Neither Piper nor espeak-ng is available.")

    def unload(self) -> None:
        return

    def synth(self, text: str, character) -> tuple[int, np.ndarray]:
        self.load(character)
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        words_per_minute = int(175 / max(0.3, character.speech_rate))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out = Path(handle.name)
        try:
            subprocess.run(
                [binary, "-w", str(out), "-s", str(words_per_minute), text],
                check=True,
                capture_output=True,
            )
            with wave.open(str(out), "rb") as wav:
                rate = wav.getframerate()
                data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            return rate, data
        finally:
            out.unlink(missing_ok=True)

    def stream(self, text: str, character) -> Iterator[np.ndarray]:
        _, audio = self.synth(text, character)
        yield audio


_speaker: PiperSpeaker | None = None
_fallback = EspeakSpeaker()
_lock = threading.Lock()


def speaker(character=None):
    """Return Piper if the character's voice is installed, else espeak-ng."""
    global _speaker
    with _lock:
        if _speaker is None:
            _speaker = PiperSpeaker()
    if character is not None:
        try:
            voice_path(character.voice)
        except TTSUnavailable:
            if _fallback.ready:
                return _fallback
            raise
    return _speaker


# --------------------------------------------------------------------------- #
# playback (server-side speakers, used by the hands-free voice loop)
# --------------------------------------------------------------------------- #
class Player:
    """Plays int16 audio on the local sound card and can be cut off mid-word."""

    def __init__(self) -> None:
        self._stream = None
        self._rate = 0
        self.interrupt = threading.Event()
        self._lock = threading.Lock()

    def _open(self, rate: int):
        import sounddevice as sd  # noqa: PLC0415

        if self._stream is not None and self._rate == rate:
            return self._stream
        self.close()
        device = config.get().output_device or None
        self._stream = sd.OutputStream(
            samplerate=rate, channels=1, dtype="int16",
            device=device if device else None, blocksize=1024,
        )
        self._stream.start()
        self._rate = rate
        return self._stream

    def play(self, rate: int, audio: np.ndarray) -> bool:
        """Play a block.  Returns False if playback was interrupted."""
        if audio.size == 0:
            return True
        try:
            stream = self._open(rate)
        except Exception as exc:  # pragma: no cover - no sound card in CI
            raise TTSUnavailable(f"No audio output device available: {exc}") from exc
        with self._lock:
            step = max(1024, rate // 10)
            for start in range(0, audio.size, step):
                if self.interrupt.is_set():
                    return False
                stream.write(np.ascontiguousarray(audio[start:start + step]))
        return True

    def stop(self) -> None:
        self.interrupt.set()

    def resume(self) -> None:
        self.interrupt.clear()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover
                pass
            self._stream = None
            self._rate = 0


_player: Player | None = None


def player() -> Player:
    global _player
    if _player is None:
        _player = Player()
    return _player


def save_wav(path: Path, rate: int, audio: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(audio.astype(np.int16).tobytes())
    return path
