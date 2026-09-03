"""The running companion: one active character, one conversation, one voice.

This is the layer the GUI talks to.  It owns the speech-out worker (so replies
start being spoken while the language model is still generating) and the
hands-free listening loop.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from . import (
    catalog,
    config,
    hardware,
    imagegen,
    llm,
    net,
    paths,
    tts,
    voice,
)
from . import (
    character as character_mod,
)
from . import (
    conversation as conversation_mod,
)


class Session:
    def __init__(self) -> None:
        paths.ensure_dirs()
        net.init()
        self.character = character_mod.ensure_default()
        self.conversation = conversation_mod.Conversation(self.character)
        self.speak_replies = True          # play audio on the device's speakers
        self.status = "idle"
        self._speech_queue: queue.Queue[str | None] = queue.Queue()
        self._speech_thread: threading.Thread | None = None
        self._last_reply_audio: tuple[int, np.ndarray] | None = None
        self._busy = threading.Lock()

    # ------------------------------------------------------------ character
    def use_character(self, character_id: str) -> str:
        if not character_id:
            return "No companion selected."
        try:
            self.character = character_mod.Character.load(character_id)
        except (OSError, ValueError) as exc:
            return f"Could not load '{character_id}': {exc}"
        config.update(active_character=character_id)
        self.conversation = conversation_mod.Conversation(self.character)
        llm.engine().unload()
        return f"{self.character.name} is now active."

    def refresh_character(self, updated) -> None:
        """Adopt an edited character without losing the conversation."""
        self.character = updated
        self.conversation.character = updated
        config.update(active_character=updated.id)

    # ---------------------------------------------------------------- speech
    def _speech_worker(self) -> None:
        player = tts.player()
        while True:
            item = self._speech_queue.get()
            if item is None:
                break
            text = conversation_mod.speakable(item)
            if not text:
                continue
            try:
                speaker = tts.speaker(self.character)
                rate = getattr(speaker, "sample_rate", 22050)
                for chunk in speaker.stream(text, self.character):
                    if voice.loop().interrupted.is_set():
                        break
                    if self.character.volume != 1.0:
                        chunk = np.clip(
                            chunk.astype(np.float32) * self.character.volume, -32768, 32767
                        ).astype(np.int16)
                    rate = getattr(speaker, "sample_rate", rate)
                    if not player.play(rate, chunk):
                        break
            except tts.TTSUnavailable as exc:
                self.status = f"voice unavailable: {exc}"
            except Exception as exc:  # pragma: no cover - audio stacks vary
                self.status = f"speech error: {exc}"

    def _ensure_speech_worker(self) -> None:
        if self._speech_thread is None or not self._speech_thread.is_alive():
            self._speech_thread = threading.Thread(
                target=self._speech_worker, daemon=True, name="companion-speech"
            )
            self._speech_thread.start()

    def say(self, text: str) -> None:
        """Queue a sentence for the device's speakers."""
        if not self.speak_replies or not text.strip():
            return
        self._ensure_speech_worker()
        tts.player().resume()
        voice.loop().interrupted.clear()
        self._speech_queue.put(text)

    def silence(self) -> None:
        """Drop anything queued and cut off the current utterance."""
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break
        tts.player().stop()

    def render_audio(self, text: str) -> tuple[int, np.ndarray] | None:
        """Synthesise without playing - used to send audio to the browser."""
        clean = conversation_mod.speakable(text)
        if not clean:
            return None
        speaker = tts.speaker(self.character)
        rate, audio = speaker.synth(clean, self.character)
        return (rate, audio) if audio.size else None

    # ----------------------------------------------------------------- reply
    def reply_stream(self, user_text: str):
        """Yield the growing reply text.  Sentences are spoken as they complete."""
        with self._busy:
            self.status = "thinking"
            voice.loop().speaking.set()
            voice.loop().interrupted.clear()
            tts.player().resume()
            spoke_anything = False
            try:
                for full, sentence in self.conversation.stream_reply(user_text):
                    if voice.loop().interrupted.is_set():
                        self.conversation.stop()
                        self.silence()
                        self.status = "interrupted"
                        break
                    if sentence:
                        spoke_anything = True
                        self.status = "speaking"
                        self.say(sentence)
                    yield full
            except (llm.ModelNotAvailable, RuntimeError) as exc:
                yield f"[{exc}]"
            finally:
                # Let the tail of the reply finish playing before we listen again.
                if spoke_anything and self.speak_replies:
                    self._wait_for_speech()
                voice.loop().speaking.clear()
                if self.status != "interrupted":
                    self.status = "idle"

    def _wait_for_speech(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._speech_queue.empty():
                # The worker may still be mid-utterance; give it a moment.
                time.sleep(0.2)
                if self._speech_queue.empty():
                    return
            if voice.loop().interrupted.is_set():
                return
            time.sleep(0.1)

    def reply_blocking(self, user_text: str) -> str:
        text = ""
        for text in self.reply_stream(user_text):  # noqa: B007 - we want the last chunk
            pass
        return text

    # ------------------------------------------------------------ hands free
    def start_listening(self) -> str:
        listener = voice.loop()
        if listener.running:
            return "Already listening."

        def on_utterance(text: str) -> None:
            for _ in self.reply_stream(text):
                pass

        listener.start(self.character, on_utterance)
        time.sleep(0.3)
        if not listener.running:
            return listener.last_error or "The microphone loop failed to start."
        return "Listening.  Just talk - say something and pause."

    def stop_listening(self) -> str:
        voice.loop().stop()
        self.silence()
        return "Stopped listening."

    @property
    def listening(self) -> bool:
        return voice.loop().running

    # ---------------------------------------------------------------- status
    def readiness(self) -> list[list[str]]:
        """Component-by-component readiness table for the System tab."""
        rows = []

        def check(kind: str, model_id: str, label: str) -> None:
            entry = catalog.get(kind, model_id)
            if entry is not None:
                state = "ready" if entry.installed else "not installed"
                detail = entry.name
            elif kind == "llm" and (paths.LLM_DIR / model_id).exists():
                state, detail = "ready", f"local file {model_id}"
            else:
                state, detail = "not installed", model_id or "(none chosen)"
            rows.append([label, state, detail])

        check("llm", self.character.llm_model, "Language model")
        check("asr", self.character.asr_model, "Speech recognition")
        check("tts", self.character.voice, "Voice")
        check("image", self.character.image_model, "Image model")

        rows.append(["Microphone loop", "running" if self.listening else "stopped", ""])
        rows.append(["Speak replies aloud", "on" if self.speak_replies else "off", ""])
        status = net.status()
        rows.append(["Network", status.policy, status.badge])
        return rows

    def summary(self) -> str:
        info = hardware.detect()
        estimate = imagegen.estimate_seconds(self.character)
        return (
            f"**{self.character.name}** on **{info.label}** - "
            f"{self.status}.  A {self.character.image_size}px image takes roughly "
            f"{estimate:.0f}s here."
        )


_session: Session | None = None
_lock = threading.Lock()


def get() -> Session:
    global _session
    with _lock:
        if _session is None:
            _session = Session()
        return _session
