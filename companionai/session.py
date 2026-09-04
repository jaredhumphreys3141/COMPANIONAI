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
    recall,
    tts,
    voice,
)
from . import character as character_mod
from . import conversation as conversation_mod
from . import memory as memory_mod


class Session:
    def __init__(self) -> None:
        paths.ensure_dirs()
        net.init()
        self.character = character_mod.ensure_default()
        self.memory = memory_mod.for_character(self.character.id)
        self.conversation = self._new_conversation(resume=True)
        self.speak_replies = True          # play audio on the device's speakers
        self._pending: list[tuple[str, str]] = []
        self._memory_lock = threading.Lock()
        self.status = "idle"
        self._speech_queue: queue.Queue[str | None] = queue.Queue()
        # Utterances the speaker finished, so an interrupted reply can be
        # trimmed to what the user actually heard.
        self._played: list[str] = []
        self._speech_thread: threading.Thread | None = None
        self._last_reply_audio: tuple[int, np.ndarray] | None = None
        self._busy = threading.Lock()

    # --------------------------------------------------------------- memory
    def _new_conversation(self, resume: bool = False):
        """Start a conversation, optionally reopening the last saved one."""
        conversation = conversation_mod.Conversation(self.character, self.memory)
        if resume and config.get().resume_conversations:
            if conversation.resume():
                return conversation
        self.memory.conversations += 1
        self.memory.save()
        return conversation

    def _remember(self, user_text: str, reply_text: str) -> None:
        """Index the exchange and, periodically, distil it into memory.

        Runs on a background thread after the reply has been spoken: an extra
        model call costs seconds on a Pi, and the point of the whole streaming
        pipeline is that the user never waits for one.

        Exchanges are buffered by value rather than re-read from the live
        conversation, because these threads can overlap and finish out of
        order - reading the conversation later would extract facts against
        whatever was said most recently instead of the exchange this pass is
        actually for.
        """
        settings = config.get()
        if not settings.memory_enabled:
            return
        character_id = self.character.id
        try:
            with self._memory_lock:
                if settings.recall_enabled:
                    recall.index().add(character_id, "user", user_text)
                    recall.index().add(character_id, self.character.name, reply_text)

                self.memory.messages += 2
                self._pending.append((user_text, reply_text))
                if len(self._pending) < max(1, settings.memory_extract_every):
                    self.memory.save()
                    return

                batch, self._pending = self._pending, []
                turns = []
                for said, replied in batch:
                    turns.append(conversation_mod.Turn("user", said))
                    turns.append(conversation_mod.Turn("assistant", replied))
                memory_mod.extract(self.memory, self.character, turns, llm.engine())
        except Exception as exc:  # memory is a nicety; never break the session
            self.status = f"memory error: {exc}"

    def new_conversation(self) -> None:
        """Start a fresh conversation, leaving long-term memory intact."""
        self.silence()
        self.conversation = self._new_conversation(resume=False)

    def remember_now(self) -> str:
        """Run an extraction pass immediately - the Memory tab's button."""
        with self.conversation.lock:
            recent = list(self.conversation.turns[-12:])
        if not recent:
            return "Nothing to remember yet - have a conversation first."
        added, summarised = memory_mod.extract(
            self.memory, self.character, recent, llm.engine()
        )
        parts = [f"{added} new fact{'s' if added != 1 else ''}"]
        if summarised:
            parts.append("summary updated")
        return "Remembered: " + ", ".join(parts) + "."

    # ------------------------------------------------------------ character
    def use_character(self, character_id: str) -> str:
        if not character_id:
            return "No companion selected."
        try:
            self.character = character_mod.Character.load(character_id)
        except (OSError, ValueError) as exc:
            return f"Could not load '{character_id}': {exc}"
        config.update(active_character=character_id)
        self.memory = memory_mod.for_character(self.character.id)
        self._pending = []
        self.conversation = self._new_conversation(resume=True)
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
            finished = True
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
                        finished = False
                        break
                if finished and not voice.loop().interrupted.is_set():
                    self._played.append(item)
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
            reply_text = ""
            self._played = []
            try:
                interrupted = False
                for full, sentence in self.conversation.stream_reply(user_text):
                    reply_text = full
                    if not interrupted and voice.loop().interrupted.is_set():
                        # Spoken interruption: keep only what the user heard.
                        interrupted = True
                        self.conversation.stop(
                            spoken_only=True, spoken_text=" ".join(self._played)
                        )
                        self.silence()
                        self.status = "interrupted"
                    if interrupted:
                        # Deliberately not `break`: abandoning the generator
                        # skips the code that stores the turn, so the companion
                        # would remember saying nothing at all.  Draining it
                        # lets it finish and record what was actually heard -
                        # the underlying loop stops at the next chunk anyway.
                        continue
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
                # Distil the exchange into memory once the talking is over, so
                # the extra model call never sits between question and answer.
                if user_text.strip() and reply_text.strip():
                    threading.Thread(
                        target=self._remember, args=(user_text, reply_text),
                        daemon=True, name="companion-memory",
                    ).start()

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
