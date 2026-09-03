"""Conversation state and the streaming reply pipeline.

The key trick for conversational latency: as tokens arrive we cut the reply at
sentence boundaries and hand each finished sentence to text-to-speech straight
away, so the companion starts speaking while the language model is still
writing.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import config, llm, paths

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…])$|\n{2,}")
_MIN_SPEAKABLE = 24   # characters: below this we wait for more text


@dataclass
class Turn:
    role: str
    content: str
    time: str = field(default_factory=lambda: time.strftime("%H:%M:%S"))


class Conversation:
    """History plus the reply generator for one companion."""

    def __init__(self, character, memory=None) -> None:
        self.character = character
        self.memory = memory
        self.turns: list[Turn] = []
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.started = time.strftime("%Y%m%d-%H%M%S")
        self.resumed = False
        # Text actually handed to the voice, as distinct from text generated.
        # They differ the moment the user interrupts.
        self.dispatched = ""
        self._spoken_text: str | None = None
        self._keep_spoken_only = False

    # ------------------------------------------------------------- history
    def reset(self) -> None:
        with self.lock:
            self.turns.clear()
            self.started = time.strftime("%Y%m%d-%H%M%S")

    def add(self, role: str, content: str) -> None:
        with self.lock:
            self.turns.append(Turn(role, content))

    def messages(self) -> list[dict]:
        """System prompt plus the most recent exchanges, within the memory window.

        Long-term memory rides in the system prompt: facts, the rolling summary
        and any recalled lines relevant to what was just said.  Only the last
        ``memory_turns`` exchanges appear verbatim, so the prompt stays a fixed
        size however long the relationship runs.
        """
        keep = max(2, self.character.memory_turns * 2)
        with self.lock:
            recent = self.turns[-keep:]
            latest_user = next(
                (t.content for t in reversed(self.turns) if t.role == "user"), ""
            )

        system = self.character.system_prompt()
        if self.memory is not None:
            block = self.memory.prompt_block(
                self.character, latest_user, exclude={t.content for t in recent}
            )
            if block:
                system = f"{system}\n\n{block}"

        return [{"role": "system", "content": system}] + [
            {"role": turn.role, "content": turn.content} for turn in recent
        ]

    # ------------------------------------------------------------- resuming
    def latest_transcript(self):
        """Newest saved transcript for this companion, or None."""
        folder = paths.HOME / "transcripts"
        if not folder.is_dir():
            return None
        files = sorted(
            folder.glob(f"{self.character.id}-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files[0] if files else None

    def resume(self) -> int:
        """Reload the most recent conversation.  Returns how many turns came back."""
        path = self.latest_transcript()
        if path is None:
            return 0
        try:
            data = json.loads(path.read_text("utf-8"))
            saved = data.get("turns", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            return 0
        restored = [
            Turn(t["role"], t["content"], t.get("time", ""))
            for t in saved
            if isinstance(t, dict) and t.get("role") in ("user", "assistant") and t.get("content")
        ]
        if not restored:
            return 0
        with self.lock:
            self.turns = restored
            # Keep writing to the same file rather than orphaning it.
            self.started = data.get("started") or path.stem.split("-", 1)[-1]
            self.resumed = True
        return len(restored)

    def as_chat_pairs(self) -> list[dict]:
        """Gradio ``messages``-format history."""
        with self.lock:
            return [{"role": turn.role, "content": turn.content} for turn in self.turns]

    # ---------------------------------------------------------------- reply
    def stop(self, spoken_only: bool = False, spoken_text: str | None = None) -> None:
        """Halt generation.

        ``spoken_only`` records that the user cut the companion off out loud, so
        only what they actually heard is kept.  A Stop button press in a text
        conversation keeps the whole reply, because the user read all of it.

        ``spoken_text`` is what the speaker finished playing.  The caller has to
        supply it: sentences handed to the speech queue are dropped when the
        queue is flushed on a barge-in, so "dispatched" over-counts what was
        heard by however much the model ran ahead of the voice.
        """
        self._keep_spoken_only = spoken_only
        if spoken_text is not None:
            self._spoken_text = spoken_text
        self.stop_flag.set()
        engine = llm.engine()
        if hasattr(engine, "cancel"):
            engine.cancel.set()

    def stream_reply(self, user_text: str) -> Iterator[tuple[str, str]]:
        """Yield ``(full_text_so_far, newly_completed_sentence)``.

        ``newly_completed_sentence`` is empty on most yields and holds a chunk of
        text ready to be spoken whenever a sentence boundary is crossed.
        """
        self.stop_flag.clear()
        self._keep_spoken_only = False
        self._spoken_text = None
        self.dispatched = ""
        if user_text.strip():
            self.add("user", user_text.strip())

        engine = llm.engine()
        buffer = ""
        full = ""

        for piece in engine.stream_chat(self.character, self.messages()):
            if self.stop_flag.is_set():
                break
            full += piece
            buffer += piece
            sentence, buffer = _split_ready(buffer)
            if sentence:
                self.dispatched += (" " if self.dispatched else "") + sentence
                yield full, sentence
            else:
                yield full, ""

        tail = buffer.strip()
        if tail and not self.stop_flag.is_set():
            self.dispatched += (" " if self.dispatched else "") + tail
            yield full, tail

        # Storing text the user never heard would have the companion believe it
        # said things it did not - and that belief then feeds context, the
        # transcript and memory extraction.
        if self._keep_spoken_only:
            # What the speaker finished, when the caller knows; otherwise the
            # sentences handed out, which is still far closer than everything
            # the model managed to generate.
            heard = self._spoken_text if self._spoken_text is not None else self.dispatched
        else:
            heard = full
        cleaned = _tidy(heard)
        if cleaned:
            self.add("assistant", cleaned)
            if config.get().save_transcripts:
                self.save_transcript()

    # ----------------------------------------------------------- transcript
    def transcript_path(self) -> Path:
        folder = paths.HOME / "transcripts"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{self.character.id or 'companion'}-{self.started}.json"

    def save_transcript(self) -> None:
        try:
            with self.lock:
                data = {
                    "character": self.character.id,
                    "started": self.started,
                    "turns": [{"role": t.role, "content": t.content, "time": t.time} for t in self.turns],
                }
            self.transcript_path().write_text(json.dumps(data, indent=2), "utf-8")
        except OSError:
            pass


def _split_ready(buffer: str) -> tuple[str, str]:
    """Peel off one complete, long-enough sentence from the buffer."""
    match = _SENTENCE_END.search(buffer)
    if not match:
        return "", buffer
    head, tail = buffer[: match.end()], buffer[match.end():]
    if len(head.strip()) < _MIN_SPEAKABLE:
        # Too short to be worth a separate utterance ("Sure." / "Hm.") unless it
        # is all we are going to get; keep accumulating.
        second = _SENTENCE_END.search(tail)
        if second:
            head, tail = buffer[: match.end() + second.end()], buffer[match.end() + second.end():]
        else:
            return "", buffer
    return head.strip(), tail


_MARKUP = re.compile(r"[*_`#>]+")
_STAGE = re.compile(r"\*[^*]{0,80}\*")


def _tidy(text: str) -> str:
    return text.strip()


def speakable(text: str) -> str:
    """Strip anything a voice should not read out loud."""
    text = _STAGE.sub("", text)
    text = _MARKUP.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


SpeakFn = Callable[[str], None]
