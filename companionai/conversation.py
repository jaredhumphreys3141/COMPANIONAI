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
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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

    def __init__(self, character) -> None:
        self.character = character
        self.turns: list[Turn] = []
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.started = time.strftime("%Y%m%d-%H%M%S")

    # ------------------------------------------------------------- history
    def reset(self) -> None:
        with self.lock:
            self.turns.clear()
            self.started = time.strftime("%Y%m%d-%H%M%S")

    def add(self, role: str, content: str) -> None:
        with self.lock:
            self.turns.append(Turn(role, content))

    def messages(self) -> list[dict]:
        """System prompt plus the most recent exchanges, within the memory window."""
        keep = max(2, self.character.memory_turns * 2)
        with self.lock:
            recent = self.turns[-keep:]
        return [{"role": "system", "content": self.character.system_prompt()}] + [
            {"role": turn.role, "content": turn.content} for turn in recent
        ]

    def as_chat_pairs(self) -> list[dict]:
        """Gradio ``messages``-format history."""
        with self.lock:
            return [{"role": turn.role, "content": turn.content} for turn in self.turns]

    # ---------------------------------------------------------------- reply
    def stop(self) -> None:
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
                yield full, sentence
            else:
                yield full, ""

        tail = buffer.strip()
        if tail and not self.stop_flag.is_set():
            yield full, tail

        cleaned = _tidy(full)
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
