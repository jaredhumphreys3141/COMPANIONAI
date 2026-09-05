"""What a companion remembers between conversations.

Three layers, in increasing cost:

``facts``
    Short durable statements about the user, extracted from conversation by the
    same local model that does the talking.  Injected into every system prompt.
    This is what makes a companion feel like it knows you.

``summary``
    A rolling precis of everything older than the live context window, so a long
    relationship does not reduce to the last dozen messages.

recall
    Keyword search over every past line - see :mod:`companionai.recall`.

Extraction runs on a background thread after a reply has been spoken, never
before, because on a Raspberry Pi an extra model call costs seconds and the
whole point of the streaming pipeline is that the companion answers promptly.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field, fields

from . import character as character_mod
from . import config, paths, recall

MEMORY_DIR = paths.HOME / "memory"

_MAX_FACT_CHARS = 160
_WORD = re.compile(r"[A-Za-z0-9']+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")      # model output: * is a bullet
_USER_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s*")   # user editing: * means pinned


@dataclass
class Fact:
    text: str
    created: str = ""
    pinned: bool = False

    def key(self) -> str:
        """Normalised form used to spot duplicates."""
        return re.sub(r"[^a-z0-9 ]", "", self.text.lower()).strip()


@dataclass
class Memory:
    character_id: str = ""
    facts: list[Fact] = field(default_factory=list)
    summary: str = ""
    conversations: int = 0
    messages: int = 0
    first_seen: str = ""
    last_seen: str = ""
    schema: int = 1

    def __post_init__(self) -> None:
        # Extraction runs on background threads and a fast exchange can start
        # the next pass before the previous one has finished, so every mutation
        # is guarded.  Not a dataclass field, so it stays out of asdict().
        self._lock = threading.RLock()

    # ---------------------------------------------------------- persistence
    @property
    def file(self):
        return MEMORY_DIR / f"{self.character_id}.json"

    @classmethod
    def load(cls, character_id: str) -> Memory:
        path = MEMORY_DIR / f"{character_id}.json"
        try:
            raw = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(character_id=character_id, first_seen=time.strftime("%Y-%m-%d"))
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in raw.items() if k in known}
        data["character_id"] = character_id
        data["facts"] = [Fact(**f) for f in data.get("facts", []) if isinstance(f, dict)]
        return cls(**data)

    def save(self) -> Memory:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.last_seen = time.strftime("%Y-%m-%d")
            if not self.first_seen:
                self.first_seen = self.last_seen
            payload = json.dumps(asdict(self), indent=2)
        try:
            # Write-then-rename, so a crash mid-write cannot truncate memory.
            tmp = self.file.with_suffix(".json.tmp")
            tmp.write_text(payload, "utf-8")
            tmp.replace(self.file)
        except OSError:
            pass
        return self

    # --------------------------------------------------------------- facts
    def add_facts(self, texts) -> int:
        """Merge new facts in, skipping duplicates.  Returns how many were kept."""
        cap = max(4, config.get().memory_max_facts)
        with self._lock:
            existing = {fact.key() for fact in self.facts}
            added = 0
            for text in texts:
                text = text.strip()[:_MAX_FACT_CHARS]
                if not text:
                    continue
                candidate = Fact(text=text, created=time.strftime("%Y-%m-%d"))
                if candidate.key() in existing or not candidate.key():
                    continue
                self.facts.append(candidate)
                existing.add(candidate.key())
                added += 1
            # Oldest unpinned facts fall off the end first, so the prompt cannot
            # grow without limit as the relationship goes on.
            if len(self.facts) > cap:
                pinned = [f for f in self.facts if f.pinned]
                loose = [f for f in self.facts if not f.pinned]
                room = max(0, cap - len(pinned))
                # loose[-0:] would keep everything, so slice explicitly.
                self.facts = pinned + (loose[len(loose) - room:] if room else [])
            return added

    def as_text(self) -> str:
        """Facts as editable lines; a leading ``*`` marks a pinned fact."""
        return "\n".join(("* " if f.pinned else "") + f.text for f in self.facts)

    def set_from_text(self, text: str) -> Memory:
        """Replace the fact list from the Memory tab's editor."""
        facts = []
        for line in (text or "").splitlines():  # noqa: PLR1702
            line = line.strip()
            # Read the pin marker before stripping list bullets, or "*" gets
            # eaten as a bullet and the pin is silently lost.
            pinned = line.startswith("*")
            if pinned:
                line = line[1:].strip()
            line = _USER_BULLET.sub("", line).strip()
            if line:
                facts.append(Fact(text=line[:_MAX_FACT_CHARS],
                                  created=time.strftime("%Y-%m-%d"), pinned=pinned))
        with self._lock:
            self.facts = facts
        return self

    def forget(self) -> Memory:
        with self._lock:
            self.facts.clear()
            self.summary = ""
            self.conversations = 0
            self.messages = 0
        recall.index().clear(self.character_id)
        return self.save()

    # -------------------------------------------------------- prompt block
    def prompt_block(self, character, query: str = "", exclude: set | None = None) -> str:
        """The text appended to the system prompt for this turn."""
        settings = config.get()
        if not settings.memory_enabled:
            return ""

        parts = []
        if self.facts:
            # Pinned facts first: they are the ones the user chose to protect.
            ordered = [f for f in self.facts if f.pinned] + [f for f in self.facts if not f.pinned]
            lines = "\n".join(f"- {f.text}" for f in ordered[:settings.memory_max_facts])
            parts.append(f"What you know about {character.user_name}:\n{lines}")

        if self.summary.strip():
            parts.append(f"Earlier conversations, in brief: {self.summary.strip()}")

        if settings.recall_enabled and query.strip():
            hits = recall.index().search(
                self.character_id, query, limit=settings.recall_top_k + 2
            )
            # Drop anything already quoted verbatim in the live window.
            seen = {t.strip() for t in (exclude or set())}
            hits = [h for h in hits if h[2].strip() not in seen][:settings.recall_top_k]
            if hits:
                quoted = "\n".join(f'- ({stamp}) {role}: "{text}"' for stamp, role, text in hits)
                parts.append(
                    "Possibly relevant things from past conversations - mention them only "
                    f"if they genuinely fit:\n{quoted}"
                )

        if not parts:
            return ""
        return (
            "\n\n".join(parts)
            + "\n\nTreat the above as your own memory. Do not read it out as a list, "
            "and never claim to remember something that is not there."
        )

    def stats_markdown(self) -> str:
        return (
            f"- Facts remembered: **{len(self.facts)}** "
            f"({sum(1 for f in self.facts if f.pinned)} pinned)\n"
            f"- Conversations: **{self.conversations}**\n"
            f"- Messages exchanged: **{self.messages}**\n"
            f"- Lines searchable: **{recall.index().count(self.character_id)}**\n"
            f"- Known since: **{self.first_seen or 'today'}** "
            f"(last seen {self.last_seen or 'today'})"
        )


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
_FACT_SYSTEM = (
    "You extract durable facts about a person from a conversation. "
    "A durable fact is still true next week: names, relationships, work, where "
    "they live, preferences, plans, ongoing situations. "
    "Ignore small talk, greetings, and anything the assistant said about itself. "
    "Reply with one short fact per line and nothing else. "
    "If there are no durable facts, reply with exactly: NONE"
)

_SUMMARY_SYSTEM = (
    "You keep a running summary of a friendship. "
    "Given the previous summary and a recent conversation, write an updated "
    "summary of at most three sentences, in the third person, keeping only what "
    "matters for future conversations. Reply with the summary and nothing else."
)


def _tuned(character, max_tokens: int):
    """A copy of the character wired for extraction rather than conversation.

    Sampling only.  The model path, context size, threads and GPU layers are
    left alone so the already-loaded model is reused instead of reloaded, and
    the instructions travel as a system message rather than on the character:
    the engines pass the message list straight to the model and never read a
    character's prompt fields.
    """
    data = character.as_dict()
    data.update(
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        repeat_penalty=1.05,
        max_tokens=max_tokens,
        seed=0,
    )
    return character_mod.Character.from_dict(data)


def _content_words(text: str) -> set:
    """Meaningful words, for checking a fact against what was actually said.

    Filtering by stopword rather than by length matters: "cat", "dog", "job",
    "son" are exactly the short words durable facts hang on, and a length cut
    would throw those facts away.
    """
    return {
        w.lower() for w in _WORD.findall(text or "")
        if len(w) > 2 and w.lower() not in recall.STOPWORDS
    }


def _grounded(fact: str, source: str) -> bool:
    """Reject facts with no lexical footing in the conversation.

    Small models invent plausible-sounding details.  Requiring that a fact share
    at least one content word with what was actually said removes the worst of
    it without needing a second model call to verify.
    """
    words = _content_words(fact)
    return bool(words & _content_words(source)) if words else False


def _parse_facts(reply: str, source: str) -> list[str]:
    out = []
    for line in (reply or "").splitlines():
        line = _BULLET.sub("", line).strip().strip('"')
        if not line or line.upper().startswith("NONE"):
            continue
        if len(line) > _MAX_FACT_CHARS or len(line.split()) < 2:
            continue
        if line.endswith(":") or line.lower().startswith(("here are", "sure", "okay")):
            continue
        if not _grounded(line, source):
            continue
        out.append(line)
    return out[:5]


def _complete(engine, character, system: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        return "".join(engine.stream_chat(character, messages))
    except Exception:
        return ""  # a failed extraction must never disturb the conversation


def extract(memory: Memory, character, turns, engine) -> tuple[int, bool]:
    """Update ``memory`` from recent turns.  Returns ``(facts_added, summarised)``."""
    settings = config.get()
    if not settings.memory_enabled or not turns:
        return 0, False

    exchange = "\n".join(
        f"{'User' if t.role == 'user' else character.name}: {t.content}" for t in turns
    )
    user_said = "\n".join(t.content for t in turns if t.role == "user")

    added = 0
    reply = _complete(engine, _tuned(character, 160), _FACT_SYSTEM, exchange)
    if reply:
        # Ground facts against what the user actually said, not the whole
        # exchange, so the companion's own inventions cannot validate themselves.
        added = memory.add_facts(_parse_facts(reply, user_said or exchange))

    summarised = False
    if settings.memory_summarise:
        prompt = (
            f"Previous summary: {memory.summary or '(none yet)'}\n\n"
            f"Recent conversation:\n{exchange}"
        )
        new_summary = _complete(
            engine, _tuned(character, 180), _SUMMARY_SYSTEM, prompt
        ).strip()
        if new_summary and len(new_summary) > 20:
            memory.summary = new_summary[:800]
            summarised = True

    memory.save()
    return added, summarised


_memories: dict = {}
_lock = threading.Lock()


def for_character(character_id: str) -> Memory:
    with _lock:
        if character_id not in _memories:
            _memories[character_id] = Memory.load(character_id)
        return _memories[character_id]


def reset_cache() -> None:
    with _lock:
        _memories.clear()
