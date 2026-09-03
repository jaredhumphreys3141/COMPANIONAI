"""Keyword recall over everything the companion has ever been told.

Uses SQLite's FTS5 full-text index, which ships with Python's stdlib sqlite3 on
every target platform - no vector database, no embedding model, no extra
dependency.  It matches on words rather than meaning, which makes it strong on
names, numbers and dates and weak on paraphrase; the facts store in
:mod:`companionai.memory` covers the paraphrase case from the other direction.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time

from . import paths

INDEX_FILE = paths.HOME / "memory" / "recall.sqlite"

# Words too common to be worth searching on; without this, a query like "do you
# remember the thing" matches nearly every row.
STOPWORDS = {
    "a", "about", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "been", "but", "by", "can", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "is", "it", "its", "just", "me", "my", "no",
    "not", "of", "on", "or", "our", "out", "so", "that", "the", "their", "them",
    "then", "there", "they", "this", "to", "up", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
    "remember", "think", "know", "tell", "say", "said", "thing", "really",
}
_WORD = re.compile(r"[A-Za-z0-9']+")

_lock = threading.Lock()


class RecallIndex:
    """One FTS5 table shared by every companion, partitioned by character id."""

    def __init__(self, path=None) -> None:
        self.path = path or INDEX_FILE
        self._available: bool | None = None

    # ------------------------------------------------------------- plumbing
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @property
    def available(self) -> bool:
        """False when this Python's sqlite3 was built without FTS5."""
        if self._available is None:
            try:
                with self._connect() as connection:
                    connection.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS turns "
                        "USING fts5(character, role, text, stamp UNINDEXED)"
                    )
                self._available = True
            except sqlite3.Error:
                self._available = False
        return self._available

    # --------------------------------------------------------------- writes
    def add(self, character_id: str, role: str, text: str) -> None:
        text = (text or "").strip()
        if not text or not self.available:
            return
        with _lock:
            try:
                with self._connect() as connection:
                    connection.execute(
                        "INSERT INTO turns (character, role, text, stamp) VALUES (?, ?, ?, ?)",
                        (character_id, role, text, time.strftime("%Y-%m-%d %H:%M")),
                    )
            except sqlite3.Error:
                pass  # recall is a nicety; never break a conversation over it

    def clear(self, character_id: str) -> None:
        if not self.available:
            return
        with _lock:
            try:
                with self._connect() as connection:
                    connection.execute("DELETE FROM turns WHERE character = ?", (character_id,))
            except sqlite3.Error:
                pass

    def count(self, character_id: str) -> int:
        if not self.available:
            return 0
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT count(*) FROM turns WHERE character = ?", (character_id,)
                ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

    # -------------------------------------------------------------- reading
    @staticmethod
    def _to_match(query: str) -> str:
        """Turn free text into a safe FTS5 MATCH expression.

        Every token is quoted, so punctuation and FTS operators in the user's
        message can never be interpreted as query syntax.
        """
        words = [w.lower() for w in _WORD.findall(query or "")]
        useful = [w for w in words if len(w) > 2 and w not in STOPWORDS]
        if not useful:
            return ""
        return " OR ".join(f'"{w}"' for w in useful[:12])

    def search(self, character_id: str, query: str, limit: int = 3
               ) -> list[tuple[str, str, str]]:
        """Best-matching past lines as ``(stamp, role, text)``, most relevant first."""
        match = self._to_match(query)
        if not match or not self.available:
            return []
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT stamp, role, text FROM turns "
                    "WHERE character = ? AND turns MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (character_id, match, limit),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [tuple(r) for r in rows]


_index: RecallIndex | None = None


def index() -> RecallIndex:
    global _index
    if _index is None:
        _index = RecallIndex()
    return _index
