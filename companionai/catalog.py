"""The model catalogue: what CompanionAI knows about, and what is installed.

Entries come from ``companionai/data/catalog.json`` plus an optional
``catalog.user.json`` in the data directory, so users can add their own models
without touching the code.  Nothing is fetched until :func:`install` is called,
and that goes through the network gate.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import hardware, net, paths

BUILTIN_CATALOG = Path(__file__).parent / "data" / "catalog.json"
USER_CATALOG = paths.HOME / "catalog.user.json"

KINDS = ("llm", "asr", "tts", "image")

_DEST_DIR = {
    "llm": paths.LLM_DIR,
    "asr": paths.ASR_DIR,
    "tts": paths.TTS_DIR,
    "image": paths.IMAGE_DIR,
}


@dataclass
class Entry:
    id: str
    kind: str            # llm | asr | tts | image
    name: str
    source: str          # gguf | hf-repo | piper | diffusers
    url: str = ""
    config_url: str = ""
    repo: str = ""
    filename: str = ""
    allow_patterns: list[str] = field(default_factory=list)
    size_mb: int = 0
    sha256: str = ""
    license: str = ""
    good_for: list[str] = field(default_factory=list)
    notes: str = ""
    resolution: int = 512
    steps: int = 4
    guidance: float = 0.0

    # -- where it lands on disk --------------------------------------------
    @property
    def path(self) -> Path:
        base = _DEST_DIR[self.kind]
        if self.source in ("gguf", "piper"):
            return base / (self.filename or f"{self.id}.bin")
        return base / self.id

    @property
    def installed(self) -> bool:
        path = self.path
        if not path.exists():
            return False
        if path.is_dir():
            return any(path.iterdir())
        if self.source == "piper":
            return path.with_suffix(path.suffix + ".json").exists()
        return path.stat().st_size > 0

    @property
    def size_label(self) -> str:
        if self.size_mb >= 1024:
            return f"{self.size_mb / 1024:.1f} GB"
        return f"{self.size_mb} MB"

    def recommended_here(self, device: str | None = None) -> bool:
        device = device or hardware.detect().device
        return device in self.good_for

    def as_row(self, device: str) -> list:
        return [
            "installed" if self.installed else "not installed",
            self.name,
            self.id,
            self.size_label,
            "yes" if self.recommended_here(device) else "",
            self.license,
            self.notes,
        ]


TABLE_HEADERS = ["state", "model", "id", "size", "suits this device", "license", "notes"]


def _entry_from_json(kind: str, raw: dict) -> Entry:
    return Entry(
        id=raw["id"],
        kind=kind,
        name=raw.get("name", raw["id"]),
        source=raw.get("kind", "gguf"),
        url=raw.get("url", ""),
        config_url=raw.get("config_url", ""),
        repo=raw.get("repo", ""),
        filename=raw.get("filename", ""),
        allow_patterns=raw.get("allow_patterns", []) or [],
        size_mb=int(raw.get("size_mb", 0)),
        sha256=raw.get("sha256", ""),
        license=raw.get("license", ""),
        good_for=raw.get("good_for", []) or [],
        notes=raw.get("notes", ""),
        resolution=int(raw.get("resolution", 512)),
        steps=int(raw.get("steps", 4)),
        guidance=float(raw.get("guidance", 0.0)),
    )


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load() -> dict[str, list[Entry]]:
    """All catalogue entries, built-ins first, user entries overriding by id."""
    merged: dict[str, dict[str, Entry]] = {kind: {} for kind in KINDS}
    for source in (BUILTIN_CATALOG, USER_CATALOG):
        data = _read(source)
        for kind in KINDS:
            for raw in data.get(kind, []) or []:
                try:
                    entry = _entry_from_json(kind, raw)
                except KeyError:
                    continue
                merged[kind][entry.id] = entry
    return {kind: list(items.values()) for kind, items in merged.items()}


def entries(kind: str) -> list[Entry]:
    return load().get(kind, [])


def get(kind: str, entry_id: str) -> Entry | None:
    for entry in entries(kind):
        if entry.id == entry_id:
            return entry
    return None


def installed(kind: str) -> list[Entry]:
    return [entry for entry in entries(kind) if entry.installed]


def choices(kind: str, only_installed: bool = False) -> list[str]:
    pool = installed(kind) if only_installed else entries(kind)
    return [entry.id for entry in pool]


def table(kind: str) -> list[list]:
    device = hardware.detect().device
    rows = [entry.as_row(device) for entry in entries(kind)]
    # Installed first, then models that suit this device, then the rest.
    rows.sort(key=lambda row: (row[0] != "installed", row[4] != "yes", row[1]))
    return rows


def local_gguf_files() -> list[str]:
    """GGUF files the user dropped into the models folder by hand."""
    paths.ensure_dirs()
    return sorted(p.name for p in paths.LLM_DIR.glob("*.gguf"))


def local_piper_voices() -> list[str]:
    paths.ensure_dirs()
    return sorted(p.name for p in paths.TTS_DIR.glob("*.onnx"))


# --------------------------------------------------------------------------- #
# install / remove
# --------------------------------------------------------------------------- #
ProgressFn = Callable[[float, str], None]


def install(kind: str, entry_id: str, progress: ProgressFn | None = None) -> str:
    """Download one catalogue entry.  Raises :class:`net.NetworkBlocked` if the
    user has not approved network access."""
    entry = get(kind, entry_id)
    if entry is None:
        raise ValueError(f"Unknown {kind} model: {entry_id}")
    if entry.installed:
        return f"{entry.name} is already installed."

    paths.ensure_dirs()
    free = net.disk_free_gb()
    needed = entry.size_mb / 1024
    if free < needed * 1.15:
        raise RuntimeError(
            f"Not enough free space: {entry.name} needs about {needed:.1f} GB, "
            f"{free} GB available."
        )

    def report(fraction: float, label: str) -> None:
        if progress:
            progress(fraction, f"{entry.name}: {label}")

    if entry.source in ("gguf", "piper"):
        net.download(
            entry.url, entry.path, sha256=entry.sha256 or None,
            progress=report, detail=f"install {entry.id}",
        )
        if entry.source == "piper" and entry.config_url:
            net.download(
                entry.config_url,
                entry.path.with_suffix(entry.path.suffix + ".json"),
                detail=f"install {entry.id} config",
            )
    else:  # hf-repo / diffusers
        report(0.0, "fetching repository ...")
        net.snapshot_download(
            entry.repo,
            entry.path,
            allow_patterns=entry.allow_patterns or None,
            detail=f"install {entry.id}",
        )

    report(1.0, "done")
    return f"Installed {entry.name} to {entry.path}"


def remove(kind: str, entry_id: str) -> str:
    entry = get(kind, entry_id)
    if entry is None:
        raise ValueError(f"Unknown {kind} model: {entry_id}")
    path = entry.path
    if not path.exists():
        return f"{entry.name} is not installed."
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".json").unlink(missing_ok=True)
    return f"Removed {entry.name}."


def missing_for(character) -> list[tuple[str, str]]:
    """Return ``(kind, id)`` pairs a character needs but that are not installed."""
    wanted: Iterable[tuple[str, str]] = (
        ("llm", character.llm_model),
        ("asr", character.asr_model),
        ("tts", character.voice),
        ("image", character.image_model),
    )
    out = []
    for kind, entry_id in wanted:
        if not entry_id:
            continue
        entry = get(kind, entry_id)
        if entry is not None and not entry.installed:
            out.append((kind, entry_id))
    return out
