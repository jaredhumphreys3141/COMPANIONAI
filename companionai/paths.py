"""Filesystem layout.

Everything CompanionAI writes lives under a single data directory so the app
can be moved to another machine (or an SD card) by copying one folder.
Override with the ``COMPANIONAI_HOME`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path


def _default_home() -> Path:
    env = os.environ.get("COMPANIONAI_HOME")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser() / "companionai"
    return Path.home() / ".companionai"


HOME = _default_home()

CHARACTERS_DIR = HOME / "characters"
MODELS_DIR = HOME / "models"
LLM_DIR = MODELS_DIR / "llm"
ASR_DIR = MODELS_DIR / "asr"
TTS_DIR = MODELS_DIR / "tts"
IMAGE_DIR = MODELS_DIR / "image"
GALLERY_DIR = HOME / "gallery"
CACHE_DIR = HOME / "cache"
LOG_DIR = HOME / "logs"
CONFIG_FILE = HOME / "settings.json"

ALL_DIRS = (
    HOME,
    CHARACTERS_DIR,
    MODELS_DIR,
    LLM_DIR,
    ASR_DIR,
    TTS_DIR,
    IMAGE_DIR,
    GALLERY_DIR,
    CACHE_DIR,
    LOG_DIR,
)


def ensure_dirs() -> None:
    """Create the data directory tree.  Safe to call repeatedly."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
