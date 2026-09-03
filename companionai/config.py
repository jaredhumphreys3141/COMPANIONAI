"""Persisted application settings.

A single JSON file under the data directory.  Anything the user can change from
the Settings tab lives here; per-companion parameters live in the character
files instead.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field, fields

from . import paths

# Network policy values -------------------------------------------------------
OFFLINE = "offline"          # nothing may leave the machine (default)
SESSION = "session"          # approved until the app is restarted
ALWAYS = "always"            # approved and remembered across restarts
POLICIES = (OFFLINE, SESSION, ALWAYS)


@dataclass
class Settings:
    # -- network ------------------------------------------------------------
    network_policy: str = OFFLINE
    hf_endpoint: str = "https://huggingface.co"
    allow_hosts: list[str] = field(
        default_factory=lambda: ["huggingface.co", "cdn-lfs.huggingface.co", "github.com"]
    )

    # -- server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 7860
    open_browser: bool = True
    theme: str = "soft"

    # -- audio --------------------------------------------------------------
    input_device: str = ""       # empty means the system default
    output_device: str = ""
    mic_gain: float = 1.0
    vad_aggressiveness: int = 2  # 0 (permissive) .. 3 (strict), webrtcvad scale
    silence_ms: int = 700        # end-of-utterance silence before we transcribe
    min_speech_ms: int = 250     # ignore shorter blips (door slams, keyboard)
    barge_in: bool = True        # let the user interrupt the companion

    # -- runtime ------------------------------------------------------------
    active_character: str = ""
    save_transcripts: bool = True
    llm_backend: str = "llama.cpp"   # "llama.cpp" or "ollama"
    ollama_url: str = "http://127.0.0.1:11434"

    # -- upgrade hook -------------------------------------------------------
    schema: int = 1

    @classmethod
    def load(cls) -> Settings:
        paths.ensure_dirs()
        if paths.CONFIG_FILE.exists():
            try:
                raw = json.loads(paths.CONFIG_FILE.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
        else:
            raw = {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        paths.ensure_dirs()
        tmp = paths.CONFIG_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), "utf-8")
        tmp.replace(paths.CONFIG_FILE)

    def as_dict(self) -> dict:
        return asdict(self)


_lock = threading.Lock()
_settings: Settings | None = None


def get() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    with _lock:
        if _settings is None:
            _settings = Settings.load()
        return _settings


def update(**changes) -> Settings:
    """Apply changes to the singleton and persist them."""
    settings = get()
    known = {f.name for f in fields(Settings)}
    with _lock:
        for key, value in changes.items():
            if key in known:
                setattr(settings, key, value)
        settings.save()
    return settings
