"""The network gate.

CompanionAI is offline by default.  Every byte that leaves or enters the
machine has to pass through this module, which refuses unless the user has
explicitly approved network access in the GUI.  Each attempt - allowed or
blocked - is appended to an audit log the Settings tab displays, so the user
can see exactly what the app did.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import config, paths

AUDIT_FILE = paths.LOG_DIR / "network-audit.jsonl"
_USER_AGENT = "CompanionAI/0.1 (local; no telemetry)"

_lock = threading.RLock()
_recent: deque[dict] = deque(maxlen=200)
_session_override: bool = False   # set by "approve for this session"


class NetworkBlocked(RuntimeError):
    """Raised when code tries to use the network without user approval."""


@dataclass(frozen=True)
class Status:
    policy: str
    allowed: bool
    reason: str

    @property
    def badge(self) -> str:
        if not self.allowed:
            return "OFFLINE - no data can leave this device"
        if self.policy == config.ALWAYS:
            return "ONLINE - approved (remembered)"
        return "ONLINE - approved for this session only"


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
def status() -> Status:
    settings = config.get()
    policy = settings.network_policy
    allowed = policy == config.ALWAYS or (policy == config.SESSION and _session_override)
    if allowed:
        reason = "user approved" + (" (remembered)" if policy == config.ALWAYS else " for this session")
    else:
        reason = "offline by default"
    return Status(policy=policy, allowed=allowed, reason=reason)


def is_allowed() -> bool:
    return status().allowed


def approve(scope: str = config.SESSION, note: str = "") -> Status:
    """Grant network access.  ``scope`` is ``session`` or ``always``."""
    global _session_override
    if scope not in (config.SESSION, config.ALWAYS):
        raise ValueError(f"unknown scope {scope!r}")
    with _lock:
        _session_override = True
        config.update(network_policy=scope)
    _apply_env()
    _audit("approve", scope, note or "user granted network access from the GUI", True)
    return status()


def revoke(note: str = "") -> Status:
    """Return to fully offline operation."""
    global _session_override
    with _lock:
        _session_override = False
        config.update(network_policy=config.OFFLINE)
    _apply_env()
    _audit("revoke", "offline", note or "user revoked network access", True)
    return status()


def _apply_env() -> None:
    """Keep HuggingFace / Transformers offline switches in sync with policy."""
    offline = "0" if is_allowed() else "1"
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = offline
    os.environ.setdefault("HF_HOME", str(paths.CACHE_DIR / "huggingface"))
    endpoint = config.get().hf_endpoint.strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint


def init() -> Status:
    """Called once at start-up.  Session approvals never survive a restart."""
    global _session_override
    _session_override = False
    settings = config.get()
    if settings.network_policy == config.SESSION:
        # A session grant is deliberately not persistent: drop back to offline.
        config.update(network_policy=config.OFFLINE)
    _apply_env()
    return status()


# --------------------------------------------------------------------------- #
# audit trail
# --------------------------------------------------------------------------- #
def _audit(action: str, target: str, detail: str, allowed: bool) -> None:
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "target": target,
        "detail": detail,
        "allowed": allowed,
    }
    with _lock:
        _recent.append(entry)
        try:
            paths.LOG_DIR.mkdir(parents=True, exist_ok=True)
            with AUDIT_FILE.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # the audit log is best effort; never break the app over it


def audit_log(limit: int = 50) -> list[dict]:
    with _lock:
        entries = list(_recent)[-limit:]
    if entries or not AUDIT_FILE.exists():
        return list(reversed(entries))
    try:
        lines = AUDIT_FILE.read_text("utf-8").splitlines()[-limit:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))


def audit_markdown(limit: int = 25) -> str:
    entries = audit_log(limit)
    if not entries:
        return "_No network activity recorded._"
    rows = ["| time | action | target | result |", "| --- | --- | --- | --- |"]
    for entry in entries:
        result = "allowed" if entry.get("allowed") else "BLOCKED"
        rows.append(
            f"| {entry.get('time','')} | {entry.get('action','')} | "
            f"`{entry.get('target','')}` | {result} |"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# guarded access
# --------------------------------------------------------------------------- #
def host_of(url: str) -> str:
    return urllib.parse.urlparse(url).hostname or ""


def _host_permitted(host: str) -> bool:
    allow = config.get().allow_hosts
    if not allow:
        return True
    host = host.lower()
    return any(host == entry.lower() or host.endswith("." + entry.lower()) for entry in allow)


def require(target: str, detail: str = "") -> None:
    """Raise :class:`NetworkBlocked` unless network access is approved."""
    if not is_allowed():
        _audit("blocked", target, detail or "network access is not approved", False)
        raise NetworkBlocked(
            "Network access is turned off.  Open Settings -> Network and approve "
            "access before downloading anything."
        )
    host = host_of(target) if "://" in target else target
    if host and not _host_permitted(host):
        _audit("blocked", target, f"host {host!r} is not on the allow list", False)
        raise NetworkBlocked(
            f"'{host}' is not on the allowed-hosts list.  Add it in Settings -> Network."
        )
    _audit("allow", target, detail or "approved request", True)


@contextmanager
def guarded(target: str, detail: str = "") -> Iterator[None]:
    """Context manager form of :func:`require`."""
    require(target, detail)
    yield


# --------------------------------------------------------------------------- #
# downloads
# --------------------------------------------------------------------------- #
ProgressFn = Callable[[float, str], None]


def download(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    progress: ProgressFn | None = None,
    detail: str = "",
) -> Path:
    """Download ``url`` to ``dest`` after checking the network gate.

    Downloads to a ``.part`` file and renames on success, so an interrupted
    transfer never leaves a half-written model that looks valid.
    """
    require(url, detail or f"download {dest.name}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 18)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    fraction = done / total if total else 0.0
                    progress(fraction, f"{done / 1e6:.1f} MB" + (f" / {total / 1e6:.1f} MB" if total else ""))
    except urllib.error.URLError as exc:
        part.unlink(missing_ok=True)
        _audit("error", url, f"download failed: {exc}", True)
        raise RuntimeError(f"Download failed: {exc}") from exc

    if sha256 and digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        _audit("error", url, "checksum mismatch", True)
        raise RuntimeError("Downloaded file failed its checksum check; it was discarded.")

    part.replace(dest)
    _audit("downloaded", url, f"saved to {dest}", True)
    return dest


def snapshot_download(repo_id: str, dest: Path, *, allow_patterns: list[str] | None = None,
                      detail: str = "") -> Path:
    """Fetch a HuggingFace repo (used for diffusers and faster-whisper models)."""
    require(config.get().hf_endpoint, detail or f"download model repo {repo_id}")
    from huggingface_hub import snapshot_download as _snapshot  # noqa: PLC0415

    dest.mkdir(parents=True, exist_ok=True)
    path = _snapshot(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        cache_dir=str(paths.CACHE_DIR / "huggingface"),
    )
    _audit("downloaded", repo_id, f"saved to {path}", True)
    return Path(path)


# --------------------------------------------------------------------------- #
# interface sign-in
# --------------------------------------------------------------------------- #
def set_password(password: str, username: str = "") -> bool:
    """Set (or with an empty password, clear) the sign-in for the web UI."""
    settings = config.get()
    username = (username or settings.ui_username or "companion").strip()
    if not password:
        config.update(ui_password_hash="", ui_password_salt="", ui_username=username)
        _audit("auth", "interface", "sign-in disabled", True)
        return False
    salt = secrets.token_hex(16)
    config.update(
        ui_username=username,
        ui_password_salt=salt,
        ui_password_hash=_hash_password(password, salt),
    )
    _audit("auth", "interface", f"sign-in enabled for {username!r}", True)
    return True


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), 240_000
    ).hex()


def password_set() -> bool:
    settings = config.get()
    return bool(settings.ui_password_hash and settings.ui_password_salt)


def check_login(username: str, password: str) -> bool:
    """Gradio's auth callback.  Constant-time, so it leaks no timing signal."""
    settings = config.get()
    if not password_set():
        return True
    expected = _hash_password(password or "", settings.ui_password_salt)
    return (
        secrets.compare_digest(username or "", settings.ui_username)
        and secrets.compare_digest(expected, settings.ui_password_hash)
    )


def is_public_bind(host: str) -> bool:
    """True when this address exposes the interface beyond this machine."""
    return (host or "").strip() not in ("127.0.0.1", "localhost", "::1", "")


def disk_free_gb(path: Path | None = None) -> float:
    usage = shutil.disk_usage(str(path or paths.HOME))
    return round(usage.free / (1024 ** 3), 1)
