"""Speech recognition with faster-whisper (CTranslate2).

CTranslate2 is the reason this works across all three targets from one code
path: int8 on Raspberry Pi CPU, float16 on the Jetson iGPU or a desktop GPU.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

from . import catalog, hardware, net, paths

SAMPLE_RATE = 16000


class ASRUnavailable(RuntimeError):
    pass


def _resolve_model(model_id: str) -> str:
    """Local directory if the model is installed, otherwise the plain size name."""
    entry = catalog.get("asr", model_id)
    if entry is not None and entry.installed:
        return str(entry.path)
    local = paths.ASR_DIR / model_id
    if local.is_dir() and any(local.iterdir()):
        return str(local)
    if not net.is_allowed():
        raise ASRUnavailable(
            f"Speech model '{model_id}' is not installed.  Install it from the Models tab "
            "(that needs one-off network approval), or choose a model you already have."
        )
    return model_id  # faster-whisper will fetch it; the gate is already open


class Transcriber:
    """Holds one loaded Whisper model, reloading only when the choice changes."""

    def __init__(self) -> None:
        self._model = None
        self._key: tuple | None = None
        self._lock = threading.Lock()

    def load(self, character) -> None:
        info = hardware.detect()
        device = "cuda" if info.cuda else "cpu"
        compute_type = character.asr_compute_type or ("float16" if info.cuda else "int8")
        signature = (character.asr_model, device, compute_type)
        with self._lock:
            if self._key == signature and self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise ASRUnavailable(
                    "faster-whisper is not installed.  Run scripts/install.sh, or "
                    "`pip install faster-whisper`."
                ) from exc

            target = _resolve_model(character.asr_model)
            try:
                self._model = WhisperModel(
                    target,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(paths.ASR_DIR),
                    cpu_threads=max(1, min(info.cpu_count, 4)),
                )
            except (ValueError, RuntimeError):
                # Older CPUs and some Jetson builds reject float16/int8_float16.
                self._model = WhisperModel(
                    target,
                    device=device,
                    compute_type="int8" if device == "cpu" else "float32",
                    download_root=str(paths.ASR_DIR),
                )
            self._key = signature

    @property
    def ready(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._key = None

    # ---------------------------------------------------------------- decode
    def transcribe(self, audio, character) -> str:
        """``audio`` is a float32 mono numpy array at 16 kHz, or a path to a file."""
        self.load(character)
        if isinstance(audio, (str, Path)):
            audio = str(audio)
        else:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio.size == 0:
                return ""
        language = character.asr_language or None
        if language in ("", "auto"):
            language = None
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=max(1, character.asr_beam_size),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


_transcriber: Transcriber | None = None
_lock = threading.Lock()


def transcriber() -> Transcriber:
    global _transcriber
    with _lock:
        if _transcriber is None:
            _transcriber = Transcriber()
        return _transcriber


# --------------------------------------------------------------------------- #
# audio helpers shared with the voice loop and the browser upload path
# --------------------------------------------------------------------------- #
def load_wav(path: str | Path) -> np.ndarray:
    """Read any wav file as mono float32 at 16 kHz."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    data /= float(np.iinfo(dtype).max)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        data = resample(data, rate, SAMPLE_RATE)
    return data


def resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resampling - good enough for 16 kHz speech and dependency free."""
    if source_rate == target_rate or data.size == 0:
        return data
    duration = data.size / source_rate
    target_length = int(duration * target_rate)
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    positions = np.linspace(0, data.size - 1, target_length, dtype=np.float32)
    return np.interp(positions, np.arange(data.size), data).astype(np.float32)


def from_gradio(value) -> np.ndarray:
    """Normalise whatever a Gradio audio component hands us into 16 kHz float32."""
    if value is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(value, (str, Path)):
        return load_wav(value)
    rate, data = value
    data = np.asarray(data)
    if data.dtype.kind in "iu":
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return resample(data, rate, SAMPLE_RATE)
