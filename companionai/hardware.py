"""Device detection and per-platform defaults.

CompanionAI targets three very different machines, so instead of asking the
user to tune a dozen knobs we detect the board and pick a profile.  Every value
a profile suggests remains editable in the GUI.
"""

from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

PC = "pc"
RPI5 = "raspberry-pi-5"
JETSON = "jetson"


@dataclass
class Profile:
    """Suggested defaults for a class of machine."""

    device: str
    label: str
    torch_device: str = "cpu"
    llm_model: str = "qwen2.5-1.5b-instruct-q4"
    llm_threads: int = 4
    llm_gpu_layers: int = 0
    llm_context: int = 4096
    asr_model: str = "base.en"
    asr_compute_type: str = "int8"
    image_model: str = "sd-turbo"
    image_steps: int = 4
    image_size: int = 512
    notes: str = ""


@dataclass
class HardwareInfo:
    device: str
    label: str
    machine: str
    system: str
    cpu_count: int
    total_ram_gb: float
    model_string: str = ""
    cuda: bool = False
    cuda_name: str = ""
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _device_tree_model() -> str:
    for candidate in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            return Path(candidate).read_bytes().decode("utf-8", "ignore").strip("\x00 \n")
        except OSError:
            continue
    return ""


def _total_ram_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024 ** 3), 1)
    except (ValueError, OSError, AttributeError):
        return 0.0


def _is_jetson(model_string: str) -> bool:
    if "jetson" in model_string.lower() or "tegra" in model_string.lower():
        return True
    return Path("/etc/nv_tegra_release").exists()


def _is_rpi(model_string: str) -> bool:
    return "raspberry pi" in model_string.lower()


def _cuda_available() -> tuple[bool, str]:
    """Check for a usable CUDA GPU without importing torch (which is slow)."""
    try:
        import torch  # noqa: PLC0415  - optional, imported only if already installed

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch missing or broken install
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return True, out.stdout.strip().splitlines()[0]
        except Exception:  # pragma: no cover
            pass
    # Jetson exposes the iGPU without nvidia-smi on older JetPack releases.
    if Path("/dev/nvhost-gpu").exists() or Path("/dev/nvgpu").exists():
        return True, "NVIDIA Tegra iGPU"
    return False, ""


@functools.lru_cache(maxsize=1)
def detect() -> HardwareInfo:
    """Identify the machine we are running on."""
    model_string = _device_tree_model()
    cuda, cuda_name = _cuda_available()

    if _is_jetson(model_string):
        device, label = JETSON, model_string or "NVIDIA Jetson"
    elif _is_rpi(model_string):
        device, label = RPI5, model_string or "Raspberry Pi"
    else:
        device, label = PC, f"{platform.system()} {platform.machine()}"

    return HardwareInfo(
        device=device,
        label=label,
        machine=platform.machine(),
        system=platform.system(),
        cpu_count=os.cpu_count() or 1,
        total_ram_gb=_total_ram_gb(),
        model_string=model_string,
        cuda=cuda,
        cuda_name=cuda_name,
        details={"python": platform.python_version(), "release": platform.release()},
    )


def profile_for(info: HardwareInfo | None = None) -> Profile:
    """Return sensible defaults for the detected (or supplied) hardware."""
    info = info or detect()
    threads = max(1, min(info.cpu_count, 8))

    if info.device == JETSON:
        return Profile(
            device=JETSON,
            label=info.label,
            torch_device="cuda" if info.cuda else "cpu",
            llm_model="qwen2.5-3b-instruct-q4",
            llm_threads=threads,
            # -1 offloads every layer to the iGPU; this is what makes
            # conversational latency possible on the Jetson.
            llm_gpu_layers=-1 if info.cuda else 0,
            llm_context=4096,
            asr_model="small.en",
            asr_compute_type="float16" if info.cuda else "int8",
            image_model="sd-turbo",
            image_steps=4,
            image_size=512,
            notes="CUDA offload enabled: expect real-time speech in and out.",
        )

    if info.device == RPI5:
        return Profile(
            device=RPI5,
            label=info.label,
            torch_device="cpu",
            llm_model="qwen2.5-1.5b-instruct-q4",
            llm_threads=min(threads, 4),
            llm_gpu_layers=0,
            llm_context=2048,
            asr_model="tiny.en",
            asr_compute_type="int8",
            image_model="sd-turbo",
            image_steps=2,
            image_size=384,
            notes="CPU only: replies stream within a few seconds, images take minutes.",
        )

    return Profile(
        device=PC,
        label=info.label,
        torch_device="cuda" if info.cuda else "cpu",
        llm_model="qwen2.5-3b-instruct-q4",
        llm_threads=threads,
        llm_gpu_layers=-1 if info.cuda else 0,
        llm_context=8192,
        asr_model="base.en" if not info.cuda else "small.en",
        asr_compute_type="float16" if info.cuda else "int8",
        image_model="sd-turbo",
        image_steps=4,
        image_size=512,
        notes="Desktop profile." + ("" if info.cuda else "  No GPU found, running on CPU."),
    )


def summary() -> str:
    """One-paragraph human readable description, shown on the System tab."""
    info = detect()
    prof = profile_for(info)
    gpu = f"{info.cuda_name} (CUDA)" if info.cuda else "none detected"
    return (
        f"**{info.label}**\n\n"
        f"- Platform: {info.system} / {info.machine}\n"
        f"- CPU cores: {info.cpu_count}\n"
        f"- RAM: {info.total_ram_gb} GB\n"
        f"- GPU: {gpu}\n"
        f"- Profile: `{prof.device}` - {prof.notes}"
    )
