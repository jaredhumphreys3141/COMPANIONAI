"""CompanionAI - a fully local AI companion with voice and image generation.

Runs on a PC, a Raspberry Pi 5 or an NVIDIA Jetson.  Everything (LLM, speech
recognition, speech synthesis, image generation) executes on the device.  The
process never reaches the network unless the user explicitly approves it from
the GUI.
"""

from __future__ import annotations

import os

__version__ = "0.1.0"


def _silence_third_party_telemetry() -> None:
    """Turn off phone-home behaviour in our dependencies.

    This runs at import time, before Gradio / HuggingFace / Torch are loaded,
    because several of them read these variables once at import.  The network
    gate in :mod:`companionai.net` is the real enforcement point; this is the
    belt-and-braces layer that stops libraries chattering on their own.
    """
    for key, value in {
        "GRADIO_ANALYTICS_ENABLED": "False",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "BITSANDBYTES_NOWELCOME": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(key, value)


_silence_third_party_telemetry()

__all__ = ["__version__"]
