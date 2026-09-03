"""Command line entry point: ``python -m companionai`` or ``companionai``."""

from __future__ import annotations

import argparse
import sys

from . import config, hardware, net, paths


def _doctor() -> int:
    """Report what is installed and what is missing, without touching the network."""
    from . import catalog, session

    print(hardware.summary().replace("**", ""))
    print()

    checks = [
        ("gradio", "the graphical interface"),
        ("numpy", "audio handling"),
        ("llama_cpp", "language models (llama.cpp)"),
        ("faster_whisper", "speech recognition"),
        ("piper", "speech synthesis"),
        ("sounddevice", "microphone and speakers"),
        ("webrtcvad", "voice activity detection (optional)"),
        ("diffusers", "image generation (optional)"),
        ("torch", "image generation (optional)"),
    ]
    print("Python packages")
    missing_required = False
    for module, why in checks:
        try:
            __import__(module)
            state = "ok     "
        except ImportError:
            state = "MISSING"
            if module in ("gradio", "numpy"):
                missing_required = True
        print(f"  {state}  {module:<16} {why}")

    print("\nInstalled models")
    for kind in catalog.KINDS:
        installed = catalog.installed(kind)
        names = ", ".join(entry.id for entry in installed) or "(none)"
        print(f"  {kind:<6} {names}")

    print(f"\nData directory: {paths.HOME}")
    print(f"Network policy: {net.status().badge}")

    state = session.get()
    print(f"\nActive companion: {state.character.name} ({state.character.id})")
    for row in state.readiness():
        print(f"  {row[0]:<22} {row[1]:<15} {row[2]}")
    return 1 if missing_required else 0


def _list_devices() -> int:
    from . import voice

    inputs, outputs = voice.list_devices()
    if not inputs and not outputs:
        print("No audio devices found (is sounddevice installed?).")
        return 1
    print("Microphones:")
    for name in inputs:
        print(f"  {name}")
    print("Speakers:")
    for name in outputs:
        print(f"  {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="companionai",
        description="A local AI companion: voice in, voice out, pictures included.",
    )
    parser.add_argument("--host", help="bind address (default from settings)")
    parser.add_argument("--port", type=int, help="port (default from settings)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window on start")
    parser.add_argument("--share", action="store_true",
                        help="create a public Gradio tunnel (off by default; this "
                             "sends traffic through Gradio's servers)")
    parser.add_argument("--doctor", action="store_true",
                        help="print an installation and readiness report, then exit")
    parser.add_argument("--list-devices", action="store_true",
                        help="list audio devices, then exit")
    parser.add_argument("--home", help="use this directory for models and settings")
    args = parser.parse_args(argv)

    if args.home:
        import os

        os.environ["COMPANIONAI_HOME"] = args.home
        # paths were resolved at import; re-resolve so --home takes effect.
        from importlib import reload

        reload(paths)

    paths.ensure_dirs()
    net.init()

    if args.doctor:
        return _doctor()
    if args.list_devices:
        return _list_devices()

    if args.share:
        print(
            "WARNING: --share opens a public tunnel through Gradio's servers.  Your "
            "conversation stays on this device, but the web interface becomes reachable "
            "from the internet.  Press Ctrl+C now if that is not what you want.",
            file=sys.stderr,
        )

    from .ui import launch

    settings = config.get()
    host = args.host or settings.host
    port = args.port or settings.port
    print(f"CompanionAI - http://{host}:{port}   ({net.status().badge})")
    launch(host=host, port=port, share=args.share,
           open_browser=False if args.no_browser else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
