"""Character image generation with Diffusers.

The same pipeline object serves the portrait on the Character Studio tab and
free-form scenes on the Images tab.  Keeping the character's seed fixed is what
makes the same face come back shot after shot.
"""

from __future__ import annotations

import threading
import time

from . import catalog, hardware, net, paths


class ImageUnavailable(RuntimeError):
    pass


def _resolve(model_id: str) -> str:
    entry = catalog.get("image", model_id)
    if entry is not None and entry.installed:
        return str(entry.path)
    local = paths.IMAGE_DIR / model_id
    if local.is_dir() and any(local.iterdir()):
        return str(local)
    if entry is not None and net.is_allowed():
        return entry.repo  # diffusers will fetch it; the gate is open
    raise ImageUnavailable(
        f"Image model '{model_id}' is not installed.  Install it from the Models tab "
        "(that needs one-off network approval)."
    )


class ImageGenerator:
    def __init__(self) -> None:
        self._pipe = None
        self._key: str | None = None
        self._lock = threading.Lock()

    def load(self, model_id: str, progress=None) -> None:
        target = _resolve(model_id)
        with self._lock:
            if self._key == target and self._pipe is not None:
                return
            try:
                import torch  # noqa: PLC0415
                from diffusers import AutoPipelineForText2Image  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise ImageUnavailable(
                    "diffusers/torch are not installed.  Run scripts/install.sh with the "
                    "image extra, or `pip install 'companionai[image]'`."
                ) from exc

            info = hardware.detect()
            device = "cuda" if info.cuda else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            if progress:
                progress(0.1, f"loading {model_id} on {device} ...")

            self._pipe = None  # release VRAM before loading the replacement
            pipe = AutoPipelineForText2Image.from_pretrained(
                target,
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=not net.is_allowed(),
                cache_dir=str(paths.CACHE_DIR / "huggingface"),
            )
            pipe.set_progress_bar_config(disable=True)
            pipe = pipe.to(device)
            # These two keep an SDXL-class model inside a Jetson's shared memory
            # and make SD 1.5 fit on a Pi at all.
            pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            self._pipe = pipe
            self._key = target

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            self._key = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(
        self,
        character,
        scene: str = "",
        *,
        steps: int | None = None,
        guidance: float | None = None,
        size: int | None = None,
        seed: int | None = None,
        progress=None,
    ):
        """Render one image and return ``(PIL image, saved path, seed used)``."""
        import torch  # noqa: PLC0415

        self.load(character.image_model, progress=progress)
        info = hardware.detect()
        device = "cuda" if info.cuda else "cpu"

        steps = int(steps if steps is not None else character.image_steps)
        guidance = float(guidance if guidance is not None else character.image_guidance)
        size = int(size if size is not None else character.image_size)
        size = max(128, (size // 64) * 64)  # UNet needs a multiple of 64
        used_seed = int(seed if seed is not None else character.image_seed)
        if used_seed < 0:
            used_seed = int(time.time() * 1000) % (2 ** 31)

        generator = torch.Generator(device=device).manual_seed(used_seed)
        prompt = character.image_prompt(scene)
        kwargs = dict(
            prompt=prompt,
            num_inference_steps=max(1, steps),
            guidance_scale=guidance,
            width=size,
            height=size,
            generator=generator,
        )
        # Turbo pipelines run without classifier-free guidance, so a negative
        # prompt is meaningless (and errors on some versions).
        if guidance > 1.0 and character.negative_prompt:
            kwargs["negative_prompt"] = character.negative_prompt

        if progress:
            progress(0.4, f"rendering {size}x{size}, {steps} steps ...")
        image = self._pipe(**kwargs).images[0]

        paths.GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = paths.GALLERY_DIR / f"{character.id or 'companion'}-{stamp}-{used_seed}.png"
        image.save(out)
        (out.with_suffix(".txt")).write_text(
            f"prompt: {prompt}\nnegative: {character.negative_prompt}\n"
            f"model: {character.image_model}\nsteps: {steps}\nguidance: {guidance}\n"
            f"size: {size}\nseed: {used_seed}\n",
            "utf-8",
        )
        if progress:
            progress(1.0, "done")
        return image, out, used_seed


_generator: ImageGenerator | None = None
_lock = threading.Lock()


def generator() -> ImageGenerator:
    global _generator
    with _lock:
        if _generator is None:
            _generator = ImageGenerator()
        return _generator


def gallery(limit: int = 60) -> list[str]:
    paths.ensure_dirs()
    files = sorted(paths.GALLERY_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(path) for path in files[:limit]]


def estimate_seconds(character) -> float:
    """Rough wall-clock estimate, used to warn before a very slow render."""
    info = hardware.detect()
    pixels = (character.image_size / 512) ** 2
    per_step = {"jetson": 1.2, "raspberry-pi-5": 22.0, "pc": 0.35 if info.cuda else 9.0}
    return per_step.get(info.device, 5.0) * max(1, character.image_steps) * pixels
