"""Language-model backends.

Two backends, both entirely local:

``llama.cpp``
    ``llama-cpp-python`` loading a GGUF file.  Works on every target: pure CPU
    on a Raspberry Pi 5, CUDA-offloaded on a Jetson or a desktop GPU.

``ollama``
    Talks to an Ollama daemon on localhost.  Useful if the user already
    maintains their models there.  Still local - nothing leaves the machine.

Both expose the same ``stream_chat`` generator so the rest of the app does not
care which is in use.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

from . import catalog, config, paths

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class ModelNotAvailable(RuntimeError):
    """The requested model is not installed, or its runtime is missing."""


def resolve_gguf(model_id: str):
    """Map a catalogue id or a bare filename to a GGUF path on disk."""
    if not model_id:
        raise ModelNotAvailable("No language model has been chosen for this companion.")
    entry = catalog.get("llm", model_id)
    if entry is not None:
        if entry.installed:
            return entry.path
        raise ModelNotAvailable(
            f"'{entry.name}' is not installed yet.  Open the Models tab and install it."
        )
    direct = paths.LLM_DIR / model_id
    if direct.exists():
        return direct
    raise ModelNotAvailable(
        f"No GGUF file named '{model_id}' in {paths.LLM_DIR}.  Install a model from the "
        "Models tab, or copy a .gguf file into that folder."
    )


class LlamaCppEngine:
    """A loaded GGUF model.  Reloads only when the relevant settings change."""

    def __init__(self) -> None:
        self._llm = None
        self._key: tuple | None = None
        self._lock = threading.Lock()
        # One llama.cpp context is not safe to generate from concurrently, and
        # memory extraction runs on a background thread while the user may be
        # sending the next message.  Generations queue instead of colliding.
        self._gen_lock = threading.RLock()
        self.cancel = threading.Event()

    # ------------------------------------------------------------------ load
    def _signature(self, character) -> tuple:
        return (
            str(resolve_gguf(character.llm_model)),
            character.context_size,
            character.threads,
            character.gpu_layers,
        )

    def load(self, character) -> None:
        signature = self._signature(character)
        with self._lock:
            if self._key == signature and self._llm is not None:
                return
            try:
                from llama_cpp import Llama  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ModelNotAvailable(
                    "llama-cpp-python is not installed.  Run scripts/install.sh, or "
                    "`pip install llama-cpp-python`."
                ) from exc

            self._llm = None  # free the previous model before allocating a new one
            self._llm = Llama(
                model_path=signature[0],
                n_ctx=character.context_size,
                n_threads=character.threads,
                n_gpu_layers=character.gpu_layers,
                verbose=False,
                # Keeping the prompt cache warm is what makes turn 2 onwards feel
                # instant in a voice conversation.
                use_mlock=False,
            )
            self._key = signature

    def unload(self) -> None:
        with self._lock:
            self._llm = None
            self._key = None

    @property
    def ready(self) -> bool:
        return self._llm is not None

    @property
    def model_path(self) -> str:
        return self._key[0] if self._key else ""

    # ------------------------------------------------------------------ chat
    def stream_chat(self, character, messages: list[Message]) -> Iterator[str]:
        self.load(character)
        self.cancel.clear()
        kwargs = dict(
            messages=messages,
            temperature=character.temperature,
            top_p=character.top_p,
            top_k=character.top_k,
            repeat_penalty=character.repeat_penalty,
            max_tokens=character.max_tokens,
            stream=True,
        )
        if character.seed is not None and character.seed >= 0:
            kwargs["seed"] = character.seed
        # Held for the life of the generator; released when it finishes or the
        # caller closes it.
        with self._gen_lock:
            for chunk in self._llm.create_chat_completion(**kwargs):
                if self.cancel.is_set():
                    break
                delta = chunk["choices"][0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    yield piece


class OllamaEngine:
    """Thin client for a locally running Ollama daemon."""

    def __init__(self) -> None:
        self.cancel = threading.Event()

    @property
    def base_url(self) -> str:
        return config.get().ollama_url.rstrip("/")

    def _post(self, route: str, payload: dict, stream: bool = False):
        request = urllib.request.Request(
            f"{self.base_url}{route}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=None if stream else 15)

    def available_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as response:
                data = json.loads(response.read())
            return [model["name"] for model in data.get("models", [])]
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
            return []

    def load(self, character) -> None:
        if not self.available_models():
            raise ModelNotAvailable(
                f"No Ollama daemon answered at {self.base_url}.  Start it with `ollama serve`, "
                "or switch the backend back to llama.cpp in Settings."
            )

    @property
    def ready(self) -> bool:
        return bool(self.available_models())

    @property
    def model_path(self) -> str:
        return self.base_url

    def unload(self) -> None:
        return

    def stream_chat(self, character, messages: list[Message]) -> Iterator[str]:
        self.cancel.clear()
        payload = {
            "model": character.llm_model or "llama3.2",
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": character.temperature,
                "top_p": character.top_p,
                "top_k": character.top_k,
                "repeat_penalty": character.repeat_penalty,
                "num_predict": character.max_tokens,
                "num_ctx": character.context_size,
            },
        }
        try:
            response = self._post("/api/chat", payload, stream=True)
        except (urllib.error.URLError, OSError) as exc:
            raise ModelNotAvailable(f"Could not reach Ollama: {exc}") from exc
        with response:
            for line in response:
                if self.cancel.is_set():
                    break
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = data.get("message", {}).get("content")
                if piece:
                    yield piece
                if data.get("done"):
                    break


_engines: dict[str, object] = {}
_engine_lock = threading.Lock()


def engine(backend: str | None = None):
    """Return the shared engine for the configured backend."""
    backend = backend or config.get().llm_backend
    with _engine_lock:
        if backend not in _engines:
            _engines[backend] = OllamaEngine() if backend == "ollama" else LlamaCppEngine()
        return _engines[backend]
