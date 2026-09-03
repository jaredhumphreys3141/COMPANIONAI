"""End-to-end check against real models.

Everything else in the test suite runs against stand-in engines, which proves
the plumbing but never that ``llama-cpp-python``, ``piper-tts`` and
``faster-whisper`` actually work on this machine.  This does:

    1. install the smallest usable model of each kind (network approval needed)
    2. generate a reply with the language model
    3. speak it with Piper
    4. transcribe that audio back with Whisper
    5. check the round trip recovered the words

Step 4 is the trick: feeding synthesised speech straight back into speech
recognition exercises both halves of the voice pipeline without a microphone
or a speaker, so it works on a headless box and in CI.

    companionai --smoke-test                  # models must already be installed
    companionai --smoke-test --allow-network  # fetch what is missing first
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import catalog, config, net

# The smallest entries in the catalogue: about half a gigabyte all told.
SMOKE_MODELS = {
    "llm": "qwen2.5-0.5b-instruct-q4",
    "tts": "en_US-amy-low",
    "asr": "tiny.en",
}

PROMPT = "Say hello and tell me the sky is blue. Two short sentences."


@dataclass
class Stage:
    name: str
    ok: bool = False
    detail: str = ""
    seconds: float = 0.0

    def line(self) -> str:
        mark = "pass" if self.ok else "FAIL"
        return f"  [{mark}] {self.name:<26} {self.seconds:6.1f}s  {self.detail}"


@dataclass
class Result:
    stages: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(stage.ok for stage in self.stages)

    def report(self) -> str:
        lines = [stage.line() for stage in self.stages]
        lines.append("")
        lines.append("Smoke test PASSED" if self.ok else "Smoke test FAILED")
        return "\n".join(lines)


def _timed(result: Result, name: str):
    stage = Stage(name)
    result.stages.append(stage)
    stage._start = time.time()  # noqa: SLF001 - local bookkeeping
    return stage


def _done(stage: Stage, ok: bool, detail: str = "") -> bool:
    stage.seconds = time.time() - stage._start
    stage.ok = ok
    stage.detail = detail
    return ok


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 2}


def _overlap(said: str, heard: str) -> float:
    """Fraction of the spoken words that came back from transcription."""
    spoken, recovered = _words(said), _words(heard)
    if not spoken:
        return 0.0
    return len(spoken & recovered) / len(spoken)


def run(allow_network: bool = False, keep_audio: bool = False) -> Result:
    """Run the whole pipeline once and report on each stage."""
    from . import asr, llm, paths, tts
    from . import character as character_mod

    result = Result()

    # -- 0. models ----------------------------------------------------------
    stage = _timed(result, "models available")
    missing = [
        (kind, name) for kind, name in SMOKE_MODELS.items()
        if catalog.get(kind, name) is None or not catalog.get(kind, name).installed
    ]
    if missing:
        if not allow_network:
            _done(stage, False,
                  "missing " + ", ".join(f"{k}:{v}" for k, v in missing)
                  + " - re-run with --allow-network to fetch them")
            return result
        net.approve(config.SESSION, "smoke test model download")
        try:
            for kind, name in missing:
                catalog.install(kind, name)
        except Exception as exc:
            _done(stage, False, f"download failed: {exc}")
            return result
        finally:
            net.revoke("smoke test finished")
    _done(stage, True, ", ".join(f"{k}:{v}" for k, v in SMOKE_MODELS.items()))

    subject = character_mod.Character(
        id="smoke-test", name="Smoke", llm_model=SMOKE_MODELS["llm"],
        voice=SMOKE_MODELS["tts"], asr_model=SMOKE_MODELS["asr"],
        max_tokens=48, temperature=0.3, seed=1, context_size=1024,
    )

    # -- 1. language model --------------------------------------------------
    stage = _timed(result, "language model replies")
    try:
        engine = llm.engine("llama.cpp")
        reply = "".join(engine.stream_chat(subject, [{"role": "user", "content": PROMPT}]))
    except Exception as exc:
        _done(stage, False, str(exc)[:120])
        return result
    if not reply.strip():
        _done(stage, False, "the model produced no text")
        return result
    _done(stage, True, f"{len(reply.split())} words: {reply.strip()[:60]!r}")

    # -- 2. speech synthesis ------------------------------------------------
    stage = _timed(result, "speech synthesis")
    spoken = "Hello there. The sky is blue today."
    try:
        speaker = tts.speaker(subject)
        rate, audio = speaker.synth(spoken, subject)
    except Exception as exc:
        _done(stage, False, str(exc)[:120])
        return result
    if audio.size == 0:
        _done(stage, False, "no audio produced")
        return result
    seconds = audio.size / rate
    wav_path = paths.CACHE_DIR / "smoke-test.wav"
    tts.save_wav(wav_path, rate, audio)
    _done(stage, True, f"{seconds:.1f}s of audio at {rate} Hz")

    # -- 3. speech recognition on our own audio -----------------------------
    stage = _timed(result, "speech recognition")
    try:
        samples = asr.load_wav(wav_path)
        heard = asr.transcriber().transcribe(samples, subject)
    except Exception as exc:
        _done(stage, False, str(exc)[:120])
        return result
    finally:
        if not keep_audio:
            wav_path.unlink(missing_ok=True)
    if not heard.strip():
        _done(stage, False, "nothing transcribed")
        return result
    _done(stage, True, repr(heard.strip()[:60]))

    # -- 4. did the round trip survive? -------------------------------------
    stage = _timed(result, "voice round trip")
    score = _overlap(spoken, heard)
    _done(stage, score >= 0.6,
          f"{score:.0%} of spoken words recovered (need 60%)")

    return result


def main(allow_network: bool = False, keep_audio: bool = False) -> int:
    from . import hardware, paths

    paths.ensure_dirs()
    print(f"CompanionAI smoke test on {hardware.detect().label}\n")
    result = run(allow_network=allow_network, keep_audio=keep_audio)
    print(result.report())
    return 0 if result.ok else 1
