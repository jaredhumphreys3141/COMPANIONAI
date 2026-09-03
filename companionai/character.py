"""A companion is one JSON file.

Everything the Character Studio tab exposes - persona, model choices and every
sampling/voice/image parameter - is a field on :class:`Character`.  Characters
are saved under ``<data dir>/characters/<id>.json`` so they can be copied
between a PC, a Pi and a Jetson.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, fields

from . import hardware, paths

TRAITS = [
    "warm", "playful", "witty", "calm", "curious", "encouraging", "blunt",
    "sarcastic", "formal", "nerdy", "poetic", "practical", "patient",
    "enthusiastic", "protective", "mischievous",
]

ART_STYLES = {
    "photographic": "photorealistic portrait, natural light, 50mm lens, shallow depth of field",
    "anime": "anime illustration, clean line art, cel shading, vibrant colours",
    "digital painting": "digital painting, painterly brush strokes, dramatic lighting, artstation quality",
    "watercolour": "delicate watercolour illustration, soft washes, paper texture",
    "3d render": "stylised 3d character render, subsurface scattering, studio lighting",
    "comic": "comic book ink and colour, bold outlines, halftone shading",
    "pixel art": "16-bit pixel art portrait, limited palette",
}

DEFAULT_NEGATIVE = (
    "blurry, lowres, deformed, extra limbs, extra fingers, bad anatomy, "
    "watermark, text, signature, jpeg artifacts"
)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "companion"


@dataclass
class Character:
    # -- identity -----------------------------------------------------------
    id: str = ""
    name: str = "Aria"
    tagline: str = "A local AI companion."
    persona: str = (
        "You are a supportive companion who enjoys long conversations, remembers what "
        "matters to your friend, and has opinions of your own."
    )
    traits: list[str] = field(default_factory=lambda: ["warm", "curious", "witty"])
    greeting: str = "Hey, good to see you. What's on your mind?"
    user_name: str = "friend"
    custom_system_prompt: str = ""     # if set, replaces the generated prompt
    reply_style: str = "conversational"  # conversational | brief | detailed

    # -- language model -----------------------------------------------------
    llm_model: str = ""
    temperature: float = 0.75
    top_p: float = 0.92
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 320
    context_size: int = 4096
    threads: int = 4
    gpu_layers: int = 0
    seed: int = -1                     # -1 means random each turn
    memory_turns: int = 12             # how many exchanges stay in the prompt

    # -- speech recognition -------------------------------------------------
    asr_model: str = "base.en"
    asr_compute_type: str = "int8"
    asr_language: str = "en"
    asr_beam_size: int = 1

    # -- speech synthesis ---------------------------------------------------
    voice: str = "en_US-amy-medium"
    speech_rate: float = 1.0           # >1 is slower (piper length_scale)
    voice_variation: float = 0.667     # piper noise_scale
    voice_cadence: float = 0.8         # piper noise_w
    speaker_index: int = 0             # multi-speaker piper voices
    volume: float = 1.0

    # -- appearance / image generation --------------------------------------
    image_model: str = "sd-turbo"
    appearance: str = (
        "a friendly woman in her late twenties, shoulder-length auburn hair, "
        "green eyes, freckles, soft knitted jumper"
    )
    art_style: str = "digital painting"
    negative_prompt: str = DEFAULT_NEGATIVE
    image_steps: int = 4
    image_guidance: float = 0.0
    image_size: int = 512
    image_seed: int = 1234             # fixed seed keeps the face consistent
    portrait: str = ""                 # path to the saved reference portrait

    # -- bookkeeping --------------------------------------------------------
    created: str = ""
    modified: str = ""
    schema: int = 1

    # ---------------------------------------------------------------- prompt
    def system_prompt(self) -> str:
        if self.custom_system_prompt.strip():
            return self.custom_system_prompt.strip()

        style = {
            "brief": "Keep replies to one or two sentences unless asked for more.",
            "detailed": "Give thorough, well-structured answers when the topic deserves it.",
        }.get(
            self.reply_style,
            "Keep replies short enough to speak aloud - usually two to four sentences.",
        )
        traits = ", ".join(self.traits) if self.traits else "warm"
        return "\n".join(
            [
                f"You are {self.name}. {self.tagline}",
                "",
                self.persona.strip(),
                "",
                f"Your manner is {traits}.",
                f"You are talking with {self.user_name}.",
                style,
                "Your replies are spoken aloud by a text-to-speech voice, so write plain "
                "prose: no markdown, no bullet points, no emoji, no stage directions.",
                "You run entirely on this device and have no internet access. If you are "
                "asked for live information, say so plainly instead of inventing it.",
            ]
        )

    def image_prompt(self, scene: str = "", *, portrait: bool = True) -> str:
        style = ART_STYLES.get(self.art_style, self.art_style)
        parts = [self.appearance.strip()]
        if scene.strip():
            parts.append(scene.strip())
        elif portrait:
            parts.append("head and shoulders portrait, looking at the viewer, neutral background")
        parts.append(style)
        parts.append("highly detailed, coherent face")
        return ", ".join(part for part in parts if part)

    # ------------------------------------------------------------ persistence
    @property
    def file(self):
        return paths.CHARACTERS_DIR / f"{self.id or slugify(self.name)}.json"

    def save(self) -> Character:
        paths.ensure_dirs()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.id:
            self.id = slugify(self.name)
        if not self.created:
            self.created = now
        self.modified = now
        self.file.write_text(json.dumps(asdict(self), indent=2), "utf-8")
        return self

    def delete(self) -> None:
        self.file.unlink(missing_ok=True)

    def copy_as(self, new_name: str) -> Character:
        data = asdict(self)
        data.update(id=slugify(new_name), name=new_name, created="", modified="", portrait="")
        return Character(**data)

    @classmethod
    def from_dict(cls, raw: dict) -> Character:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def load(cls, character_id: str) -> Character:
        path = paths.CHARACTERS_DIR / f"{character_id}.json"
        return cls.from_dict(json.loads(path.read_text("utf-8")))

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# collection helpers
# --------------------------------------------------------------------------- #
def list_ids() -> list[str]:
    paths.ensure_dirs()
    return sorted(p.stem for p in paths.CHARACTERS_DIR.glob("*.json"))


def load_all() -> list[Character]:
    out = []
    for character_id in list_ids():
        try:
            out.append(Character.load(character_id))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return out


def new_from_profile(name: str = "Aria") -> Character:
    """A blank companion pre-tuned for the machine we are running on."""
    profile = hardware.profile_for()
    character = Character(
        id=slugify(name),
        name=name,
        llm_model=profile.llm_model,
        threads=profile.llm_threads,
        gpu_layers=profile.llm_gpu_layers,
        context_size=profile.llm_context,
        asr_model=profile.asr_model,
        asr_compute_type=profile.asr_compute_type,
        image_model=profile.image_model,
        image_steps=profile.image_steps,
        image_size=profile.image_size,
    )
    return character


def ensure_default() -> Character:
    """Guarantee at least one companion exists, and return the active one."""
    from . import config  # local import to avoid a cycle at module load

    existing = list_ids()
    settings = config.get()
    if settings.active_character and settings.active_character in existing:
        return Character.load(settings.active_character)
    if existing:
        character = Character.load(existing[0])
        config.update(active_character=character.id)
        return character
    character = new_from_profile().save()
    config.update(active_character=character.id)
    return character
