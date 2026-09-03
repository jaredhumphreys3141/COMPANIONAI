"""The CompanionAI graphical interface.

Gradio was chosen deliberately: it gives us a touch-friendly UI that works on
the device's own screen and from another machine on the LAN, without writing
any front-end code.  Six tabs:

    Talk       - the conversation, by keyboard or by voice
    Studio     - build a companion: persona, models, every parameter
    Images     - generate pictures of the companion
    Models     - browse and install models (the only place that uses the network)
    Settings   - network policy, audio devices, server options
    System     - what hardware we detected and what is ready
"""

from __future__ import annotations

import gradio as gr

from .. import (
    catalog,
    config,
    hardware,
    imagegen,
    llm,
    net,
    paths,
    session,
    tts,
    voice,
)
from .. import character as character_mod
from .theme import CSS, theme

# Gradio moved `theme`/`css` from Blocks() to launch(), and made the messages
# chat format the only one, in version 6.  Handling both keeps CompanionAI
# working on Gradio 5 as well, which is what a Raspberry Pi or JetPack image
# may already have pinned.  (Gradio 4 is not supported: it cannot import
# against huggingface_hub 1.x.)
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_LEGACY_GRADIO = _GRADIO_MAJOR < 6


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _compat(component, **kwargs):
    """Build a Gradio component, dropping arguments this version does not accept.

    Gradio renames and retires constructor arguments between majors; filtering
    against the real signature is cheaper than pinning every board to one wheel.
    """
    import inspect  # noqa: PLC0415

    try:
        accepted = set(inspect.signature(component.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic component
        return component(**kwargs)
    return component(**{k: v for k, v in kwargs.items() if k in accepted})


def _chatbot(**kwargs) -> gr.Chatbot:
    if _LEGACY_GRADIO:
        kwargs["type"] = "messages"
    return _compat(gr.Chatbot, **kwargs)
def _err(exc: Exception) -> str:
    if isinstance(exc, net.NetworkBlocked):
        return f"Blocked: {exc}"
    return f"{type(exc).__name__}: {exc}"


def _net_badge() -> str:
    status = net.status()
    css_class = "net-offline" if not status.allowed else "net-online"
    short = "OFFLINE" if not status.allowed else "ONLINE"
    # Both labels are emitted and CSS picks one, so a 7" Pi touchscreen shows
    # "OFFLINE" while a desktop shows the full sentence.
    return (
        f"<span class='net-badge {css_class}' title='{status.badge}'>"
        f"<span class='net-badge-long'>{status.badge}</span>"
        f"<span class='net-badge-short'>{short}</span>"
        "</span>"
    )


def _character_choices() -> list[str]:
    return character_mod.list_ids() or [""]


def _model_choices(kind: str) -> list[str]:
    ids = catalog.choices(kind)
    if kind == "llm":
        ids += [name for name in catalog.local_gguf_files() if name not in ids]
    if kind == "tts":
        ids += [name for name in catalog.local_piper_voices() if name not in ids]
    return ids or [""]


def _labelled(kind: str) -> list[tuple[str, str]]:
    """Dropdown choices as ``(label, value)`` with an installed marker."""
    out = []
    for entry in catalog.entries(kind):
        mark = "" if entry.installed else "  (not installed)"
        out.append((f"{entry.name}{mark}", entry.id))
    extra = catalog.local_gguf_files() if kind == "llm" else (
        catalog.local_piper_voices() if kind == "tts" else []
    )
    known = {value for _, value in out}
    out += [(f"{name}  (local file)", name) for name in extra if name not in known]
    return out or [("(none available)", "")]


# --------------------------------------------------------------------------- #
# Character Studio field plumbing
#
# One ordered list drives the whole form: it builds the inputs, reads them back
# and writes them onto the Character.  Adding a parameter means adding a name
# here and one component below - no per-field wiring.
# --------------------------------------------------------------------------- #
FIELDS = [
    "name", "tagline", "user_name", "persona", "traits", "greeting", "reply_style",
    "custom_system_prompt",
    "llm_model", "temperature", "top_p", "top_k", "repeat_penalty", "max_tokens",
    "context_size", "threads", "gpu_layers", "seed", "memory_turns",
    "voice", "speech_rate", "voice_variation", "voice_cadence", "speaker_index", "volume",
    "asr_model", "asr_compute_type", "asr_language", "asr_beam_size",
    "image_model", "appearance", "art_style", "negative_prompt",
    "image_steps", "image_guidance", "image_size", "image_seed",
]


def _to_character(base: character_mod.Character, values: list) -> character_mod.Character:
    data = base.as_dict()
    for name, value in zip(FIELDS, values, strict=True):
        if value is None:
            continue
        data[name] = value
    updated = character_mod.Character.from_dict(data)
    updated.id = base.id or character_mod.slugify(updated.name)
    return updated


def _from_character(character: character_mod.Character) -> list:
    return [getattr(character, name) for name in FIELDS]


# --------------------------------------------------------------------------- #
# the app
# --------------------------------------------------------------------------- #
def build() -> gr.Blocks:
    state = session.get()

    blocks_kwargs = {"title": "CompanionAI"}
    if _LEGACY_GRADIO:
        blocks_kwargs.update(theme=theme(config.get().theme), css=CSS)

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Row(elem_classes="companion-header"):
            gr.HTML("<h1>CompanionAI</h1>")
            net_badge = gr.HTML(_net_badge())
            header_status = gr.Markdown(state.summary(), elem_classes="companion-subtle")

        with gr.Tabs():
            talk_refs = _talk_tab(state, net_badge, header_status)
            studio_refs = _studio_tab(state, header_status, talk_refs)
            _images_tab(state)
            _models_tab(state, net_badge, studio_refs)
            _settings_tab(state, net_badge)
            _system_tab(state)

    return demo


# --------------------------------------------------------------------------- #
# Talk
# --------------------------------------------------------------------------- #
def _talk_tab(state: session.Session, net_badge, header_status) -> dict:
    with gr.Tab("Talk") as talk_tab:
        with gr.Row():
            with gr.Column(scale=3):
                chat = _chatbot(
                    height=460,
                    elem_classes="chat-window",
                    label=None,
                    show_copy_button=True,
                    value=[{"role": "assistant", "content": state.character.greeting}],
                )
                with gr.Row():
                    box = gr.Textbox(
                        placeholder="Say something ...",
                        show_label=False,
                        scale=8,
                        autofocus=True,
                        submit_btn=True,
                    )
                    stop_btn = gr.Button("Stop", variant="stop", scale=1)
                status = gr.Markdown("", elem_classes="status-line")
                reply_audio = gr.Audio(
                    label="Spoken reply", autoplay=True, visible=False, interactive=False
                )

            with gr.Column(scale=1, min_width=240):
                # Guard against the active companion having been deleted from
                # disk: an out-of-range value renders as a blank dropdown.
                _choices = _character_choices()
                _active = (state.character.id if state.character.id in _choices
                           else (_choices[0] or None))
                who = gr.Dropdown(
                    choices=_choices,
                    value=_active,
                    label="Companion",
                    interactive=True,
                )
                mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Push to talk",
                    format="wav",
                )
                gr.Markdown("**Hands-free**")
                with gr.Row():
                    listen_btn = gr.Button("Start listening", variant="primary", size="sm")
                    hush_btn = gr.Button("Stop", size="sm")
                speak_where = gr.Radio(
                    ["this device's speakers", "my browser", "text only"],
                    value="this device's speakers",
                    label="Speak replies on",
                )
                clear_btn = gr.Button("New conversation", size="sm")

        # ---------------------------------------------------------- handlers
        def send(message: str, history: list, speak_target: str):
            message = (message or "").strip()
            if not message:
                yield history, "", gr.update(), ""
                return
            state.speak_replies = speak_target == "this device's speakers"
            history = list(history) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            yield history, "", gr.update(), "thinking ..."
            reply = ""
            for reply in state.reply_stream(message):
                history[-1]["content"] = reply
                yield history, "", gr.update(), state.status
            audio_update = gr.update()
            if speak_target == "my browser" and reply:
                try:
                    rendered = state.render_audio(reply)
                    if rendered:
                        audio_update = gr.update(value=rendered, visible=True)
                except Exception as exc:
                    yield history, "", gr.update(), _err(exc)
                    return
            yield history, "", audio_update, "ready"

        box.submit(send, [box, chat, speak_where], [chat, box, reply_audio, status])

        def from_mic(audio, history: list, speak_target: str):
            if audio is None:
                yield history, gr.update(), "No audio captured."
                return
            from .. import asr

            yield history, gr.update(), "transcribing ..."
            try:
                samples = asr.from_gradio(audio)
                text = asr.transcriber().transcribe(samples, state.character)
            except Exception as exc:
                yield history, gr.update(), _err(exc)
                return
            if not text.strip():
                yield history, gr.update(), "Didn't catch that."
                return
            for chunk in send(text, history, speak_target):
                yield chunk[0], chunk[2], chunk[3]

        mic.stop_recording(from_mic, [mic, chat, speak_where], [chat, reply_audio, status])

        def do_stop():
            state.conversation.stop()
            state.silence()
            return "Stopped."

        stop_btn.click(do_stop, None, status, queue=False)

        def start_listening():
            state.speak_replies = True
            return state.start_listening()

        listen_btn.click(start_listening, None, status)
        hush_btn.click(lambda: state.stop_listening(), None, status, queue=False)

        def switch(character_id: str):
            message = state.use_character(character_id)
            history = [{"role": "assistant", "content": state.character.greeting}]
            return history, message, state.summary()

        who.change(switch, who, [chat, status, header_status])

        def clear():
            state.conversation.reset()
            state.silence()
            return [{"role": "assistant", "content": state.character.greeting}], "New conversation."

        clear_btn.click(clear, None, [chat, status], queue=False)

        # While the hands-free loop is running the conversation advances on a
        # background thread, so poll it into the chat window.
        timer = gr.Timer(1.5)

        def poll():
            if not state.listening:
                return gr.update(), gr.update(), gr.update()
            events = voice.loop().drain()
            note = state.status
            for event in events:
                if event["type"] == "error":
                    note = event.get("message", "error")
                elif event["type"] == "transcribing":
                    note = "transcribing ..."
                elif event["type"] == "interrupt":
                    note = "interrupted - go ahead"
            return state.conversation.as_chat_pairs(), note, _net_badge()

        timer.tick(poll, None, [chat, status, net_badge])

        # The dropdown's choices are baked in when the page is built, so
        # companions created later have to be pushed in.  The Studio does that
        # directly; re-reading on tab open covers everything else.
        talk_tab.select(
            lambda: gr.update(choices=_character_choices()), None, who, queue=False
        )

    return {"who": who, "chat": chat, "status": status}


# --------------------------------------------------------------------------- #
# Character Studio
# --------------------------------------------------------------------------- #
def _studio_tab(state: session.Session, header_status, talk_refs: dict) -> dict:
    character = state.character

    with gr.Tab("Studio"):
        gr.Markdown(
            "Build a companion.  Every model and parameter below is editable; "
            "the defaults were chosen for the hardware this is running on."
        )
        with gr.Row():
            picker = gr.Dropdown(
                choices=_character_choices(), value=character.id,
                label="Editing", scale=3, interactive=True,
            )
            new_name = gr.Textbox(label="New companion name", placeholder="Nova", scale=2)
            new_btn = gr.Button("Create", variant="primary", scale=1)
            dup_btn = gr.Button("Duplicate", scale=1)
            del_btn = gr.Button("Delete", variant="stop", scale=1)

        studio_status = gr.Markdown("", elem_classes="status-line")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Identity")
                name = gr.Textbox(label="Name", value=character.name)
                tagline = gr.Textbox(label="Tagline", value=character.tagline)
                user_name = gr.Textbox(label="What should it call you?", value=character.user_name)
                persona = gr.Textbox(
                    label="Persona", value=character.persona, lines=5,
                    info="Who they are, what they care about, how they behave.",
                )
                traits = gr.CheckboxGroup(
                    choices=character_mod.TRAITS, value=character.traits, label="Manner",
                )
                greeting = gr.Textbox(label="Opening line", value=character.greeting)
                reply_style = gr.Radio(
                    ["brief", "conversational", "detailed"],
                    value=character.reply_style, label="Reply length",
                )
                custom_system_prompt = gr.Textbox(
                    label="Custom system prompt (advanced)",
                    value=character.custom_system_prompt, lines=4,
                    info="Leave empty to use the prompt built from the fields above.",
                )
                prompt_preview = gr.Textbox(
                    label="System prompt preview", value=character.system_prompt(),
                    lines=8, interactive=False,
                )

            with gr.Column(scale=1):
                with gr.Accordion("Language model", open=True):
                    llm_model = gr.Dropdown(
                        choices=_labelled("llm"), value=character.llm_model,
                        label="Model", interactive=True, allow_custom_value=True,
                    )
                    temperature = gr.Slider(0.0, 1.6, character.temperature, step=0.05,
                                            label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, character.top_p, step=0.01, label="Top-p")
                    top_k = gr.Slider(0, 200, character.top_k, step=1, label="Top-k")
                    repeat_penalty = gr.Slider(1.0, 1.6, character.repeat_penalty, step=0.01,
                                               label="Repetition penalty")
                    max_tokens = gr.Slider(32, 2048, character.max_tokens, step=16,
                                           label="Max reply tokens")
                    context_size = gr.Slider(512, 32768, character.context_size, step=512,
                                             label="Context window")
                    threads = gr.Slider(1, 16, character.threads, step=1, label="CPU threads")
                    gpu_layers = gr.Slider(-1, 100, character.gpu_layers, step=1,
                                           label="GPU layers (-1 = all)")
                    seed = gr.Number(value=character.seed, label="Seed (-1 = random)",
                                     precision=0)
                    memory_turns = gr.Slider(2, 40, character.memory_turns, step=1,
                                            label="Exchanges kept in memory")

                with gr.Accordion("Voice", open=False):
                    voice_dd = gr.Dropdown(
                        choices=_labelled("tts"), value=character.voice,
                        label="Voice", interactive=True, allow_custom_value=True,
                    )
                    speech_rate = gr.Slider(0.5, 2.0, character.speech_rate, step=0.05,
                                            label="Pace (higher is slower)")
                    voice_variation = gr.Slider(0.0, 1.5, character.voice_variation, step=0.01,
                                                label="Expressiveness")
                    voice_cadence = gr.Slider(0.0, 1.5, character.voice_cadence, step=0.01,
                                              label="Cadence variation")
                    speaker_index = gr.Slider(0, 20, character.speaker_index, step=1,
                                              label="Speaker (multi-speaker voices)")
                    volume = gr.Slider(0.1, 2.0, character.volume, step=0.05, label="Volume")
                    with gr.Row():
                        preview_btn = gr.Button("Preview voice", size="sm")
                    voice_sample = gr.Audio(label="Sample", interactive=False)

                with gr.Accordion("Listening", open=False):
                    asr_model = gr.Dropdown(
                        choices=_labelled("asr"), value=character.asr_model,
                        label="Speech recognition model", interactive=True,
                        allow_custom_value=True,
                    )
                    asr_compute_type = gr.Radio(
                        ["int8", "int8_float16", "float16", "float32"],
                        value=character.asr_compute_type, label="Precision",
                    )
                    asr_language = gr.Textbox(value=character.asr_language,
                                              label="Language code (blank = detect)")
                    asr_beam_size = gr.Slider(1, 5, character.asr_beam_size, step=1,
                                              label="Beam size")

                with gr.Accordion("Appearance", open=False):
                    image_model = gr.Dropdown(
                        choices=_labelled("image"), value=character.image_model,
                        label="Image model", interactive=True, allow_custom_value=True,
                    )
                    appearance = gr.Textbox(label="Looks", value=character.appearance, lines=3)
                    art_style = gr.Dropdown(
                        choices=list(character_mod.ART_STYLES), value=character.art_style,
                        label="Art style", allow_custom_value=True,
                    )
                    negative_prompt = gr.Textbox(label="Avoid", value=character.negative_prompt,
                                                 lines=2)
                    image_steps = gr.Slider(1, 50, character.image_steps, step=1, label="Steps")
                    image_guidance = gr.Slider(0.0, 12.0, character.image_guidance, step=0.1,
                                               label="Guidance (0 for turbo models)")
                    image_size = gr.Slider(256, 1024, character.image_size, step=64,
                                           label="Size (px)")
                    image_seed = gr.Number(value=character.image_seed, precision=0,
                                           label="Face seed (keep fixed for a consistent look)")
                    portrait_btn = gr.Button("Generate reference portrait", variant="primary")
                    portrait = gr.Image(
                        label="Reference portrait", height=320,
                        value=character.portrait or None, interactive=False,
                    )

        with gr.Row():
            save_btn = gr.Button("Save companion", variant="primary", scale=2)
            activate_btn = gr.Button("Save and make active", scale=2)

        inputs = [
            name, tagline, user_name, persona, traits, greeting, reply_style,
            custom_system_prompt,
            llm_model, temperature, top_p, top_k, repeat_penalty, max_tokens,
            context_size, threads, gpu_layers, seed, memory_turns,
            voice_dd, speech_rate, voice_variation, voice_cadence, speaker_index, volume,
            asr_model, asr_compute_type, asr_language, asr_beam_size,
            image_model, appearance, art_style, negative_prompt,
            image_steps, image_guidance, image_size, image_seed,
        ]
        assert len(inputs) == len(FIELDS), "studio form and FIELDS are out of step"

        # ---------------------------------------------------------- handlers
        def _current(values: list) -> character_mod.Character:
            return _to_character(state.character, values)

        def save(*values, activate: bool = False):
            updated = _current(list(values))
            missing = catalog.missing_for(updated)
            try:
                updated.save()
            except OSError as exc:
                return (gr.update(), f"Could not save: {exc}", gr.update(),
                        gr.update(), gr.update())
            if activate or updated.id == state.character.id:
                state.refresh_character(updated)
            note = f"Saved **{updated.name}**."
            if missing:
                listed = ", ".join(f"{kind}:{model}" for kind, model in missing)
                note += f"  Still to install from the Models tab: {listed}."
            # Only move the Talk tab's selection when this companion actually
            # became the active one; a plain Save must not reset the chat.
            talk_update = gr.update(choices=_character_choices())
            if activate:
                talk_update = gr.update(choices=_character_choices(), value=updated.id)
            return (
                gr.update(choices=_character_choices(), value=updated.id),
                note,
                updated.system_prompt(),
                state.summary(),
                talk_update,
            )

        save_btn.click(save, inputs,
                       [picker, studio_status, prompt_preview, header_status, talk_refs["who"]])
        activate_btn.click(
            lambda *values: save(*values, activate=True),
            inputs, [picker, studio_status, prompt_preview, header_status, talk_refs["who"]],
        )

        def load_character(character_id: str):
            if not character_id:
                return [gr.update()] * (len(FIELDS) + 3)
            try:
                loaded = character_mod.Character.load(character_id)
            except Exception as exc:
                return [gr.update()] * len(FIELDS) + [f"Could not load: {_err(exc)}",
                                                      gr.update(), gr.update()]
            state.refresh_character(loaded)
            state.conversation.character = loaded
            return _from_character(loaded) + [
                f"Editing **{loaded.name}**.",
                loaded.system_prompt(),
                loaded.portrait or None,
            ]

        picker.change(load_character, picker, inputs + [studio_status, prompt_preview, portrait])

        def create(name_value: str):
            name_value = (name_value or "").strip() or "New companion"
            fresh = character_mod.new_from_profile(name_value).save()
            state.refresh_character(fresh)
            return (
                gr.update(choices=_character_choices(), value=fresh.id),
                f"Created **{fresh.name}** with defaults for this hardware.",
                "",
                gr.update(choices=_character_choices(), value=fresh.id),
            )

        new_btn.click(create, new_name, [picker, studio_status, new_name, talk_refs["who"]])

        def duplicate(name_value: str, *values):
            source = _current(list(values))
            target = (name_value or "").strip() or f"{source.name} copy"
            clone = source.copy_as(target).save()
            return (
                gr.update(choices=_character_choices(), value=clone.id),
                f"Duplicated to **{clone.name}**.",
                "",
                gr.update(choices=_character_choices()),
            )

        dup_btn.click(duplicate, [new_name] + inputs,
                      [picker, studio_status, new_name, talk_refs["who"]])

        def delete(character_id: str):
            ids = character_mod.list_ids()
            if len(ids) <= 1:
                return gr.update(), "Keep at least one companion.", gr.update()
            try:
                character_mod.Character.load(character_id).delete()
            except Exception as exc:
                return gr.update(), _err(exc), gr.update()
            remaining = character_mod.list_ids()
            state.use_character(remaining[0])
            return (
                gr.update(choices=remaining, value=remaining[0]),
                f"Deleted `{character_id}`.",
                gr.update(choices=remaining, value=remaining[0]),
            )

        del_btn.click(delete, picker, [picker, studio_status, talk_refs["who"]])

        def preview_voice(*values):
            updated = _current(list(values))
            try:
                speaker = tts.speaker(updated)
                rate, audio = speaker.synth(
                    f"Hi, I'm {updated.name}. {updated.greeting}", updated
                )
            except Exception as exc:
                return None, _err(exc)
            if audio.size == 0:
                return None, "The voice produced no audio."
            return (rate, audio), "Voice preview ready."

        preview_btn.click(preview_voice, inputs, [voice_sample, studio_status])

        def make_portrait(*values, progress=gr.Progress()):
            updated = _current(list(values))
            estimate = imagegen.estimate_seconds(updated)
            progress(0.05, desc=f"about {estimate:.0f}s on this hardware")
            try:
                image, path, _ = imagegen.generator().generate(
                    updated, progress=lambda f, d: progress(f, desc=d)
                )
            except Exception as exc:
                return gr.update(), _err(exc)
            updated.portrait = str(path)
            updated.save()
            if updated.id == state.character.id:
                state.refresh_character(updated)
            return str(path), f"Portrait saved to `{path}`."

        portrait_btn.click(make_portrait, inputs, [portrait, studio_status])

        # Keep the preview honest as the persona is edited.
        for component in (name, tagline, user_name, persona, traits, greeting,
                          reply_style, custom_system_prompt):
            component.change(
                lambda *values: _current(list(values)).system_prompt(),
                inputs, prompt_preview, show_progress="hidden",
            )

    return {"picker": picker, "llm": llm_model, "tts": voice_dd,
            "asr": asr_model, "image": image_model}


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def _images_tab(state: session.Session) -> None:
    with gr.Tab("Images"):
        gr.Markdown(
            "Pictures of the active companion.  The face seed from the Studio tab is "
            "reused unless you override it, which is what keeps the character "
            "recognisable across images."
        )
        with gr.Row():
            with gr.Column(scale=1):
                scene = gr.Textbox(
                    label="Scene",
                    placeholder="sitting in a rainy cafe, warm lamplight, holding a mug",
                    lines=3,
                )
                with gr.Row():
                    steps = gr.Slider(1, 50, state.character.image_steps, step=1, label="Steps")
                    guidance = gr.Slider(0.0, 12.0, state.character.image_guidance, step=0.1,
                                         label="Guidance")
                with gr.Row():
                    size = gr.Slider(256, 1024, state.character.image_size, step=64, label="Size")
                    seed = gr.Number(value=state.character.image_seed, precision=0,
                                     label="Seed (-1 = random)")
                with gr.Row():
                    go = gr.Button("Generate", variant="primary", scale=2)
                    unload = gr.Button("Free memory", scale=1)
                image_status = gr.Markdown("", elem_classes="status-line")
            with gr.Column(scale=1):
                out = gr.Image(label="Result", height=430, interactive=False)

        gallery = gr.Gallery(label="Gallery", columns=6, height=220, value=imagegen.gallery())
        refresh_gallery = gr.Button("Refresh gallery", size="sm")

        def generate(scene_text, steps_value, guidance_value, size_value, seed_value,
                     progress=gr.Progress()):
            character = state.character
            estimate = imagegen.estimate_seconds(character)
            progress(0.02, desc=f"about {estimate:.0f}s on this hardware")
            try:
                image, path, used_seed = imagegen.generator().generate(
                    character,
                    scene_text or "",
                    steps=int(steps_value),
                    guidance=float(guidance_value),
                    size=int(size_value),
                    seed=int(seed_value),
                    progress=lambda f, d: progress(f, desc=d),
                )
            except Exception as exc:
                return gr.update(), _err(exc), gr.update()
            return str(path), f"Seed `{used_seed}` - saved to `{path}`.", imagegen.gallery()

        go.click(generate, [scene, steps, guidance, size, seed], [out, image_status, gallery])
        unload.click(
            lambda: (imagegen.generator().unload(), "Image model unloaded.")[1],
            None, image_status, queue=False,
        )
        refresh_gallery.click(imagegen.gallery, None, gallery, queue=False)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
_KIND_LABELS = {
    "llm": "Language models",
    "asr": "Speech recognition",
    "tts": "Voices",
    "image": "Image models",
}


def _models_tab(state: session.Session, net_badge, studio_refs: dict) -> None:
    with gr.Tab("Models"):
        gr.Markdown(
            "Everything here is optional and nothing downloads by itself.  Installing a "
            "model is the only action in CompanionAI that touches the network, and it is "
            "refused unless you approve access on the Settings tab first."
        )
        gate_note = gr.Markdown(_gate_note())

        for kind, label in _KIND_LABELS.items():
            with gr.Tab(label):
                grid = gr.Dataframe(
                    headers=catalog.TABLE_HEADERS,
                    value=catalog.table(kind),
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    picker = gr.Dropdown(
                        choices=_model_choices(kind), label="Model", scale=3, interactive=True,
                    )
                    install_btn = gr.Button("Install", variant="primary", scale=1)
                    remove_btn = gr.Button("Remove", variant="stop", scale=1)
                    refresh_btn = gr.Button("Refresh", scale=1)
                note = gr.Markdown("", elem_classes="status-line")

                def do_install(model_id, _kind=kind, progress=gr.Progress()):
                    try:
                        message = catalog.install(
                            _kind, model_id, progress=lambda f, d: progress(f, desc=d)
                        )
                    except Exception as exc:
                        return catalog.table(_kind), _err(exc), _net_badge(), gr.update()
                    return (
                        catalog.table(_kind),
                        message,
                        _net_badge(),
                        gr.update(choices=_labelled(_kind)),
                    )

                def do_remove(model_id, _kind=kind):
                    try:
                        message = catalog.remove(_kind, model_id)
                    except Exception as exc:
                        return catalog.table(_kind), _err(exc), gr.update()
                    return catalog.table(_kind), message, gr.update(choices=_labelled(_kind))

                studio_component = studio_refs.get(kind)
                install_btn.click(
                    do_install, picker, [grid, note, net_badge, studio_component]
                )
                remove_btn.click(do_remove, picker, [grid, note, studio_component])
                refresh_btn.click(
                    lambda _kind=kind: (
                        catalog.table(_kind),
                        gr.update(choices=_model_choices(_kind)),
                        _gate_note(),
                    ),
                    None, [grid, picker, gate_note], queue=False,
                )

        with gr.Accordion("Bring your own models", open=False):
            gr.Markdown(
                f"""
CompanionAI never requires the internet.  You can copy models in by hand instead:

- **Language models** - any `.gguf` file into `{paths.LLM_DIR}`
- **Voices** - a Piper `*.onnx` **and** its `*.onnx.json` into `{paths.TTS_DIR}`
- **Speech recognition** - a CTranslate2 Whisper folder into `{paths.ASR_DIR}/<name>`
- **Image models** - a Diffusers folder into `{paths.IMAGE_DIR}/<name>`

They show up in the dropdowns after pressing Refresh.  To add your own download
links, create `{paths.HOME / 'catalog.user.json'}` using the same shape as the
built-in catalogue.
"""
            )


def _gate_note() -> str:
    status = net.status()
    if status.allowed:
        return (
            f"Network access is **on** ({status.reason}).  Downloads are permitted until "
            "you turn it off again on the Settings tab."
        )
    return (
        "Network access is **off**.  Installing a model will be refused until you approve "
        "access on the Settings tab."
    )


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def _settings_tab(state: session.Session, net_badge) -> None:
    settings = config.get()

    with gr.Tab("Settings"):
        with gr.Tab("Network"):
            gr.Markdown(
                """
### Nothing leaves this device unless you say so

CompanionAI starts offline every time it launches.  While it is offline the app
cannot download models, cannot check for updates and sends no telemetry - the
language model, speech recognition, speech synthesis and image generation all run
on this machine.

Approving access here only lets **you** pull models down from the Models tab.
Your conversations, audio and images are never uploaded, whatever this is set to.
"""
            )
            policy = gr.Radio(
                choices=[
                    ("Offline - block all network access (default)", config.OFFLINE),
                    ("Allow for this session only", config.SESSION),
                    ("Allow and remember", config.ALWAYS),
                ],
                value=net.status().policy,
                label="Network policy",
            )
            hosts = gr.Textbox(
                value="\n".join(settings.allow_hosts),
                label="Allowed hosts (one per line, empty means any host)",
                lines=4,
            )
            endpoint = gr.Textbox(value=settings.hf_endpoint, label="Model hub endpoint")
            net_status_md = gr.Markdown(_gate_note())
            audit = gr.Markdown(net.audit_markdown())
            with gr.Row():
                apply_net = gr.Button("Apply", variant="primary")
                refresh_audit = gr.Button("Refresh activity log")

            def apply_network(policy_value, hosts_value, endpoint_value):
                config.update(
                    allow_hosts=[h.strip() for h in (hosts_value or "").splitlines() if h.strip()],
                    hf_endpoint=(endpoint_value or "").strip(),
                )
                if policy_value == config.OFFLINE:
                    net.revoke()
                else:
                    net.approve(policy_value)
                return _gate_note(), _net_badge(), net.audit_markdown()

            apply_net.click(
                apply_network, [policy, hosts, endpoint], [net_status_md, net_badge, audit]
            )
            refresh_audit.click(net.audit_markdown, None, audit, queue=False)

        with gr.Tab("Audio"):
            inputs_list, outputs_list = voice.list_devices()
            input_device = gr.Dropdown(
                choices=[""] + inputs_list, value=settings.input_device,
                label="Microphone", allow_custom_value=True,
            )
            output_device = gr.Dropdown(
                choices=[""] + outputs_list, value=settings.output_device,
                label="Speakers", allow_custom_value=True,
            )
            mic_gain = gr.Slider(0.2, 5.0, settings.mic_gain, step=0.1, label="Microphone gain")
            vad = gr.Slider(0, 3, settings.vad_aggressiveness, step=1,
                            label="Voice detection strictness")
            silence_ms = gr.Slider(200, 2000, settings.silence_ms, step=50,
                                   label="End-of-sentence silence (ms)")
            min_speech_ms = gr.Slider(100, 1000, settings.min_speech_ms, step=50,
                                      label="Shortest sound treated as speech (ms)")
            barge_in = gr.Checkbox(value=settings.barge_in,
                                   label="Let me interrupt while the companion is speaking")
            audio_note = gr.Markdown("", elem_classes="status-line")
            save_audio = gr.Button("Save audio settings", variant="primary")

            def apply_audio(*values):
                keys = ["input_device", "output_device", "mic_gain", "vad_aggressiveness",
                        "silence_ms", "min_speech_ms", "barge_in"]
                payload = dict(zip(keys, values, strict=True))
                payload["vad_aggressiveness"] = int(payload["vad_aggressiveness"])
                payload["silence_ms"] = int(payload["silence_ms"])
                payload["min_speech_ms"] = int(payload["min_speech_ms"])
                config.update(**payload)
                restart = ""
                if state.listening:
                    state.stop_listening()
                    restart = state.start_listening()
                return f"Audio settings saved.  {restart}"

            save_audio.click(
                apply_audio,
                [input_device, output_device, mic_gain, vad, silence_ms, min_speech_ms, barge_in],
                audio_note,
            )

        with gr.Tab("Runtime"):
            backend = gr.Radio(
                [("llama.cpp (GGUF files)", "llama.cpp"), ("Ollama daemon on localhost", "ollama")],
                value=settings.llm_backend, label="Language model backend",
            )
            ollama_url = gr.Textbox(value=settings.ollama_url, label="Ollama URL")
            transcripts = gr.Checkbox(value=settings.save_transcripts,
                                      label="Save conversation transcripts to disk")
            ui_theme = gr.Radio(["soft", "default", "monochrome", "glass"],
                                value=settings.theme, label="Theme (applies on restart)")
            host = gr.Textbox(value=settings.host, label="Bind address",
                              info="127.0.0.1 keeps the UI on this machine; "
                                   "0.0.0.0 lets other devices on your LAN reach it.")
            port = gr.Number(value=settings.port, precision=0, label="Port")
            runtime_note = gr.Markdown("", elem_classes="status-line")
            save_runtime = gr.Button("Save runtime settings", variant="primary")

            def apply_runtime(backend_value, url_value, transcripts_value, theme_value,
                              host_value, port_value):
                config.update(
                    llm_backend=backend_value,
                    ollama_url=(url_value or "").strip(),
                    save_transcripts=bool(transcripts_value),
                    theme=theme_value,
                    host=(host_value or "127.0.0.1").strip(),
                    port=int(port_value or 7860),
                )
                llm.engine().unload()
                extra = ""
                if backend_value == "ollama":
                    models = llm.OllamaEngine().available_models()
                    extra = (f"  Ollama models found: {', '.join(models[:6])}"
                             if models else "  No Ollama daemon answered yet.")
                return "Saved.  Bind address and theme take effect next launch." + extra

            save_runtime.click(
                apply_runtime,
                [backend, ollama_url, transcripts, ui_theme, host, port],
                runtime_note,
            )


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
def _system_tab(state: session.Session) -> None:
    with gr.Tab("System"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Hardware")
                hardware_md = gr.Markdown(hardware.summary())
            with gr.Column():
                gr.Markdown("### Ready to use")
                readiness = gr.Dataframe(
                    headers=["component", "state", "detail"],
                    value=state.readiness(),
                    interactive=False,
                    wrap=True,
                )
        with gr.Row():
            refresh = gr.Button("Refresh", variant="primary")
            free_btn = gr.Button("Unload all models (free memory)")
        system_note = gr.Markdown("", elem_classes="status-line")

        gr.Markdown(
            f"""
### Where things live

| what | path |
| --- | --- |
| data directory | `{paths.HOME}` |
| companions | `{paths.CHARACTERS_DIR}` |
| models | `{paths.MODELS_DIR}` |
| generated images | `{paths.GALLERY_DIR}` |
| transcripts | `{paths.HOME / 'transcripts'}` |
| network activity log | `{net.AUDIT_FILE}` |

Set `COMPANIONAI_HOME` before launching to move all of it somewhere else - an
external SSD on a Raspberry Pi, for instance.
"""
        )
        activity = gr.Markdown(net.audit_markdown())

        def do_refresh():
            return hardware.summary(), state.readiness(), net.audit_markdown(), ""

        refresh.click(do_refresh, None, [hardware_md, readiness, activity, system_note],
                      queue=False)

        def free_all():
            llm.engine().unload()
            imagegen.generator().unload()
            from .. import asr

            asr.transcriber().unload()
            tts.speaker().unload()
            return "Unloaded the language, speech and image models."

        free_btn.click(free_all, None, system_note, queue=False)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def launch(host: str | None = None, port: int | None = None, share: bool = False,
           open_browser: bool | None = None) -> None:
    settings = config.get()
    demo = build()
    demo.queue(default_concurrency_limit=4)
    launch_kwargs = dict(
        server_name=host or settings.host,
        server_port=int(port or settings.port),
        share=share,               # off unless the user explicitly asks for it
        inbrowser=settings.open_browser if open_browser is None else open_browser,
        show_api=False,
    )
    if not _LEGACY_GRADIO:
        launch_kwargs.update(theme=theme(settings.theme), css=CSS)

    import inspect

    accepted = set(inspect.signature(demo.launch).parameters)
    demo.launch(**{k: v for k, v in launch_kwargs.items() if k in accepted})
