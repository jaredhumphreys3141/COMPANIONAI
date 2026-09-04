"""Interruption truncation, interface sign-in, and the smoke test's guard rails."""

import pytest

from companionai import character as character_mod
from companionai import config, net
from companionai import conversation as conversation_mod


class ChattyEngine:
    """Streams several sentences so a reply can be cut off part-way."""

    # Each sentence is comfortably longer than the splitter's minimum, so they
    # are dispatched one at a time rather than merged into pairs.
    TEXT = (
        "The first thing I noticed was how quiet the street had become. "
        "Then the second thing, which was the smell of rain on warm stone. "
        "A third observation followed about the light behind the rooftops. "
        "And finally a fourth remark that nobody was ever meant to hear."
    )

    def stream_chat(self, character, messages):
        for word in self.TEXT.split(" "):
            yield word + " "

    def unload(self):
        pass


# --------------------------------------------------------------------------- #
# interruption
# --------------------------------------------------------------------------- #
def test_barge_in_stores_only_what_was_spoken(monkeypatch):
    """Cutting the companion off must not leave it believing it said the rest.

    The stored turn feeds context, the transcript and memory extraction, so
    keeping unspoken text makes the companion refer to things the user never
    heard.
    """
    from companionai import llm

    monkeypatch.setattr(llm, "engine", lambda backend=None: ChattyEngine())
    conversation = conversation_mod.Conversation(character_mod.Character())

    spoken = []
    for _full, sentence in conversation.stream_reply("go"):
        if sentence:
            spoken.append(sentence)
            if len(spoken) == 2:          # user interrupts out loud
                conversation.stop(spoken_only=True)

    stored = conversation.as_chat_pairs()[-1]
    assert stored["role"] == "assistant"
    assert stored["content"] == " ".join(spoken)
    assert "fourth remark" not in stored["content"]


def test_only_played_audio_is_kept_when_the_model_ran_ahead(monkeypatch):
    """The case that actually happens on a device.

    Speech is queued sentence by sentence and played in order, so the model is
    normally several sentences ahead of the voice.  On a barge-in the queue is
    flushed, and everything past the sentence being played was never heard -
    including sentences already handed to the speech worker.
    """
    from companionai import llm

    monkeypatch.setattr(llm, "engine", lambda backend=None: ChattyEngine())
    conversation = conversation_mod.Conversation(character_mod.Character())

    heard_aloud = "The first thing I noticed was how quiet the street had become."
    dispatched = []
    for _full, sentence in conversation.stream_reply("go"):
        if sentence:
            dispatched.append(sentence)
            if len(dispatched) == 3:
                # Three sentences queued, only the first finished playing.
                conversation.stop(spoken_only=True, spoken_text=heard_aloud)

    stored = conversation.as_chat_pairs()[-1]["content"]
    assert stored == heard_aloud
    assert len(dispatched) >= 3, "the model should have run ahead of the voice"
    assert "second thing" not in stored, "queued but unplayed text must be dropped"
    assert "third observation" not in stored


def test_stop_button_in_a_text_chat_keeps_the_whole_reply(monkeypatch):
    """A Stop press is not a barge-in: the user read what was on screen."""
    from companionai import llm

    monkeypatch.setattr(llm, "engine", lambda backend=None: ChattyEngine())
    conversation = conversation_mod.Conversation(character_mod.Character())

    last_full = ""
    for full, sentence in conversation.stream_reply("go"):
        last_full = full
        if sentence and "second thing" in sentence:
            conversation.stop()           # no spoken_only
    stored = conversation.as_chat_pairs()[-1]["content"]
    assert stored == last_full.strip()


def test_an_uninterrupted_reply_is_stored_whole(monkeypatch):
    from companionai import llm

    monkeypatch.setattr(llm, "engine", lambda backend=None: ChattyEngine())
    conversation = conversation_mod.Conversation(character_mod.Character())
    list(conversation.stream_reply("go"))
    assert conversation.as_chat_pairs()[-1]["content"].strip() == ChattyEngine.TEXT.strip()


def test_interruption_before_any_speech_stores_nothing(monkeypatch):
    from companionai import llm

    monkeypatch.setattr(llm, "engine", lambda backend=None: ChattyEngine())
    conversation = conversation_mod.Conversation(character_mod.Character())
    generator = conversation.stream_reply("go")
    next(generator)
    conversation.stop(spoken_only=True)
    list(generator)
    # The user turn is there; no assistant turn, because nothing was heard.
    roles = [turn["role"] for turn in conversation.as_chat_pairs()]
    assert roles == ["user"]


# --------------------------------------------------------------------------- #
# sign-in
# --------------------------------------------------------------------------- #
def test_no_sign_in_by_default():
    assert not net.password_set()
    assert net.check_login("anyone", "anything")


def test_password_is_stored_hashed_not_in_plain_text():
    from companionai import paths

    net.set_password("correct horse battery staple", "jared")
    assert net.password_set()
    saved = paths.CONFIG_FILE.read_text("utf-8")
    assert "correct horse battery staple" not in saved
    assert config.get().ui_password_hash
    assert config.get().ui_password_salt


def test_sign_in_accepts_only_the_right_credentials():
    net.set_password("hunter2", "jared")
    assert net.check_login("jared", "hunter2")
    assert not net.check_login("jared", "Hunter2")
    assert not net.check_login("bob", "hunter2")
    assert not net.check_login("jared", "")


def test_sign_in_can_be_turned_off():
    net.set_password("hunter2", "jared")
    net.set_password("")
    assert not net.password_set()
    assert net.check_login("", "")


def test_two_passwords_get_different_salts():
    net.set_password("same-password")
    first = config.get().ui_password_hash
    net.set_password("same-password")
    assert config.get().ui_password_hash != first, "salt must be regenerated"


@pytest.mark.parametrize(
    ("host", "public"),
    [("0.0.0.0", True), ("192.168.1.10", True), ("127.0.0.1", False),
     ("localhost", False), ("::1", False), ("", False)],
)
def test_public_bind_detection(host, public):
    assert net.is_public_bind(host) is public


# --------------------------------------------------------------------------- #
# smoke test guard rails
# --------------------------------------------------------------------------- #
def test_smoke_test_refuses_to_download_without_permission():
    from companionai import smoke

    result = smoke.run(allow_network=False)
    assert not result.ok
    first = result.stages[0]
    assert not first.ok
    assert "--allow-network" in first.detail
    # It must not have gone on to load anything.
    assert len(result.stages) == 1


def test_smoke_report_names_every_stage_it_reached():
    from companionai import smoke

    report = smoke.run(allow_network=False).report()
    assert "models available" in report
    assert "FAILED" in report


def test_smoke_models_are_the_smallest_in_the_catalogue():
    from companionai import catalog, smoke

    for kind, model_id in smoke.SMOKE_MODELS.items():
        entry = catalog.get(kind, model_id)
        assert entry is not None, f"{kind}:{model_id} is not in the catalogue"
        smallest = min(e.size_mb for e in catalog.entries(kind))
        assert entry.size_mb == smallest, f"{kind} smoke model is not the smallest"


def test_word_overlap_scoring():
    from companionai.smoke import _overlap

    assert _overlap("the sky is blue today", "the sky is blue today") == 1.0
    assert _overlap("the sky is blue today", "The Sky is BLUE today") == 1.0
    assert _overlap("the sky is blue today", "completely different words") == 0.0
    assert 0.4 < _overlap("hello there the sky is blue", "hello the sky") < 0.9


# --------------------------------------------------------------------------- #
# generator lifetime and locking
# --------------------------------------------------------------------------- #
def test_abandoned_generation_releases_the_lock_from_another_thread():
    """Reported from a Pi as "cannot release un-acquired lock".

    The engine holds its generation lock across yields.  When the user
    interrupts, the generator is abandoned and finalised on whichever thread
    drops the last reference - not the one that acquired the lock.  An
    owner-checked RLock raises there; a plain Lock does not.
    """
    import threading

    from companionai import llm

    engine = llm.LlamaCppEngine()
    assert not isinstance(engine._gen_lock, type(threading.RLock())), (
        "the generation lock is held across yields, so it must be releasable "
        "by a thread other than the one that acquired it"
    )

    engine._llm = object()          # never reached; load() is bypassed below
    engine._key = ("stub", 0, 0, 0)

    def fake_completion(**_kwargs):
        for index in range(100):
            yield {"choices": [{"delta": {"content": f"word{index} "}}]}

    engine._llm = type("Stub", (), {"create_chat_completion": staticmethod(fake_completion)})()
    engine.load = lambda character: None

    character = character_mod.Character()
    holder = []

    def start_and_abandon():
        stream = engine.stream_chat(character, [{"role": "user", "content": "hi"}])
        next(stream)
        holder.append(stream)       # lock now held, acquired on this thread

    thread = threading.Thread(target=start_and_abandon)
    thread.start()
    thread.join()

    errors = []

    def finalise_elsewhere():
        try:
            holder[0].close()       # a different thread unwinds the finally
        except Exception as exc:    # pragma: no cover - the bug being fixed
            errors.append(exc)

    other = threading.Thread(target=finalise_elsewhere)
    other.start()
    other.join()
    assert not errors, f"closing from another thread raised {errors}"

    # And the lock really is free, so the next reply does not hang.
    assert engine._gen_lock.acquire(timeout=2), "generation lock was never released"
    engine._gen_lock.release()


def test_barge_in_through_the_session_still_records_the_turn(monkeypatch):
    """The session used to `break` out of the reply generator on a barge-in.

    That abandoned it before the code that stores the assistant turn, so an
    interrupted companion remembered saying nothing at all - and the
    truncation fix never actually ran in the real path.
    """
    from companionai import llm, session, voice

    monkeypatch.setattr(llm, "engine", lambda backend=None: ChattyEngine())
    session._session = None
    state = session.get()
    state.speak_replies = False

    # Stand in for the speech worker: an utterance handed to say() is treated
    # as having finished playing, which is what _played records.
    monkeypatch.setattr(state, "say", lambda text: state._played.append(text))
    # The user cuts in once one sentence has been heard.
    monkeypatch.setattr(
        voice.loop().interrupted, "is_set", lambda: len(state._played) >= 1
    )

    list(state.reply_stream("go"))

    history = state.conversation.as_chat_pairs()
    assert [turn["role"] for turn in history] == ["user", "assistant"], (
        "an interrupted reply must still be recorded"
    )
    stored = history[-1]["content"]
    assert stored == " ".join(state._played)
    assert "first thing" in stored
    assert "fourth remark" not in stored, "unheard text must not be remembered"
    assert state.status == "interrupted"
