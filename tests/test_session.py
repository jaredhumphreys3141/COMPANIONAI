"""End-to-end check of the reply pipeline with a stand-in language model."""

import pytest

from companionai import llm, session


class FakeEngine:
    """Streams a fixed reply one word at a time, like llama.cpp would."""

    def __init__(self, text):
        self.text = text
        self.calls = []

    def stream_chat(self, character, messages):
        self.calls.append(messages)
        for word in self.text.split(" "):
            yield word + " "

    def unload(self):
        pass


@pytest.fixture
def fake_session(monkeypatch):
    engine = FakeEngine(
        "Good to hear from you. I was just thinking about that map you mentioned. "
        "Shall we pick it up again?"
    )
    monkeypatch.setattr(llm, "engine", lambda backend=None: engine)
    session._session = None
    state = session.get()
    state.speak_replies = False          # no sound card in a test runner
    return state, engine


def test_reply_streams_and_is_remembered(fake_session):
    state, engine = fake_session
    chunks = list(state.reply_stream("Hi there"))

    assert chunks, "the reply produced nothing"
    assert chunks[-1].startswith("Good to hear from you.")
    # The user turn and the completed reply are both in the history.
    history = state.conversation.as_chat_pairs()
    assert history[0] == {"role": "user", "content": "Hi there"}
    assert history[1]["role"] == "assistant"
    assert "map you mentioned" in history[1]["content"]


def test_system_prompt_leads_the_request(fake_session):
    state, engine = fake_session
    list(state.reply_stream("Hello"))
    messages = engine.calls[0]
    assert messages[0]["role"] == "system"
    assert state.character.name in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Hello"}


def test_sentences_are_handed_to_speech_as_they_complete(fake_session, monkeypatch):
    state, _ = fake_session
    state.speak_replies = True
    spoken = []
    monkeypatch.setattr(state, "say", spoken.append)
    monkeypatch.setattr(state, "_wait_for_speech", lambda timeout=0: None)

    list(state.reply_stream("Hi"))

    assert len(spoken) >= 2, "the reply should be spoken in more than one piece"
    assert spoken[0].endswith(("?", ".", "!")), spoken[0]
    # Nothing is lost between the pieces and the final text.
    assert "".join(spoken).replace("  ", " ").startswith("Good to hear from you.")


def test_second_turn_keeps_the_first_in_context(fake_session):
    state, engine = fake_session
    list(state.reply_stream("First question"))
    list(state.reply_stream("Second question"))
    messages = engine.calls[1]
    contents = [m["content"] for m in messages]
    assert "First question" in contents
    assert "Second question" in contents


def test_missing_model_surfaces_as_a_readable_message(monkeypatch):
    class BrokenEngine:
        def stream_chat(self, character, messages):
            raise llm.ModelNotAvailable("no model installed")
            yield  # pragma: no cover

        def unload(self):
            pass

    monkeypatch.setattr(llm, "engine", lambda backend=None: BrokenEngine())
    session._session = None
    state = session.get()
    state.speak_replies = False
    output = list(state.reply_stream("hello"))
    assert output == ["[no model installed]"]


def test_readiness_reports_every_component(fake_session):
    state, _ = fake_session
    labels = [row[0] for row in state.readiness()]
    for expected in ("Language model", "Speech recognition", "Voice", "Image model", "Network"):
        assert expected in labels
