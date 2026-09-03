from companionai import character as character_mod
from companionai.conversation import Conversation, _split_ready, speakable


def test_split_ready_waits_for_a_full_sentence():
    sentence, rest = _split_ready("I was thinking about")
    assert sentence == ""
    assert rest == "I was thinking about"


def test_split_ready_emits_complete_sentences():
    sentence, rest = _split_ready("That sounds lovely, honestly. And then ")
    assert sentence == "That sounds lovely, honestly."
    assert rest == "And then "


def test_split_ready_merges_a_short_opener():
    # "Sure." on its own is too short to be worth a separate utterance.
    sentence, rest = _split_ready("Sure. Let me think about that for a second. Then")
    assert sentence.startswith("Sure.")
    assert "think about that" in sentence
    assert rest == "Then"


def test_speakable_strips_markup_and_stage_directions():
    assert speakable("*smiles* **Hello** there") == "Hello there"
    assert speakable("- one\n- two") == "- one - two"


def test_memory_window_limits_prompt_length():
    character = character_mod.Character(memory_turns=2)
    conversation = Conversation(character)
    for index in range(10):
        conversation.add("user", f"u{index}")
        conversation.add("assistant", f"a{index}")
    messages = conversation.messages()
    assert messages[0]["role"] == "system"
    assert len(messages) == 1 + 4          # system + 2 exchanges
    assert messages[-1]["content"] == "a9"


def test_reset_clears_history():
    conversation = Conversation(character_mod.Character())
    conversation.add("user", "hi")
    conversation.reset()
    assert conversation.as_chat_pairs() == []
