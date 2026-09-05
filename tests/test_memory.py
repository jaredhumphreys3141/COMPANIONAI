"""The memory system: facts, rolling summary, recall and resuming."""


from companionai import character as character_mod
from companionai import config, recall
from companionai import conversation as conversation_mod
from companionai import memory as memory_mod


class FakeEngine:
    """Answers extraction prompts; refuses to invent anything else."""

    def __init__(self, facts="", summary=""):
        self.facts = facts
        self.summary = summary
        self.prompts = []

    def stream_chat(self, character, messages):
        # Read the system message, exactly as llama.cpp and Ollama do - a real
        # engine never looks at the character's prompt fields.
        assert messages[0]["role"] == "system", "extraction must send instructions"
        system = messages[0]["content"]
        self.prompts.append(messages[-1]["content"])
        reply = self.summary if "running summary" in system else self.facts
        for word in reply.split(" "):
            yield word + " "

    def unload(self):
        pass


def _turns(*pairs):
    return [conversation_mod.Turn(role, text) for role, text in pairs]


# --------------------------------------------------------------------------- #
# facts
# --------------------------------------------------------------------------- #
def test_facts_are_deduplicated():
    memory = memory_mod.Memory(character_id="aria")
    assert memory.add_facts(["Has a cat called Biscuit"]) == 1
    assert memory.add_facts(["has a cat called biscuit."]) == 0
    assert len(memory.facts) == 1


def test_fact_list_is_capped_and_pinned_facts_survive():
    config.update(memory_max_facts=5)
    memory = memory_mod.Memory(character_id="aria")
    memory.add_facts([f"Fact number {i}" for i in range(10)])
    assert len(memory.facts) == 5
    memory.facts[0].pinned = True
    pinned_text = memory.facts[0].text
    memory.add_facts([f"Later fact {i}" for i in range(10)])
    assert len(memory.facts) == 5
    assert any(f.text == pinned_text and f.pinned for f in memory.facts)


def test_editing_facts_as_text_round_trips_with_pins():
    memory = memory_mod.Memory(character_id="aria")
    memory.set_from_text("* Lives in Leeds\n- Learning Portuguese\n\n  \n")
    assert [f.text for f in memory.facts] == ["Lives in Leeds", "Learning Portuguese"]
    assert memory.facts[0].pinned and not memory.facts[1].pinned
    assert memory.as_text() == "* Lives in Leeds\nLearning Portuguese"


def test_memory_round_trips_to_disk():
    memory = memory_mod.Memory(character_id="aria")
    memory.add_facts(["Drinks tea, never coffee"])
    memory.summary = "They talked about kitchens."
    memory.save()

    reloaded = memory_mod.Memory.load("aria")
    assert [f.text for f in reloaded.facts] == ["Drinks tea, never coffee"]
    assert reloaded.summary == "They talked about kitchens."


# --------------------------------------------------------------------------- #
# hallucination guard
# --------------------------------------------------------------------------- #
def test_ungrounded_facts_are_discarded():
    source = "I have been trying to get my sourdough starter going."
    kept = memory_mod._parse_facts(
        "Is baking sourdough bread\nOwns a yacht in Monaco\nNONE", source
    )
    assert kept == ["Is baking sourdough bread"]


def test_parse_facts_handles_none_and_preamble():
    assert memory_mod._parse_facts("NONE", "anything") == []
    assert memory_mod._parse_facts("Here are the facts:", "anything") == []
    assert memory_mod._parse_facts("- Works as a welder", "I work as a welder") == [
        "Works as a welder"
    ]


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def test_extraction_stores_facts_and_summary():
    character = character_mod.Character(name="Aria")
    memory = memory_mod.Memory(character_id="aria")
    engine = FakeEngine(
        facts="Has a cat called Biscuit\nWorks night shifts",
        summary="They discussed the cat and the night shifts.",
    )
    turns = _turns(
        ("user", "My cat Biscuit keeps me up, and I work night shifts anyway"),
        ("assistant", "That sounds exhausting."),
    )
    added, summarised = memory_mod.extract(memory, character, turns, engine)

    assert added == 2
    assert summarised
    assert memory.summary.startswith("They discussed")
    assert memory_mod.Memory.load("aria").facts, "extraction must persist"


def test_extraction_survives_a_broken_engine():
    class Broken:
        def stream_chat(self, character, messages):
            raise RuntimeError("model exploded")
            yield  # pragma: no cover

    memory = memory_mod.Memory(character_id="aria")
    added, summarised = memory_mod.extract(
        memory, character_mod.Character(), _turns(("user", "hello")), Broken()
    )
    assert (added, summarised) == (0, False)


# --------------------------------------------------------------------------- #
# the prompt block
# --------------------------------------------------------------------------- #
def test_prompt_block_carries_facts_and_summary():
    character = character_mod.Character(name="Aria", user_name="Jared")
    memory = memory_mod.Memory(character_id="aria")
    memory.add_facts(["Has a cat called Biscuit"])
    memory.summary = "They have talked about kitchens."

    block = memory.prompt_block(character)
    assert "Biscuit" in block
    assert "kitchens" in block
    assert "Jared" in block


def test_prompt_block_is_empty_when_memory_is_off():
    config.update(memory_enabled=False)
    memory = memory_mod.Memory(character_id="aria")
    memory.add_facts(["Has a cat called Biscuit"])
    assert memory.prompt_block(character_mod.Character()) == ""


def test_prompt_block_recalls_by_keyword():
    config.update(memory_enabled=True, recall_enabled=True, recall_top_k=3)
    recall.index().add("aria", "user", "My sourdough starter keeps dying on me")
    recall.index().add("aria", "user", "the weather is lovely")

    memory = memory_mod.Memory(character_id="aria")
    block = memory.prompt_block(character_mod.Character(), query="how is the sourdough going")
    assert "sourdough starter" in block
    assert "weather" not in block


def test_memory_reaches_the_system_prompt():
    character = character_mod.Character(name="Aria")
    memory = memory_mod.Memory(character_id="aria")
    memory.add_facts(["Has a cat called Biscuit"])

    conversation = conversation_mod.Conversation(character, memory)
    conversation.add("user", "how are you")
    system = conversation.messages()[0]["content"]
    assert "Biscuit" in system
    assert character.name in system, "the persona must survive alongside memory"


# --------------------------------------------------------------------------- #
# resuming
# --------------------------------------------------------------------------- #
def test_conversation_resumes_from_the_last_transcript():
    character = character_mod.Character(name="Aria").save()
    first = conversation_mod.Conversation(character)
    first.add("user", "remember the kitchen paint")
    first.add("assistant", "Soft clay, I said.")
    first.save_transcript()

    second = conversation_mod.Conversation(character)
    assert second.resume() == 2
    assert second.resumed
    assert second.as_chat_pairs()[0]["content"] == "remember the kitchen paint"
    # It keeps writing to the same file instead of orphaning it.
    assert second.started == first.started


def test_resume_is_a_no_op_with_no_history():
    character = character_mod.Character(name="Nobody")
    assert conversation_mod.Conversation(character).resume() == 0


def test_session_reopens_the_last_conversation(monkeypatch):
    from companionai import llm, session

    character = character_mod.ensure_default()
    prior = conversation_mod.Conversation(character)
    prior.add("user", "we were talking about paint")
    prior.add("assistant", "We were.")
    prior.save_transcript()

    config.update(resume_conversations=True)
    monkeypatch.setattr(llm, "engine", lambda backend=None: FakeEngine())
    session._session = None
    state = session.get()
    assert state.conversation.resumed
    assert any("paint" in t["content"] for t in state.conversation.as_chat_pairs())


def test_session_starts_fresh_when_resume_is_off(monkeypatch):
    from companionai import llm, session

    character = character_mod.ensure_default()
    prior = conversation_mod.Conversation(character)
    prior.add("user", "old talk")
    prior.save_transcript()

    config.update(resume_conversations=False)
    monkeypatch.setattr(llm, "engine", lambda backend=None: FakeEngine())
    session._session = None
    state = session.get()
    assert not state.conversation.resumed
    assert state.conversation.as_chat_pairs() == []


def test_forgetting_clears_facts_summary_and_recall():
    recall.index().add("aria", "user", "something memorable about otters")
    memory = memory_mod.Memory(character_id="aria")
    memory.add_facts(["Likes otters"])
    memory.summary = "otters"
    memory.forget()

    assert memory.facts == [] and memory.summary == ""
    assert recall.index().search("aria", "otters") == []


# --------------------------------------------------------------------------- #
# recall index
# --------------------------------------------------------------------------- #
def test_recall_query_is_injection_safe():
    recall.index().add("aria", "user", "normal line")
    assert recall.index().search("aria", '" OR 1=1 --') == []
    assert recall.index().search("aria", "MATCH * (") == []


def test_recall_ignores_stopword_only_queries():
    recall.index().add("aria", "user", "a real line about carpentry")
    assert recall.index().search("aria", "do you remember the thing") == []
    assert recall.index().search("aria", "carpentry") != []


def test_recall_is_scoped_per_companion():
    recall.index().add("aria", "user", "aria knows about kayaks")
    recall.index().add("nova", "user", "nova knows about kayaks")
    hits = recall.index().search("aria", "kayaks")
    assert len(hits) == 1
    assert "aria knows" in hits[0][2]


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #
def test_extraction_uses_the_exchange_it_was_given_not_the_live_conversation(monkeypatch):
    """Background memory passes can finish out of order.

    They used to re-read the conversation at execution time, so a pass spawned
    for exchange 1 could run after exchange 3 and extract facts against the
    wrong text - silently losing everything said earlier.
    """
    from companionai import llm, session

    engine = FakeEngine(facts="Has a dog called Rex", summary="")
    monkeypatch.setattr(llm, "engine", lambda backend=None: engine)
    config.update(memory_enabled=True, memory_extract_every=1, memory_summarise=False)

    session._session = None
    state = session.get()
    state.speak_replies = False

    # The live conversation is about something else entirely.
    state.conversation.add("user", "let's talk about bicycles instead")
    state.conversation.add("assistant", "Bicycles it is.")

    state._remember("I have a dog called Rex", "Rex sounds lovely.")

    assert [f.text for f in state.memory.facts] == ["Has a dog called Rex"]
    assert "Rex" in engine.prompts[0], "extraction saw the wrong exchange"


def test_concurrent_memory_passes_do_not_lose_facts(monkeypatch):
    import threading

    from companionai import llm, session

    replies = {
        "cat": "Has a cat called Biscuit",
        "shift": "Works night shifts",
    }

    class PerExchangeEngine:
        def stream_chat(self, character, messages):
            prompt = messages[-1]["content"]
            key = "cat" if "cat" in prompt else "shift"
            for word in replies[key].split(" "):
                yield word + " "

        def unload(self):
            pass

    monkeypatch.setattr(llm, "engine", lambda backend=None: PerExchangeEngine())
    config.update(memory_enabled=True, memory_extract_every=1, memory_summarise=False)

    session._session = None
    state = session.get()
    state.speak_replies = False

    threads = [
        threading.Thread(target=state._remember, args=("my cat keeps me up", "oh dear")),
        threading.Thread(target=state._remember, args=("I work night shift", "rough")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert sorted(f.text for f in state.memory.facts) == [
        "Has a cat called Biscuit",
        "Works night shifts",
    ]
    assert state.memory.messages == 4
