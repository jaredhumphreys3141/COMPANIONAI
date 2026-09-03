from companionai import character as character_mod


def test_round_trip(tmp_path):
    original = character_mod.new_from_profile("Nova")
    original.persona = "Loves maps and terrible puns."
    original.temperature = 0.42
    original.save()

    loaded = character_mod.Character.load("nova")
    assert loaded.name == "Nova"
    assert loaded.persona == original.persona
    assert loaded.temperature == 0.42


def test_system_prompt_mentions_identity_and_offline_status():
    character = character_mod.Character(name="Kip", traits=["blunt"], user_name="Sam")
    prompt = character.system_prompt()
    assert "Kip" in prompt and "Sam" in prompt and "blunt" in prompt
    assert "no internet access" in prompt


def test_custom_system_prompt_wins():
    character = character_mod.Character(custom_system_prompt="You are a lighthouse.")
    assert character.system_prompt() == "You are a lighthouse."


def test_image_prompt_includes_style_and_scene():
    character = character_mod.Character(art_style="anime", appearance="silver hair")
    prompt = character.image_prompt("standing on a bridge at night")
    assert "silver hair" in prompt
    assert "standing on a bridge at night" in prompt
    assert "anime" in prompt


def test_duplicate_gets_a_new_id_and_no_portrait():
    source = character_mod.Character(name="Ada", portrait="/tmp/x.png").save()
    clone = source.copy_as("Ada Two")
    assert clone.id == "ada-two"
    assert clone.portrait == ""


def test_ensure_default_creates_one_companion():
    assert character_mod.list_ids() == []
    created = character_mod.ensure_default()
    assert character_mod.list_ids() == [created.id]
    # Calling again returns the same companion rather than making another.
    assert character_mod.ensure_default().id == created.id


def test_slugify():
    assert character_mod.slugify("  Captain  Nemo! ") == "captain-nemo"
    assert character_mod.slugify("***") == "companion"
