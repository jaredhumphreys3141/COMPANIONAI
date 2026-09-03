from companionai import catalog, hardware


def test_every_kind_has_entries():
    for kind in catalog.KINDS:
        assert catalog.entries(kind), f"no entries for {kind}"


def test_entries_are_well_formed():
    for kind in catalog.KINDS:
        for entry in catalog.entries(kind):
            assert entry.id and entry.name
            assert entry.source in ("gguf", "hf-repo", "piper", "diffusers")
            if entry.source in ("gguf", "piper"):
                assert entry.url.startswith("https://")
                assert entry.filename
            else:
                assert entry.repo
            assert entry.size_mb > 0


def test_nothing_is_installed_in_a_fresh_data_directory():
    for kind in catalog.KINDS:
        assert catalog.installed(kind) == []


def test_each_target_board_has_a_recommended_model():
    for device in (hardware.PC, hardware.RPI5, hardware.JETSON):
        for kind in catalog.KINDS:
            suitable = [e for e in catalog.entries(kind) if device in e.good_for]
            assert suitable, f"no {kind} model suits {device}"


def test_profile_defaults_exist_in_the_catalogue():
    for device in (hardware.PC, hardware.RPI5, hardware.JETSON):
        info = hardware.HardwareInfo(
            device=device, label=device, machine="aarch64", system="Linux",
            cpu_count=4, total_ram_gb=8.0,
        )
        profile = hardware.profile_for(info)
        assert catalog.get("llm", profile.llm_model) is not None
        assert catalog.get("asr", profile.asr_model) is not None
        assert catalog.get("image", profile.image_model) is not None


def test_user_catalogue_extends_the_builtin(tmp_path):
    import json

    from companionai import paths

    (paths.HOME / "catalog.user.json").write_text(
        json.dumps({"llm": [{"id": "my-model", "name": "Mine", "kind": "gguf",
                             "url": "https://example.com/m.gguf",
                             "filename": "m.gguf", "size_mb": 10}]}),
        "utf-8",
    )
    assert catalog.get("llm", "my-model") is not None


def test_table_shape_matches_headers():
    for kind in catalog.KINDS:
        for row in catalog.table(kind):
            assert len(row) == len(catalog.TABLE_HEADERS)
