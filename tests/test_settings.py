from companionai import config


def test_defaults_are_offline_and_local():
    settings = config.Settings()
    assert settings.network_policy == config.OFFLINE
    assert settings.host == "127.0.0.1"


def test_update_persists():
    config.update(port=7999, mic_gain=1.5)
    reloaded = config.Settings.load()
    assert reloaded.port == 7999
    assert reloaded.mic_gain == 1.5


def test_unknown_keys_are_ignored_on_load():
    from companionai import paths

    paths.CONFIG_FILE.write_text('{"port": 1234, "not_a_setting": true}', "utf-8")
    settings = config.Settings.load()
    assert settings.port == 1234
    assert not hasattr(settings, "not_a_setting")
