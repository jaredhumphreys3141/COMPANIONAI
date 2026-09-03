"""The offline guarantee is the feature most worth a regression test."""

import pytest

from companionai import config, net


def test_offline_by_default():
    assert net.status().policy == config.OFFLINE
    assert not net.is_allowed()


def test_require_blocks_when_offline():
    with pytest.raises(net.NetworkBlocked):
        net.require("https://huggingface.co/some/model", "test download")


def test_blocked_attempt_is_audited():
    with pytest.raises(net.NetworkBlocked):
        net.require("https://huggingface.co/some/model")
    entries = net.audit_log()
    assert entries and entries[0]["action"] == "blocked"
    assert entries[0]["allowed"] is False


def test_approval_then_revocation():
    net.approve(config.SESSION)
    assert net.is_allowed()
    net.require("https://huggingface.co/model")     # does not raise
    net.revoke()
    assert not net.is_allowed()
    with pytest.raises(net.NetworkBlocked):
        net.require("https://huggingface.co/model")


def test_host_allow_list_is_enforced():
    net.approve(config.SESSION)
    config.update(allow_hosts=["huggingface.co"])
    net.require("https://huggingface.co/model")
    with pytest.raises(net.NetworkBlocked):
        net.require("https://telemetry.example.com/beacon")


def test_session_approval_does_not_survive_restart():
    net.approve(config.SESSION)
    assert net.is_allowed()
    net.init()                       # simulates the next launch
    assert not net.is_allowed()


def test_remembered_approval_does_survive_restart():
    net.approve(config.ALWAYS)
    net.init()
    assert net.is_allowed()


def test_download_refuses_while_offline(tmp_path):
    with pytest.raises(net.NetworkBlocked):
        net.download("https://huggingface.co/model.gguf", tmp_path / "model.gguf")
    assert not (tmp_path / "model.gguf").exists()


def test_offline_env_vars_track_policy():
    import os

    net.revoke()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    net.approve(config.ALWAYS)
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_catalog_install_is_gated():
    from companionai import catalog

    with pytest.raises(net.NetworkBlocked):
        catalog.install("tts", "en_US-amy-medium")
