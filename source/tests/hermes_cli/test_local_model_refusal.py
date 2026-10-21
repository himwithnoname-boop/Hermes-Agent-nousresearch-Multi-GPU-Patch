"""Managed-model admission failures reach runtime startup before assistant construction."""

import pytest


@pytest.fixture
def refused_runtime(tmp_path, monkeypatch):
    from hermes_cli import config, runtime_provider
    from hermes_cli.local_runtime import endpoint, presets

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    config.save_config({"model": {"provider": "llamacpp", "default": "Large-Model"},
                        "local_runtime": {"enabled": True}})
    managed = {"base_url": "http://127.0.0.1:8123/v1", "api_key": "private-fixture-key"}
    monkeypatch.setattr(endpoint, "resolve_llamacpp_endpoint", lambda **_: managed)
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: managed)
    monkeypatch.setattr(runtime_provider, "_get_named_custom_provider", lambda *_: None)
    refusal = ("Large-Model: needs ~71.6 GiB at the 8K floor but only ~57.3 GiB "
               "of Windows commit remain after desktop headroom - close other apps, "
               "make more Windows paging-file capacity available, or use a smaller model/quant")
    from hermes_cli.local_runtime.binaries import runtimes_root
    presets.write_presets([presets.PresetEntry("Large-Model", 0, False, refusal=refusal)],
                          runtimes_root() / "presets.ini")
    return runtime_provider, endpoint, managed


@pytest.mark.parametrize("requested,target,explicit", [
    (None, None, False),
    ("llamacpp", "Large-Model", False),
    ("llama.cpp", "Large-Model", False),
    ("llama-cpp", "Large-Model", True),
    ("custom", "Large-Model", True),
])
def test_refused_managed_model_reports_capacity_before_building_agent(refused_runtime, requested, target, explicit):
    rp, _, managed = refused_runtime
    kwargs = {"requested": requested, "target_model": target}
    if explicit:
        kwargs["explicit_base_url"] = managed["base_url"]
    with pytest.raises(ValueError) as failure:
        rp.resolve_runtime_provider(**kwargs)
    message = str(failure.value)
    assert "71.6 GiB" in message and "57.3 GiB" in message
    assert "Settings" in message and "Local models" in message
    assert "private-fixture-key" not in message and managed["base_url"] not in message


@pytest.mark.parametrize("recorded", [False, True])
def test_refused_saved_default_does_not_block_session_replacement(refused_runtime, recorded):
    rp, _, _ = refused_runtime
    if recorded:
        from hermes_cli.local_runtime import presets
        from hermes_cli.local_runtime.binaries import runtimes_root
        presets.write_presets([presets.PresetEntry("Replacement", 8192, False, keys={"model": "fixture.gguf"})],
                              runtimes_root() / "presets.ini")
    runtime = rp.resolve_runtime_provider(requested="llamacpp", target_model="Replacement")
    assert runtime["source"] == "local-runtime"


@pytest.mark.parametrize("requested", ["llamacpp", "custom"])
def test_external_server_with_same_model_id_is_not_subject_to_local_refusal(refused_runtime, requested, monkeypatch):
    rp, endpoint, _ = refused_runtime
    external = {"base_url": "http://127.0.0.1:9999/v1", "api_key": ""}
    monkeypatch.setattr(endpoint, "resolve_llamacpp_endpoint", lambda **_: external)
    kwargs = {"requested": requested, "target_model": "Large-Model"}
    if requested == "custom":
        kwargs["explicit_base_url"] = external["base_url"]
    runtime = rp.resolve_runtime_provider(**kwargs)
    assert runtime["base_url"] == external["base_url"]


@pytest.mark.parametrize("session", [False, True])
def test_agent_runtime_selection_preserves_refusal_before_assistant_construction(refused_runtime, session):
    from tui_gateway.server import _resolve_agent_model_runtime

    _, _, managed = refused_runtime
    override = ({"model": "Large-Model", "provider": "custom", "base_url": managed["base_url"]}
                if session else None)
    with pytest.raises(ValueError, match="71.6 GiB"):
        _resolve_agent_model_runtime(override, None)


def test_agent_session_picker_replacement_wins_over_refused_saved_default(refused_runtime):
    from tui_gateway.server import _resolve_agent_model_runtime

    _, _, managed = refused_runtime
    model, runtime = _resolve_agent_model_runtime(
        {"model": "Replacement", "provider": "custom", "base_url": managed["base_url"]}, None)
    assert model == "Replacement"
    assert runtime["base_url"] == managed["base_url"]


@pytest.mark.parametrize("state", [None, "{invalid-json", '{"pid": 0, "base_url": "http://127.0.0.1:8123/v1"}'])
def test_unowned_or_malformed_local_state_never_blocks_generic_remote_provider(tmp_path, monkeypatch, state):
    from hermes_cli import runtime_provider
    from hermes_cli.local_runtime.binaries import runtimes_root

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    if state is not None:
        path = runtimes_root() / "server.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(state, encoding="utf-8")
    runtime = runtime_provider.resolve_runtime_provider(
        requested="custom", explicit_base_url="https://fixture.invalid/v1", target_model="Large-Model")
    assert runtime["base_url"] == "https://fixture.invalid/v1"


def test_broken_local_state_reader_does_not_block_switching_to_remote(refused_runtime, monkeypatch):
    rp, endpoint, _ = refused_runtime

    def unavailable():
        raise OSError("local state unavailable")

    monkeypatch.setattr(endpoint, "_state_endpoint", unavailable)
    runtime = rp.resolve_runtime_provider(
        requested="custom", explicit_base_url="https://fixture.invalid/v1", target_model="Large-Model")
    assert runtime["base_url"] == "https://fixture.invalid/v1"


@pytest.mark.parametrize("configured,target,blocked", [
    ("Large-Model", None, True),
    ("Large-Model", "Replacement", False),
    ("Replacement", None, False),
])
def test_named_managed_endpoint_uses_effective_provider_model(refused_runtime, monkeypatch, configured, target, blocked):
    rp, _, managed = refused_runtime
    monkeypatch.setattr(rp, "_get_named_custom_provider", lambda *_: {
        "name": "Localbox", "base_url": managed["base_url"], "api_key": "fixture", "model": configured})
    if blocked:
        with pytest.raises(ValueError, match="71.6 GiB"):
            rp.resolve_runtime_provider(requested="Localbox", target_model=target)
    else:
        runtime = rp.resolve_runtime_provider(requested="Localbox", target_model=target)
        assert runtime["base_url"] == managed["base_url"]


def test_changed_admission_decision_allows_retry_after_memory_is_available(refused_runtime):
    from hermes_cli.local_runtime import presets
    from hermes_cli.local_runtime.binaries import runtimes_root

    rp, _, _ = refused_runtime
    with pytest.raises(ValueError, match="71.6 GiB"):
        rp.resolve_runtime_provider()
    presets.write_presets([presets.PresetEntry("Large-Model", 8192, False, keys={"model": "fixture.gguf"})],
                          runtimes_root() / "presets.ini")
    runtime = rp.resolve_runtime_provider()
    assert runtime["source"] == "local-runtime"
