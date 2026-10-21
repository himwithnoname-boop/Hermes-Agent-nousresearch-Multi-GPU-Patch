"""Dedicated GPU allocations and host caches compete for Windows commit backing."""
from dataclasses import replace
from types import SimpleNamespace
import threading

from hermes_cli.local_runtime import bootstrap, context_policy as policy, hardware, presets
from hermes_cli.local_runtime.estimator import (
    HardwareBudget, LayerKind, ModelProfile, PhysicsRefusal, footprint_bytes, physics_check)

G = 1 << 30


def _card(commit=None):
    return HardwareBudget(80 * G, 90 * G, 50 * G, commit_room_bytes=commit)


def _profile(weights=60):
    return ModelProfile("multi-card", weights * G, 0, 262144, [(LayerKind.FULL, 65536)])


def test_gpu_resident_floor_cannot_spend_physical_ram_as_extra_commit():
    # Constrained host commit must limit the simulated cache grant.
    refusal = physics_check(_profile(), _card(G), policy.FLOOR,
                            overhead_bytes=policy.RUNTIME_OVERHEAD_BYTES)
    assert isinstance(refusal, PhysicsRefusal)
    assert refusal.available_bytes == G
    assert "Windows commit" in refusal.message


def test_commit_room_narrows_window_and_recovers_after_unload():
    profile = _profile()
    capacity = policy.plan_launch(profile, _card())
    limited = policy.fit_to_free_memory(capacity, profile, _card(68 * G))
    assert not limited.decision.spilled
    assert policy.FLOOR <= limited.decision.window < capacity.decision.window
    assert footprint_bytes(limited.profile, limited.decision.window,
                           overhead_bytes=limited.overhead_bytes) <= 68 * G
    recovered = policy.fit_to_free_memory(capacity, profile, _card(86 * G))
    assert recovered.decision.window == capacity.decision.window


def test_commit_floor_refusal_is_not_replaced_by_the_resident_floor_fallback():
    profile = _profile()
    capacity = policy.plan_launch(profile, _card())
    live = replace(_card(50 * G), usable_vram_bytes=G)
    fitted = policy.fit_to_free_memory(capacity, profile, live)
    assert isinstance(fitted.decision, PhysicsRefusal)
    assert "Windows commit" in fitted.decision.message


def test_discrete_prompt_cache_reserves_full_gpu_cost_without_uma_multiplier():
    assert hardware.prompt_cache_bytes(_card(84 * G), 80 * G) == 2 * G
    assert hardware.prompt_cache_bytes(_card(80 * G), 80 * G) == 0
    assert hardware.prompt_cache_bytes(_card(), 80 * G) == 8 * G


def test_resident_count_respects_full_model_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "staged_in", lambda d: [tmp_path / "a.gguf", tmp_path / "b.gguf"])
    monkeypatch.setattr(presets, "_launch_footprint", lambda *args: 28 * G)
    assert presets.admitted_residency_count(tmp_path, _card(), 4) == 2
    assert presets.admitted_residency_count(tmp_path, _card(), 4, live=_card(50 * G)) == 1


def test_growth_never_grants_a_rung_exceeding_discrete_commit():
    decision = policy.growth_decision(
        _profile(), _card(60 * G), current_window=65536, session_tokens=64000,
        measured_decode_tok_s=55, server_idle=True)
    assert decision.action == "compress-default"


def test_idle_refit_retries_when_commit_recovers_without_gpu_memory_changing(tmp_path, monkeypatch):
    path = tmp_path / "presets.ini"
    path.write_text("")
    live = [_card(50 * G)]
    clock = [0.0]
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: _card())
    monkeypatch.setattr(bootstrap, "_launch_budget", lambda capacity: live[0])
    calls = []
    monkeypatch.setattr(presets, "plan_presets", lambda *a, **kw: calls.append(kw["live"]) or [])
    router = SimpleNamespace(preset_path=path, models_max=1, _refit_usable=None,
                             _lifecycle_lock=threading.RLock(), proc=SimpleNamespace(poll=lambda: None),
                             models=lambda **kw: {}, reload_presets=lambda: None)
    assert bootstrap.refit_idle_presets(router) is False
    assert bootstrap.refit_idle_presets(router) is False
    live[0] = _card(84 * G)
    assert bootstrap.refit_idle_presets(router) is False
    assert len(calls) == 1  # A larger budget must first remain stable.
    clock[0] = 61.0
    assert bootstrap.refit_idle_presets(router) is False
    assert len(calls) == 2


def test_low_commit_cache_keeps_commit_uncertainty_room_and_small_models_keep_default():
    live = _card(int(86.9 * G))
    cache = hardware.prompt_cache_bytes(live, int(80.66120012942702 * G))
    assert 3.11 * G < cache < 3.13 * G
    assert hardware.prompt_cache_bytes(live, 70 * G) == 8 * G


def test_low_commit_growth_defers_without_stopping_the_running_model(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import estimator, gguf, growth

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    model = tmp_path / "multi-card.gguf"
    model.touch()
    header = SimpleNamespace(path=model, sampling_defaults={})
    monkeypatch.setattr(bootstrap, "staged_models", lambda: [model])
    monkeypatch.setattr(bootstrap, "get_supervisor", lambda: SimpleNamespace(is_idle=lambda mid: True))
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda url: True)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: _card())
    monkeypatch.setattr(bootstrap, "_launch_budget", lambda *a, **kw: _card(G))
    monkeypatch.setattr(gguf, "read_gguf_header", lambda path: header)
    monkeypatch.setattr(presets, "read_gguf_header", lambda path: header)
    monkeypatch.setattr(estimator, "profile_from_gguf", lambda header: _profile())
    monkeypatch.setattr(presets, "profile_from_gguf", lambda header: _profile())
    monkeypatch.setattr(presets, "read_preset_decisions", dict)
    monkeypatch.setattr(presets, "resident_footprint", lambda *a, **kw: 60 * G)
    calls = []
    monkeypatch.setattr(bootstrap, "refresh_local_runtime", lambda: calls.append("stopped") or True)
    monkeypatch.setattr(growth, "save_window_override", lambda *a: calls.append("override"))
    assert growth.maybe_grow_window("multi-card", base_url="http://localhost/v1",
                                    current_window=65536, session_tokens=64000,
                                    measured_decode_tok_s=55) is None
    assert calls == []


def test_equal_window_and_gpu_pressure_fallback_reprice_mtp_posture_against_commit():
    profile = ModelProfile("mtp", 60 * G, 0, 65536, [(LayerKind.FULL, 65536)], n_vocab=200000)
    capacity = HardwareBudget(80 * G, 90 * G, 100 * G)
    original = policy.plan_launch(profile, capacity, mtp_capable=True)
    assert original.mtp_prefill
    for gpu_free in (1, 66):
        live = HardwareBudget(gpu_free * G, 90 * G, 100 * G, commit_room_bytes=65 * G)
        fitted = policy.fit_to_free_memory(original, profile, live, mtp_capable=True)
        assert fitted.decision.window == 65536 and not fitted.decision.spilled
        assert not fitted.mtp_prefill
        assert footprint_bytes(fitted.profile, fitted.decision.window,
                               overhead_bytes=fitted.overhead_bytes) <= 65 * G
