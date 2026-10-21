"""Charge the complete CPU expert selector, not only a capacity deficit.

The geometry below was captured from a split qwen4exp GGUF's metadata only:
a 1.18 GiB capacity deficit emitted a 55.43 GiB CPU expert override.
"""
from dataclasses import replace
from types import SimpleNamespace
import re

import pytest

from hermes_cli.local_runtime import catalog, presets
from hermes_cli.local_runtime.context_policy import (
    RUNTIME_OVERHEAD_BYTES, host_spill_bytes, plan_launch, spill_overrides,
)
from hermes_cli.local_runtime.estimator import HardwareBudget, LayerKind, ModelProfile

GIB = 1 << 30
EXPERT_BYTES = 59_519_795_200
COMPANION_BYTES = 5_044_972_224


def captured_profile():
    return ModelProfile(
        name="captured-moe", weights_bytes=64_871_421_440,
        embd_table_bytes=675_430_400, n_ctx_train=262144,
        layers=[(LayerKind.FULL, 2048) if i % 4 == 3 else (LayerKind.RECURRENT, 0)
                for i in range(48)],
        moe=True, architecture="qwen4exp", n_vocab=248320,
        lazy_bytes=28_800_138_240, recurrent_state_bytes=117_669_888,
        tensor_table_complete=True,
        tensor_bytes_by_name={
            "blk.0.ffn_up_exps.weight": EXPERT_BYTES,
            "blk.0.ffn_up.weight": 5_351_626_240,
            "per_layer_token_embd.weight": 28_800_138_240,
        },
    )


def capacity():
    return HardwareBudget(87_914_646_078, 96_609_501_184, 64 * GIB, platform="win32")


def preset(tmp_path, monkeypatch, live):
    profile = captured_profile()
    model = tmp_path / "captured-moe.gguf"
    model.touch()
    monkeypatch.setattr(catalog, "entry_for_model", lambda _: None)
    monkeypatch.setattr(presets, "read_gguf_header", lambda _: SimpleNamespace(
        sampling_defaults={}, architecture=profile.architecture))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda _: profile)
    monkeypatch.setattr(presets, "_companions", lambda _: presets._Companions(nbytes=COMPANION_BYTES))
    return presets.preset_for_model(
        model, capacity(), {model.stem}, requested_window=262144,
        live=live, engine_tag="b11706")


def test_complete_expert_selector_is_charged_for_small_spill():
    p, budget = captured_profile(), capacity()
    plan = plan_launch(p, budget, mtp_capable=True,
                       fixed_overhead=RUNTIME_OVERHEAD_BYTES + COMPANION_BYTES,
                       requested_window=262144)
    spill = plan.decision.spill_bytes
    assert 0 < spill < 2 * GIB
    selector = spill_overrides(p, spill)[1].removesuffix("=CPU")
    selected = sum(size for name, size in p.tensor_bytes_by_name.items() if re.search(selector, name))
    assert selected == EXPERT_BYTES
    assert host_spill_bytes(p, spill, budget=budget) == selected


def test_real_preset_refuses_expert_set_larger_than_live_host_ram(tmp_path, monkeypatch):
    live = replace(capacity(), ram_available_bytes=32 * GIB, commit_room_bytes=120 * GIB)
    result = preset(tmp_path, monkeypatch, live)
    assert result.refusal and "host RAM" in result.refusal
    assert result.keys is None and result.window == 0


def test_real_preset_cache_reserves_all_cpu_experts(tmp_path, monkeypatch):
    live = replace(capacity(), ram_available_bytes=56 * GIB, commit_room_bytes=120 * GIB)
    result = preset(tmp_path, monkeypatch, live)
    assert result.refusal is None and result.window == 262144
    cache_bytes = int(result.keys["cache-ram"]) << 20
    assert 0 <= cache_bytes <= live.ram_available_bytes - EXPERT_BYTES
    assert result.keys["parallel"] == "1"


def test_real_preset_commit_remainder_still_bounds_cache(tmp_path, monkeypatch):
    p, budget = captured_profile(), capacity()
    plan = plan_launch(p, budget, mtp_capable=True,
                       fixed_overhead=RUNTIME_OVERHEAD_BYTES + COMPANION_BYTES,
                       requested_window=262144)
    from hermes_cli.local_runtime.estimator import footprint_bytes
    footprint = footprint_bytes(plan.profile, plan.decision.window, overhead_bytes=plan.overhead_bytes)
    live = replace(budget, commit_room_bytes=footprint + 4 * GIB)
    result = preset(tmp_path, monkeypatch, live)
    assert result.refusal is None and result.window == 262144
    assert int(result.keys["cache-ram"]) << 20 <= 2 * GIB


def test_unspilled_moe_has_no_host_expert_charge():
    assert host_spill_bytes(captured_profile(), 0, budget=capacity()) == 0


@pytest.mark.parametrize("table,complete", [(None, False), ({"blk.0.ffn_up_exps.weight": 1}, False)])
def test_incomplete_tensor_table_uses_conservative_weight_charge(table, complete):
    p = replace(captured_profile(), tensor_bytes_by_name=table, tensor_table_complete=complete)
    assert host_spill_bytes(p, GIB, budget=capacity()) == p.weights_bytes


def test_dense_spill_estimate_is_unchanged():
    p = replace(captured_profile(), moe=False)
    assert host_spill_bytes(p, GIB, budget=capacity()) == GIB


def test_selector_that_cannot_cover_deficit_is_refused():
    p = replace(captured_profile(), tensor_bytes_by_name={"blk.0.ffn_up_exps.weight": GIB})
    with pytest.raises(ValueError, match="cannot cover"):
        host_spill_bytes(p, 2 * GIB, budget=capacity())

def test_unsafe_growth_does_not_persist_or_restart(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import bootstrap, estimator, gguf, growth, hardware

    live = replace(capacity(), ram_available_bytes=32 * GIB, commit_room_bytes=120 * GIB)
    preset(tmp_path, monkeypatch, live)  # Real preset seam remains patched to captured geometry.
    model = tmp_path / "captured-moe.gguf"
    monkeypatch.setattr(bootstrap, "get_supervisor", lambda: SimpleNamespace(is_idle=lambda _: True))
    monkeypatch.setattr(bootstrap, "staged_models", lambda: [model])
    monkeypatch.setattr(bootstrap, "_launch_budget", lambda *a, **kw: live)
    monkeypatch.setattr(growth, "is_managed_endpoint", lambda _: True)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: capacity())
    monkeypatch.setattr(gguf, "read_gguf_header", lambda _: object())
    monkeypatch.setattr(estimator, "profile_from_gguf", lambda _: captured_profile())
    monkeypatch.setattr(presets, "resident_footprint", lambda *a, **kw: 0)
    restarts = []
    monkeypatch.setattr(bootstrap, "refresh_local_runtime", lambda: restarts.append(True))
    for _ in range(2):
        assert growth.maybe_grow_window(
            model.stem, base_url="http://127.0.0.1:18434/v1", session_tokens=190000,
            current_window=221184, measured_decode_tok_s=50) is None
    assert growth.load_window_overrides() == {}
    assert restarts == []
