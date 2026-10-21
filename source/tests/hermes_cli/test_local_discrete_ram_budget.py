"""Discrete GPUs must leave host RAM/commit for offload and prompt caches."""
from dataclasses import replace
from hermes_cli.local_runtime import bootstrap, context_policy, hardware, presets
from hermes_cli.local_runtime.estimator import HardwareBudget, ModelProfile, PhysicsRefusal

G = 1 << 30


def _capacity():
    return HardwareBudget(40 * G, 48 * G, 64 * G)


def test_discrete_launch_limits_host_spill_to_current_ram_and_commit(monkeypatch):
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (64 * G, 16 * G))
    monkeypatch.setattr(hardware, "_windows_commit_room", lambda: 8 * G)
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: (48 * G, 48 * G, "GPU", None))
    live = hardware.launch_budget(_capacity())
    assert live.ram_available_bytes == 8 * G
    assert live.commit_room_bytes == 8 * G
    assert live.usable_vram_bytes == 40 * G


def test_discrete_launch_keeps_host_headroom_without_crediting_gpu_bytes(monkeypatch):
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (64 * G, 8 * G))
    monkeypatch.setattr(hardware, "_windows_commit_room", lambda: None)
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: (48 * G, 48 * G, "GPU", None))
    live = hardware.launch_budget(_capacity(), own_bytes=32 * G)
    assert live.ram_available_bytes == 8 * G - int(64 * G * hardware._MARGIN_FRACTION)


def test_discrete_cache_shares_host_room_after_spill_without_uma_multiplier():
    live = replace(_capacity(), ram_available_bytes=8 * G, commit_room_bytes=92 * G)
    assert hardware.prompt_cache_bytes(live, 42 * G, residents=2) == 2 * G
    assert hardware.prompt_cache_bytes(live, 44 * G, residents=2) == 0


def test_discrete_dflash_does_not_use_uma_cache_conversion():
    live = replace(_capacity(), ram_available_bytes=8 * G, commit_room_bytes=8 * G)
    assert hardware.usable_beside_prompt_cache(live, residents=2) == 40 * G


def test_spilled_capacity_plan_rechecks_actual_host_room():
    profile = ModelProfile("large", 54 * G, 0, 65536, [])
    plan = context_policy.plan_launch(profile, _capacity())
    assert plan.decision.spilled
    live = replace(_capacity(), ram_available_bytes=8 * G)
    fitted = context_policy.fit_to_free_memory(plan, profile, live)
    assert isinstance(fitted.decision, PhysicsRefusal)
    assert fitted.decision.available_bytes == 48 * G


def test_discrete_residency_count_respects_other_programs_gpu_memory(tmp_path, monkeypatch):
    files = [tmp_path / "one.gguf", tmp_path / "two.gguf"]
    monkeypatch.setattr(bootstrap, "staged_in", lambda directory: files)
    monkeypatch.setattr(presets, "_launch_footprint", lambda *args: 18 * G)
    assert presets.admitted_residency_count(tmp_path, _capacity(), 4) == 2
    live = replace(_capacity(), usable_vram_bytes=24 * G, ram_available_bytes=8 * G)
    assert presets.admitted_residency_count(tmp_path, _capacity(), 4, live=live) == 1


def test_boot_passes_live_discrete_budget_to_residency_admission(tmp_path, monkeypatch):
    capacity = _capacity()
    live = replace(capacity, usable_vram_bytes=24 * G)
    monkeypatch.setattr(hardware, "probe_budget", lambda **kw: capacity)
    monkeypatch.setattr(bootstrap, "_launch_budget", lambda budget: live)
    received = []
    monkeypatch.setattr(presets, "admitted_residency_count", lambda *args, **kw:
                        received.append(kw["live"]) or 1)
    assert bootstrap._admitted_models_max(tmp_path, 4) == 1
    assert received == [live]


def test_cuda_launch_keeps_live_host_limits(monkeypatch):
    from hermes_cli.local_runtime import multigpu

    capacity = replace(_capacity(), cuda_devices=(object(),))
    monkeypatch.setattr(multigpu, "cuda_launch_budget", lambda *a, **kw: capacity)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (64 * G, 16 * G))
    monkeypatch.setattr(hardware, "_windows_commit_room", lambda: 8 * G)
    assert hardware.launch_budget(capacity).ram_available_bytes == 8 * G


def test_optional_spill_draft_cannot_spend_capacity_ram(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.local_runtime import catalog

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    main, draft = tmp_path / "large.gguf", tmp_path / "draft.gguf"
    draft.touch()
    entry = SimpleNamespace(draft=object(), draft_spec_type="draft-dspark", mmproj=None,
                            mtp_head=None, mtp=False, mtp_draft_depth=3, sampling={})
    main_profile = ModelProfile("large", 54 * G, 0, 65536, [])
    draft_profile = ModelProfile("draft", G, 0, 65536, [])
    monkeypatch.setattr(catalog, "entry_for_model", lambda mid: entry)
    monkeypatch.setattr(presets, "_asset_path", lambda asset: draft if asset is entry.draft else None)
    monkeypatch.setattr(presets, "read_gguf_header", lambda path:
                        SimpleNamespace(path=path, sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda header:
                        draft_profile if header.path == draft else main_profile)
    capacity = presets.preset_for_model(main, _capacity(), set())
    assert capacity.keys["model-draft"] == str(draft)
    live = replace(_capacity(), ram_available_bytes=16 * G)
    fitted = presets.preset_for_model(main, _capacity(), set(), live=live)
    assert fitted.spilled and "model-draft" not in fitted.keys

    # Extra model room already allocated to prompt caching cannot also fund the optional draft.
    cached = presets.preset_for_model(main, _capacity(), set(),
                                     live=replace(_capacity(), ram_available_bytes=22 * G))
    assert int(cached.keys["cache-ram"]) > 0
    assert "model-draft" not in cached.keys
