"""Multiple CUDA cards must be priced and launched as the same device set."""
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import hardware, presets
from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile

GIB = 1 << 30


def _inventory(monkeypatch, tmp_path, sizes, free=None):
    from hermes_cli.local_runtime import multigpu

    from hermes_cli.local_runtime import cuda_memory

    monkeypatch.setattr(cuda_memory, "bound_memory", lambda binary, devices: devices)
    monkeypatch.setattr(multigpu, "_cache", {})
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (256 * GIB, 200 * GIB))
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: (48 * GIB, 47 * GIB, "physical first", None))
    monkeypatch.setattr(hardware, "_unified_pool_bytes", lambda *_: None)
    monkeypatch.setattr(hardware, "_configured_engine", lambda: SimpleNamespace(
        backend="cuda", binary=tmp_path / "llama-server.exe"))
    rows = [{"name": f"CUDA{i}", "description": f"card {i}", "type": 1,
             "total": size * GIB, "free": (free or sizes)[i] * GIB}
            for i, size in enumerate(sizes)]
    monkeypatch.setattr(multigpu, "_list_devices", lambda *_: "\n".join(
        f"  {d['name']}: {d['description']} ({d['total'] >> 20} MiB, {d['free'] >> 20} MiB free)"
        for d in rows))
    return rows


@pytest.mark.parametrize("sizes", [(48, 48), (48, 48, 32), (48, 48, 96)])
def test_all_visible_cards_fund_a_model_and_reach_the_router_preset(tmp_path, monkeypatch, sizes):
    _inventory(monkeypatch, tmp_path, sizes)
    budget = hardware.probe_budget(planning=True)
    assert budget.total_device_bytes == sum(sizes) * GIB
    assert budget.usable_vram_bytes == sum(
        size * GIB - max(hardware._MARGIN_FLOOR, int(size * GIB * hardware._MARGIN_FRACTION))
        for size in sizes)
    monkeypatch.setattr(presets, "read_gguf_header", lambda _: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda _: ModelProfile(
        "larger-than-one-card", 60 * GIB, 0, 8192, [(LayerKind.FULL, 4096)] * 64))
    gguf = tmp_path / "larger-than-one-card.gguf"
    gguf.touch()
    entry = presets.preset_for_model(gguf, budget, set())
    assert entry.refusal is None and not entry.spilled
    ini = tmp_path / "presets.ini"
    ini.write_text(presets.render_presets([entry]), encoding="utf-8")
    keys = presets.read_preset_decisions(ini)[entry.model_id].keys
    assert keys["device"] == ",".join(f"CUDA{i}" for i in range(len(sizes)))
    assert keys["split-mode"] == "layer"
    assert keys["flash-attn"] == "on"
    assert keys["cache-type-k"] == keys["cache-type-v"] == "q8_0"
    assert "override-tensor" not in keys
    ratios = [float(x) for x in keys["tensor-split"].split(",")]
    assert ratios[0] / ratios[-1] == pytest.approx(sizes[0] / sizes[-1], rel=0.02)


def test_engine_visibility_and_free_memory_bound_each_card(tmp_path, monkeypatch):
    rows = _inventory(monkeypatch, tmp_path, (48, 32), (8, 30))
    capacity = hardware.probe_budget(planning=True)
    live = hardware.launch_budget(capacity)
    assert live.usable_vram_bytes < capacity.usable_vram_bytes
    assert live.cuda_split_bytes[0] <= 7 * GIB
    assert live.cuda_split_bytes[1] <= 29 * GIB
    # An engine masked to one card must never count physical GPUs hidden from CUDA.
    rows.pop(0)
    from hermes_cli.local_runtime import multigpu

    multigpu._cache.clear()
    budget = hardware.probe_budget(planning=True)
    assert budget.total_device_bytes == 32 * GIB


def test_hidden_devices_and_failed_queries_do_not_invent_capacity(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import multigpu

    rows = _inventory(monkeypatch, tmp_path, (48, 48))
    rows.clear()
    masked = hardware.probe_budget(planning=True)
    assert masked.uma and not masked.cuda_devices
    # A driver/probe failure retains the old conservative single-card fallback.
    def fail(_):
        raise OSError("driver unavailable")

    multigpu._cache.clear()
    monkeypatch.setattr(multigpu, "_list_devices", fail)
    fallback = hardware.probe_budget(planning=True)
    assert fallback.total_device_bytes == 48 * GIB
    assert not fallback.cuda_devices

def test_every_staged_model_gets_placement_including_models_added_later(tmp_path, monkeypatch):
    _inventory(monkeypatch, tmp_path, (48, 48))
    budget = hardware.probe_budget(planning=True)
    monkeypatch.setattr(presets, "read_gguf_header", lambda p: SimpleNamespace(
        name=p.stem, sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda h: ModelProfile(
        h.name, 60 * GIB, 0, 8192, [(LayerKind.FULL, 4096)] * 64))
    models = tmp_path / "models"
    models.mkdir()
    ini = tmp_path / "presets.ini"
    for model_id in ("local-a", "local-b", "local-c"):
        (models / f"{model_id}.gguf").touch()
        presets.generate_presets(models, budget, ini)
        entries = presets.read_preset_decisions(ini)
        assert set(entries) == {p.stem for p in models.glob("*.gguf")}
        for entry in entries.values():
            assert not entry.refusal and not entry.spilled
            assert entry.keys["device"] == "CUDA0,CUDA1"
            assert entry.keys["split-mode"] == "layer"
            assert entry.keys["tensor-split"]
            assert entry.keys["model"] == str(models / f"{entry.model_id}.gguf")

def test_ui_start_replans_legacy_presets_and_changed_gpu_inventory(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import binaries, bootstrap, multigpu

    rows = _inventory(monkeypatch, tmp_path, (48, 48))
    budget = hardware.probe_budget(planning=True)
    models = tmp_path / "models"
    models.mkdir()
    model = models / "existing-ui-model.gguf"
    model.touch()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    ini = runtime / "presets.ini"
    monkeypatch.setattr(bootstrap, "models_dir", lambda: models)
    monkeypatch.setattr(binaries, "runtimes_root", lambda: runtime)
    ini.write_text(presets.render_presets([presets.PresetEntry(
        model.stem, 8192, True, keys={"model": str(model), "ctx-size": "8192"})]))
    assert bootstrap._presets_stale(), "The UI must not adopt a pre-patch single-card launch policy"
    monkeypatch.setattr(presets, "read_gguf_header", lambda _: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda _: ModelProfile(
        "existing-ui-model", 60 * GIB, 0, 8192, [(LayerKind.FULL, 4096)] * 64))
    presets.generate_presets(models, budget, ini)
    assert not bootstrap._presets_stale()
    rows.append({"name": "CUDA2", "description": "third card", "type": 1,
                 "total": 32 * GIB, "free": 32 * GIB})
    multigpu._cache.clear()
    assert bootstrap._presets_stale(), "Adding a card must update existing model policies"


def test_refused_models_replan_once_for_new_gpu_capacity(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import multigpu

    _inventory(monkeypatch, tmp_path, (48, 48))
    legacy = presets.PresetEntry("too-large", 0, False, refusal="Single card too small")
    assert multigpu.presets_need_migration({legacy.model_id: legacy}, [legacy.model_id])
    budget = hardware.probe_budget(planning=True)
    monkeypatch.setattr(presets, "read_gguf_header", lambda _: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda _: ModelProfile(
        "still-too-large", 400 * GIB, 0, 8192, [(LayerKind.FULL, 4096)] * 64))
    model = tmp_path / "still-too-large.gguf"
    model.touch()
    entry = presets.preset_for_model(model, budget, set())
    assert entry.refusal
    ini = tmp_path / "presets.ini"
    presets.write_presets([entry], ini)
    known = presets.read_preset_decisions(ini)
    assert not multigpu.presets_need_migration(known, [entry.model_id])


@pytest.mark.platforms("windows")
def test_windows_memory_uses_pci_identity_in_visible_cuda_order(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import cuda_memory, multigpu

    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "nvidia-smi")
    # Identical cards, with CUDA_VISIBLE_DEVICES reversing their physical order.
    monkeypatch.setattr(cuda_memory, "_buses", lambda _, **kw: ("0000:42:00.0", "0000:21:00.0"))
    monkeypatch.setattr(cuda_memory, "_run", lambda *_: (
        "00000000:21:00.0, 27000, 18000, 30\n00000000:42:00.0, 23000, 22000, 40\n"))
    cards = tuple(multigpu.CudaDevice(f"CUDA{i}", "identical RTX", 46068 << 20, 44000 << 20)
                  for i in range(2))
    corrected = cuda_memory.bound_memory(tmp_path / "llama-server.exe", cards)
    assert [d.free >> 20 for d in corrected] == [23000, 27000]
    assert [d.used >> 20 for d in corrected] == [22000, 18000]
    assert [d.utilization for d in corrected] == [40, 30]
    assert [d.name for d in corrected] == ["CUDA0", "CUDA1"]
    capacity = multigpu.cuda_budget(corrected, 64 * GIB, planning=True)
    monkeypatch.setattr(multigpu, "cuda_devices", lambda **_: corrected)
    live = multigpu.cuda_launch_budget(capacity)
    assert live.cuda_split_bytes[0] <= (23000 << 20) - hardware._LAUNCH_HEADROOM
    assert live.cuda_split_bytes[1] <= (27000 << 20) - hardware._LAUNCH_HEADROOM


@pytest.mark.platforms("windows")
def test_windows_memory_does_not_guess_for_mig_or_increase_cuda_free(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import cuda_memory, multigpu

    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "nvidia-smi")
    monkeypatch.setattr(cuda_memory, "_buses", lambda _, **kw: ("0000:21:00.0", "0000:21:00.0"))
    monkeypatch.setattr(cuda_memory, "_run", lambda *_: "00000000:21:00.0, 27000, 18000, 30\n")
    cards = tuple(multigpu.CudaDevice(f"CUDA{i}", "MIG", 10000 << 20, 5000 << 20) for i in range(2))
    assert cuda_memory.bound_memory(tmp_path / "llama-server.exe", cards) == cards
    monkeypatch.setattr(cuda_memory, "_buses", lambda _, **kw: ("0000:21:00.0",))
    card = multigpu.CudaDevice("CUDA0", "RTX", 46068 << 20, 5000 << 20)
    corrected = cuda_memory.bound_memory(tmp_path / "llama-server.exe", (card,))
    assert corrected[0].free == card.free and corrected[0].used == 18000 << 20


def test_growth_never_assigns_unmeasured_outgoing_bytes_to_another_card(tmp_path, monkeypatch):
    _inventory(monkeypatch, tmp_path, (48, 48), (4, 8))
    capacity = hardware.probe_budget(planning=True)
    # The outgoing model may hold 40 GiB on CUDA1 while another application
    # occupies CUDA0. A 20/20 credit would invent free memory on CUDA0.
    without_credit = hardware.launch_budget(capacity)
    growing = hardware.launch_budget(capacity, own_bytes=40 * GIB)
    assert growing.usable_vram_bytes == without_credit.usable_vram_bytes
    assert growing.cuda_split_bytes[0] <= 4 * GIB - hardware._LAUNCH_HEADROOM
    assert growing.cuda_split_bytes[1] <= 8 * GIB - hardware._LAUNCH_HEADROOM
