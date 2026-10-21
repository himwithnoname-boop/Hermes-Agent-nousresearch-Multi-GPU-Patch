"""Hardware-independent CUDA cache identity and stale placement contracts.

Every hardware query is stubbed: these tests never load CUDA, run GPU binaries,
or inspect the user's devices.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import cuda_memory, hardware, multigpu

GIB = 1 << 30


@pytest.mark.parametrize("variable,initial,changed", [
    ("CUDA_VISIBLE_DEVICES", None, ""),
    ("GGML_CUDA_DEVICES", None, "3"),
])
def test_inventory_cache_tracks_effective_device_visibility(tmp_path, monkeypatch, variable, initial, changed):
    monkeypatch.setattr(multigpu, "_cache", {})
    monkeypatch.setattr(cuda_memory, "bound_memory", lambda binary, devices: devices)
    monkeypatch.setattr(hardware, "_configured_engine", lambda: SimpleNamespace(
        backend="cuda", binary=tmp_path / "llama-server.exe"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("GGML_CUDA_DEVICES", raising=False)
    if initial is not None:
        monkeypatch.setenv(variable, initial)
    calls = []

    def inventory(_):
        import os
        calls.append(True)
        count = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else int(os.environ.get("GGML_CUDA_DEVICES", "2"))
        return "\n".join(f"CUDA{i}: accelerator ({16384} MiB, {15360} MiB free)" for i in range(count))

    monkeypatch.setattr(multigpu, "_list_devices", inventory)
    assert len(multigpu.cuda_devices()) == 2
    monkeypatch.setenv(variable, changed)
    expected = 0 if variable == "CUDA_VISIBLE_DEVICES" else 3
    assert len(multigpu.cuda_devices()) == expected
    assert len(calls) == 2


@pytest.mark.parametrize("current", [
    (),
    (multigpu.CudaDevice("CUDA0", "smaller", 16 * GIB, 15 * GIB),),
    (multigpu.CudaDevice("CUDA0", "replacement", 24 * GIB, 23 * GIB),
     multigpu.CudaDevice("CUDA1", "second", 32 * GIB, 31 * GIB)),
    tuple(multigpu.CudaDevice(f"CUDA{i}", "accelerator", size * GIB, (size - 1) * GIB)
          for i, size in enumerate((16, 32, 24))),
])
def test_known_topology_change_returns_no_old_device_memory_grant(monkeypatch, current):
    original = (multigpu.CudaDevice("CUDA0", "first", 16 * GIB, 15 * GIB),
                multigpu.CudaDevice("CUDA1", "second", 32 * GIB, 31 * GIB))
    capacity = multigpu.cuda_budget(original, 64 * GIB, planning=True)
    monkeypatch.setattr(multigpu, "cuda_devices", lambda **kw: current)
    live = multigpu.cuda_launch_budget(capacity)
    assert live is not None, "A known change must be distinguishable from an unavailable probe"
    assert live.cuda_devices == current
    assert live.total_device_bytes == sum(device.total for device in current)
    assert live.usable_vram_bytes == 0
    assert live.cuda_split_bytes == (0,) * len(current)


def test_unavailable_device_probe_remains_distinct(monkeypatch):
    cards = (multigpu.CudaDevice("CUDA0", "accelerator", 16 * GIB, 15 * GIB),)
    capacity = multigpu.cuda_budget(cards, 64 * GIB, planning=True)
    monkeypatch.setattr(multigpu, "cuda_devices", lambda **kw: None)
    assert multigpu.cuda_launch_budget(capacity) is None


@pytest.mark.platforms("windows")
def test_changed_logical_inventory_rebinds_windows_memory_immediately(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda_memory, "_bus_cache", {})
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "nvidia-smi")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    buses = ["0000:10:00.0", "0000:20:00.0"]
    pci_calls = []

    def run(argv, cwd):
        if "--query-gpu=pci.bus_id,memory.free,memory.used,utilization.gpu" in argv:
            return "0000:10:00.0, 10240, 6144, 1\n0000:20:00.0, 20480, 12288, 2\n0000:30:00.0, 2048, 22528, 3\n"
        pci_calls.append(True)
        return json.dumps(buses)

    monkeypatch.setattr(cuda_memory, "_run", run)
    cards = tuple(multigpu.CudaDevice(f"CUDA{i}", f"accelerator-{i}", size * GIB, (size - 1) * GIB)
                  for i, size in enumerate((16, 32)))
    assert cuda_memory.bound_memory(tmp_path / "llama-server.exe", cards)[0].free == 10 * GIB
    # A different set/order arrives before the cache TTL. New CUDA0 is not the old physical GPU.
    buses[:] = ["0000:30:00.0", "0000:10:00.0", "0000:20:00.0"]
    changed = tuple(multigpu.CudaDevice(f"CUDA{i}", f"accelerator-{i}", size * GIB, (size - 1) * GIB)
                    for i, size in enumerate((24, 16, 32)))
    corrected = cuda_memory.bound_memory(tmp_path / "llama-server.exe", changed)
    assert tuple(device.free for device in corrected) == (2 * GIB, 10 * GIB, 20 * GIB)
    assert len(pci_calls) == 2


def test_successful_pci_cache_expires_for_same_shape_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda_memory, "_bus_cache", {})
    clock = [0]
    monkeypatch.setattr(cuda_memory, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    answers = iter(['["0000:10:00.0"]', '["0000:20:00.0"]'])
    monkeypatch.setattr(cuda_memory, "_run", lambda *args: next(answers))
    binary = Path(tmp_path / "llama-server.exe")
    assert cuda_memory._buses(binary) == ("0000:10:00.0",)
    clock[0] = 10
    assert cuda_memory._buses(binary) == ("0000:10:00.0",)
    clock[0] = 61
    assert cuda_memory._buses(binary) == ("0000:20:00.0",)
@pytest.mark.parametrize("devices,keys,expected", [
    ((), {"device": "CUDA0,CUDA1", "split-mode": "layer", "tensor-split": "1,1"}, True),
    ((), {"ctx-size": "8192"}, False),
    (None, {"device": "CUDA0"}, False),
])
def test_migration_replans_known_disappearance_but_not_unknown_probe(monkeypatch, devices, keys, expected):
    monkeypatch.setattr(multigpu, "cuda_devices", lambda **kw: devices)
    entry = SimpleNamespace(refusal=None, keys=keys)
    assert multigpu.presets_need_migration({"model": entry}, ["model"]) is expected