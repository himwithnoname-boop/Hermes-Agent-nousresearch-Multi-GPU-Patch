"""Cache admission must respect host RAM independently of GPU probe and commit room."""
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import hardware, presets
from hermes_cli.local_runtime.estimator import HardwareBudget, allocation_limit_bytes
from hermes_cli.local_runtime.multigpu import CudaDevice, cuda_budget

G = 1 << 30


def stat(physical_free):
    return SimpleNamespace(ullTotalPhys=64 * G, ullAvailPhys=physical_free,
                           ullTotalPageFile=512 * G, ullAvailPageFile=400 * G)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("physical_free,residents", [(8 * G, 1), (8 * G, 3), (6 * G, 1)])
def test_unified_host_cache_cannot_spend_pagefile_as_physical_ram(monkeypatch, physical_free, residents):
    monkeypatch.setattr(hardware, "_windows_memory_status", lambda: stat(physical_free))
    capacity = HardwareBudget(120 * G, 150 * G, 0, uma=True)
    live = hardware.launch_budget(capacity)
    cache = hardware.prompt_cache_bytes(live, 10 * G, residents)
    host_room = max(0, physical_free - int(64 * G * hardware._MARGIN_FRACTION))
    assert cache * residents <= host_room
    assert cache <= hardware._PROMPT_CACHE_MAX
    # Host cache room cannot become extra model capacity or shrink the allocator pool.
    assert live.usable_vram_bytes == capacity.usable_vram_bytes
    assert live.ram_available_bytes == 0
    assert allocation_limit_bytes(live) == live.usable_vram_bytes


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("sizes", [(16,), (16, 24), (12, 24, 16)])
def test_unknown_cuda_probe_still_bounds_host_cache(monkeypatch, sizes):
    from hermes_cli.local_runtime import multigpu

    devices = tuple(CudaDevice(f"CUDA{i}", f"device {i}", n * G, n * G) for i, n in enumerate(sizes))
    capacity = cuda_budget(devices, 64 * G, planning=True)
    monkeypatch.setattr(multigpu, "cuda_launch_budget", lambda *a, **kw: None)
    monkeypatch.setattr(hardware, "_windows_memory_status", lambda: stat(G))
    live = hardware.launch_budget(capacity)
    assert live is not None
    assert live.ram_available_bytes == 0
    assert hardware.prompt_cache_bytes(live, 4 * G) == 0
    # A missed query cannot invent any additional GPU capacity.
    assert live.usable_vram_bytes == capacity.usable_vram_bytes
    assert live.cuda_devices == capacity.cuda_devices


@pytest.mark.parametrize("current_sizes", [(16, 16), (16,), (16, 16, 16)])
def test_changed_cuda_inventory_is_refused_before_planning(tmp_path, monkeypatch, current_sizes):
    capacity_devices = tuple(CudaDevice(f"CUDA{i}", f"device {i}", n * G, n * G)
                             for i, n in enumerate((16, 24)))
    current_devices = tuple(CudaDevice(f"CUDA{i}", f"device {i}", n * G, n * G)
                            for i, n in enumerate(current_sizes))
    capacity = cuda_budget(capacity_devices, 32 * G, planning=True)
    live = HardwareBudget(0, sum(d.total for d in current_devices), 16 * G,
                          cuda_devices=current_devices, cuda_split_bytes=tuple(0 for _ in current_devices))
    # The read seam fails if stale placement proceeds to ordinary capacity planning.
    def no_header_read(_):
        raise AssertionError("stale CUDA placement reached model planning")
    monkeypatch.setattr(presets, "read_gguf_header", no_header_read)
    entry = presets.preset_for_model(tmp_path / "model.gguf", capacity, set(), live=live)
    assert entry.keys is None and entry.window == 0
    assert "CUDA device inventory changed" in entry.refusal
