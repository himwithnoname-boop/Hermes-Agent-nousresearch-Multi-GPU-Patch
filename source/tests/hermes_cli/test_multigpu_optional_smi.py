"""The serving CUDA engine remains authoritative when optional nvidia-smi is absent."""
import pytest

from hermes_cli.local_runtime import hardware
from tests.hermes_cli import test_local_multigpu

GIB = 1 << 30


@pytest.mark.parametrize("sizes", [(16,), (16, 24), (12, 24, 16)])
def test_cuda_inventory_survives_missing_legacy_nvidia_smi(tmp_path, monkeypatch, sizes):
    test_local_multigpu._inventory(monkeypatch, tmp_path, sizes)
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hardware, "_accelerator_device", lambda **kw: None)
    monkeypatch.setattr(hardware, "_windows_igpu_bytes", lambda device: None)
    monkeypatch.setattr(hardware, "gpu_class", lambda: "nvidia")

    budget = hardware.probe_budget(planning=True)

    assert not budget.uma, "A failed optional physical-memory probe cannot turn discrete CUDA into UMA"
    assert budget.total_device_bytes == sum(sizes) * GIB
    assert [device.name for device in budget.cuda_devices] == [f"CUDA{i}" for i in range(len(sizes))]
    assert len(budget.cuda_split_bytes) == len(sizes)


def test_explicitly_hidden_cuda_devices_still_use_cpu_budget_without_nvidia_smi(tmp_path, monkeypatch):
    test_local_multigpu._inventory(monkeypatch, tmp_path, ())
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hardware, "_accelerator_device", lambda **kw: None)
    monkeypatch.setattr(hardware, "_windows_igpu_bytes", lambda device: None)
    monkeypatch.setattr(hardware, "gpu_class", lambda: "nvidia")
    budget = hardware.probe_budget(planning=True)
    assert budget.uma and not budget.cuda_devices
