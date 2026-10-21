"""Bad CUDA telemetry is uncertainty, never evidence that devices were hidden."""
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import cuda_memory, hardware, multigpu

VALID = "CUDA0: accelerator (16384 MiB, 15360 MiB free)"


@pytest.fixture
def inventory_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(multigpu, "_cache", {})
    monkeypatch.setattr(hardware, "_configured_engine", lambda: SimpleNamespace(
        backend="cuda", binary=tmp_path / "llama-server.exe"))
    monkeypatch.setattr(cuda_memory, "bound_memory", lambda binary, devices: devices)


@pytest.mark.parametrize("output", [
    "CUDA0: accelerator (16384 MiB, 20000 MiB free)",
    "CUDA0: accelerator (0 MiB, 0 MiB free)",
    "CUDA0: accelerator (16384 MiB, -1 MiB free)",
    "CUDA0: accelerator total=16384 free=15360",
    "CUDA-1: accelerator (16384 MiB, 15360 MiB free)",
    VALID + "\n" + VALID,
    VALID + "\nCUDA1: unsupported memory format",
])
def test_invalid_cuda_rows_remain_unknown_and_are_not_cached(inventory_probe, monkeypatch, output):
    monkeypatch.setattr(multigpu, "_list_devices", lambda binary: output)
    assert multigpu.cuda_devices() is None
    assert not multigpu._cache
    # A later valid answer immediately recovers; malformed data cannot poison the TTL.
    monkeypatch.setattr(multigpu, "_list_devices", lambda binary: VALID)
    assert len(multigpu.cuda_devices()) == 1


@pytest.mark.parametrize("output", ["", "Available devices:\n", "ggml_cuda_init: found 0 CUDA devices\n"])
def test_successful_zero_device_output_is_still_known_empty(inventory_probe, monkeypatch, output):
    monkeypatch.setattr(multigpu, "_list_devices", lambda binary: output)
    assert multigpu.cuda_devices() == ()


def test_valid_device_labels_and_diagnostic_lines_keep_native_order(inventory_probe, monkeypatch):
    output = ("ggml_cuda_init: found 2 CUDA devices\nAvailable devices:\n"
              "  CUDA3: accelerator (large memory variant) (32768 MiB, 28672 MiB free)\n"
              "  CUDA1: accelerator (16384 MiB, 8192 MiB free)\n")
    monkeypatch.setattr(multigpu, "_list_devices", lambda binary: output)
    devices = multigpu.cuda_devices()
    assert [device.name for device in devices] == ["CUDA3", "CUDA1"]
    assert [device.free for device in devices] == [28672 << 20, 8192 << 20]
