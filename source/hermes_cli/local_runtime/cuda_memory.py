"""Bound Windows CUDA's allocator view by physical, PCI-matched NVIDIA residency."""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

_bus_cache: dict[tuple[str, str | None, str | None, tuple], tuple[float, tuple[str, ...]]] = {}


def _pci_key(value: str) -> tuple[int, int, int, int]:
    domain, bus, tail = value.strip().split(":")
    device, function = tail.split(".")
    return tuple(int(part, 16) for part in (domain, bus, device, function))


def _run(argv: list[str], cwd: Path) -> str:
    from hermes_cli._subprocess_compat import windows_hide_flags
    from hermes_cli.local_runtime.processes import server_child_env

    return subprocess.run(argv, cwd=str(cwd), stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=15, check=True,
                          creationflags=windows_hide_flags(), env=server_child_env(os.environ)).stdout


def _buses(binary: Path, *, inventory: tuple = ()) -> tuple[str, ...]:
    # Device descriptions/capacities detect changed logical order without keying on volatile free
    # memory. Expiry also refreshes replacements whose inventory has the same shape.
    topology = tuple((d.name, d.description, d.total) for d in inventory)
    key = (str(binary), os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_DEVICE_ORDER"), topology)
    now = time.monotonic()
    cached = _bus_cache.get(key)
    if cached is not None and now - cached[0] < 60:
        return cached[1]
    buses = tuple(json.loads(_run([sys.executable, "-I", str(Path(__file__).resolve()),
                                  str(binary.parent)], binary.parent)))
    if not all(isinstance(bus, str) for bus in buses):
        raise ValueError("Invalid CUDA PCI inventory")
    _bus_cache[key] = (now, buses)
    return buses


def bound_memory(binary: Path, devices: tuple):
    """Keep the engine's order/capacity; use dedicated free/used memory for each matched card.

    WDDM can offer CUDA allocations that would evict another context's residency. That is
    not free physical VRAM. Device names and physical indices cannot match identical or
    reordered cards; the CUDA runtime supplies PCI identities in its own visible order.
    """
    if sys.platform != "win32" or not devices:
        return devices
    from hermes_cli.local_runtime.hardware import _nvidia_smi_path

    try:
        smi = _nvidia_smi_path()
        if not smi:
            return devices
        buses = tuple(_pci_key(bus) for bus in _buses(binary, inventory=devices))
        output = _run([smi, "--query-gpu=pci.bus_id,memory.free,memory.used,utilization.gpu",
                       "--format=csv,noheader,nounits"], binary.parent)
        physical = {}
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            if len(row) != 4:
                continue
            try:
                bus, free, used = _pci_key(row[0]), int(row[1]) << 20, int(row[2]) << 20
            except ValueError:
                continue
            try:
                utilization = int(row[3])
            except ValueError:
                utilization = None
            if free >= 0 and used >= 0:
                physical[bus] = (free, used, utilization if utilization is not None and 0 <= utilization <= 100 else None)
        counts = Counter(buses)
        out = []
        for device in devices:
            ordinal = int(device.name.removeprefix("CUDA"))
            bus = buses[ordinal] if ordinal < len(buses) else None
            memory = physical.get(bus) if bus is not None and counts[bus] == 1 else None
            # Shared PCI identity (MIG) is not evidence for a physical-card memory grant.
            if memory is not None and memory[1] <= device.total:
                device = replace(device, free=min(device.free, memory[0]), used=memory[1], utilization=memory[2])
            out.append(device)
        return tuple(out)
    except (OSError, ValueError, TypeError, IndexError, subprocess.SubprocessError):
        return devices


def _runtime_buses(directory: Path) -> list[str]:
    """Run only in the disposable standard-library child, never the UI process."""
    import ctypes

    libraries = sorted(directory.glob("cudart64_*.dll"))
    if len(libraries) != 1:
        return []
    with os.add_dll_directory(str(directory)):
        runtime = ctypes.CDLL(str(libraries[0]))
        runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaGetDeviceCount.restype = ctypes.c_int
        runtime.cudaDeviceGetPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
        runtime.cudaDeviceGetPCIBusId.restype = ctypes.c_int
        count = ctypes.c_int()
        if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0 or not 0 <= count.value <= 64:
            return []
        buses = []
        for ordinal in range(count.value):
            buffer = ctypes.create_string_buffer(32)
            if runtime.cudaDeviceGetPCIBusId(buffer, len(buffer), ordinal) != 0:
                return []
            buses.append(buffer.value.decode("ascii"))
        return buses


if __name__ == "__main__":
    print(json.dumps(_runtime_buses(Path(sys.argv[1]))))
