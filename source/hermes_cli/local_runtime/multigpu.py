"""CUDA placement uses the engine's logical devices, never nvidia-smi's physical order."""
from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import os
import re
import subprocess
import time

logger = logging.getLogger(__name__)
_cache: dict[tuple[str, str | None, str | None, str | None], tuple[float, tuple[CudaDevice, ...]]] = {}


@dataclass(frozen=True)
class CudaDevice:
    name: str
    description: str
    total: int
    free: int
    used: int | None = None
    utilization: int | None = None


def _parse_device_inventory(output: str) -> tuple[CudaDevice, ...]:
    """Malformed device rows are uncertainty, not a known-empty or partial inventory."""
    devices: list[CudaDevice] = []
    names: set[str] = set()
    for line in output.splitlines():
        match = re.fullmatch(r"\s*(CUDA\d+):\s*(.+)\s+\((\d+) MiB, (\d+) MiB free\)\s*", line)
        if match is None:
            if re.match(r"\s*CUDA[^:\s]*\s*:", line):
                raise ValueError("Unrecognized CUDA device inventory row")
            continue
        device = CudaDevice(match[1], match[2].strip(), int(match[3]) << 20, int(match[4]) << 20)
        if device.name in names or device.total <= 0 or not 0 <= device.free <= device.total:
            raise ValueError("Duplicate CUDA device or invalid memory telemetry")
        names.add(device.name)
        devices.append(device)
    return tuple(devices)


def cuda_devices(*, fresh: bool = False) -> tuple[CudaDevice, ...] | None:
    """Ask the selected CUDA engine, which honors CUDA_VISIBLE_DEVICES and its ordering."""
    from hermes_cli.local_runtime.hardware import _configured_engine

    try:
        engine = _configured_engine()
        if engine is None or engine.backend != "cuda":
            return None
        key = (str(engine.binary), os.environ.get("CUDA_VISIBLE_DEVICES"),
               os.environ.get("CUDA_DEVICE_ORDER"), os.environ.get("GGML_CUDA_DEVICES"))
        now = time.monotonic()
        cached = _cache.get(key)
        if not fresh and cached is not None and now - cached[0] < 4:
            return cached[1]
        result = _parse_device_inventory(_list_devices(engine.binary))
        from hermes_cli.local_runtime.cuda_memory import bound_memory

        result = bound_memory(engine.binary, result)
        _cache[key] = (now, result)
        return result
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.warning("CUDA device inventory unavailable: %s", exc)
        return None


def _list_devices(binary) -> str:
    from hermes_cli._subprocess_compat import windows_hide_flags
    from hermes_cli.local_runtime.processes import server_child_env

    # The executable owns DLL search/bootstrap (the CUDA DLL can fail under a ctypes host).
    result = subprocess.run([str(binary), "--list-devices"], cwd=str(binary.parent),
                            stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=20, check=True,
                            creationflags=windows_hide_flags(), env=server_child_env(os.environ))
    return result.stdout + result.stderr


def cuda_budget(devices: tuple[CudaDevice, ...], ram: int, *, planning: bool):
    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_cli.local_runtime.hardware import _MARGIN_FLOOR, _MARGIN_FRACTION
    import sys

    split = tuple(max(0, (d.total if planning else d.free)
                      - max(_MARGIN_FLOOR, int(d.total * _MARGIN_FRACTION))) for d in devices)
    return HardwareBudget(sum(split), sum(d.total for d in devices), ram,
                          gpu_name=" + ".join(d.description for d in devices), platform=sys.platform,
                          cuda_devices=devices, cuda_split_bytes=split)


def cuda_launch_budget(capacity, *, own_bytes: int = 0):
    """Keep the admitted placement ratios, narrowed by the most constrained GPU.

    Layer placement, KV caches and output buffers need not follow tensor-split ratios.
    An aggregate outgoing-model estimate therefore cannot establish how many bytes
    each card will release. Credit only measured free memory until it has unloaded;
    keeping the admitted ratios makes growth decisions comparable to capacity.
    """
    from hermes_cli.local_runtime.hardware import _LAUNCH_HEADROOM

    current = cuda_devices(fresh=True)
    if current is None:
        return None
    if tuple((d.name, d.total) for d in current) != tuple(
            (d.name, d.total) for d in capacity.cuda_devices):
        # A known inventory change is not an unavailable probe. Preset admission must refuse
        # this stale placement before fit can retain its old capacity-sized floor.
        return replace(capacity, usable_vram_bytes=0, total_device_bytes=sum(d.total for d in current),
                       gpu_name=" + ".join(d.description for d in current), cuda_devices=current,
                       cuda_split_bytes=(0,) * len(current))
    total = sum(capacity.cuda_split_bytes)
    if not total:
        return replace(capacity, usable_vram_bytes=0)
    ratios = capacity.cuda_split_bytes
    scale = min((min(limit, max(0, d.free - _LAUNCH_HEADROOM)) / limit
                 for d, limit in zip(current, ratios) if limit), default=0)
    split = tuple(int(limit * scale) for limit in ratios)
    return replace(capacity, usable_vram_bytes=sum(split), cuda_devices=current, cuda_split_bytes=split)


def placement_keys(budget) -> dict[str, str]:
    if not budget.cuda_devices:
        return {}
    # Bytes stay proportional without rounding a small card's share down to zero.
    return {"device": ",".join(d.name for d in budget.cuda_devices), "split-mode": "layer",
            "tensor-split": ",".join(str(n) for n in budget.cuda_split_bytes)}


def presets_need_migration(known, model_ids: list[str]) -> bool:
    """A UI restart must not adopt pre-patch or different-hardware launch policies."""
    devices = cuda_devices()
    if devices is None:
        return False
    placements = ((known[mid].placement if known[mid].refusal else known[mid].keys) or {}
                  for mid in model_ids)
    if not devices:
        # An authoritative empty inventory must retire a saved GPU selection. A failed query
        # above is uncertainty and does not justify rebuilding a working preset.
        return any(keys.get("device", "").strip().lower() not in {"", "none"}
                   or keys.get("tensor-split") for keys in placements)
    desired = placement_keys(cuda_budget(devices, 0, planning=True))
    return any(any(keys.get(k) != v for k, v in desired.items()) for keys in placements)
