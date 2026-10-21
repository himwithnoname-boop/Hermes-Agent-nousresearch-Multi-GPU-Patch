"""Optional CUDA placement provider for independently installable runtime upgrades.

The common runtime retains upstream single-device planning when the MultiGPU
provider is absent. A present but broken provider is an error, never absence.
"""
from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
import re

_PROVIDER = "hermes_cli.local_runtime.multigpu"


def _provider():
    return import_module(_PROVIDER) if find_spec(_PROVIDER) is not None else None


def cuda_devices(*, fresh: bool = False):
    provider = _provider()
    return provider.cuda_devices(fresh=fresh) if provider is not None else None


def cuda_budget(devices, ram: int, *, planning: bool):
    provider = _provider()
    return provider.cuda_budget(devices, ram, planning=planning) if provider is not None else None


def cuda_launch_budget(capacity, *, own_bytes: int = 0):
    provider = _provider()
    return provider.cuda_launch_budget(capacity, own_bytes=own_bytes) if provider is not None else None


def placement_keys(budget) -> dict[str, str]:
    provider = _provider()
    return provider.placement_keys(budget) if provider is not None else {}


def _generated_cuda_placement(keys: dict) -> bool:
    """Recognize the complete managed addon signature, never a generic device override."""
    devices = str(keys.get("device", "")).split(",")
    split = str(keys.get("tensor-split", "")).split(",")
    return (keys.get("split-mode") == "layer"
            and len(devices) == len(split)
            and all(re.fullmatch(r"CUDA\d+", value) for value in devices)
            and all(re.fullmatch(r"\d+", value) for value in split)
            and any(value.strip("0") for value in split))


def presets_need_migration(known, model_ids: list[str]) -> bool:
    provider = _provider()
    if provider is not None:
        return provider.presets_need_migration(known, model_ids)
    # Normal boot regeneration removes only the old addon's generated placement
    # policy. Do not inspect unrelated presets or mutate the running router here.
    for model_id in model_ids:
        entry = known.get(model_id)
        if entry is not None:
            keys = (entry.placement if entry.refusal else entry.keys) or {}
            if _generated_cuda_placement(keys):
                return True
    return False
