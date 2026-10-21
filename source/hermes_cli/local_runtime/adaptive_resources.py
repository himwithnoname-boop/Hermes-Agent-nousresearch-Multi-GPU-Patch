"""Bounded host-cache policy and passive admission pressure, independent of GPU names.

Cache settings are launch grants. They never resize or purge an active server.
Unmeasured GPU splits, CPU affinities and accelerator backends remain unchanged.
"""

from __future__ import annotations

import logging

from hermes_cli.local_runtime.request_target import bounded_number

logger = logging.getLogger(__name__)
_MIB = 1 << 20


def resource_config() -> dict:
    from hermes_cli.config import load_config_readonly

    runtime = load_config_readonly().get("local_runtime") or {}
    value = runtime.get("adaptive_resources") or {}
    return value if isinstance(value, dict) else {}


def cache_grant(room: int, residents: int, *, settings: dict | None = None) -> int:
    """Each resident receives at most its share of independently admitted host room.

    The default ceiling limits cache growth. Raising the cap is explicit and the fraction
    leaves additional spare room as workloads grow; negative/unlimited is refused.
    """
    cfg = resource_config() if settings is None else settings
    cap = int(bounded_number(cfg.get("prompt_cache_max_mib"), 8192, 0, 262144)) * _MIB
    fraction = bounded_number(cfg.get("prompt_cache_room_fraction"), 1, 0, 1)
    grant = min(cap, int(max(0, room) / max(1, residents) * fraction))
    return grant if grant >= 512 * _MIB else 0


def admission_pressure() -> tuple[bool, str]:
    """Pause new dispatch under current host pressure without touching active work."""
    from hermes_cli.local_runtime.hardware import _ram_bytes, _windows_commit_room

    cfg = resource_config()
    if cfg.get("pause_on_memory_pressure") is False:
        return False, ""
    try:
        total, free = _ram_bytes()
        commit_room = _windows_commit_room()
    except (OSError, ValueError):
        # Missing measurements are not fabricated as exhausted memory. Launch
        # admission still applies its independent host/VRAM model-footprint checks.
        logger.debug("request memory-pressure probe unavailable", exc_info=True)
        return False, "unknown"
    fraction = bounded_number(cfg.get("physical_reserve_fraction"), 0.05, 0, 0.5)
    reserve = min(total // 4, max(512 * _MIB, int(total * fraction))) if total > 0 else 0
    if total > 0 and free < reserve:
        return True, "available physical RAM"
    if commit_room is not None and commit_room < 256 * _MIB:
        return True, "Windows commit headroom"
    return False, ""


def refit_ready(supervisor, resources: tuple, *, now: float, settings: dict | None = None) -> bool:
    """Require stable increases and a cooldown; accept decreases immediately.

    Called only after the existing no-resident/no-loading check. Pending state is
    owned by the profile-bound supervisor, never by a global hardware singleton.
    """
    cfg = resource_config() if settings is None else settings
    old = getattr(supervisor, "_refit_resources", None)
    if old is None:
        return True
    if any(a is not None and b is not None and b < a for a, b in zip(old, resources)):
        supervisor._adaptive_refit_candidate = None
        return True
    candidate = getattr(supervisor, "_adaptive_refit_candidate", None)
    dwell = bounded_number(cfg.get("growth_stability_seconds"), 30, 0, 600)
    cooldown = bounded_number(cfg.get("refit_cooldown_seconds"), 60, 0, 3600)
    if candidate is None or any(
            (a is None) != (b is None) or (a is not None and abs(a - b) >= 256 * _MIB)
            for a, b in zip(candidate[0], resources)):
        supervisor._adaptive_refit_candidate = (resources, now)
        return dwell == 0 and now - getattr(supervisor, "_adaptive_refit_at", float("-inf")) >= cooldown
    return (now - candidate[1] >= dwell
            and now - getattr(supervisor, "_adaptive_refit_at", float("-inf")) >= cooldown)
