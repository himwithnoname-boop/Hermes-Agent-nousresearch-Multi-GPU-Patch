"""Ownership-fenced, read-only runtime facts for request admission."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import urllib.error

from hermes_cli.local_runtime.request_queue import RequestQueue


def generation_of(state: dict) -> str:
    identity = [state.get(key) for key in
                ("base_url", "pid", "create_time", "start_time", "executable", "owner_pid", "owner_create_time")]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def scheduler_config() -> dict:
    from hermes_cli.config import load_config_readonly

    runtime = load_config_readonly().get("local_runtime") or {}
    value = runtime.get("request_scheduler") or {}
    return value if isinstance(value, dict) else {}


def bounded_number(value, default: float, low: float, high: float) -> float:
    import math

    if isinstance(value, bool):
        return default
    try:
        result = float(value)
        return max(low, min(high, result)) if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


@dataclass(frozen=True)
class RequestTarget:
    state: dict
    path: Path
    model: str
    slots: int
    settings: dict

    @property
    def generation(self) -> str:
        return generation_of(self.state)

    def valid(self) -> bool:
        from hermes_cli.local_runtime.recovery import read_state, recorded_process

        current = read_state()
        return generation_of(current) == self.generation and recorded_process(current) is not None

    def idle(self, model: str) -> bool:
        """Uncertain/disconnected requests block until the same router is idle."""
        from hermes_cli.local_runtime.endpoint import child_query, managed_get_json

        if not self.valid():
            return False
        base = self.state["base_url"].rstrip("/").removesuffix("/v1")
        try:
            slots = managed_get_json(base, self.state.get("api_key", ""),
                                     "/slots?" + child_query(model), timeout_s=2)
            idle = isinstance(slots, list) and bool(slots) and all(
                isinstance(slot, dict) and slot.get("is_processing") is False for slot in slots)
        except urllib.error.HTTPError as exc:
            body = exc.read(4096)
            idle = (exc.code == 400 and b"model is not loaded" in body.lower()) or (
                exc.code in (400, 404) and _missing_model_response(body, model, exc.code))
        except (OSError, ValueError):
            return False
        return idle and self.valid()

    def queue(self) -> RequestQueue:
        return RequestQueue(self.path, self.generation, capacity=self.slots,
                            max_waiters=int(bounded_number(self.settings.get("max_waiters"), 64, 1, 1024)),
                            aging_seconds=bounded_number(self.settings.get("foreground_advantage_seconds"), 30, 0, 300))


def _missing_model_response(body: bytes, model: str, status: int) -> bool:
    """An owned router cannot still execute a model it explicitly says is absent."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    return (isinstance(error, dict) and error.get("code") == status
            and error.get("type") == "invalid_request_error"
            and error.get("message") == f"model '{model}' not found")


def resolve_target(base_url: str, model: str) -> RequestTarget | None:
    """Only schedule a confirmed owned router, never an arbitrary local/cloud URL."""
    from hermes_cli.local_runtime.recovery import read_state, recorded_process
    from hermes_cli.local_runtime.supervisor import state_path

    if not isinstance(base_url, str) or not isinstance(model, str) or not model:
        return None
    state = read_state()
    if base_url.rstrip("/") != str(state.get("base_url", "")).rstrip("/"):
        return None
    if recorded_process(state) is None:
        return None
    settings = scheduler_config()
    if settings.get("enabled") is False:
        return None
    # MTP and unknown/auto slot counts stay serial. Explicit non-MTP capacity is
    # bounded by the preset; GPU count is never interpreted as request capacity.
    from hermes_cli.local_runtime.presets import read_preset_decisions

    preset = read_preset_decisions().get(model)
    keys = (preset.keys or {}) if preset is not None else {}
    slots = 1
    if not keys.get("spec-type"):
        slots = int(bounded_number(keys.get("parallel"), 1, 1, 32))
    return RequestTarget(state, state_path().parent / "request-admission.sqlite3", model, slots, settings)
