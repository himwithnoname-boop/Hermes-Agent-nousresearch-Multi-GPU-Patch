"""Owned endpoint discovery and passive orphan-release proofs."""

import io
from types import SimpleNamespace
import urllib.error

import pytest

from hermes_cli.local_runtime import request_target as target


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    state = {"base_url": "http://127.0.0.1:12345/v1", "pid": 123, "create_time": 100,
             "owner_pid": 12, "owner_create_time": 90, "executable": "llama-server"}
    monkeypatch.setattr("hermes_cli.local_runtime.recovery.read_state", lambda: dict(state))
    monkeypatch.setattr("hermes_cli.local_runtime.recovery.recorded_process", lambda value: object())
    monkeypatch.setattr("hermes_cli.local_runtime.supervisor.state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(target, "scheduler_config", dict)
    monkeypatch.setattr("hermes_cli.local_runtime.presets.read_preset_decisions", dict)
    return state


def test_foreign_endpoint_and_disabled_scheduler_create_no_database(runtime, tmp_path, monkeypatch):
    assert target.resolve_target("https://remote.example/v1", "m") is None
    monkeypatch.setattr(target, "scheduler_config", lambda: {"enabled": False})
    assert target.resolve_target(runtime["base_url"], "m") is None
    assert not (tmp_path / "request-admission.sqlite3").exists()


@pytest.mark.parametrize(("keys", "expected"), [
    ({}, 1), ({"parallel": "-1"}, 1), ({"parallel": "4"}, 4),
    ({"parallel": "1000"}, 32), ({"parallel": "4", "spec-type": "draft-mtp"}, 1),
    ({"parallel": "4", "spec-type": "draft-dflash"}, 1),
])
def test_capacity_is_from_preset_never_gpu_count(runtime, monkeypatch, keys, expected):
    monkeypatch.setattr("hermes_cli.local_runtime.presets.read_preset_decisions",
                        lambda: {"m": SimpleNamespace(keys=keys)})
    assert target.resolve_target(runtime["base_url"], "m").slots == expected


def test_restart_and_lost_ownership_invalidate_target(runtime, monkeypatch):
    resolved = target.resolve_target(runtime["base_url"], "m")
    assert resolved.valid()
    runtime["create_time"] += 1
    assert not resolved.valid()
    runtime["create_time"] -= 1
    monkeypatch.setattr("hermes_cli.local_runtime.recovery.recorded_process", lambda state: None)
    assert not resolved.valid()
    assert target.resolve_target(runtime["base_url"], "m") is None


@pytest.mark.parametrize(("slots", "idle"), [
    ([{"is_processing": False}], True), ([{"is_processing": True}], False),
    ([], False), ({}, False), ([{}], False),
])
def test_only_explicit_idle_slots_release_orphan(runtime, monkeypatch, slots, idle):
    def probe(base, key, route, timeout_s):
        assert "autoload=false" in route and "model=m" in route
        assert route.startswith("/slots?") and timeout_s == 2
        return slots
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.managed_get_json", probe)
    assert target.resolve_target(runtime["base_url"], "m").idle("m") is idle


@pytest.mark.parametrize(("code", "body", "idle"), [
    (400, b'{"error":"model is not loaded"}', True),
    (400, b'unsupported slots', False), (401, b'model is not loaded', False),
    (500, b'model is not loaded', False),
])
def test_probe_errors_do_not_silently_free_server_slot(runtime, monkeypatch, code, body, idle):
    def probe(*args, **kwargs):
        raise urllib.error.HTTPError("http://local", code, "error", {}, io.BytesIO(body))
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.managed_get_json", probe)
    assert target.resolve_target(runtime["base_url"], "m").idle("m") is idle


def test_restart_during_idle_probe_is_not_release_proof(runtime, monkeypatch):
    def probe(*args, **kwargs):
        runtime["create_time"] += 1
        return [{"is_processing": False}]
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.managed_get_json", probe)
    assert not target.resolve_target(runtime["base_url"], "m").idle("m")
