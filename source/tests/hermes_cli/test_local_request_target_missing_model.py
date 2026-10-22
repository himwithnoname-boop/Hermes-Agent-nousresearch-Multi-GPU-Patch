"""A missing router model must not quarantine unrelated model requests."""
import io
import json
import urllib.error
from types import SimpleNamespace
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

@pytest.mark.parametrize(("status", "message", "kind", "expected"), [
    (400, "model 'missing' not found", "invalid_request_error", True),
    (404, "model 'missing' not found", "invalid_request_error", True),
    (401, "model 'missing' not found", "invalid_request_error", False),
    (403, "model 'missing' not found", "invalid_request_error", False),
    (500, "model 'missing' not found", "invalid_request_error", False),
    (400, "model 'different' not found", "invalid_request_error", False),
    (400, "model 'missing' not found", "server_error", False),
    (400, "slot not found", "invalid_request_error", False),
])
def test_missing_model_release_requires_exact_authenticated_response(runtime, monkeypatch, status, message, kind, expected):
    def probe(*args, **kwargs):
        body = json.dumps({"error": {"code": status, "message": message, "type": kind}}).encode()
        raise urllib.error.HTTPError("http://local", status, "error", {}, io.BytesIO(body))
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.managed_get_json", probe)
    assert target.resolve_target(runtime["base_url"], "qwen").idle("missing") is expected

def test_missing_model_drain_does_not_block_next_model(runtime, tmp_path, monkeypatch):
    from agent.local_request_scheduler import Lease
    from hermes_cli.local_runtime.endpoint import child_query
    resolved = target.resolve_target(runtime["base_url"], "qwen")
    queue = resolved.queue()
    old = queue.enqueue("missing")
    assert queue.try_admit(old)
    queue.finish(old, uncertain=True)
    def probe(base, key, route, timeout_s):
        assert route == "/slots?" + child_query("missing")
        body = b'{"error":{"code":400,"message":"model \'missing\' not found","type":"invalid_request_error"}}'
        raise urllib.error.HTTPError("http://local", 400, "error", {}, io.BytesIO(body))
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.managed_get_json", probe)
    monkeypatch.setattr("hermes_cli.local_runtime.adaptive_resources.admission_pressure", lambda: (False, ""))
    assert resolved.idle("missing"), "Router rejection must release the failed request before a new model can run"
    lease = Lease(resolved)
    lease.acquire()
    assert queue.snapshot()["draining"] == 0
    assert queue.snapshot()["active"] == 1
    lease.finish()
    assert queue.snapshot()["active"] == 0

def test_restart_during_missing_model_response_remains_blocked(runtime, monkeypatch):
    def probe(*args, **kwargs):
        runtime["create_time"] += 1
        body = b'{"error":{"code":400,"message":"model \'missing\' not found","type":"invalid_request_error"}}'
        raise urllib.error.HTTPError("http://local", 400, "error", {}, io.BytesIO(body))
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.managed_get_json", probe)
    assert not target.resolve_target(runtime["base_url"], "qwen").idle("missing")
