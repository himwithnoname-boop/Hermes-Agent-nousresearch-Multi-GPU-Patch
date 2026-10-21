"""Adopted-router ejection needs a generation-bound request and a confirmed unload receipt."""
from __future__ import annotations

import inspect
import json
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.local_runtime import recovery
from hermes_cli.web_routers import local_models as lm

_PUBLIC_KEY = 'stable-session-key-fixture'
_OPERATION_KEY = 'private-generation-one-key-fixture'
_NEXT_OPERATION_KEY = 'private-generation-two-key-fixture'


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = tmp_path / 'config.yaml'
    config.write_text('model: selected\nprovider: llamacpp\n', encoding='utf-8')
    state = {'base_url': 'http://127.0.0.1:19876/v1', 'api_key': _PUBLIC_KEY,
             'operation_api_key': _OPERATION_KEY, 'pid': 100, 'create_time': 2.0,
             'start_time': 2.0, 'executable': 'fixture-engine', 'owner_pid': 99,
             'owner_create_time': 1.0, 'owner_start_time': 1.0}
    monkeypatch.setattr(lm.bootstrap, 'get_supervisor', lambda: None)
    monkeypatch.setattr(recovery, 'read_state', lambda: dict(state))
    monkeypatch.setattr(recovery, 'recorded_process', lambda _state: SimpleNamespace(pid=100))
    monkeypatch.setattr(lm, '_state_endpoint', lambda: {
        'base_url': state['base_url'], 'api_key': state['api_key']})
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(lm.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(lm.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))
    app = FastAPI()
    app.include_router(lm.router)
    with TestClient(app) as client:
        yield SimpleNamespace(state=state, clock=clock, client=client, config=config)
    assert config.read_text(encoding='utf-8') == 'model: selected\nprovider: llamacpp\n'


def _status(value):
    return {'data': [{'id': 'selected', 'status': {'value': value}}]}


def _script(monkeypatch, statuses):
    states = iter(statuses)
    calls = []

    def request(endpoint, path, *, timeout, payload=None):
        calls.append((dict(endpoint), path, timeout, payload))
        if payload is not None:
            return None
        result = next(states)
        if isinstance(result, BaseException):
            raise result
        return result if isinstance(result, dict) else _status(result)

    monkeypatch.setattr(lm, '_router_request', request)
    return calls


def _eject(runtime):
    return runtime.client.post('/api/local-models/eject', json={'model_id': 'selected'})


@pytest.mark.parametrize('terminal', ['unloaded', 'downloaded', None])
def test_eject_waits_for_terminal_state_and_uses_private_generation_key(runtime, monkeypatch, terminal):
    final = {'data': []} if terminal is None else terminal
    calls = _script(monkeypatch, ['loading', 'sleeping', 'unloading', 'loaded', final])
    result = _eject(runtime)
    assert result.status_code == 200 and result.json() == {'ok': True}
    assert calls[0][1:] == ('/models/unload', 120, {'model': 'selected'})
    assert [call[1] for call in calls[1:]] == ['/models?autoload=false'] * 5
    assert all(endpoint['api_key'] == _OPERATION_KEY for endpoint, *_ in calls)
    assert _OPERATION_KEY not in result.text and _PUBLIC_KEY not in result.text


@pytest.mark.parametrize('payload', [
    {}, {'data': None}, {'data': 'unknown'}, {'data': [None]},
    {'data': [{'id': 9}]}, {'data': [{'id': 'selected'}]},
    _status('unknown'), _status([]),
    {'data': _status('unloaded')['data'] * 2},
])
def test_unknown_state_never_acknowledges_memory_release(runtime, monkeypatch, payload):
    _script(monkeypatch, [payload])
    result = _eject(runtime)
    assert result.status_code == 502 and 'unconfirmed' in result.json()['detail']


@pytest.mark.parametrize('status', ['loading', 'sleeping', 'loaded', 'ready', 'unloading'])
def test_nonterminal_state_times_out_without_claiming_release(runtime, monkeypatch, status):
    calls = _script(monkeypatch, [status] * 100)
    result = _eject(runtime)
    assert result.status_code == 502 and 'timed out' in result.json()['detail']
    assert runtime.clock.now <= 15.01
    assert all(timeout <= 3 for _, _, timeout, _ in calls[1:])


def test_transport_failure_is_uncertain_and_does_not_expose_credentials(runtime, monkeypatch):
    _script(monkeypatch, [OSError(_OPERATION_KEY)])
    result = _eject(runtime)
    assert result.status_code == 502 and 'confirm' in result.json()['detail']
    assert _OPERATION_KEY not in result.text


def test_missing_operation_key_requires_restart_without_stable_fallback(runtime, monkeypatch):
    runtime.state.pop('operation_api_key')
    calls = _script(monkeypatch, [])
    result = _eject(runtime)
    assert result.status_code == 409 and 'Restart' in result.json()['detail']
    assert calls == []


def test_unowned_record_is_refused_before_http(runtime, monkeypatch):
    monkeypatch.setattr(recovery, 'recorded_process', lambda _state: None)
    calls = _script(monkeypatch, [])
    result = _eject(runtime)
    assert result.status_code == 409 and calls == []


@pytest.mark.parametrize('field,value', [('pid', 101), ('create_time', 3.0),
                                         ('owner_start_time', 2.0), ('operation_api_key', _NEXT_OPERATION_KEY)])
def test_replacement_after_unload_ack_is_not_a_release_receipt(runtime, monkeypatch, field, value):
    calls = []

    def request(endpoint, path, *, timeout, payload=None):
        calls.append(path)
        runtime.state[field] = value

    monkeypatch.setattr(lm, '_router_request', request)
    result = _eject(runtime)
    assert result.status_code == 502 and 'changed' in result.json()['detail']
    assert calls == ['/models/unload']


def test_replacement_during_poll_cannot_turn_old_unloaded_state_into_success(runtime, monkeypatch):
    def request(endpoint, path, *, timeout, payload=None):
        if payload is None:
            runtime.state['operation_api_key'] = _NEXT_OPERATION_KEY
            return _status('unloaded')
        return None

    monkeypatch.setattr(lm, '_router_request', request)
    result = _eject(runtime)
    assert result.status_code == 502 and 'changed' in result.json()['detail']


def test_deadline_covers_a_late_unload_ack(runtime, monkeypatch):
    calls = []

    def request(endpoint, path, *, timeout, payload=None):
        calls.append(path)
        runtime.clock.now += 136

    monkeypatch.setattr(lm, '_router_request', request)
    result = _eject(runtime)
    assert result.status_code == 502 and 'timed out' in result.json()['detail']
    assert calls == ['/models/unload']


def test_eject_route_stays_sync_and_delegates_process_local_supervisor(runtime, monkeypatch):
    calls = []
    sup = SimpleNamespace(unload_model=lambda model: calls.append(model))
    monkeypatch.setattr(lm.bootstrap, 'get_supervisor', lambda: sup)
    assert not inspect.iscoroutinefunction(lm.local_models_eject)
    assert _eject(runtime).status_code == 200
    assert calls == ['selected']


@pytest.fixture
def router_http(runtime):
    observed = []
    model = {'value': 'loaded', 'polls': 0}

    class Handler(BaseHTTPRequestHandler):
        def answer(self, status, body):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self):
            key = self.headers.get('Authorization')
            observed.append((self.command, self.path, key))
            if key not in {'Bearer ' + _PUBLIC_KEY, 'Bearer ' + runtime.state['operation_api_key']}:
                self.answer(401, {'error': 'Unauthorized'})
                return False
            return True

        def do_POST(self):
            if not self.authorized():
                return
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body == {'model': 'selected'} and self.path == '/models/unload'
            model['value'] = 'unloading'
            self.answer(200, {'success': True})

        def do_GET(self):
            if not self.authorized():
                return
            assert self.path == '/models?autoload=false'
            model['polls'] += 1
            if model['polls'] == 2:
                model['value'] = 'unloaded'
            self.answer(200, _status(model['value']))

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    runtime.state['base_url'] = f'http://127.0.0.1:{server.server_address[1]}/v1'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield observed, model
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_real_http_eject_waits_and_never_uses_public_session_key(runtime, router_http):
    observed, model = router_http
    assert _eject(runtime).status_code == 200
    assert model['value'] == 'unloaded' and model['polls'] == 2
    assert [method for method, *_ in observed] == ['POST', 'GET', 'GET']
    assert all(auth == 'Bearer ' + _OPERATION_KEY for _, _, auth in observed)


def test_same_port_replacement_at_dispatch_rejects_captured_old_key(runtime, router_http, monkeypatch):
    observed, model = router_http
    urlopen = lm.urllib.request.urlopen
    swapped = False

    def replace_at_dispatch(request, *args, **kwargs):
        nonlocal swapped
        if not swapped and request.full_url.endswith('/models/unload'):
            swapped = True
            runtime.state['operation_api_key'] = _NEXT_OPERATION_KEY
            runtime.state['pid'] = 101
        return urlopen(request, *args, **kwargs)

    monkeypatch.setattr(lm.urllib.request, 'urlopen', replace_at_dispatch)
    result = _eject(runtime)
    assert result.status_code == 502 and 'confirm' in result.json()['detail']
    assert model['value'] == 'loaded' and model['polls'] == 0
    assert observed == [('POST', '/models/unload', 'Bearer ' + _OPERATION_KEY)]
    assert _OPERATION_KEY not in result.text and _NEXT_OPERATION_KEY not in result.text
