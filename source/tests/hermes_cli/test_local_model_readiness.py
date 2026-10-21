"""Router load acknowledgement is asynchronous; release requires a confirmed terminal state."""
from __future__ import annotations

from types import SimpleNamespace
import urllib.error

import pytest

from hermes_cli.local_runtime import supervisor


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "_stable_api_key", lambda: "test-private-key")
    sup = supervisor.LlamaServerSupervisor(tmp_path / "engine", tmp_path, port=19876)
    sup.proc = SimpleNamespace(poll=lambda: None)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock.now)

    def advance(seconds):
        clock.now += seconds
        return sup._stop_event.is_set()

    monkeypatch.setattr(sup._stop_event, "wait", advance)
    return sup, clock


def response(value, **extra):
    return {"data": [{"id": "selected", "status": {"value": value, **extra}}]}


def scripted_router(sup, statuses, *, duplicate_load=False):
    states = iter(statuses)
    calls = []
    last = "unloaded"

    def request(route, body=None, timeout_s=30, **kwargs):
        nonlocal last
        calls.append((route, body, timeout_s))
        if route.startswith("/models?") or route == "/models":
            item = next(states)
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, dict):
                return item
            last = item
            return response(item)
        if route == "/models/load" and duplicate_load:
            raise RuntimeError("duplicate asynchronous load request")
        if route == "/v1/chat/completions":
            assert last in {"loaded", "ready", "sleeping"}, "generated before load completed"
            return {"choices": [{"message": {"content": "Paris"}}]}
        return {"success": True}

    sup._request = request
    return calls


def test_ensure_waits_for_async_load_before_generation(runtime):
    sup, _ = runtime
    calls = scripted_router(sup, ["unloaded", "loading", "loaded"])
    assert sup.ensure_model_ready("selected", timeout_s=5)
    assert [route for route, _, _ in calls] == [
        "/models", "/models/load", "/models?autoload=false", "/models?autoload=false", "/v1/chat/completions"]
    assert sup.primary_model is None


def test_ensure_joins_existing_load_without_posting_duplicate(runtime):
    sup, _ = runtime
    calls = scripted_router(sup, ["loading", "loading", "loaded"], duplicate_load=True)
    assert sup.ensure_model_ready("selected", timeout_s=5)
    assert all(route != "/models/load" for route, _, _ in calls)


def test_load_model_remains_async(runtime):
    sup, _ = runtime
    calls = scripted_router(sup, [])
    sup.load_model("selected", timeout_s=5)
    assert calls == [("/models/load", {"model": "selected"}, 5)]


def test_failed_child_is_not_retried_or_touch_generated(runtime):
    sup, _ = runtime
    calls = scripted_router(sup, [response("unloaded", failed=True, exit_code=17, args=["--api-key", "test-private-key"])])
    with pytest.raises(RuntimeError, match="17") as error:
        sup.wait_until_loaded("selected", timeout_s=5)
    assert "test-private-key" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("status", ["unloaded", "unloading", "unknown"])
def test_interrupted_load_fails_without_implicit_reload(runtime, status):
    sup, _ = runtime
    calls = scripted_router(sup, [status])
    with pytest.raises(RuntimeError):
        sup.wait_until_loaded("selected", timeout_s=5)
    assert len(calls) == 1


def test_loading_wait_timeout_is_bounded_and_does_not_generate(runtime):
    sup, clock = runtime
    calls = scripted_router(sup, ["loading"] * 20)
    with pytest.raises(TimeoutError):
        sup.wait_until_loaded("selected", timeout_s=1)
    assert clock.now <= 1.01
    assert all(route == "/models?autoload=false" and timeout <= 1 for route, _, timeout in calls)


def test_load_wait_cancelled_during_status_request(runtime):
    sup, _ = runtime

    def request(*args, **kwargs):
        sup._stop_event.set()
        return response("loaded")

    sup._request = request
    with pytest.raises(RuntimeError, match="cancel"):
        sup.wait_until_loaded("selected", timeout_s=5)


def test_load_wait_rejects_replaced_router(runtime):
    sup, _ = runtime

    def request(*args, **kwargs):
        sup.proc = SimpleNamespace(poll=lambda: None)
        return response("loaded")

    sup._request = request
    with pytest.raises(RuntimeError, match="replaced"):
        sup.wait_until_loaded("selected", timeout_s=5)


def test_load_wait_exited_router_does_not_probe_new_owner(runtime):
    sup, _ = runtime
    sup.proc = SimpleNamespace(poll=lambda: 7)
    calls = scripted_router(sup, [])
    with pytest.raises(RuntimeError, match="exited"):
        sup.wait_until_loaded("selected", timeout_s=5)
    assert calls == []


def test_ensure_shares_deadline_with_generation(runtime):
    sup, clock = runtime
    calls = scripted_router(sup, ["unloaded", "loading", "loaded"])
    request = sup._request

    def slow_request(route, body=None, timeout_s=30, **kwargs):
        clock.now += 0.2
        return request(route, body, timeout_s, **kwargs)

    sup._request = slow_request
    assert sup.ensure_model_ready("selected", timeout_s=2)
    generation = [call for call in calls if call[0] == "/v1/chat/completions"]
    assert len(generation) == 1
    assert 0 < generation[0][2] < 2
    assert clock.now < 2


def test_sleeping_model_is_woken_by_touch_not_reloaded(runtime):
    sup, _ = runtime
    calls = scripted_router(sup, ["sleeping"], duplicate_load=True)
    assert sup.ensure_model_ready("selected", timeout_s=2)
    assert [route for route, _, _ in calls] == ["/models", "/v1/chat/completions"]


def test_unload_waits_for_loading_child_to_exit(runtime):
    sup, _ = runtime
    calls = scripted_router(sup, ["loading", "loading", "unloaded"])
    sup.unload_model("selected")
    assert sum(route == "/models" for route, _, _ in calls) == 3
    assert sup.primary_model is None


def test_unload_poll_error_never_claims_memory_released(runtime):
    sup, _ = runtime
    scripted_router(sup, [urllib.error.URLError("status unavailable")])
    with pytest.raises(RuntimeError, match="confirm release") as error:
        sup.unload_model("selected")
    assert isinstance(error.value.__cause__, urllib.error.URLError)


def test_unload_timeout_is_explicit(runtime):
    sup, clock = runtime
    scripted_router(sup, ["loading"] * 100)
    with pytest.raises(TimeoutError):
        sup.unload_model("selected")
    assert clock.now <= 15.01


def test_ensure_rejects_replaced_router_before_touch(runtime):
    sup, _ = runtime

    def request(*args, **kwargs):
        sup.proc = SimpleNamespace(poll=lambda: None)
        return response("loaded")

    sup._request = request
    with pytest.raises(RuntimeError, match="replaced"):
        sup.ensure_model_ready("selected", timeout_s=5)


def test_async_readiness_uses_authenticated_read_only_http(runtime):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    sup, _ = runtime
    observed = []
    state = {"value": "unloaded", "load_polls": 0}

    class Handler(BaseHTTPRequestHandler):
        def send(self, value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            observed.append((self.path, self.headers.get("Authorization")))
            if state["value"] == "loading":
                state["load_polls"] += 1
                if state["load_polls"] == 2:
                    state["value"] = "loaded"
            self.send(response(state["value"]))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append((self.path, self.headers.get("Authorization")))
            assert body["model"] == "selected"
            if self.path == "/models/load":
                state["value"] = "loading"
                self.send({"success": True})
            else:
                assert state["value"] == "loaded"
                self.send({"choices": [{"message": {"content": "Paris"}}]})

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sup.port = server.server_address[1]
    try:
        assert sup.ensure_model_ready("selected", timeout_s=5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert all(auth == "Bearer " + sup._operation_api_key for _, auth in observed)
    assert [path for path, _ in observed].count("/models?autoload=false") == 2


@pytest.mark.parametrize("change", ["cancelled", "replaced"])
def test_ensure_rechecks_owner_after_touch(runtime, change):
    sup, _ = runtime

    def request(route, body=None, timeout_s=30, **kwargs):
        if route == "/models":
            return response("loaded")
        if change == "cancelled":
            sup._stop_event.set()
        else:
            sup.proc = SimpleNamespace(poll=lambda: None)
        return {"choices": [{"message": {"content": "Paris"}}]}

    sup._request = request
    with pytest.raises(RuntimeError, match=change):
        sup.ensure_model_ready("selected", timeout_s=5)


@pytest.mark.parametrize("change", ["cancelled", "replaced"])
def test_unload_rechecks_owner_before_release_confirmation(runtime, change):
    sup, _ = runtime

    def request(route, body=None, timeout_s=30, **kwargs):
        if route == "/models/unload":
            return {"success": True}
        if change == "cancelled":
            sup._stop_event.set()
        else:
            sup.proc = SimpleNamespace(poll=lambda: None)
        return response("unloaded")

    sup._request = request
    with pytest.raises(RuntimeError, match=change):
        sup.unload_model("selected")


def test_missing_model_during_wait_does_not_autoload(runtime):
    sup, _ = runtime
    calls = scripted_router(sup, [{"data": []}])
    with pytest.raises(KeyError):
        sup.wait_until_loaded("selected", timeout_s=5)
    assert [route for route, _, _ in calls] == ["/models?autoload=false"]


def test_ensure_does_not_claim_in_budget_readiness_from_late_touch(runtime):
    sup, clock = runtime

    def request(route, body=None, timeout_s=30, **kwargs):
        if route == "/models":
            return response("loaded")
        clock.now += 3
        return {"choices": [{"message": {"content": "Paris"}}]}

    sup._request = request
    with pytest.raises(TimeoutError):
        sup.ensure_model_ready("selected", timeout_s=2)


class _GenerationServer:
    """Small real loopback server with the pinned engine's startup-frozen key-set behavior."""
    def __init__(self, keys, *, port=0, state="unloaded", hold=None):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from urllib.parse import urlsplit

        self.accepted = []
        self.rejected = 0
        self.closed = False
        owner = self
        accepted_keys = frozenset(keys)

        class Handler(BaseHTTPRequestHandler):
            def send(self, code, value):
                raw = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def authenticated(self):
                key = self.headers.get("Authorization", "").removeprefix("Bearer ")
                if key not in accepted_keys:
                    owner.rejected += 1
                    self.send(401, {"error": {"message": "Invalid API Key"}})
                    return False
                owner.accepted.append(self.path)
                return True

            def do_GET(self):
                if self.path == "/health":
                    self.send(200, {"status": "ok"})
                elif self.authenticated():
                    path = urlsplit(self.path).path
                    if path == "/models":
                        self.send(200, response(state))
                    elif path == "/slots":
                        self.send(200, [])
                    else:
                        self.send(200, {})

            def do_POST(self):
                if not self.authenticated():
                    return
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.path == "/v1/chat/completions":
                    if hold is not None:
                        hold[0].set()
                        hold[1].wait(timeout=5)
                    self.send(200, {"choices": [{"message": {"content": "Paris"}}]})
                else:
                    self.send(200, {"success": True})

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        if not self.closed:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)
            self.closed = True
            assert not self.thread.is_alive()


@pytest.mark.parametrize("action,route", [
    ("load", "/models/load"),
    ("unload", "/models/unload"),
    ("reload", "/models?reload=1"),
    ("touch", "/v1/chat/completions"),
    ("ensure", "/models/load"),
])
def test_stale_operation_cannot_mutate_replacement_on_same_port(runtime, monkeypatch, action, route):
    import urllib.request

    sup, _ = runtime
    original_key = sup._operation_api_key
    server = _GenerationServer({sup.api_key, original_key})
    sup.port = server.port
    servers = [server]
    original_open = urllib.request.urlopen
    replaced = False

    def replace_before_dispatch(request, *args, **kwargs):
        nonlocal replaced
        if request.full_url.removeprefix(sup.base_url.removesuffix("/v1")) == route and not replaced:
            replaced = True
            server.close()
            sup.proc = SimpleNamespace(poll=lambda: None)
            sup._operation_api_key = "replacement-operation-key"
            servers.append(_GenerationServer({sup.api_key, sup._operation_api_key}, port=sup.port))
        return original_open(request, *args, **kwargs)

    # The public session credential works before and after the listener is replaced.
    public_request = urllib.request.Request(sup.base_url.removesuffix("/v1") + "/models",
                                            headers={"Authorization": "Bearer " + sup.api_key})
    with original_open(public_request, timeout=2) as result:
        assert result.status == 200
    monkeypatch.setattr(urllib.request, "urlopen", replace_before_dispatch)
    actions = {"load": lambda: sup.load_model("selected"), "unload": lambda: sup.unload_model("selected"),
               "reload": sup.reload_presets, "touch": lambda: sup.touch_generate("selected"),
               "ensure": lambda: sup.ensure_model_ready("selected", timeout_s=5)}
    try:
        if action == "touch":
            assert actions[action]() is False
        else:
            with pytest.raises(urllib.error.HTTPError) as error:
                actions[action]()
            assert error.value.code == 401
        assert replaced
        replacement = servers[-1]
        assert replacement.rejected == 1
        assert replacement.accepted == [], "stale operation reached replacement mutation handler"
        # Saved public clients remain compatible with both listener incarnations.
        request = urllib.request.Request(sup.base_url.removesuffix("/v1") + "/models",
                                         headers={"Authorization": "Bearer " + sup.api_key})
        with original_open(request, timeout=2) as result:
            assert result.status == 200
    finally:
        for item in servers:
            item.close()


def test_stop_is_not_blocked_by_inflight_generation(runtime, monkeypatch):
    import threading

    sup, _ = runtime
    started, release, stopped = threading.Event(), threading.Event(), threading.Event()
    server = _GenerationServer({sup.api_key, sup._operation_api_key}, hold=(started, release))
    sup.port = server.port
    terminated = []
    monkeypatch.setattr(sup, "_terminate_tree", lambda proc: terminated.append(proc))
    results = []
    worker = threading.Thread(target=lambda: results.append(sup.touch_generate("selected", timeout_s=5)))
    stopper = threading.Thread(target=lambda: (sup.stop(), stopped.set()))
    worker.start()
    try:
        assert started.wait(timeout=5)
        stopper.start()
        assert stopped.wait(timeout=2), "stop was blocked behind the long HTTP request"
        assert terminated
    finally:
        release.set()
        worker.join(timeout=5)
        if stopper.ident is not None:
            stopper.join(timeout=5)
        server.close()
    assert not worker.is_alive() and not stopper.is_alive()
    assert results == [False]


def test_spawn_rotates_private_key_preserves_public_key_and_redacts_log(runtime, monkeypatch):
    sup, _ = runtime
    spawned = []
    monkeypatch.setattr(supervisor, "_direct_io_args", lambda binary: ())

    def spawn(cmd, **kwargs):
        spawned.append(cmd)
        return SimpleNamespace(pid=1234 + len(spawned), poll=lambda: None), None

    monkeypatch.setattr(supervisor, "spawn_server", spawn)
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    public = sup.api_key
    sup._spawn()
    first = sup._request_target()
    sup._spawn()
    second = sup._request_target()
    assert first.operation_api_key != second.operation_api_key
    assert sup.api_key == public
    assert spawned[0][spawned[0].index("--api-key") + 1] == public + "," + first.operation_api_key
    assert spawned[1][spawned[1].index("--api-key") + 1] == public + "," + second.operation_api_key
    log = sup.log_path.read_text(encoding="utf-8")
    assert public not in log and first.operation_api_key not in log and second.operation_api_key not in log
    assert first.operation_api_key not in repr(first)
    sup._log_handle.close()


def test_start_cannot_adopt_router_replaced_after_its_spawn(runtime, monkeypatch):
    sup, _ = runtime
    original_wait = sup._wait_health
    monkeypatch.setattr(sup, "_spawn", lambda: None)

    def replace_before_health(timeout_s, **kwargs):
        sup.proc = SimpleNamespace(poll=lambda: None)
        sup._operation_api_key = "replacement-operation-key"
        original_wait(timeout_s, **kwargs)

    monkeypatch.setattr(sup, "_wait_health", replace_before_health)
    with pytest.raises(RuntimeError, match="replaced"):
        sup.start(timeout_s=5)
    assert sup._watchdog is None


def test_public_health_cannot_authenticate_foreign_listener(runtime):
    sup, _ = runtime
    server = _GenerationServer({sup.api_key, "another-generation-key"})
    sup.port = server.port
    try:
        with pytest.raises(TimeoutError):
            sup._wait_health(timeout_s=1)
        assert server.rejected >= 1 and server.accepted == []
    finally:
        server.close()


def test_watchdog_retries_own_failed_spawn_without_adopting_another_generation(runtime, monkeypatch):
    sup, _ = runtime
    sup.proc = SimpleNamespace(pid=101, poll=lambda: 1)
    previous_key = sup._operation_api_key
    attempts = []
    replacement = SimpleNamespace(pid=202, poll=lambda: None)
    monkeypatch.setattr(supervisor, "_direct_io_args", lambda binary: ())

    def spawn(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("controlled spawn failure")
        return replacement, None

    monkeypatch.setattr(supervisor, "spawn_server", spawn)
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    monkeypatch.setattr(sup, "_reap_orphaned_children", lambda: None)
    monkeypatch.setattr(sup, "_wait_health", lambda *args, **kwargs: sup._stop_event.set())
    sup._watch()
    assert len(attempts) == 2
    assert sup.proc is replacement and sup._operation_api_key != previous_key
    sup._log_handle.close()


def test_failed_state_publication_stops_only_new_owned_child_and_restores_lineage(runtime, monkeypatch):
    sup, _ = runtime
    previous_proc, previous_key = sup.proc, sup._operation_api_key
    child = SimpleNamespace(pid=202, poll=lambda: None)
    closed, terminated = [], []
    job = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(supervisor, "_direct_io_args", lambda binary: ())
    monkeypatch.setattr(supervisor, "spawn_server", lambda *args, **kwargs: (child, job))
    monkeypatch.setattr(sup, "_terminate_tree", lambda proc: terminated.append(proc))

    def fail_state():
        raise OSError("controlled state publication failure")

    monkeypatch.setattr(sup, "_write_state", fail_state)
    with pytest.raises(OSError, match="publication"):
        sup._spawn()
    assert closed == [True] and terminated == [child]
    assert sup.proc is previous_proc and sup._operation_api_key == previous_key
    assert sup._log_handle is None


def test_watchdog_retries_child_that_exits_immediately_after_successful_spawn(runtime, monkeypatch):
    sup, _ = runtime
    sup.proc = SimpleNamespace(pid=101, poll=lambda: 1)
    dead_child = SimpleNamespace(pid=202, poll=lambda: 9)
    healthy_child = SimpleNamespace(pid=303, poll=lambda: None)
    attempts = []
    monkeypatch.setattr(supervisor, "_direct_io_args", lambda binary: ())

    def spawn(*args, **kwargs):
        attempts.append(True)
        return (dead_child if len(attempts) == 1 else healthy_child), None

    original_health = sup._wait_health

    def health(timeout_s, **kwargs):
        if kwargs["_target"].proc is dead_child:
            return original_health(timeout_s, **kwargs)
        sup._stop_event.set()

    monkeypatch.setattr(supervisor, "spawn_server", spawn)
    monkeypatch.setattr(sup, "_write_state", lambda: None)
    monkeypatch.setattr(sup, "_reap_orphaned_children", lambda: None)
    monkeypatch.setattr(sup, "_wait_health", health)
    sup._watch()
    assert len(attempts) == 2 and sup.proc is healthy_child
    sup._log_handle.close()


@pytest.mark.parametrize("processing", [0, 1])
def test_idle_result_is_uncertain_if_router_changes_during_metrics(runtime, monkeypatch, processing):
    from contextlib import contextmanager

    sup, _ = runtime
    monkeypatch.setattr(sup, "_request", lambda *args, **kwargs: [])

    def read():
        sup.proc = SimpleNamespace(poll=lambda: None)
        sup._operation_api_key = "replacement-operation-key"
        return f"llamacpp:requests_processing {processing}\n".encode()

    @contextmanager
    def metrics(*args, **kwargs):
        yield SimpleNamespace(read=read)

    monkeypatch.setattr(sup, "_open", metrics)
    assert sup._probe_idle("selected") is None


def test_watchdog_thread_inherits_callers_profile_context(runtime, monkeypatch, tmp_path):
    import threading
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    sup, _ = runtime
    profile_home = tmp_path / "profiles" / "research"
    observed, finished = [], threading.Event()
    expected_proc = sup.proc
    monkeypatch.setattr(sup, "_spawn", lambda: None)
    monkeypatch.setattr(sup, "_wait_health", lambda *args, **kwargs: None)

    def watch(target):
        observed.append((get_hermes_home(), target.proc))
        finished.set()

    monkeypatch.setattr(sup, "_watch", watch)
    token = set_hermes_home_override(profile_home)
    try:
        sup.start(timeout_s=2)
    finally:
        reset_hermes_home_override(token)
    assert finished.wait(timeout=5)
    sup._watchdog.join(timeout=5)
    assert not sup._watchdog.is_alive()
    assert sup._watchdog.daemon and sup._watchdog.name == "llamacpp-supervisor"
    assert observed == [(profile_home, expected_proc)]
