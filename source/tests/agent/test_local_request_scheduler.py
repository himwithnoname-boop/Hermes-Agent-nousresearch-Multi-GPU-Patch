"""Production entrypoint coverage for waiting, cancellation and stream ownership."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from agent import local_request_scheduler as scheduler
from hermes_cli.local_runtime.request_queue import QueueExpired, RequestQueue
from tests.waiting import wait_until


class Target:
    def __init__(self, path):
        self.path, self.model, self.generation = path, "m", "g"
        self.settings, self.slots = {"queue_timeout_seconds": 10}, 1
        self.valid_now, self.idle_now = True, True

    def valid(self):
        return self.valid_now

    def idle(self, model):
        return self.idle_now

    def queue(self):
        return RequestQueue(self.path, self.generation)


@pytest.fixture
def target(tmp_path, monkeypatch):
    value = Target(tmp_path / "admission.sqlite3")
    monkeypatch.setattr(scheduler, "resolve_target", lambda base, model: value if base == "local" else None)
    monkeypatch.setattr("hermes_cli.local_runtime.adaptive_resources.admission_pressure", lambda: (False, ""))
    return value


def agent():
    return SimpleNamespace(base_url="local", _interrupt_requested=False,
                           _touch_activity=lambda message: None)


def test_real_stream_entrypoint_waits_before_constructing_watchdogs(target, monkeypatch):
    from agent import chat_completion_helpers as helpers

    held = target.queue().enqueue("m")
    assert target.queue().try_admit(held)
    entered = threading.Event()

    class Call:
        def __init__(self, *args):
            entered.set()

        def run(self):
            return "complete"

    monkeypatch.setattr(helpers, "_StreamingCall", Call)
    a = agent()
    a.api_mode = "chat_completions"
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(helpers.interruptible_streaming_api_call, a, {"model": "m"})
        assert wait_until(lambda: target.queue().snapshot()["waiting"] == 1, timeout=10)
        assert not entered.is_set()
        target.queue().finish(held)
        assert result.result(timeout=10) == "complete"
    assert target.queue().snapshot()["active"] == 0


def test_stop_removes_waiter_without_calling_model(target):
    held = target.queue().enqueue("m")
    assert target.queue().try_admit(held)
    a, dispatched = agent(), []

    @scheduler.scheduled_agent_call
    def invoke(instance, kwargs):
        dispatched.append(True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(invoke, a, {"model": "m"})
        assert wait_until(lambda: target.queue().snapshot()["waiting"] == 1, timeout=10)
        a._interrupt_requested = True
        with pytest.raises(InterruptedError):
            result.result(timeout=10)
    assert not dispatched
    assert target.queue().snapshot()["waiting"] == 0


def test_restart_fences_waiter_and_does_not_send_request(target):
    target.valid_now = False
    lease = scheduler.Lease(target)
    with pytest.raises(QueueExpired):
        lease.acquire()
    assert target.queue().snapshot()["waiting"] == 0


def test_pressure_wait_is_cancelable_and_does_not_admit(target, monkeypatch):
    monkeypatch.setattr("hermes_cli.local_runtime.adaptive_resources.admission_pressure",
                        lambda: (True, "available physical RAM"))
    cancel, notice = threading.Event(), threading.Event()
    lease = scheduler.Lease(target)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(lease.acquire, cancel.is_set, lambda message: notice.set())
        assert notice.wait(10)
        assert target.queue().snapshot()["active"] == 0
        cancel.set()
        with pytest.raises(InterruptedError):
            result.result(timeout=10)


def test_stream_permit_lasts_until_consumption_or_close(target):
    client = SimpleNamespace(base_url="local")

    @scheduler.scheduled_aux_call(streaming=True)
    def create(client, kwargs):
        return iter(["one", "two"])

    stream = create(client, {"model": "m"})
    assert target.queue().snapshot()["active"] == 1
    assert list(stream) == ["one", "two"]
    assert target.queue().snapshot()["active"] == 0
    stream = create(client, {"model": "m"})
    stream.close()
    assert target.queue().snapshot()["draining"] == 1


def test_exception_quarantines_until_router_reports_idle(target):
    target.idle_now = False

    @scheduler.scheduled_agent_call
    def invoke(instance, kwargs):
        raise OSError("closed transport")

    with pytest.raises(OSError):
        invoke(agent(), {"model": "m"})
    assert target.queue().snapshot()["draining"] == 1
    target.idle_now = True
    lease = scheduler.Lease(target)
    lease.acquire()
    lease.finish()
    assert target.queue().snapshot()["draining"] == 0


def test_worker_surviving_cancel_retains_capacity(target):
    worker_done, worker_started = threading.Event(), threading.Event()

    @scheduler.scheduled_agent_call
    def invoke(instance, kwargs):
        def worker():
            worker_started.set()
            worker_done.wait(10)
        thread = threading.Thread(target=worker)
        scheduler.track_thread(thread)
        thread.start()
        assert worker_started.wait(10)
        raise InterruptedError("owner canceled")

    try:
        with pytest.raises(InterruptedError):
            invoke(agent(), {"model": "m"})
        assert target.queue().snapshot()["active"] == 1
    finally:
        worker_done.set()
    assert wait_until(lambda: target.queue().snapshot()["draining"] == 1, timeout=10)


def test_nested_dispatch_uses_one_permit_and_remote_bypasses(target):
    @scheduler.scheduled_agent_call
    def inner(instance, kwargs):
        assert target.queue().snapshot()["active"] == 1
        assert scheduler.network_retry_limit(2) == 0
        return "ok"

    @scheduler.scheduled_agent_call
    def outer(instance, kwargs):
        return inner(instance, kwargs)

    assert outer(agent(), {"model": "m"}) == "ok"
    assert scheduler.network_retry_limit(2) == 2
    a = agent()
    a.base_url = "remote"

    @scheduler.scheduled_agent_call
    def remote(instance, kwargs):
        return target.queue().snapshot()["active"]

    assert remote(a, {"model": "m"}) == 0


@pytest.mark.asyncio
async def test_async_cancel_removes_queued_ticket(target):
    from tests.waiting import async_wait_until

    held = target.queue().enqueue("m")
    assert target.queue().try_admit(held)

    @scheduler.scheduled_async_aux_call
    async def invoke(client, kwargs):
        raise AssertionError("must not dispatch")

    task = asyncio.create_task(invoke(SimpleNamespace(base_url="local"), {"model": "m"}))
    assert await async_wait_until(lambda: target.queue().snapshot()["waiting"] == 1, timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert target.queue().snapshot()["waiting"] == 0


@pytest.mark.asyncio
async def test_real_auxiliary_async_and_sync_entrypoints_hold_permit(target, monkeypatch):
    from agent import auxiliary_client as aux

    monkeypatch.setattr(aux, "_relay_auxiliary_metadata", lambda **kwargs: None)
    client = SimpleNamespace(base_url="local")

    def create(request):
        assert target.queue().snapshot()["active"] == 1
        return "sync"

    async def acreate(request):
        assert target.queue().snapshot()["active"] == 1
        return "async"

    assert aux._relay_sync_completion(client, {"model": "m"}, create=create) == "sync"
    assert await aux._relay_async_completion(client, {"model": "m"}, create=acreate) == "async"
    assert target.queue().snapshot()["active"] == 0


@pytest.mark.parametrize("change", ["restart", "model"])
def test_final_provider_fence_rejects_changed_route(target, change):
    called = []

    @scheduler.scheduled_agent_call
    def invoke(instance, kwargs):
        if change == "restart":
            target.valid_now = False
        else:
            kwargs = {"model": "different"}
        return scheduler.guarded_provider(lambda request: called.append(request))(kwargs)

    with pytest.raises(QueueExpired):
        invoke(agent(), {"model": "m"})
    assert called == []


def test_queue_deadline_sends_nothing_and_removes_ticket(target, monkeypatch):
    lease = scheduler.Lease(target)
    lease.started -= 11
    with pytest.raises(QueueExpired, match="deadline"):
        lease.acquire()
    assert target.queue().snapshot()["waiting"] == 0


def test_real_nonstream_entrypoint_acquires_before_watchdog_constructor(target, monkeypatch):
    from agent import chat_completion_helpers as helpers
    from agent import chat_completion_nonstream as nonstream

    a = agent()
    a.api_mode, a.provider, a.platform = "chat_completions", "llamacpp", "cli"
    held = target.queue().enqueue("m")
    assert target.queue().try_admit(held)
    entered = threading.Event()

    class Call:
        def __init__(self, *args):
            entered.set()

        def run(self):
            return "complete"

    monkeypatch.setattr(nonstream, "_NonStreamRequest", Call)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(helpers.interruptible_api_call, a, {"model": "m"})
        assert wait_until(lambda: target.queue().snapshot()["waiting"] == 1, timeout=10)
        assert not entered.is_set()
        target.queue().finish(held)
        assert result.result(timeout=10) == "complete"


def test_two_real_auxiliary_calls_never_overlap_at_one_slot(target, monkeypatch):
    from agent import auxiliary_client as aux

    monkeypatch.setattr(aux, "_relay_auxiliary_metadata", lambda **kwargs: None)
    entered, release = threading.Event(), threading.Event()
    client = SimpleNamespace(base_url="local")
    calls = []

    def create(request):
        calls.append(request["model"])
        entered.set()
        assert release.wait(10)
        return "done"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(aux._relay_sync_completion, client, {"model": "m"}, create=create)
        assert entered.wait(10)
        second = pool.submit(aux._relay_sync_completion, client, {"model": "m"}, create=create)
        try:
            assert wait_until(lambda: target.queue().snapshot()["waiting"] == 1, timeout=10)
            assert calls == ["m"]
        finally:
            release.set()
        assert first.result(timeout=10) == second.result(timeout=10) == "done"
    assert calls == ["m", "m"]


def test_managed_stream_keeps_admission_and_upstream_prompt_progress(target, monkeypatch):
    """The new local progress frames stay invisible while the request owns its slot."""
    from unittest.mock import MagicMock
    from run_agent import AIAgent

    base = "http://127.0.0.1:18434/v1"
    monkeypatch.setattr(scheduler, "resolve_target", lambda url, model: target if url == base else None)
    client = MagicMock()
    sent = []

    def chunk(content=None, progress=None, finish=None):
        delta = SimpleNamespace(role="assistant", content=content, tool_calls=None,
                                reasoning_content=None, reasoning=None, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish)],
                               model="m", usage=None, id="fixture", model_extra=progress or {})

    def create(**kwargs):
        assert target.queue().snapshot()["active"] == 1
        sent.append(kwargs)
        return iter([
            chunk(progress={"prompt_progress": {"total": 100, "processed": 50, "cache": 0}}),
            chunk(content="complete", finish="stop"),
        ])

    client.chat.completions.create.side_effect = create
    monkeypatch.setattr(AIAgent, "_create_request_openai_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(AIAgent, "_close_request_openai_client", lambda *args, **kwargs: None)
    instance = AIAgent(api_key="fixture", base_url=base, provider="llamacpp", model="m",
                       quiet_mode=True, skip_context_files=True, skip_memory=True)
    deltas = []
    instance.stream_delta_callback = deltas.append
    response = instance._interruptible_streaming_api_call(
        {"model": "m", "messages": [{"role": "user", "content": "hello"}]})
    assert sent[0]["extra_body"]["return_progress"] is True
    assert response.choices[0].message.content == "complete"
    assert "".join(deltas) == "complete"
    assert target.queue().snapshot()["active"] == 0

def test_completed_auxiliary_stream_response_releases_permit(target, monkeypatch):
    """Adapters may return a complete response even when streaming was requested."""
    from agent import auxiliary_client as aux

    monkeypatch.setattr(aux, "_relay_auxiliary_metadata", lambda **kwargs: None)
    completed = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done"))])
    client = SimpleNamespace(base_url="local", chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kwargs: completed)))
    result = aux._relay_sync_stream(client, {"model": "m", "messages": [], "stream": True})
    assert target.queue().snapshot()["active"] == 0
    assert result is completed
@pytest.mark.asyncio
async def test_explicit_host_stop_cancels_async_queue_without_dispatch(target, monkeypatch):
    """A queue worker captures the owner's thread-local Stop source before crossing threads."""
    from agent import auxiliary_client as aux
    from tests.waiting import async_wait_until

    held = target.queue().enqueue("m")
    assert target.queue().try_admit(held)
    target.settings["queue_timeout_seconds"] = 60
    stopped = threading.Event()
    called = []
    monkeypatch.setattr(aux, "_relay_auxiliary_metadata", lambda **kwargs: None)

    async def create(request):
        called.append(request)
        return "must not send"

    with aux.aux_interrupt_protection(cancel_event=stopped):
        task = asyncio.create_task(aux._relay_async_completion(
            SimpleNamespace(base_url="local"), {"model": "m"}, create=create))
        try:
            assert await async_wait_until(lambda: target.queue().snapshot()["waiting"] == 1, timeout=10)
            stopped.set()
            assert await async_wait_until(task.done, timeout=3), "explicit host Stop was lost across queue thread"
            with pytest.raises(aux.AuxiliaryExplicitCancellation):
                await task
            assert target.queue().snapshot()["waiting"] == 0
            assert called == []
        finally:
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            target.queue().finish(held)


def test_explicit_host_stop_in_sync_queue_is_not_retryable(target, monkeypatch):
    """Host Stop stays the frozen non-retryable signal at the queue boundary."""
    from agent import auxiliary_client as aux

    stopped = threading.Event()
    stopped.set()
    called = []
    monkeypatch.setattr(aux, "_relay_auxiliary_metadata", lambda **kwargs: None)
    with aux.aux_interrupt_protection(cancel_event=stopped):
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            aux._relay_sync_completion(SimpleNamespace(base_url="local"), {"model": "m"},
                                       create=lambda request: called.append(request))
    assert target.queue().snapshot()["waiting"] == 0
    assert called == []
