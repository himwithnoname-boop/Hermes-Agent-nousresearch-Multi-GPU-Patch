"""Schedule managed model attempts before their transport watchdog starts.

Tool work remains concurrent. Permits cover stream consumption and surviving
request workers, including protected auxiliary callbacks that outlive cancellation.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from functools import wraps
import logging
import threading
import time

from hermes_cli.local_runtime.request_queue import QueueExpired
from hermes_cli.local_runtime.request_target import bounded_number, resolve_target

logger = logging.getLogger(__name__)
_current: ContextVar[Lease | None] = ContextVar("local_model_admission", default=None)


class Lease:
    def __init__(self, target, *, background=False):
        self.target = target
        self.queue = target.queue()
        self.token = self.queue.enqueue(target.model, background=background)
        self.started = time.monotonic()
        self.admitted = None
        self.workers = []
        self.requests = []
        self.finished = False

    def acquire(self, cancelled=lambda: False, progress=lambda message: None):
        timeout = bounded_number(self.target.settings.get("queue_timeout_seconds"), 3600, 1, 86400)
        last_notice = float("-inf")
        drain_probe_at = float("-inf")
        pressure_at, pressured, reason = float("-inf"), False, ""
        try:
            while True:
                if cancelled():
                    raise InterruptedError("Canceled while waiting for the local model; no request was sent.")
                if not self.target.valid():
                    raise QueueExpired("Managed model server changed while queued; no request was sent.")
                if time.monotonic() - self.started >= timeout:
                    raise QueueExpired("Local model queue deadline reached; no request was sent.")
                if time.monotonic() - drain_probe_at >= 1:
                    for token, model in self.queue.draining():
                        if self.target.idle(model):
                            self.queue.clear_drained([token])
                    drain_probe_at = time.monotonic()
                if time.monotonic() - pressure_at >= 1:
                    from hermes_cli.local_runtime.adaptive_resources import admission_pressure

                    pressured, reason = admission_pressure()
                    pressure_at = time.monotonic()
                if not pressured and self.queue.try_admit(self.token):
                    self.admitted = time.monotonic()
                    logger.info("Local model admission: queue_wait_ms=%.0f capacity=%d",
                                1000 * (self.admitted - self.started), self.target.slots)
                    return
                if time.monotonic() - last_notice >= 5:
                    progress("Waiting for " + reason if pressured else "Waiting in local model queue")
                    last_notice = time.monotonic()
                threading.Event().wait(0.25)
        except BaseException:
            self.queue.finish(self.token)
            raise

    def _threads(self):
        threads = list(self.workers)
        for request in self.requests:
            thread = getattr(request, "worker", None) or getattr(request, "thread", None)
            if thread is not None:
                threads.append(thread)
        return [thread for thread in threads if thread is not threading.current_thread() and thread.is_alive()]

    def finish(self, *, uncertain=False):
        if self.finished:
            return
        self.finished = True
        uncertain = uncertain or any(
            bool(getattr(request, "result", {}).get("error")) or bool(getattr(request, "cancelled", False))
            for request in self.requests)
        threads = self._threads()
        if threads:
            from agent.memory_provider import spawn_context_thread

            def drain():
                for thread in threads:
                    thread.join()
                self.queue.finish(self.token, uncertain=True)

            spawn_context_thread(drain, name="local-admission-drain").start()
        else:
            self.queue.finish(self.token, uncertain=uncertain)
        if self.admitted is not None:
            logger.info("Local model admission finished: request_ms=%.0f draining=%s",
                        1000 * (time.monotonic() - self.admitted), bool(threads) or uncertain)


def track_request(request):
    lease = _current.get()
    if lease is not None:
        lease.requests.append(request)


def track_thread(thread):
    lease = _current.get()
    if lease is not None:
        lease.workers.append(thread)


def network_retry_limit(default: int) -> int:
    """Managed failures re-enter admission after cleanup instead of racing orphaned work."""
    return 0 if _current.get() is not None else default


def guarded_provider(callback):
    """Recheck ownership after hooks and immediately before opening the wire."""
    @wraps(callback)
    def create(request):
        lease = _current.get()
        if lease is not None:
            if not lease.target.valid() or request.get("model") != lease.target.model:
                raise QueueExpired("Managed model route changed before dispatch; no request was sent.")
        return callback(request)
    return create


def run_streaming_request(agent, api_kwargs, on_first_delta):
    """Dispatch under admission, retaining the facade's provider/test seams."""
    from agent import chat_completion_helpers as helpers

    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted before streaming API call")
    if agent.api_mode == "codex_responses":
        return helpers._stream_codex_passthrough(agent, api_kwargs, on_first_delta)
    if agent.api_mode == "bedrock_converse":
        return helpers._BedrockStream(agent, api_kwargs, on_first_delta).run()
    helpers._check_stale_giveup(agent)
    return helpers._StreamingCall(agent, api_kwargs, on_first_delta).run()


def _new_lease(base_url, model, *, background=False):
    target = resolve_target(base_url, model)
    if target is None:
        return None
    current = _current.get()
    if current is not None and current.target.generation == target.generation:
        return None  # one nested physical attempt, not another slot
    return Lease(target, background=background)


def scheduled_agent_call(function):
    @wraps(function)
    def call(agent, api_kwargs, *args, **kwargs):
        from agent.delegation_context import is_delegated_child_process_context

        lease = _new_lease(getattr(agent, "base_url", ""), api_kwargs.get("model", ""),
                           background=is_delegated_child_process_context())
        if lease is None:
            return function(agent, api_kwargs, *args, **kwargs)
        lease.acquire(lambda: bool(getattr(agent, "_interrupt_requested", False)),
                      lambda message: agent._touch_activity(message))
        token = _current.set(lease)
        failed = True
        try:
            result = function(agent, api_kwargs, *args, **kwargs)
            failed = bool(getattr(agent, "_interrupt_requested", False))
            return result
        finally:
            _current.reset(token)
            lease.finish(uncertain=failed)
    return call


class ScheduledStream:
    def __init__(self, stream, lease):
        self.stream, self.lease = stream, lease
        self.complete = False

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __iter__(self):
        try:
            yield from self.stream
            self.complete = True
        finally:
            self.close()

    def close(self):
        try:
            close = getattr(self.stream, "close", None)
            if callable(close):
                close()
        finally:
            self.lease.finish(uncertain=not self.complete)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _capture_aux_cancellation():
    from agent.auxiliary_client import (
        AuxiliaryExplicitCancellation, _capture_aux_cancel_check, _captured_aux_cancel_requested,
    )

    # The owner uses thread-local protection; asyncio.to_thread cannot inherit it.
    check = _capture_aux_cancel_check()

    def cancelled():
        if check is not None and _captured_aux_cancel_requested(check):
            # Preserve the host's non-retryable signal rather than a connection error.
            raise AuxiliaryExplicitCancellation()
        return False

    return cancelled


def scheduled_aux_call(*, streaming=False):
    def decorate(function):
        @wraps(function)
        def call(client, kwargs, *args, **options):
            lease = _new_lease(str(getattr(client, "base_url", "")), kwargs.get("model", ""), background=True)
            if lease is None:
                return function(client, kwargs, *args, **options)
            lease.acquire(_capture_aux_cancellation())
            token = _current.set(lease)
            transferred = succeeded = False
            try:
                result = function(client, kwargs, *args, **options)
                # Relay/adapters can complete a streaming request without returning an iterator.
                if streaming and not hasattr(result, "choices"):
                    transferred = True
                    return ScheduledStream(result, lease)
                succeeded = True
                return result
            finally:
                _current.reset(token)
                if not transferred:
                    lease.finish(uncertain=not succeeded)
        return call
    return decorate


def scheduled_async_aux_call(function):
    @wraps(function)
    async def call(client, kwargs, *args, **options):
        lease = _new_lease(str(getattr(client, "base_url", "")), kwargs.get("model", ""), background=True)
        if lease is None:
            return await function(client, kwargs, *args, **options)
        canceled = threading.Event()
        host_cancelled = _capture_aux_cancellation()
        waiter = asyncio.create_task(asyncio.to_thread(
            lease.acquire, lambda: canceled.is_set() or host_cancelled()))
        try:
            await asyncio.shield(waiter)
        except BaseException:
            canceled.set()
            try:
                await asyncio.shield(waiter)
            except (InterruptedError, QueueExpired):
                pass
            finally:
                lease.finish()
            raise
        token = _current.set(lease)
        succeeded = False
        try:
            result = await function(client, kwargs, *args, **options)
            succeeded = True
            return result
        finally:
            _current.reset(token)
            lease.finish(uncertain=not succeeded)
    return call
