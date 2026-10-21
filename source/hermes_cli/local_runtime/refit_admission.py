"""Keep idle preset reloads outside admitted model requests."""

from contextlib import contextmanager
import logging
import sqlite3

from hermes_cli.local_runtime.request_queue import QueueFull

logger = logging.getLogger(__name__)


@contextmanager
def idle_refit_guard():
    """Nonblocking maintenance admission; normal requests retain FIFO precedence.

    Existing ownership checks still govern unmanaged/disabled scheduling. Once a
    managed queue exists, a refit holds its sole permit through the final resident
    check and reload, closing the no-resident-check / new-request race.
    """
    from hermes_cli.local_runtime.recovery import read_state
    from hermes_cli.local_runtime.request_target import resolve_target

    state = read_state()
    target = resolve_target(state.get("base_url", ""), "__hermes_idle_refit__")
    if target is None:
        yield True
        return
    queue, ticket = None, None
    try:
        queue = target.queue()
        ticket = queue.enqueue(target.model, background=True)
        admitted = queue.try_admit(ticket) and target.valid()
    except (OSError, sqlite3.Error, QueueFull):
        logger.debug("idle refit admission unavailable", exc_info=True)
        admitted = False
    try:
        yield admitted
    finally:
        if ticket is not None:
            queue.finish(ticket)
