"""Real SQLite/process contracts for local model admission."""

import multiprocessing
import os
import sqlite3

import psutil
import pytest

from hermes_cli.local_runtime.request_queue import QueueFull, RequestQueue, owner_alive


def queue(tmp_path, **kwargs):
    return RequestQueue(tmp_path / "queue.sqlite3", "generation", **kwargs)


def test_fifo_capacity_and_release(tmp_path):
    q = queue(tmp_path)
    first, second = q.enqueue("m"), q.enqueue("m")
    assert not q.try_admit(second)
    assert q.try_admit(first)
    assert not q.try_admit(second)
    q.finish(first)
    assert q.try_admit(second)
    q.finish(second)
    assert q.snapshot()["active"] == 0


def test_bounded_queue_and_cancelled_waiter(tmp_path):
    q = queue(tmp_path, max_waiters=1)
    first = q.enqueue("m")
    with pytest.raises(QueueFull):
        q.enqueue("m")
    q.finish(first)
    assert q.try_admit(q.enqueue("m"))


def test_aging_gives_foreground_bounded_advantage(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.local_runtime.request_queue.time.time", lambda: 1000)
    q = queue(tmp_path, aging_seconds=30)
    child = q.enqueue("m", background=True)
    parent = q.enqueue("m")
    assert q.try_admit(parent)
    q.finish(parent)
    monkeypatch.setattr("hermes_cli.local_runtime.request_queue.time.time", lambda: 1031)
    later_parent = q.enqueue("m")
    assert not q.try_admit(later_parent)
    assert q.try_admit(child)


def test_explicit_capacity_and_model_switch_are_safe(tmp_path):
    q = queue(tmp_path, capacity=2)
    a, b, other = q.enqueue("a"), q.enqueue("a"), q.enqueue("b")
    assert q.try_admit(a) and q.try_admit(b)
    q.finish(a)
    assert not q.try_admit(other)
    q.finish(b)
    assert q.try_admit(other)


def test_changed_capacity_cannot_overbook_live_holder(tmp_path):
    q = queue(tmp_path, capacity=1)
    first = q.enqueue("m")
    assert q.try_admit(first)
    other = queue(tmp_path, capacity=4)
    assert not other.try_admit(other.enqueue("m"))


def test_failed_request_stays_draining_until_confirmed_idle(tmp_path):
    q = queue(tmp_path)
    first = q.enqueue("m")
    assert q.try_admit(first)
    q.finish(first, uncertain=True)
    second = q.enqueue("m")
    assert not q.try_admit(second)
    assert q.draining() == [(first, "m")]
    q.clear_drained([first])
    assert q.try_admit(second)


def test_old_generation_cannot_delete_new_generation_work(tmp_path):
    old = queue(tmp_path)
    new = RequestQueue(old.path, "new-generation")
    ticket = new.enqueue("m")
    assert new.try_admit(ticket)
    old.enqueue("m")
    assert new.snapshot()["active"] == 1


def test_pid_reuse_and_unknown_owner_are_distinct(monkeypatch):
    assert not owner_alive(os.getpid(), psutil.Process().create_time() - 100)
    def denied(pid):
        raise psutil.AccessDenied(pid)
    monkeypatch.setattr("hermes_cli.local_runtime.request_queue.psutil.Process", denied)
    assert owner_alive(123, 456)


def _hold_in_child(path, connection):
    q = RequestQueue(path, "generation")
    ticket = q.enqueue("m")
    assert q.try_admit(ticket)
    connection.send(ticket)
    connection.recv()


def test_real_process_exit_does_not_immediately_release_server_slot(tmp_path):
    from pathlib import Path

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_hold_in_child, args=(Path(tmp_path / "queue.sqlite3"), child))
    proc.start()
    try:
        assert parent.poll(20)
        ticket = parent.recv()
        q = queue(tmp_path)
        second = q.enqueue("m")
        assert not q.try_admit(second)
        parent.send("exit")
        proc.join(20)
        assert proc.exitcode == 0
        assert q.draining() == [(ticket, "m")]
        assert not q.try_admit(second)
        q.clear_drained([ticket])
        assert q.try_admit(second)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        parent.close()
        child.close()


def test_dead_waiter_does_not_block_fifo(tmp_path):
    q = queue(tmp_path)
    abandoned = q.enqueue("m")
    with sqlite3.connect(q.path) as db:
        db.execute("UPDATE tickets SET birth=birth-100 WHERE id=?", (abandoned,))
    live = q.enqueue("m")
    assert q.try_admit(live)
