"""Durable admission tickets shared by local-runtime clients.

Transactions never span inference or a network probe. A dead request owner leaves
a draining ticket until the owning router confirms idle; age alone never evicts
an active request. The database contains identities and timings, never prompts.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import time
import uuid

import psutil


class QueueFull(RuntimeError):
    """The bounded local admission queue cannot accept another request."""


class QueueExpired(TimeoutError):
    """A request exhausted its queue budget before opening a connection."""


def owner_alive(pid: int, birth: float) -> bool:
    """Unknown/access-denied owners remain live; PID reuse is not ownership."""
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and abs(proc.create_time() - birth) < 0.001
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True


class RequestQueue:
    def __init__(self, path: Path, generation: str, *, capacity: int = 1,
                 max_waiters: int = 64, aging_seconds: float = 30):
        self.path = path
        self.generation = generation
        self.capacity = max(1, capacity)
        self.max_waiters = max_waiters
        self.aging_seconds = aging_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS tickets ("
                "id TEXT PRIMARY KEY, generation TEXT NOT NULL, model TEXT NOT NULL,"
                "pid INTEGER NOT NULL, birth REAL NOT NULL, rank REAL NOT NULL,"
                "enqueued REAL NOT NULL, admitted REAL, state TEXT NOT NULL, capacity INTEGER NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS ticket_generation ON tickets(generation, state)")

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=2, isolation_level=None)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def enqueue(self, model: str, *, background: bool = False) -> str:
        token, now = uuid.uuid4().hex, time.time()
        with self._transaction() as db:
            self._reap(db)
            count = db.execute(
                "SELECT count(*) FROM tickets WHERE generation=? AND state='waiting'",
                (self.generation,),
            ).fetchone()[0]
            if count >= self.max_waiters:
                raise QueueFull("Local model request queue is full; no request was sent.")
            db.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'waiting', ?)",
                       (token, self.generation, model, os.getpid(), psutil.Process().create_time(),
                        now + (self.aging_seconds if background else 0), now, self.capacity))
        return token

    def _reap(self, db):
        owners = {}
        for token, generation, pid, birth, state in db.execute(
                "SELECT id,generation,pid,birth,state FROM tickets").fetchall():
            identity = (pid, birth)
            if identity not in owners:
                owners[identity] = owner_alive(pid, birth)
            if not owners[identity]:
                if state == "waiting" or generation != self.generation:
                    db.execute("DELETE FROM tickets WHERE id=?", (token,))
                else:
                    db.execute("UPDATE tickets SET state='draining' WHERE id=?", (token,))

    def draining(self) -> list[tuple[str, str]]:
        with self._transaction() as db:
            self._reap(db)
            return db.execute(
                "SELECT id,model FROM tickets WHERE generation=? AND state='draining'",
                (self.generation,),
            ).fetchall()

    def clear_drained(self, tokens: list[str]) -> None:
        with self._transaction() as db:
            db.executemany("DELETE FROM tickets WHERE id=? AND generation=? AND state='draining'",
                           [(token, self.generation) for token in tokens])

    def try_admit(self, token: str) -> bool:
        with self._transaction() as db:
            self._reap(db)
            held = db.execute(
                "SELECT model,capacity FROM tickets WHERE generation=? AND state!='waiting'",
                (self.generation,),
            ).fetchall()
            own = db.execute("SELECT model FROM tickets WHERE id=?", (token,)).fetchone()
            if own is None:
                raise QueueExpired("Local admission ticket was retired before dispatch.")
            capacity = min([self.capacity, *(row[1] for row in held)])
            if len(held) >= capacity or any(row[0] != own[0] for row in held):
                return False
            first = db.execute(
                "SELECT id FROM tickets WHERE generation=? AND state='waiting' ORDER BY rank,enqueued,id LIMIT 1",
                (self.generation,),
            ).fetchone()
            if first is None or first[0] != token:
                return False
            db.execute("UPDATE tickets SET state='active',admitted=? WHERE id=?", (time.time(), token))
            return True

    def finish(self, token: str, *, uncertain: bool = False) -> None:
        with self._transaction() as db:
            if uncertain:
                db.execute("UPDATE tickets SET state='draining' WHERE id=? AND state='active'", (token,))
            else:
                db.execute("DELETE FROM tickets WHERE id=?", (token,))

    def snapshot(self) -> dict:
        with self._transaction() as db:
            counts = dict(db.execute(
                "SELECT state,count(*) FROM tickets WHERE generation=? GROUP BY state", (self.generation,)))
            oldest = db.execute(
                "SELECT min(enqueued) FROM tickets WHERE generation=? AND state='waiting'",
                (self.generation,),
            ).fetchone()[0]
        return {"capacity": self.capacity, "waiting": counts.get("waiting", 0),
                "active": counts.get("active", 0), "draining": counts.get("draining", 0),
                "oldest_wait_seconds": max(0, time.time() - oldest) if oldest else 0}
