"""Portable memory bounds and idle-refit hysteresis, without model execution."""

from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import adaptive_resources as resources
from hermes_cli.local_runtime import hardware
from hermes_cli.local_runtime.estimator import HardwareBudget

G = 1 << 30


@pytest.mark.parametrize("room,residents", [(0, 1), (G // 4, 1), (G, 1), (16 * G, 4), (100 * G, 2)])
def test_cache_default_preserves_bounded_per_resident_grant(room, residents):
    grant = resources.cache_grant(room, residents, settings={})
    assert grant <= 8 * G
    assert grant * residents <= room
    assert grant == 0 or grant >= G // 2


def test_operator_can_use_more_spare_ram_but_cannot_exceed_room():
    cfg = {"prompt_cache_max_mib": 32768, "prompt_cache_room_fraction": 0.5}
    assert resources.cache_grant(100 * G, 2, settings=cfg) == 25 * G
    assert resources.cache_grant(2 * G, 2, settings=cfg) == G // 2
    assert resources.cache_grant(100 * G, 1, settings={"prompt_cache_max_mib": 0}) == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "bad", True, -1])
def test_bad_or_unlimited_cache_setting_never_grants_unbounded_memory(value):
    assert resources.cache_grant(1000 * G, 1, settings={"prompt_cache_max_mib": value}) <= 8 * G


@pytest.mark.parametrize("uma", [False, True])
def test_large_cache_cap_still_obeys_independent_physical_and_commit_limits(monkeypatch, uma):
    monkeypatch.setattr(resources, "resource_config", lambda: {"prompt_cache_max_mib": 32768})
    budget = HardwareBudget(20 * G, 24 * G, 3 * G, uma=uma,
                            commit_room_bytes=80 * G, host_cache_ram_bytes=3 * G)
    assert hardware.prompt_cache_bytes(budget, 10 * G) <= 3 * G
    budget = HardwareBudget(20 * G, 24 * G, 100 * G, uma=uma,
                            commit_room_bytes=G, host_cache_ram_bytes=100 * G)
    assert hardware.prompt_cache_bytes(budget, 10 * G) == 0


def test_pressure_detects_ram_and_commit_independently(monkeypatch):
    monkeypatch.setattr(resources, "resource_config", dict)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (128 * G, G))
    monkeypatch.setattr(hardware, "_windows_commit_room", lambda: 100 * G)
    assert resources.admission_pressure() == (True, "available physical RAM")
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (128 * G, 100 * G))
    monkeypatch.setattr(hardware, "_windows_commit_room", lambda: 0)
    assert resources.admission_pressure() == (True, "Windows commit headroom")
    monkeypatch.setattr(hardware, "_windows_commit_room", lambda: None)
    assert resources.admission_pressure()[0] is False


def test_refit_growth_stabilizes_and_pressure_reduction_is_immediate():
    sup = SimpleNamespace(_refit_resources=(10 * G, 10 * G, None), _adaptive_refit_at=0)
    higher = (12 * G, 20 * G, None)
    assert not resources.refit_ready(sup, higher, now=10, settings={})
    assert not resources.refit_ready(sup, higher, now=41, settings={})
    assert resources.refit_ready(sup, higher, now=61, settings={})
    assert resources.refit_ready(sup, (8 * G, 10 * G, None), now=62, settings={})


def test_refit_profiles_do_not_share_pending_state():
    a = SimpleNamespace(_refit_resources=(G, G, None))
    b = SimpleNamespace(_refit_resources=(G, G, None))
    assert not resources.refit_ready(a, (2 * G, 2 * G, None), now=1, settings={})
    assert resources.refit_ready(a, (2 * G, 2 * G, None), now=32, settings={})
    assert not resources.refit_ready(b, (2 * G, 2 * G, None), now=32, settings={})


def test_refit_and_requests_share_exclusive_admission(tmp_path, monkeypatch):
    from hermes_cli.local_runtime.refit_admission import idle_refit_guard
    from hermes_cli.local_runtime.request_queue import RequestQueue

    q = RequestQueue(tmp_path / "queue.sqlite3", "generation")
    target = SimpleNamespace(model="maintenance", queue=lambda: q, valid=lambda: True)
    monkeypatch.setattr("hermes_cli.local_runtime.request_target.resolve_target", lambda *args: target)
    with idle_refit_guard() as admitted:
        assert admitted
        request = q.enqueue("m")
        assert not q.try_admit(request)
    assert q.try_admit(request)
    with idle_refit_guard() as admitted:
        assert not admitted
    assert q.snapshot()["waiting"] == 0
    q.finish(request)


def test_refit_does_not_jump_a_waiting_request(tmp_path, monkeypatch):
    from hermes_cli.local_runtime.refit_admission import idle_refit_guard
    from hermes_cli.local_runtime.request_queue import RequestQueue

    q = RequestQueue(tmp_path / "queue.sqlite3", "generation")
    target = SimpleNamespace(model="maintenance", queue=lambda: q, valid=lambda: True)
    monkeypatch.setattr("hermes_cli.local_runtime.request_target.resolve_target", lambda *args: target)
    request = q.enqueue("m")
    with idle_refit_guard() as admitted:
        assert not admitted
    assert q.try_admit(request)
