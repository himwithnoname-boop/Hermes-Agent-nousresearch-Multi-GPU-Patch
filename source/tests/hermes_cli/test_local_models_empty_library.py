"""Local Models remains usable after deleting the selected model.

Exercise the real authenticated management routes against isolated profile state,
including incomplete GGUF splits and selecting the completed replacement. Only
machine probes, the server process and remote model validation are substituted.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from tests.waiting import wait_until


@pytest.fixture
def empty_library(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    from hermes_cli import config, web_server
    from hermes_cli.local_runtime import binaries, bootstrap, catalog, endpoint, hardware
    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_cli.web_routers import local_models

    config.save_config({
        "model": {"provider": "llamacpp", "default": "Deleted-Model"},
        "local_runtime": {"enabled": True, "backend": "cpu"},
        "agent": {"max_turns": 37},
    })
    monkeypatch.setattr(catalog, "refresh_catalog_soon", lambda: None)
    monkeypatch.setattr(hardware, "probe_budget", lambda **_: HardwareBudget(96 << 30, 96 << 30, 64 << 30))
    monkeypatch.setattr(binaries, "installed_engine", lambda *_: binaries.Engine("cpu", "b11706", tmp_path / "engine"))
    monkeypatch.setattr(binaries, "pinned_tag", lambda *_: "b11706")
    monkeypatch.setattr(local_models, "_state_endpoint", lambda: None)
    monkeypatch.setattr(endpoint, "resolve_llamacpp_endpoint", lambda **_: None)
    monkeypatch.setattr(local_models, "_JOBS", {})
    monkeypatch.setattr(local_models, "_RUNNING", {})
    no_assistant = Mock(side_effect=AssertionError("Model management must not construct an assistant"))
    monkeypatch.setattr("run_agent.AIAgent", no_assistant)

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return SimpleNamespace(client=client, directory=bootstrap.models_dir(), no_assistant=no_assistant)


def _management_is_available(client):
    status = client.get("/api/local-models/status")
    assert status.status_code == 200
    assert status.json()["server_running"] is False
    assert status.json()["models"] == []
    assert status.json()["active_model_id"] == "Deleted-Model"
    assert client.get("/api/config").status_code == 200
    catalog = client.get("/api/local-models/catalog")
    assert catalog.status_code == 200
    assert isinstance(catalog.json()["models"], list)
    assert client.get("/api/local-models/jobs").status_code == 200


def test_deleted_default_does_not_block_settings_or_incomplete_download(empty_library):
    state = empty_library
    _management_is_available(state.client)
    state.directory.mkdir(parents=True, exist_ok=True)
    (state.directory / "Replacement-00001-of-00002.gguf").write_bytes(b"GGUF" + bytes(32))
    (state.directory / "Replacement-00002-of-00002.gguf.part").write_bytes(bytes(16))
    _management_is_available(state.client)

    rejected = state.client.post("/api/local-models/activate", json={"model_id": "Replacement"})
    assert rejected.status_code == 404
    assert state.client.get("/api/config").json()["model"] == "Deleted-Model"
    assert state.client.get("/api/local-models/jobs").json()["jobs"] == []
    state.no_assistant.assert_not_called()


def test_completed_replacement_can_be_selected_without_old_model_or_assistant(empty_library, monkeypatch):
    from hermes_cli import config, models_validate
    from hermes_cli.local_runtime import bootstrap, endpoint

    state = empty_library
    state.directory.mkdir(parents=True, exist_ok=True)
    for part in (1, 2):
        (state.directory / f"Replacement-{part:05d}-of-00002.gguf").write_bytes(b"GGUF" + bytes(part * 32))
    listed = state.client.get("/api/local-models/status").json()["models"]
    assert [row["id"] for row in listed] == ["Replacement"]
    assert listed[0]["size_bytes"] == 104

    starts = []

    def start_router(cfg, *, force):
        assert force is True
        assert cfg["model"]["default"] == "Deleted-Model"
        starts.append(True)
        monkeypatch.setattr(endpoint, "resolve_llamacpp_endpoint", lambda **_: {
            "base_url": "http://127.0.0.1:1/v1", "api_key": "fixture-only"})
        # Deliberately offers no load/inference method: activation only needs the listing.
        return SimpleNamespace(models=lambda: {"Replacement": {}})

    monkeypatch.setattr(bootstrap, "ensure_local_runtime", start_router)
    monkeypatch.setattr(models_validate, "validate_requested_model", lambda *_, **__: {
        "accepted": True, "persist": True, "recognized": True})
    activated = state.client.post("/api/local-models/activate", json={"model_id": "Replacement"})
    assert activated.status_code == 200
    job_id = activated.json()["job_id"]

    def completed_job():
        row = state.client.get(f"/api/local-models/jobs/{job_id}").json()
        return row if row["status"] in {"done", "error"} else None

    result = wait_until(completed_job, what="replacement activation")
    assert result["status"] == "done", result
    assert starts == [True]
    saved = config.load_config()
    assert saved["model"]["default"] == "Replacement"
    assert saved["model"]["provider"] == "llamacpp"
    assert saved["local_runtime"]["enabled"] is True
    assert saved["agent"]["max_turns"] == 37
    assert state.client.get("/api/local-models/status").json()["active_model_id"] == "Replacement"
    state.no_assistant.assert_not_called()
