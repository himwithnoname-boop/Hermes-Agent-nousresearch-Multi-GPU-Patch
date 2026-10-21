"""Production MTP policy survives boot, persisted-policy migration and idle refit."""
from contextlib import nullcontext
from dataclasses import replace
from threading import RLock
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import binaries, bootstrap, catalog, endpoint, hardware, cuda_placement, presets, supervisor
from tests.hermes_cli.test_local_mtp_head import BUILD, ENTRY, UNIFIED_128


@pytest.fixture
def staged_mtp(tmp_path, monkeypatch):
    mdir = tmp_path / "models"
    mdir.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    model = mdir / (BUILD.model_id + ".gguf")
    model.touch()
    monkeypatch.setattr(bootstrap, "models_dir", lambda: mdir)
    monkeypatch.setattr(bootstrap, "runtimes_root", lambda: runtime)
    monkeypatch.setattr(binaries, "runtimes_root", lambda: runtime)
    monkeypatch.setattr(bootstrap, "staged_models", lambda: [model])
    monkeypatch.setattr(bootstrap, "staged_in", lambda path: [model])
    monkeypatch.setattr(hardware, "probe_budget", lambda **kwargs: UNIFIED_128)
    monkeypatch.setattr(bootstrap, "_launch_budget", lambda capacity: UNIFIED_128)
    monkeypatch.setattr(presets, "read_gguf_header", lambda path: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda header: ENTRY.profile(BUILD))
    monkeypatch.setattr(catalog, "entry_for_model", lambda model_id: replace(ENTRY, id="renamed-admitted-model"))
    head = bootstrap.assets_dir() / ENTRY.mtp_head.local_name
    head.parent.mkdir(parents=True, exist_ok=True)
    head.touch()
    monkeypatch.setattr(cuda_placement, "presets_need_migration", lambda known, ids: False)
    return mdir, runtime / "presets.ini", model


def _read(path, model):
    result = presets.read_preset_decisions(path)[model.stem]
    assert result.refusal is None
    return result.keys


@pytest.mark.parametrize("engine_tag,expected,incumbent", [("b11706", "1", False),
                         ("b11707", None, False), (None, None, False), ("b11706", "1", True)])
def test_runtime_boot_renders_guard_for_the_selected_engine(staged_mtp, monkeypatch, engine_tag, expected, incumbent):
    mdir, path, model = staged_mtp
    engine = SimpleNamespace(tag=engine_tag, backend="cuda", binary=mdir / "llama-server.exe")
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_SERVING_ENGINE", None)
    monkeypatch.setattr(bootstrap, "adopt_legacy_models", lambda: None)
    monkeypatch.setattr(bootstrap, "_cross_process_boot_lock", nullcontext)
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: {} if incumbent else None)
    monkeypatch.setattr(hardware, "_configured_engine", lambda: engine)
    stops = []
    monkeypatch.setattr(bootstrap, "_stop_state_server", lambda: stops.append(True))
    if incumbent:
        old = {"model": str(model), "ctx-size": "32768", "spec-type": "draft-mtp", "parallel": "4"}
        presets.write_presets([presets.PresetEntry(model.stem, 32768, False, keys=old)], path)
    monkeypatch.setattr(binaries, "installed_engine", lambda backend: engine)
    monkeypatch.setattr(bootstrap, "_admitted_models_max", lambda *args: 1)
    monkeypatch.setattr(bootstrap, "_stop_at_exit", lambda: None)
    monkeypatch.setattr(bootstrap, "_start_idle_sweeper", lambda router: None)
    routers = []

    class Router:
        def __init__(self, binary, models_dir, **kwargs):
            self.base_url = "http://fixture.invalid"
            self.preset_path = kwargs["preset_path"]
            routers.append(self)

        def start(self):
            assert self.preset_path == path and path.exists()

    monkeypatch.setattr(supervisor, "LlamaServerSupervisor", Router)
    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True, "backend": "cuda"}}) is routers[0]
    assert bootstrap.serving_engine() is engine
    keys = _read(path, model)
    assert keys.get("parallel") == expected
    assert keys["spec-type"] == "draft-mtp"
    assert keys["backend-sampling"] == "on"
    assert stops == ([True] if incumbent else [])
    if incumbent:
        monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
        assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}}) is None
        assert stops == [True] and len(routers) == 1


def test_saved_mtp_migration_converges_after_production_generation(staged_mtp, monkeypatch):
    mdir, path, model = staged_mtp
    monkeypatch.setattr(hardware, "_configured_engine", lambda: SimpleNamespace(tag="b11706"))
    old = {"model": str(model), "ctx-size": "32768", "spec-type": "draft-mtp", "parallel": "4"}
    presets.write_presets([presets.PresetEntry(model.stem, 32768, False, keys=old)], path)
    assert bootstrap._presets_stale()
    assert bootstrap._generate_presets(mdir, path, engine_tag="b11706") == path
    assert _read(path, model)["parallel"] == "1"
    assert not bootstrap._presets_stale()
    before = path.read_bytes()
    bootstrap._generate_presets(mdir, path, engine_tag="b11706")
    assert path.read_bytes() == before


@pytest.mark.parametrize("engine_tag,expected", [("b11706", "1"), ("b11707", None), (None, None)])
def test_idle_refit_keeps_serving_engine_policy(staged_mtp, monkeypatch, engine_tag, expected):
    _, path, model = staged_mtp
    # Deliberately old policy forces a rendered rewrite, rather than testing a no-op.
    old = {"model": str(model), "ctx-size": "32768", "spec-type": "draft-mtp", "parallel": "4"}
    presets.write_presets([presets.PresetEntry(model.stem, 32768, False, keys=old)], path)
    engine = SimpleNamespace(tag=engine_tag) if engine_tag is not None else None
    monkeypatch.setattr(bootstrap, "_SERVING_ENGINE", engine)
    # A different profile's configured backend must not replace the serving engine identity.
    monkeypatch.setattr(binaries, "installed_engine", lambda backend: pytest.fail("refit consulted configuration"))
    reloads = []
    router = SimpleNamespace(preset_path=path, models_max=1, proc=SimpleNamespace(poll=lambda: None),
                             _lifecycle_lock=RLock(), models=lambda **kwargs: {model.stem: "unloaded"},
                             reload_presets=lambda: reloads.append(True))
    assert bootstrap.refit_idle_presets(router)
    assert reloads == [True]
    assert _read(path, model).get("parallel") == expected
    assert not bootstrap.refit_idle_presets(router)
    assert reloads == [True]
