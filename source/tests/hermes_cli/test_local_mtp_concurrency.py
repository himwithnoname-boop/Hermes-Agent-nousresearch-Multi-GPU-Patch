"""Serialize admitted draft-MTP on engines with the shared-KV verification-batch bug.

The guard follows the emitted speculative mode, independently of model/catalog names.
It does not enable MTP, change memory policy, or expand to unverified engine releases.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import bootstrap, catalog, hardware, cuda_placement, presets
from tests.hermes_cli.test_local_mtp_head import BUILD, ENTRY, UNIFIED_128


@pytest.mark.parametrize("engine_tag,entry_id,expected", [
    ("b11706", "qwen3.8-flash-next", "1"),
    ("b11706", "renamed-mtp-model", "1"),
    ("b11671", "renamed-mtp-model", None),
    ("b11707", "renamed-mtp-model", None),
    ("unknown", "renamed-mtp-model", None),
    (None, "renamed-mtp-model", None),
])
def test_preset_queues_only_the_affected_mtp_combination(tmp_path, monkeypatch, engine_tag, entry_id, expected):
    entry = replace(ENTRY, id=entry_id)
    monkeypatch.setattr(catalog, "entry_for_model", lambda mid: entry)
    monkeypatch.setattr(presets, "read_gguf_header", lambda path: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda header: entry.profile(BUILD))
    head = bootstrap.assets_dir() / entry.mtp_head.local_name
    head.parent.mkdir(parents=True, exist_ok=True)
    head.touch()
    model = tmp_path / (BUILD.model_id + ".gguf")

    before = presets.preset_for_model(model, UNIFIED_128, set(), engine_tag="b11671")
    after = presets.preset_for_model(model, UNIFIED_128, set(), engine_tag=engine_tag)

    assert after.keys.get("parallel") == expected
    assert after.window == before.window
    assert {key: value for key, value in after.keys.items() if key != "parallel"} == before.keys
    assert after.keys["spec-type"] == "draft-mtp"
    assert after.keys["model-draft"] == str(head)


@pytest.mark.parametrize("mtp,engine_tag,expected", [
    (True, "b11706", "1"),
    (True, "b11707", None),
    (True, None, None),
    (False, "b11706", None),
])
def test_imported_model_uses_emitted_mtp_mode(tmp_path, monkeypatch, mtp, engine_tag, expected):
    model = tmp_path / "User-Renamed-Model.gguf"
    model_id = model.stem
    monkeypatch.setattr(catalog, "entry_for_model", lambda mid: None)
    monkeypatch.setattr(presets, "read_gguf_header", lambda path: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda header: ENTRY.profile(BUILD))
    capable = {model_id} if mtp else set()

    before = presets.preset_for_model(model, UNIFIED_128, capable, engine_tag="b11671")
    after = presets.preset_for_model(model, UNIFIED_128, capable, engine_tag=engine_tag)

    assert after.keys.get("parallel") == expected
    assert (after.keys.get("spec-type") == "draft-mtp") is mtp
    assert after.window == before.window
    assert {key: value for key, value in after.keys.items() if key != "parallel"} == before.keys


@pytest.mark.parametrize("spec_type", [None, "none", "draft-dflash", "draft-dspark", "draft-mtp"])
@pytest.mark.parametrize("engine_tag", ["b11706", "b11707", None])
def test_migration_follows_saved_speculative_mode_without_catalog_lookup(monkeypatch, spec_type, engine_tag):
    model_id = "unlisted-renamed-import"
    keys = {"model": "user-import.gguf", "ctx-size": "32768", "temp": "0.7", "parallel": "4"}
    if spec_type is not None:
        keys["spec-type"] = spec_type
    known = {model_id: presets.PresetEntry(model_id, 32768, False, keys=keys.copy())}
    monkeypatch.setattr(catalog, "entry_for_model", lambda mid: pytest.fail("Migration must inspect saved preset keys"))
    monkeypatch.setattr(hardware, "_configured_engine", lambda: SimpleNamespace(tag=engine_tag))

    assert presets.mtp_presets_need_migration(known, [model_id]) is (spec_type == "draft-mtp" and engine_tag == "b11706")
    assert known[model_id].keys == keys


@pytest.mark.parametrize("parallel,refusal,expected", [(None, None, True), ("4", None, True), ("1", None, False), (None, "model refused", False)])
def test_migration_preserves_current_or_refused_presets(monkeypatch, parallel, refusal, expected):
    keys = {"model": "user-import.gguf", "spec-type": "draft-mtp"}
    if parallel is not None:
        keys["parallel"] = parallel
    known = {"user-import": presets.PresetEntry("user-import", 32768, False, refusal=refusal, keys=keys)}
    monkeypatch.setattr(hardware, "_configured_engine", lambda: SimpleNamespace(tag="b11706"))
    assert presets.mtp_presets_need_migration(known, ["user-import"]) is expected


def test_migration_without_configured_engine_is_not_assumed_affected(monkeypatch):
    known = {"user-import": presets.PresetEntry("user-import", 32768, False, keys={"spec-type": "draft-mtp"})}
    monkeypatch.setattr(hardware, "_configured_engine", lambda: None)
    assert not presets.mtp_presets_need_migration(known, ["user-import"])


@pytest.mark.parametrize("parallel,stale", [(None, True), ("4", True), ("1", False)])
def test_existing_imported_mtp_preset_migrates_when_gpu_placement_is_current(tmp_path, monkeypatch, parallel, stale):
    model_id = "Renamed-User-Model"
    mdir = bootstrap.models_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / (model_id + ".gguf")).touch()
    keys = {"model": str(mdir / (model_id + ".gguf")), "ctx-size": "221184", "spec-type": "draft-mtp"}
    if parallel is not None:
        keys["parallel"] = parallel
    path = bootstrap.runtimes_root() / "presets.ini"
    presets.write_presets([presets.PresetEntry(model_id, 221184, False, keys=keys)], path)
    monkeypatch.setattr(catalog, "entry_for_model", lambda mid: pytest.fail("Unknown imported models still require migration"))
    monkeypatch.setattr(hardware, "_configured_engine", lambda: SimpleNamespace(tag="b11706"))
    monkeypatch.setattr(cuda_placement, "presets_need_migration", lambda known, mids: False)

    assert bootstrap._presets_stale() is stale
