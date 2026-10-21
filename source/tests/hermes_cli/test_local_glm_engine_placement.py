"""Only verified engine semantics may drive the custom per-card GLM placement."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import bootstrap, glm, presets
from hermes_cli.local_runtime.estimator import HardwareBudget
from hermes_cli.local_runtime.multigpu import CudaDevice
from tests.hermes_cli import test_local_glm

split_model = test_local_glm.split_model

GIB = 1 << 30


@pytest.fixture
def glm_budget(split_model, monkeypatch):
    first, _second, header = split_model
    header.metadata.update({"glm5next.attention.head_count_kv": [0, 0, 0, 1],
                            "glm5next.expert_count": 288})
    header.tensor_bytes = 4 * GIB
    monkeypatch.setattr(presets, "read_gguf_header", lambda path: deepcopy(header))
    monkeypatch.setattr(bootstrap, "serving_engine", lambda: None)
    cards = (CudaDevice("CUDA0", "small accelerator", 32 * GIB, 30 * GIB),
             CudaDevice("CUDA1", "large accelerator", 48 * GIB, 44 * GIB))
    budget = HardwareBudget(70 * GIB, 80 * GIB, 64 * GIB,
                            cuda_devices=cards, cuda_split_bytes=(28 * GIB, 42 * GIB))
    return first, budget


@pytest.mark.parametrize("engine_tag", ["b11671", "b11707", "b99999", "custom", "", None])
def test_unverified_glm_engine_cannot_emit_a_multicard_launch_preset(glm_budget, monkeypatch, engine_tag):
    model, budget = glm_budget

    def unexpected_preparation(*args):
        pytest.fail("Unverified GLM placement must be refused before preparing model files")

    monkeypatch.setattr(glm, "preset_keys", unexpected_preparation)
    result = presets.preset_for_model(model, budget, set(), engine_tag=engine_tag)
    assert result.refusal is not None and "placement" in result.refusal.lower()
    assert "b11706" in result.refusal
    assert result.keys is None and result.window == 0
    assert result.placement["device"] == "CUDA0,CUDA1"
    assert "[" + result.model_id + "]" not in presets.render_presets([result])


def test_verified_glm_engine_keeps_unequal_device_placement(glm_budget):
    model, budget = glm_budget
    result = presets.preset_for_model(model, budget, set(), engine_tag="b11706")
    assert result.refusal is None
    assert result.keys["device"] == "CUDA0,CUDA1"
    assert result.keys["split-mode"] == "layer"
    assert result.keys["tensor-split"] == ",".join(map(str, budget.cuda_split_bytes))
    assert result.keys["model"]


@pytest.mark.parametrize("engine_tag", ["b11671", "b11707", "b99999"])
def test_single_card_glm_does_not_need_custom_multicard_mapping(glm_budget, engine_tag):
    model, budget = glm_budget
    budget = replace(budget, cuda_devices=budget.cuda_devices[:1],
                     cuda_split_bytes=budget.cuda_split_bytes[:1])
    result = presets.preset_for_model(model, budget, set(), engine_tag=engine_tag)
    assert result.refusal is None and result.keys["model"]


def test_current_serving_engine_identity_can_resolve_an_omitted_tag(glm_budget, monkeypatch):
    model, budget = glm_budget
    monkeypatch.setattr(bootstrap, "serving_engine", lambda: SimpleNamespace(tag="b11706"))
    result = presets.preset_for_model(model, budget, set())
    assert result.refusal is None and result.keys["device"] == "CUDA0,CUDA1"


def test_explicit_unverified_engine_does_not_borrow_verified_serving_identity(glm_budget, monkeypatch):
    model, budget = glm_budget
    monkeypatch.setattr(bootstrap, "serving_engine", lambda: SimpleNamespace(tag="b11706"))
    result = presets.preset_for_model(model, budget, set(), engine_tag="b11707")
    assert result.refusal is not None and result.keys is None
