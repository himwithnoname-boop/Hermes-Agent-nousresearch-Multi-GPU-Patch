"""DFlash retains portable CUDA placement and memory safeguards."""
from dataclasses import replace

import pytest

from hermes_cli.local_runtime import presets
from hermes_cli.local_runtime.multigpu import CudaDevice, cuda_budget
from tests.hermes_cli import test_local_external_draft

model = test_local_external_draft.model
GIB = 1 << 30


@pytest.mark.parametrize("engine_tag", ["b11671", "b11706"])
def test_dflash_retains_unequal_cuda_placement_and_host_cache_bounds(model, engine_tag):
    gguf, draft, _ = model
    devices = (
        CudaDevice("CUDA0", "small card", 16 * GIB, 16 * GIB),
        CudaDevice("CUDA1", "large card", 32 * GIB, 32 * GIB),
    )
    capacity = cuda_budget(devices, 32 * GIB, planning=True)
    live = replace(capacity, commit_room_bytes=80 * GIB)

    entry = presets.preset_for_model(gguf, capacity, set(), live=live, engine_tag=engine_tag)

    assert entry.refusal is None and not entry.spilled
    assert entry.keys["spec-type"] == "draft-dflash"
    assert entry.keys["model-draft"] == str(draft)
    assert entry.keys["parallel"] == "1"
    assert entry.keys["device"] == "CUDA0,CUDA1"
    assert entry.keys["split-mode"] == "layer"
    split = [float(part) for part in entry.keys["tensor-split"].split(",")]
    assert split[1] / split[0] == pytest.approx(
        capacity.cuda_split_bytes[1] / capacity.cuda_split_bytes[0])
    assert 0 <= int(entry.keys["cache-ram"]) <= 8192

    # A missing optional draft returns to MTP without losing placement or the
    # engine-specific one-slot mitigation. Unaffected engines keep their policy.
    draft.unlink()
    fallback = presets.preset_for_model(gguf, capacity, set(), live=live, engine_tag=engine_tag)
    assert fallback.refusal is None
    assert fallback.keys["spec-type"] == "draft-mtp"
    assert fallback.keys.get("parallel") == ("1" if engine_tag == "b11706" else None)
    assert fallback.keys["device"] == entry.keys["device"]
    assert fallback.keys["tensor-split"] == entry.keys["tensor-split"]
    assert 0 <= int(fallback.keys["cache-ram"]) <= 8192