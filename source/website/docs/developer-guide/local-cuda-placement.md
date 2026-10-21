---
title: "Optional CUDA Placement Provider"
description: "Source boundary for independently installable local runtime upgrades"
---

# Optional CUDA placement provider

The shared local runtime uses `hermes_cli/local_runtime/cuda_placement.py` for
CUDA inventory, capacity, launch headroom, preset placement and placement
migration. The MultiGPU package adds the implementation in `multigpu.py` and
its allocator measurement helper, `cuda_memory.py`. The shared runtime source
is identical with either package installed, so installation order does not
select a different copy of the scheduler or planner.

The adapter discovers that exact optional module with `find_spec` before
importing it. Missing provider files mean the existing upstream single-device
planning path remains active. Exceptions while importing an installed provider
propagate; a broken dependency must not silently disable placement safeguards.
The estimator's `CudaDevice` import is only a type-checking annotation and must
not make the provider a runtime requirement.

With the provider installed, the adapter delegates without changing device
order, budget calculations, per-card allocation bounds, tensor split, or engine
checks. The provider's logical inventory continues to determine placement;
physical GPU count is not request concurrency. Without the provider, placement
keys are empty and all shared lifecycle, host-memory, queue, model compatibility,
MTP, DFlash and reasoning behavior remains available.

Placement migration is read-only until the next normal managed runtime boot.
When the provider is absent, the adapter requests ordinary preset regeneration
only for requested known model IDs whose saved decision has the complete old
addon signature: CUDA device names, `split-mode=layer`, and a corresponding list
of integer byte weights in `tensor-split`. It includes saved refusal placement,
but leaves unrelated model IDs and generic/manual device override shapes alone.
Regeneration uses the upstream single-device path without the old addon's
placement flags. The adapter never stops, reloads, or changes a running router.

When porting to a newer Hermes version, keep the small import boundary in
`hardware.py`, `presets.py`, and `bootstrap.py`; reconcile the implementation
against the new upstream bodies instead of replacing those files wholesale.
Preserve `multigpu.py` and `cuda_memory.py` as the optional feature pair. New GPU
planning calls belong behind this adapter rather than importing the optional
provider from common runtime code.

`tests/hermes_cli/test_local_cuda_placement_optional.py` executes copied packages
in fresh native Python subprocesses with the provider present, omitted, and
broken. It verifies single-device fallback, logical multi-device placement,
error propagation, and managed preset migration. Compatibility validation also
runs queue/reasoning, MTP and upstream DFlash tests against a complete source
snapshot with both provider files omitted. Hardware observations are fixtures;
these tests do not start a model or probe a live GPU.
