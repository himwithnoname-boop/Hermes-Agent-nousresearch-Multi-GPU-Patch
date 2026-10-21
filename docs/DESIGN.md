# Design and behavior

## Engine identity first

`hermes_cli/local_runtime/multigpu.py` queries the selected engine's device listing. Logical CUDA IDs and that listing's order determine the usable devices. Visibility and order environment settings participate in cache identity. A missing optional physical-inventory probe does not erase a valid engine-provided CUDA inventory.

Malformed, repeated or impossible rows invalidate the probe instead of silently creating an empty or partial device set. Known changes in device count or total capacity invalidate old grants. An unavailable probe and a valid empty inventory are distinct states.

## Capacity splits and live memory grants

For each device, the planner retains the upstream capacity reserve of `max(2 GiB, 9% of capacity)`. Remaining capacity determines the layer split. A separate live grant respects measured free memory and the retained 1 GiB launch headroom; the most constrained device bounds a launch while preserving the intended split ratios.

The planner does not assume that an outgoing model's aggregate memory will be freed evenly across GPUs. Until unload is observed, only measured free memory is credited. This is conservative memory planning, not a benchmark-driven search over split ratios.

On Windows, `cuda_memory.py` can use unique PCI identities to bound engine allocator figures by physical residency reported by `nvidia-smi`. It keeps the engine's logical order. Missing or ambiguous identity, including shared PCI/MIG ambiguity, does not establish a physical-card grant. The optional helper accepts at most 64 devices. Device inventory has a four-second cache; the optional PCI probe has a sixty-second cache.

## Host memory and idle adaptation

Dedicated VRAM, physical RAM and Windows commit headroom are separate constraints. MoE host spill and prompt caches consume host resources. UMA retains the upstream treatment of its shared pool rather than double-counting RAM as extra dedicated VRAM.

Adaptive policies include a default 8 GiB prompt-cache ceiling, a 512 MiB minimum grant, a RAM reserve based on 5% of total memory with bounds, and a 256 MiB commit-pressure admission threshold. These are configurable policy defaults, not measured universal optima. Increasing grants requires stability; the default growth observation period is thirty seconds and the refit cooldown is sixty seconds.

Refits take place only when no model is loaded/loading/sleeping/ready and the lifecycle and request-admission boundaries are held. An active model is not redistributed during inference. Pressure can pause new admissions while existing work retains its lifecycle and cancellation ownership.

## Request and lifecycle prerequisites

The patch carries the shared runtime prerequisites needed for placement to work reliably: endpoint-wide admission, cancellation ownership, readiness/refusal checks, process identity and coordinated unload/refit behavior. It retains upstream agent context/history behavior. These prerequisites explain why the delta covers more than one GPU-planning module.

## Engine-specific behavior

Custom multi-card GLM tensor mapping is limited to the source-verified llama.cpp build `b11706`; a numerically newer or unknown engine does not automatically qualify. The single-device GLM minimum remains `b11368`. The MTP single-slot mitigation applies to final draft-MTP mode on `b11706`. Upstream DFlash planning is retained.

## Boundaries

Placement does not depend on a fixed GPU model or device count. Tests use synthetic device descriptions. The implementation does not optimize interconnect bandwidth, peer-to-peer capability, NUMA locality, GPU compute-speed differences, CPU affinity or integrated-GPU offload. Those require additional measurements and architecture work.

The public patch is tied to a known Hermes source baseline. Hardware adaptation and forward source compatibility are different properties: future Hermes/engine updates still need review and regression testing.
