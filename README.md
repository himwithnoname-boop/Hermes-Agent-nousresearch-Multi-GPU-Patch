# Hermes-Agent (nousresearch) - Multi GPU Patch

> **Required for custom multi-GPU GLM-5.3 placement: llama.cpp b11706.** The engine is **not bundled**. Stock Hermes v0.22.3 pins b11671; this patch does not change that pin. Follow the [engine setup guide](docs/ENGINE.md) before using custom GLM placement. Other CUDA models are not universally restricted to b11706.

A community enhancement for **Hermes Agent v0.22.3 and v0.22.4** that adds hardware-adaptive CUDA placement, memory budgeting and request coordination for local llama.cpp models. It discovers the selected engine's visible GPUs and adapts to unequal capacities and available memory.

This is the project's **first public release**. It is independent of [Nous Research](https://github.com/NousResearch/hermes-agent) and preserves the upstream features of both supported versions.

## Features

- Discovers engine-visible CUDA device IDs and order without assuming a card count or GPU model.
- Computes capacity-weighted layer splits and bounds memory grants by each device's available memory.
- Separates dedicated VRAM, host RAM and Windows commit limits, including model spill and prompt-cache budgets.
- Validates telemetry and invalidates stale budgets when device topology changes.
- Coordinates request admission, cancellation and model lifecycle operations. An unavailable model releases its failed admission so other installed models can run. Adaptive refits run only while models are unloaded.
- Preserves upstream DFlash, GGUF/recurrent metadata and context behavior.
- Displays generation t/s using accepted completion tokens and native engine generation time. A `~` prefix identifies whole-request throughput when native timings are unavailable.

See [design and limits](docs/DESIGN.md), [validation](docs/VALIDATION.md) and [performance evidence](docs/PERFORMANCE.md).

## Requirements and compatibility

| Item | Requirement |
|---|---|
| Hermes baselines | v0.22.3 (`5b147ab036d6d19131bff8d20701b5001eab844a`) and v0.22.4 (`4d04303818b61bf3aa09278af48356315515965b`) |
| Accelerators | CUDA devices visible to the selected managed llama.cpp engine |
| GPU count and capacity | Discovered at runtime |
| Custom multi-card GLM-5.3 mapping | llama.cpp b11706 |
| Windows physical-memory correction | Optional nvidia-smi with unambiguous PCI identity |
| Installer | Windows PowerShell 5.1+, Git, Hermes Python environment and upstream Desktop build prerequisites |

The planner adapts to hardware and memory pressure. It does not autotune throughput, reshard a loaded model or guarantee the fastest split on every interconnect. Future Hermes and engine versions require compatibility review. Other backends retain their upstream behavior.

## Installation

**v0.22.4 is supported.** The source patch applies unchanged, and the included installer accepts either exact baseline above. After an upstream update, check whether your local changes were restored or left in a stash. If the update produced stock source, reapply the patch and rebuild Desktop; an upstream update does not itself install this enhancement. Do not copy the `source/` snapshot over a newer checkout.

Download and extract the [release ZIP](https://github.com/himwithnoname-boop/Hermes-Agent-nousresearch-Multi-GPU-Patch/releases/latest). Finish jobs and close Hermes Desktop and its gateway, then open `installer/Install.cmd`. The installer prompts for the folder containing `hermes_cli/main.py`, checks compatibility, backs up source files locally and rebuilds Desktop without launching it. Start Hermes normally after success.

The download contains source, tests and installer scripts. The engine, models and prebuilt Desktop application are supplied separately. Building may need network access for upstream dependencies.

If you prefer to inspect and apply the changes yourself, use the complete [manual installation instructions](docs/INSTALL.md#manual-installation). Verify the extracted distribution with:

```text
python tools/verify_release.py
```

## Performance

No matched stock-versus-patched inference benchmark has been performed, so no percentage speedup or memory-saving claim is made. Regression tests establish correctness, not inference throughput. The [comparison protocol](docs/PERFORMANCE.md) describes how to measure results with a fixed model, engine, context and workload.

## Contents

- `patches/multigpu-v0.22.3.patch`: complete patch against the pinned upstream commit.
- `source/`: v0.22.3 reference source snapshot for inspection; use the patch to preserve v0.22.4 changes.
- `installer/`: Windows installer with folder selection, activity checks, source backups and Desktop build.
- `docs/`: design, engine requirements, installation, validation and performance information.
- `tools/`: checksum and baseline verification.
- `checksums.json`: SHA-256 file inventory.

Licensed under [MIT](LICENSE), retaining upstream notices. Report reproducible problems through GitHub issues and omit credentials or personal data from diagnostics.

For a chat paused after a gateway restart, see [interrupted-chat recovery](docs/RECOVERY.md).
