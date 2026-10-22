# llama.cpp engine prerequisite and setup

## What is required

**Use exact llama.cpp `b11706` for this patch's custom multi-GPU GLM-5.3 placement** (`glm5next` / `glm5-next` with more than one discrete CUDA device). Its native layer/output mapping was checked at commit `f74e2fa6a31cc33ab4ee4ae516641cfe62df3924`. A newer build is not automatically accepted for this mapping.

The engine is **not included** in this repository, source ZIP or patch. The official [llama.cpp b11706 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11706) provides platform-specific binaries and runtime archives.

| Use | Engine constraint |
|---|---|
| Custom multi-card GLM-5.3 CUDA mapping | Exact `b11706` |
| GLM-5.3 on one device | At least `b11368`; the custom multi-card restriction does not apply |
| Other CUDA models | Model/draft-specific upstream requirements; no blanket `b11706` gate added by this patch |
| MTP on `b11706` | The patch applies its single-slot mitigation when final speculative mode is `draft-mtp` |

## Why a separate setup step is needed

Both supported Hermes baselines (v0.22.3 and v0.22.4) use **`b11671` in `pm/lock.json`**. This source patch deliberately does not silently rewrite dependency pins. Runtime boot uses an installed PM-managed engine and does not download another version. A normal Local Models install follows the current PM lock, so leaving that lock at `b11671` does not obtain `b11706`.

Downloading an archive or putting a different `llama-server` on `PATH` is not sufficient to switch the managed Hermes engine. Do not rename a different build or edit its recorded tag to bypass the compatibility check.

## Set up b11706 in your patched source checkout

These commands are for the separate writable **source checkout** described in the README. Complete the clean-baseline check and apply the patch first, then provision/activate its managed Python environment according to upstream's development guide. Dependency pinning changes `pm/lock.json`, so do it after the clean-checkout patch step.

With Hermes and its model runtime closed for the engine switch, run from that activated patched checkout:

```text
python -m pm.cli lock --bump llamacpp-cuda 11706
python -m pm.cli install llamacpp-cuda
python -m pm.cli doctor
```

The numeric PM version is **`11706`**, without the `b` prefix. The first command resolves/hashes supported artifacts and changes the CUDA package pin; the second installs and verifies the pinned package for the current platform. PM manages the matching runtime dependencies. These steps require network access and can take time. In the doctor's output, confirm that `llamacpp-cuda` is verified at `11706`; review any reported failure before launching models.

This is the upstream [source dependency workflow](https://github.com/NousResearch/hermes-agent/blob/5b147ab036d6d19131bff8d20701b5001eab844a/website/docs/reference/package-management.md), not an instruction to modify a signed/bundled installation's managed payload or to replace a running server. The pin/install syntax is verified against the pinned upstream source. Native model benchmark results are not provided.

Restart the affected source-built Hermes instance after the engine is installed so it resolves the new managed engine. Confirm the engine version reported by Hermes before loading a multi-GPU GLM model. A server already running from another build will not change merely because new files were downloaded.

## Official binary downloads

Use the [official b11706 release assets](https://github.com/ggml-org/llama.cpp/releases/tag/b11706) matching the operating system, architecture and CUDA/driver requirements. Windows CUDA server archives and matching `cudart` archives are separate assets. The managed commands above select the artifacts defined by Hermes's CUDA package; a manual download alone does not register an engine with PM.

The public patch supplies hardware-adaptive planning, not CUDA drivers, model weights or a bundled engine. The operating system's CUDA driver must support the selected build. Future engine versions need compatibility review before the restricted GLM mapping is enabled.
