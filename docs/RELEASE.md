# Initial public release for Hermes v0.22.3

Hardware-adaptive CUDA placement, per-device memory grants, host/commit budgeting and coordinated request admission for local llama.cpp models. The release also includes accurate native generation-rate reporting with an explicitly marked request-throughput fallback.

**Custom multi-GPU GLM-5.3 requires llama.cpp b11706, supplied separately.** See [engine setup](ENGINE.md).

The download includes the complete source patch, changed files, tests, a Windows installer that prompts for the Hermes folder, and [manual installation instructions](INSTALL.md). The installer validates the pinned upstream source, saves source backups locally and rebuilds Desktop using upstream build tooling. Finish jobs and close Hermes before installation.

No engine, model weights or prebuilt Desktop application are bundled. No matched inference benchmark has been performed; no speedup is claimed. See [validation](VALIDATION.md) and [performance](PERFORMANCE.md).
