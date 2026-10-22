# Performance evidence and comparison protocol

## Current evidence

| Measure | Stock supported Hermes | Patched supported Hermes | Defensible conclusion |
|---|---|---|---|
| Decode tokens/second | Not measured in a matched trial | Not measured in a matched trial | No speedup claim |
| Prefill tokens/second | Not measured in a matched trial | Not measured in a matched trial | No speedup claim |
| Peak per-GPU VRAM | Not measured in a matched trial | Not measured in a matched trial | No memory-saving claim |
| Peak RAM/Windows commit | Not measured in a matched trial | Not measured in a matched trial | No memory-saving claim |
| Device/pressure/cancellation behavior | Not evaluated as a matched baseline | Covered by focused regression fixtures | Correctness evidence, not a benchmark |

Installer elapsed times measure file validation and installation, not inference.

## Reproducible future comparison

1. Use clean stock and patched checkouts from the same pinned upstream commit. Keep engine binary/hash, model and quantization, context, KV type, batch settings, speculative mode, concurrency and prompts fixed.
2. Record hardware capacities and driver/engine versions. Exclude personal identifiers, device serial numbers and sensitive prompts from published results.
3. Confirm identical device visibility and actual model residency. Report failures, offload differences and memory-fit behavior separately from like-for-like throughput.
4. Separate warmup, cold loads and warm runs. Alternate stock and patched runs (for example ABBA), repeat several times, and retain per-run observations.
5. Measure prefill, decode, time to first token, end-to-end latency, peak per-device VRAM, RAM and commit. Report medians, ranges/variation and sample counts, with the exact workload shape.
6. Repeat for a model that fits comfortably, an unequal-device configuration and a controlled near-capacity workload. Run only when the machine is idle and its owner has authorized the workload.

Calculate percentage change only between comparable successful runs. A failure-to-success improvement should be reported as a fit/reliability result, not converted into an artificial speedup percentage.
