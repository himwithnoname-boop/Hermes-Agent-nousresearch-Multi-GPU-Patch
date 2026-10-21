# Validation

## Regression coverage

| Selection | Result | Scope |
|---|---|---|
| Dynamic placement/admission/resources | 220 passed | Mock hardware and deterministic metadata |
| CUDA inventory validation | 45 passed | Invalid rows and optional probe recovery |
| GLM/DFlash/MTP policy | 125 passed | Engine compatibility and draft behavior |
| GLM placement fixtures | 12 passed | Placement edge cases |
| Broader GPU/runtime selection | 837 passed, 4 upstream skips | Runtime regression coverage |
| Generation-rate/streaming/contracts | 103 passed, 4 upstream skips | Native timings, fallback, weighting and projections |
| Desktop statusbar/usage | 13 passed | Rate display and usage updates |
| Upstream static checks | All 12 passed | Source health and repository checks |

Selections overlap and must not be added into a unique-test total. Tests use synthetic hardware and mocked model operations; they are not native inference benchmarks.

## Distribution and installer checks

The complete patch applies to the exact upstream v0.22.3 tree, matches the published source snapshot and reverses cleanly. The Windows source installer is checked for stock-patch selection, source backups, repeat application, completion receipts and refusal after external edits. The distribution includes a SHA-256 inventory verified before publication.

Tests were run on Windows. Cross-platform execution and a matched inference benchmark remain unverified. Use the upstream managed test environment, `scripts/run_tests.sh`, the relevant Desktop Vitest selection and `python scripts/check` when validating a rebuilt checkout. See [the machine-readable summary](../validation/public-release.json).
