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

## Hermes v0.22.4 compatibility

The unchanged patch was applied to exact upstream commit `4d04303818b61bf3aa09278af48356315515965b`. All 74 upstream-changed files outside the patch were preserved, and the eight locale-file overlaps retain the removal of suggestion keys. The GPU planner, engine lock and build dependency definitions are unchanged between the supported upstream versions.

On the patched v0.22.4 checkout: **630 Python tests passed (4 upstream skips), 18 Desktop UI tests passed, Desktop renderer type checking passed and all 12 upstream static checks passed**. The selection covers placement, memory pressure, lifecycle/admission, GLM/MTP policy, streaming/generation rates and the updated upstream approval/lock contracts. Fourteen installer checks verify both exact baselines, repeated application, version-separated backups and refusal of unknown revisions/external edits. No native inference benchmark was run.

## Request admission and installer recovery

The unavailable-model-to-next-model regression fails on the uncorrected implementation and passes with the release code. All **57 focused request-target, queue and scheduler tests** pass, retaining authentication, active-request and router-restart fences. **34 installer checks** pass across both supported upstream versions, including fresh installs, upgrades of an existing patch, interrupted file/receipt updates, repeat installs, read-only checks and custom-edit refusal. See [request queue](../validation/request-queue.json) and [installer recovery](../validation/installer-queue-recovery.json).

## Hermes v0.22.5 and reinstall behavior

The v0.22.5-specific source port retains upstream response-header capture, plan-usage gauges and all non-overlapping upstream changes. **668 Python tests passed, with 4 upstream skips**, including all 44 streaming cases executed in independent selections after a monolithic run exceeded its time budget. All 12 upstream checks passed. No Desktop application build or native inference benchmark was performed for this installer maintenance.

**39 installer checks** cover all three supported versions, clean installs, repeat installs, source restored by an updater, preservation of earlier backups, adoption of an already-applied patch, and refusal of unknown custom edits. **45 guard checks** passed. A native Windows fixture reproduced the original shared-writer/byte-range-lock failure and confirmed that the corrected shared reader reads the identity while live PID/birth still blocks installation. Exclusive unreadable and malformed state remain refusals. See [v0.22.5 evidence](../validation/v0.22.5.json).

## Executable preservation

The actual Windows PowerShell installer was run for install and repeat-install against an isolated v0.22.5 checkout. Existing EXE, Desktop bundle and renderer bytes **and modification timestamps** remained identical. No Desktop build, dependency installation or Windows security change was invoked. See [source-only installer verification](../validation/source-only-installer.json).
