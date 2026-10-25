# Installation

Custom multi-GPU GLM-5.3 requires **llama.cpp b11706**, supplied separately: [engine setup](ENGINE.md).

## Windows installer

1. Install the matching upstream Hermes v0.22.3 or v0.22.4 development/build prerequisites, including Git and Node as described in its Desktop build guide. Use Hermes's existing Python environment, not a new unrelated interpreter.
2. Finish jobs, then close Hermes Desktop and its gateway. Extract the patch outside the Hermes source folder.
3. Optionally inspect `installer/Install.ps1`, `apply_update.py`, the guard module and patches. Verify the distribution with `python tools/verify_release.py`.
4. Double-click `installer/Install.cmd`. Enter the folder containing `hermes_cli/main.py` (normally `%LOCALAPPDATA%\hermes\hermes-agent`). Press Enter to accept the displayed default.
5. Wait for **Update complete**, then launch Hermes normally. The installer does not launch Hermes.

PowerShell alternative, run from this distribution's root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\installer\Install.ps1
```

`-SourceRoot "D:\Apps\hermes-agent"` selects the folder explicitly. `-PythonExecutable "D:\Apps\venv\Scripts\python.exe"` supplies a custom Hermes environment; otherwise the installer discovers the managed environment or `.venv`. `-CheckOnly` verifies without applying source or building Desktop. No elevation is ordinarily required.

The guard refuses detected activity and unreadable ownership; it never terminates a job. Do not start Hermes while installation runs. The upstream Desktop builder has its own handling of locked build outputs, so keeping the application closed throughout is necessary.

The installer verifies the exact upstream Git HEAD and patch context before writing. It saves original source files and a receipt under `%LOCALAPPDATA%\Hermes-MultiGPU-Patch\<installation-key>\<Hermes-version>`. Backups stay on the installing computer. Treat a build error as an incomplete installation. After fixing prerequisites, rerun the installer to verify the applied source and resume building. Files edited after installation require manual reconciliation. The installer does not repin llama.cpp or change model presets.

## Updating Hermes

The patch supports v0.22.3 and v0.22.4. The updater may restore local modifications, keep them in a stash, or leave a clean upstream checkout depending on the update path and choices. Inspect its result. If the source is stock, use this installer to reapply the patch and rebuild Desktop. If the modifications are already present, verify them and rebuild Desktop; do not apply the same patch twice. Preserve and reconcile any stashed edits rather than discarding them. The exact-build binary installers are version-specific; this source installer is the route for the supported source baselines.

## Manual installation

These commands expose every patch/build action. You can use a separate checkout for inspection before changing an existing installation. Close Hermes and its gateway after jobs finish; do not start them during patching/building. Make a normal filesystem backup of your source and Desktop output before changing an existing installation, and retain any local modifications separately.

Place this project beside a clean upstream checkout. The example selects v0.22.4; use `5b147ab036d6d19131bff8d20701b5001eab844a` instead for v0.22.3:

```text
git -c core.autocrlf=false clone https://github.com/NousResearch/hermes-agent.git hermes-agent
git -C hermes-agent checkout --detach 4d04303818b61bf3aa09278af48356315515965b
python Hermes-Agent-nousresearch-Multi-GPU-Patch/tools/check_patch.py hermes-agent
git -C hermes-agent apply --check ../Hermes-Agent-nousresearch-Multi-GPU-Patch/patches/multigpu-v0.22.3.patch
git -C hermes-agent apply ../Hermes-Agent-nousresearch-Multi-GPU-Patch/patches/multigpu-v0.22.3.patch
```

Confirm the checkout is clean and at the exact supported commit selected above. The patch filename identifies its original source baseline and applies unchanged to both supported versions; the `source/` directory is a v0.22.3 reference snapshot, not a replacement tree for v0.22.4. If a check fails, stop and reconcile the differences; do not force the patch or overwrite custom files with the `source/` snapshot.

Follow the pinned upstream [development setup](https://github.com/NousResearch/hermes-agent/blob/5b147ab036d6d19131bff8d20701b5001eab844a/CONTRIBUTING.md#development-setup) and [Desktop build guide](https://github.com/NousResearch/hermes-agent/blob/5b147ab036d6d19131bff8d20701b5001eab844a/apps/desktop/BUILDING.md). The normal installer does not build or replace Desktop. The following command is an **optional, explicit Desktop UI rebuild**, not a required source-install step. It can produce an unsigned executable that Smart App Control blocks. Use a trusted publisher package on systems enforcing publisher trust. Only if you deliberately choose a local UI build, from `hermes-agent` run:

```text
python -m hermes_cli.main desktop --build-only --force-build
```

This builds the packaged application without launching it. Applying source alone does not replace an existing packaged Desktop. After the build succeeds, launch Hermes normally. New completions use the corrected rate; old stored conversations do not acquire missing engine timings retroactively.

To reverse an otherwise unchanged manual installation, close Hermes, run `git apply --reverse --check` followed by `git apply --reverse` against the same patch you applied, then rebuild Desktop with the command above. For installer source recovery, inspect the local receipt and restore only its listed files from `original/` after confirming no later edits; a `null` before-hash denotes a file newly added by the patch. Restart the gateway afterward; any UI rebuild is a separate explicit choice. Keep backups until the recovered installation has been verified.

The installer can also update the request-queue module in an existing installation after verifying the prior receipt and exact known source hash. Its backup is retained, repeat installs are safe, and custom edits are refused. For an interrupted conversation, use [the upstream recovery command](RECOVERY.md).

## Hermes v0.22.5

The installer selects `multigpu-v0.22.5.patch` for exact upstream `bf0203a794a07aac3ff764972ab9087baa4e4f4e`. For manual installation, follow the same clean-checkout procedure above using that commit and patch filename. Keep the v0.22.3 source snapshot as a reference; do not copy it over v0.22.5. The v0.22.5 port retains upstream response-header capture and plan-usage gauges.

For installation, finish work, close Desktop, and run `hermes gateway stop` locally if the gateway remains active. The installer never deletes a held lock or stops a process automatically. A readable held identity now produces an activity refusal instead of a malformed-file error. If an upstream update restored stock source, rerunning the installer creates a separate backup rather than overwriting the prior receipt.

## Windows publisher trust

Source installation never modifies a Desktop executable or packaged bundle. It does not disable or bypass Smart App Control. A source patch cannot give a rebuilt executable a trusted publisher signature. If a previous local build is blocked, preserve it and restore a trusted Desktop distribution through the official publisher; do not disable Windows protection as part of patch installation. Chats, models and user configuration are outside the installer payload.
