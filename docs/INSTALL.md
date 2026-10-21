# Installation

Custom multi-GPU GLM-5.3 requires **llama.cpp b11706**, supplied separately: [engine setup](ENGINE.md).

## Windows installer

1. Install the pinned upstream Hermes v0.22.3 development/build prerequisites, including Git and Node as described in its Desktop build guide. Use Hermes's existing Python environment, not a new unrelated interpreter.
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

The installer verifies the exact upstream Git HEAD and patch context before writing. It saves original source files and a receipt under `%LOCALAPPDATA%\Hermes-MultiGPU-Patch\<installation-key>\v0.22.3`. Backups stay on the installing computer. Treat a build error as an incomplete installation. After fixing prerequisites, rerun the installer to verify the applied source and resume building. Files edited after installation require manual reconciliation. The installer does not repin llama.cpp or change model presets.

## Manual installation

These commands expose every patch/build action. You can use a separate checkout for inspection before changing an existing installation. Close Hermes and its gateway after jobs finish; do not start them during patching/building. Make a normal filesystem backup of your source and Desktop output before changing an existing installation, and retain any local modifications separately.

Place this project beside a clean upstream checkout:

```text
git -c core.autocrlf=false clone https://github.com/NousResearch/hermes-agent.git hermes-agent
git -C hermes-agent checkout --detach 5b147ab036d6d19131bff8d20701b5001eab844a
python Hermes-Agent-nousresearch-Multi-GPU-Patch/tools/check_patch.py hermes-agent
git -C hermes-agent apply --check ../Hermes-Agent-nousresearch-Multi-GPU-Patch/patches/multigpu-v0.22.3.patch
git -C hermes-agent apply ../Hermes-Agent-nousresearch-Multi-GPU-Patch/patches/multigpu-v0.22.3.patch
```

Confirm the checkout is clean and at the exact commit above. If a check fails, stop and reconcile the differences; do not force the patch or overwrite custom files with the `source/` snapshot.

Follow the pinned upstream [development setup](https://github.com/NousResearch/hermes-agent/blob/5b147ab036d6d19131bff8d20701b5001eab844a/CONTRIBUTING.md#development-setup) and [Desktop build guide](https://github.com/NousResearch/hermes-agent/blob/5b147ab036d6d19131bff8d20701b5001eab844a/apps/desktop/BUILDING.md). From `hermes-agent`, using that installation's Python environment, run:

```text
python -m hermes_cli.main desktop --build-only --force-build
```

This builds the packaged application without launching it. Applying source alone does not replace an existing packaged Desktop. After the build succeeds, launch Hermes normally. New completions use the corrected rate; old stored conversations do not acquire missing engine timings retroactively.

To reverse an otherwise unchanged manual installation, close Hermes, run `git apply --reverse --check` followed by `git apply --reverse` against the same patch you applied, then rebuild Desktop with the command above. For installer source recovery, inspect the local receipt and restore only its listed files from `original/` after confirming no later edits; a `null` before-hash denotes a file newly added by the patch. Rebuild afterward. Keep backups until the recovered installation has been verified.
