"""The common runtime imports and plans without the optional CUDA placement package."""
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture
def package_copy(tmp_path):
    root = Path(__file__).resolve().parents[2]
    isolated = tmp_path / "package"
    shutil.copytree(root / "hermes_cli", isolated / "hermes_cli",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "multigpu.py", "cuda_memory.py"))
    return root, isolated


def _run(package_copy, code):
    root, isolated = package_copy
    # Only non-hermes_cli modules fall back to the checkout. Package submodules
    # must resolve wholly inside the copied package, including genuine absence.
    preamble = f"import sys; sys.path.insert(1, {str(root)!r})\n"
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", preamble + textwrap.dedent(code)],
                            cwd=isolated, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


_PLANNING = """
from pathlib import Path
from types import SimpleNamespace
from hermes_cli.local_runtime import cuda_placement, hardware, presets
from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile
GIB = 1 << 30
hardware._ram_bytes = lambda: (128 * GIB, 100 * GIB)
hardware._nvidia_vram = lambda: (16 * GIB, 15 * GIB, "physical single", None)
hardware._unified_pool_bytes = lambda *_: None
hardware._configured_engine = lambda: SimpleNamespace(backend="cuda", binary=Path("unused.exe"))
presets.read_gguf_header = lambda _: SimpleNamespace(sampling_defaults={})
presets.profile_from_gguf = lambda _: ModelProfile(
    "small-model", 2 * GIB, 0, 8192, [(LayerKind.FULL, 4096)] * 8)
"""


def test_absent_provider_keeps_single_device_planning(package_copy):
    _run(package_copy, _PLANNING + """
assert cuda_placement.cuda_devices() is None
budget = hardware.probe_budget(planning=True)
assert budget.total_device_bytes == 16 * GIB
assert budget.cuda_devices == () and budget.cuda_split_bytes == ()
entry = presets.preset_for_model(Path("small-model.gguf"), budget, set())
assert entry.refusal is None
assert not {"device", "split-mode", "tensor-split"}.intersection(entry.keys)
assert "hermes_cli.local_runtime.multigpu" not in sys.modules
""")


def test_present_provider_keeps_engine_order_and_placement(package_copy):
    root, isolated = package_copy
    for name in ("multigpu.py", "cuda_memory.py"):
        shutil.copy2(root / "hermes_cli/local_runtime" / name,
                     isolated / "hermes_cli/local_runtime" / name)
    _run(package_copy, _PLANNING + """
from hermes_cli.local_runtime import multigpu, cuda_memory
multigpu._list_devices = lambda _: (
    "CUDA0: logical first (32768 MiB, 30720 MiB free)\\n"
    "CUDA1: logical second (16384 MiB, 14336 MiB free)\\n")
cuda_memory.bound_memory = lambda binary, devices: devices
budget = hardware.probe_budget(planning=True)
assert budget.total_device_bytes == 48 * GIB
assert [device.description for device in budget.cuda_devices] == ["logical first", "logical second"]
entry = presets.preset_for_model(Path("small-model.gguf"), budget, set())
assert entry.refusal is None
assert entry.keys["device"] == "CUDA0,CUDA1"
assert entry.keys["split-mode"] == "layer"
assert tuple(map(int, entry.keys["tensor-split"].split(","))) == budget.cuda_split_bytes
live = cuda_placement.cuda_launch_budget(budget)
assert live.usable_vram_bytes < budget.usable_vram_bytes
assert cuda_placement.presets_need_migration({"m": entry}, ["m"]) is False
""")


def test_broken_provider_dependency_is_not_silently_disabled(package_copy):
    _, isolated = package_copy
    # This fixture is a present provider with a broken dependency, not absence.
    (isolated / "hermes_cli/local_runtime/multigpu.py").write_text(
        "import missing_cuda_provider_test_dependency\n", encoding="utf-8")
    _run(package_copy, """
from hermes_cli.local_runtime import cuda_placement
try:
    cuda_placement.cuda_devices()
except ModuleNotFoundError as exc:
    assert exc.name == "missing_cuda_provider_test_dependency"
else:
    raise AssertionError("A broken installed provider was hidden")
""")


def test_absent_provider_migrates_only_known_generated_placement(package_copy):
    _run(package_copy, _PLANNING + """
from copy import deepcopy
from hermes_cli.local_runtime import binaries, bootstrap
old = {"model": "small-model.gguf", "ctx-size": "8192", "device": "CUDA0,CUDA1",
       "split-mode": "layer", "tensor-split": "20000000000,10000000000"}
entry = presets.PresetEntry("small-model", 8192, False, keys=old)
known = {"small-model": entry}
before = deepcopy(known)
assert cuda_placement.presets_need_migration(known, ["small-model"])
assert not cuda_placement.presets_need_migration(known, ["unrelated"])
assert known == before
for override in ({"device": "CUDA0"}, {"device": "Vulkan0", "split-mode": "layer", "tensor-split": "1"},
                 {"device": "CUDA0,CUDA1", "split-mode": "row", "tensor-split": "2,1"},
                 {"device": "CUDA0,CUDA1", "split-mode": "layer", "tensor-split": "0.6,0.4"}):
    manual = presets.PresetEntry("manual", 8192, False, keys=override)
    assert not cuda_placement.presets_need_migration({"manual": manual}, ["manual"])
refused = presets.PresetEntry("refused", 8192, False, refusal="insufficient room", placement=old)
assert cuda_placement.presets_need_migration({"refused": refused}, ["refused"])
mdir = Path("models")
mdir.mkdir()
model = mdir / "small-model.gguf"
model.touch()
runtime = Path("runtime")
runtime.mkdir()
binaries.runtimes_root = lambda: runtime
bootstrap.models_dir = lambda: mdir
bootstrap.staged_models = lambda: [model]
presets.write_presets([entry], runtime / "presets.ini")
assert bootstrap._presets_stale()
presets.generate_presets(mdir, hardware.probe_budget(planning=True), runtime / "presets.ini")
assert not bootstrap._presets_stale()
replanned = presets.read_preset_decisions(runtime / "presets.ini")["small-model"]
assert replanned.refusal is None and replanned.keys["model"] == str(model)
assert not {"device", "split-mode", "tensor-split"}.intersection(replanned.keys)
""")
