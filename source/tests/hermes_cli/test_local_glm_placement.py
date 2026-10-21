"""Tensor-table placement admission respects individual cards and physical host residency."""
from dataclasses import replace

import pytest

from hermes_cli.local_runtime.context_policy import WindowDecision, host_spill_bytes, launch_args, spill_overrides
from hermes_cli.local_runtime.estimator import HardwareBudget, LayerKind, ModelProfile
from hermes_cli.local_runtime.multigpu import CudaDevice


def placement_fixture():
    unit = 1 << 20
    tensors = {"token_embd.weight": 2 * unit, "output.weight": unit, "output_norm.weight": unit}
    ffn = {0: 30 * unit, 1: 30 * unit, 2: 30 * unit, 3: 10 * unit}
    tensors.update({f"blk.{block}.ffn_up_exps.weight": size for block, size in ffn.items()})
    profile = ModelProfile(
        "skewed GLM", sum(tensors.values()), 2 * unit, 65536,
        [(LayerKind.RECURRENT, 0)] * 4, architecture="glm5-next", ffn_block_bytes=ffn,
        tensor_bytes_by_name=tensors, tensor_table_complete=True)
    cards = tuple(CudaDevice(f"CUDA{i}", f"card {i}", 48 * unit, 44 * unit) for i in range(2))
    budget = HardwareBudget(80 * unit, 96 * unit, 100 * unit,
                            cuda_devices=cards, cuda_split_bytes=(40 * unit, 40 * unit))
    return profile, budget


def test_skewed_weights_spill_on_the_card_that_needs_room():
    profile, budget = placement_fixture()
    # Native 5-slot partition puts blocks 0..2 on CUDA0; block 3 and output on CUDA1.
    # A proportional 15/15 cut would fail on card1 and leave card0 over budget.
    assert spill_overrides(profile, 30 << 20, budget=budget) == [
        "-ot", r"blk\.(0|1)\.ffn_.*\.weight=CPU"]
    assert host_spill_bytes(profile, 30 << 20, budget=budget) == 62 << 20


def test_input_embedding_and_rounded_ffns_share_physical_ram_admission():
    profile, budget = placement_fixture()
    with pytest.raises(ValueError, match="available host RAM"):
        spill_overrides(profile, 30 << 20, budget=replace(budget, ram_available_bytes=61 << 20))


def test_incomplete_tensor_evidence_retains_existing_conservative_policy():
    profile, budget = placement_fixture()
    profile = replace(profile, tensor_table_complete=False)
    with pytest.raises(ValueError, match="cannot cover"):
        spill_overrides(profile, 30 << 20, budget=budget)


def test_large_output_cannot_be_misrepresented_as_spillable_ffns():
    profile, budget = placement_fixture()
    tensors = dict(profile.tensor_bytes_by_name, **{"output.weight": 90 << 20})
    profile = replace(profile, weights_bytes=sum(tensors.values()), tensor_bytes_by_name=tensors)
    with pytest.raises(ValueError, match="CUDA1.*cannot cover"):
        spill_overrides(profile, 110 << 20, budget=budget)


def test_aggregate_fit_still_relieves_an_overloaded_card():
    profile, budget = placement_fixture()
    budget = replace(budget, usable_vram_bytes=120 << 20, cuda_split_bytes=(60 << 20, 60 << 20))
    decision = WindowDecision(65536, 0, True)
    args = launch_args(profile, decision, budget=budget, allocation_bytes=profile.weights_bytes)
    assert args[args.index("-ot") + 1] == r"blk\.(0|1)\.ffn_.*\.weight=CPU"
    assert host_spill_bytes(profile, 0, budget=budget,
                            allocation_bytes=profile.weights_bytes) == 62 << 20


def test_input_only_host_cost_is_checked_without_any_cpu_override():
    profile, budget = placement_fixture()
    budget = replace(budget, usable_vram_bytes=200 << 20, cuda_split_bytes=(100 << 20, 100 << 20))
    args = launch_args(profile, WindowDecision(65536, 0, True), budget=budget,
                        allocation_bytes=profile.weights_bytes)
    assert "-ot" not in args
    assert host_spill_bytes(profile, 0, budget=budget,
                            allocation_bytes=profile.weights_bytes) == 2 << 20
    with pytest.raises(ValueError, match="available host RAM"):
        launch_args(profile, WindowDecision(65536, 0, True),
                    budget=replace(budget, ram_available_bytes=1 << 20),
                    allocation_bytes=profile.weights_bytes)


def test_no_aggregate_fit_uses_entire_capacity_as_the_actual_cost():
    profile, budget = placement_fixture()
    budget = replace(budget, usable_vram_bytes=200 << 20, cuda_split_bytes=(100 << 20, 100 << 20))
    # Without the exact104MiB cost, treating spill0 as200MiB need would invent CPU spill.
    assert "-ot" not in launch_args(profile, WindowDecision(65536, 0, True), budget=budget,
                                     allocation_bytes=104 << 20)


def test_unvalidated_mtp_cannot_reuse_trunk_only_memory_accounting():
    profile, budget = placement_fixture()
    with pytest.raises(ValueError, match="MTP memory planning is not validated"):
        launch_args(profile, WindowDecision(65536, 0, True), budget=budget, mtp_capable=True,
                    allocation_bytes=profile.weights_bytes)
