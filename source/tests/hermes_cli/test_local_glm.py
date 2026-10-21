"""GLM reasoning must reflect the publisher's conditional fallback vocabulary."""
from hermes_cli.local_runtime.reasoning import capabilities_from_template

GLM_TEMPLATE = """[gMASK]<sop>
{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
{%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}
"""

def test_conditional_reasoning_fallback_is_exposed_without_off():
    caps = capabilities_from_template(GLM_TEMPLATE)
    assert caps is not None
    assert caps.efforts == ('low', 'high', 'max')
    assert caps.default_effort == 'max'
    assert not caps.supports_disable


def test_conditional_reasoning_with_publisher_loop_control_syntax():
    caps = capabilities_from_template(GLM_TEMPLATE + "{% for m in messages %}{% if m.role == 'user' %}{% break %}{% endif %}{% endfor %}")
    assert caps is not None and caps.efforts == ('low', 'high', 'max')

from copy import deepcopy
import os
from pathlib import Path
import struct

import pytest
from jinja2 import Environment

from hermes_cli.local_runtime import binaries, glm, gguf, presets, processes
from hermes_cli.local_runtime.estimator import HardwareBudget, ctx_bytes, profile_from_gguf
from hermes_cli.local_runtime.multigpu import CudaDevice


def write_header(path, architecture='glm5next', split=0):
    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    metadata = {'general.architecture': architecture, 'tokenizer.chat_template': GLM_TEMPLATE,
                architecture + '.block_count': 4, architecture + '.context_length': 1048576,
                architecture + '.embedding_length': 4096, architecture + '.attention.head_count': 64,
                architecture + '.attention.key_length': 512, architecture + '.attention.value_length': 512,
                architecture + '.ssm.conv_kernel': 4, architecture + '.kda.head_dim': 128,
                'split.no': split, 'split.count': 2}
    values = []
    for key, value in metadata.items():
        values.append(string(key) + (struct.pack('<I', 8) + string(value) if isinstance(value, str)
                                     else struct.pack('<II', 4, value)))
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, len(values)) + b''.join(values))
    return gguf._read_part(path)


@pytest.fixture
def split_model(tmp_path, monkeypatch):
    first = tmp_path / 'GLM-5.3-Flash-Q4-00001-of-00002.gguf'
    second = tmp_path / 'GLM-5.3-Flash-Q4-00002-of-00002.gguf'
    header = write_header(first)
    write_header(second, split=1)
    monkeypatch.setattr(binaries, 'runtimes_root', lambda: tmp_path / 'runtime')
    return first, second, header


def test_legacy_sidecar_preserves_all_original_shards_and_tensor_links(split_model):
    first, second, header = split_model
    before = first.read_bytes(), second.read_bytes()
    keys = glm.preset_keys(first, header)
    model = Path(keys['model'])
    rewritten = gguf._read_part(model)
    assert rewritten.architecture == 'glm5-next'
    assert rewritten.n_layer == header.n_layer
    assert rewritten.metadata['tokenizer.chat_template'] == GLM_TEMPLATE
    assert not any(key.startswith('glm5next.') for key in rewritten.metadata)
    assert os.path.samefile(model.with_name(second.name), second)
    assert (first.read_bytes(), second.read_bytes()) == before
    assert glm.preset_keys(first, header) == keys
    assert Path(keys['chat-template-file']).read_text() == GLM_TEMPLATE


def test_incomplete_or_tensor_bearing_legacy_shard_is_refused(split_model):
    first, second, header = split_model
    second.unlink()
    with pytest.raises(ValueError, match='incomplete'):
        glm.preset_keys(first, header)
    write_header(second, split=1)
    raw = bytearray(first.read_bytes())
    raw[8:16] = struct.pack('<Q', 1)
    first.write_bytes(raw[:24])
    with pytest.raises(ValueError):
        glm.preset_keys(first, header)


def test_legacy_cache_never_accepts_an_unrelated_tensor_link(split_model):
    first, second, header = split_model
    keys = glm.preset_keys(first, header)
    linked = Path(keys['model']).with_name(second.name)
    linked.unlink()
    linked.write_bytes(b'unrelated file')
    with pytest.raises(ValueError, match='differs'):
        glm.preset_keys(first, header)


@pytest.mark.parametrize('raw', [b'', b'GGUF', struct.pack('<4sIQQ', b'GGUF', 3, 1, 0),
                                struct.pack('<4sIQQ', b'GGUF', 3, 0, 100)])
def test_metadata_repair_rejects_non_metadata_inputs(raw):
    with pytest.raises(ValueError):
        glm.normalized_metadata(raw)


def test_engine_guard_is_architecture_specific_and_does_not_guess_from_name(tmp_path):
    header = write_header(tmp_path / 'model.gguf', 'glm5-next')
    assert glm.engine_refusal(header, 'b11367')
    assert glm.engine_refusal(header, None)
    assert glm.engine_refusal(header, 'b11706') is None
    header.metadata['general.architecture'] = 'qwen35'
    assert glm.engine_refusal(header, None) is None
    assert glm.preset_keys(tmp_path / 'GLM-5.3-Flash.gguf', header) == {}


@pytest.mark.parametrize('effort,expected', [(None, 'Max'), ('low', 'Low'), ('high', 'High'),
                                             ('max', 'Max'), ('medium', 'Max')])
def test_publisher_reasoning_rendering_keeps_default_and_supported_levels(effort, expected):
    values = {} if effort is None else {'reasoning_effort': effort}
    result = Environment().from_string(GLM_TEMPLATE).render(**values)
    assert 'Reasoning Effort: ' + expected in result


def test_template_indexing_fix_is_equivalent_with_real_tools_and_handles_probe_null():
    original = '''{% for tool in tools %}{{ tool.name }}{% endfor %}
{% if m.content and m.content.0.type == 'text' %}{{ m.content.0.text }}{% endif %}'''
    fixed = glm.compatible_template(original)
    values = {'m': {'content': [{'type': 'text', 'text': 'hello'}]}, 'tools': [{'name': 'echo'}]}
    assert Environment().from_string(original).render(**values) == Environment().from_string(fixed).render(**values)
    assert 'echo' not in Environment().from_string(fixed).render(**{**values, 'tools': [None]})


def test_glm_preset_keeps_multi_gpu_placement_and_f16_memory_pricing(split_model, monkeypatch):
    first, _second, header = split_model
    header.metadata.update({'glm5next.attention.head_count_kv': [0, 0, 0, 1],
                            'glm5next.expert_count': 288})
    header.tensor_bytes = 4 << 30
    monkeypatch.setattr(presets, 'read_gguf_header', lambda path: deepcopy(header))
    card0 = CudaDevice('CUDA0', 'GPU A', 48 << 30, 44 << 30)
    card1 = CudaDevice('CUDA1', 'GPU B', 48 << 30, 42 << 30)
    budget = HardwareBudget(96 << 30, 86 << 30, 64 << 30,
                            cuda_devices=(card0, card1), cuda_split_bytes=(42 << 30, 40 << 30))
    result = presets.preset_for_model(first, budget, set(), engine_tag='b11706')
    assert result.refusal is None
    assert result.keys['device'] == 'CUDA0,CUDA1'
    assert result.keys['split-mode'] == 'layer'
    assert result.keys['tensor-split']
    assert result.keys['temp'] == '1' and result.keys['top-p'] == '0.95'
    assert result.keys['flash-attn'] == 'off'
    assert result.keys['cache-type-k'] == result.keys['cache-type-v'] == 'f16'
    profile = profile_from_gguf(header)
    assert profile.recurrent_state_bytes > 3 * (4 << 20)
    assert ctx_bytes(profile, 65536) == ctx_bytes(profile, 65536, flash_attention=False)
    assert 'spec-type' not in result.keys
    header.metadata['general.sampling.temp'] = 0.8
    result = presets.preset_for_model(first, budget, set(), engine_tag='b11706')
    assert result.keys['temp'] == '0.8'


def test_precision_environment_preserves_cuda_masks_and_filters_credentials():
    original = {'CUDA_VISIBLE_DEVICES': '1,0', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'NVIDIA_TF32_OVERRIDE': '1', 'EXAMPLE_API_KEY': 'private', 'PATH': 'tools'}
    result = processes.server_child_env(original)
    assert result['NVIDIA_TF32_OVERRIDE'] == '0'
    assert result['CUDA_VISIBLE_DEVICES'] == '1,0'
    assert result['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert 'EXAMPLE_API_KEY' not in result
    assert original['NVIDIA_TF32_OVERRIDE'] == '1'


def test_model_deletion_releases_every_compatibility_hardlink(split_model):
    first, second, header = split_model
    compatible = Path(glm.preset_keys(first, header)['model'])
    assert os.path.samefile(compatible.with_name(second.name), second)
    glm.remove_model_files([first, second])
    assert not first.exists() and not second.exists()
    assert not compatible.parent.exists()


def test_non_glm_deletion_is_unchanged_and_unrelated_cache_files_block_cleanup(split_model, tmp_path):
    other = tmp_path / 'another-model.gguf'
    other.write_bytes(b'other model')
    glm.remove_model_files([other])
    assert not other.exists()
    first, second, header = split_model
    compatible = Path(glm.preset_keys(first, header)['model'])
    extra = compatible.parent / 'user-notes.txt'
    extra.write_text('keep')
    with pytest.raises(ValueError, match='Unexpected'):
        glm.remove_model_files([first, second])
    assert first.exists() and second.exists() and extra.exists()


def test_glm_spill_moves_only_enough_ffn_blocks_and_keeps_other_moe_policy():
    from dataclasses import replace
    from hermes_cli.local_runtime.context_policy import spill_overrides
    from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile

    profile = ModelProfile('GLM', 120 << 30, 0, 65536,
                           [(LayerKind.RECURRENT, 0)] * 8 + [(LayerKind.FULL, 2048)],
                           moe=True, architecture='glm5-next',
                           ffn_block_bytes={i: 3 << 30 for i in range(9)})
    assert spill_overrides(profile, 7 << 30) == ['-ot', r'blk\.(0|1|2)\.ffn_.*\.weight=CPU']
    assert spill_overrides(replace(profile, architecture='qwen35moe'), 7 << 30) == [
        '-ot', r'blk\.\d+\.ffn_.*_exps\.weight=CPU']


def test_glm_non_flash_workspace_is_priced_at_the_emitted_microbatch():
    from dataclasses import replace
    from hermes_cli.local_runtime.context_policy import launch_args, posture_profile, WindowDecision
    from hermes_cli.local_runtime.estimator import ModelProfile

    profile = ModelProfile('GLM', 120 << 30, 0, 65536, [], architecture='glm5-next')
    priced = posture_profile(profile, mtp_capable=False, mtp_prefill=False)
    args = launch_args(profile, WindowDecision(65536, 0, True))
    assert args[args.index('-ub') + 1] == '64'
    assert priced.window_compute_per_token == profile.window_compute_bytes * 64
    assert ctx_bytes(priced, 65536) == 4 << 30
    other = replace(profile, architecture='qwen35moe')
    other_args = launch_args(other, WindowDecision(65536, 0, True))
    assert other_args[other_args.index('-ub') + 1] == '2048'


@pytest.mark.parametrize(('split', 'expected'), [
    ((1, 1), [0] * 24 + [1] * 23),
    ((1, 2), [0] * 16 + [1] * 31),
    ((1, 2, 1), [0] * 12 + [1] * 24 + [2] * 11),
    ((1, 0, 1), [0] * 24 + [2] * 23),
])
def test_glm_layer_ranges_include_nextn_and_output_slot(split, expected):
    from hermes_cli.local_runtime.context_policy import _cuda_layer_devices

    # b11706 counts all 46 metadata blocks (including NextN) plus output, even without MTP.
    assert _cuda_layer_devices(46, split) == expected
    assert _cuda_layer_devices(2, (1, 1, 1)) == [0, 1, 2]


def _glm_spill_fixture(split=(1, 1), *, ram=64 << 30):
    from hermes_cli.local_runtime.estimator import LayerKind, ModelProfile

    layers = [(LayerKind.FULL if i % 4 == 3 or i == 45 else LayerKind.RECURRENT, 0)
              for i in range(46)]
    # The actual IQ1 trial's early dense FFNs are much smaller than its MoE FFNs.
    sizes = {i: (100 << 20) if i < 3 else (1 << 30) for i in range(46)}
    profile = ModelProfile('GLM', 87 << 30, 0, 1048576, layers, moe=True,
                           architecture='glm5-next', ffn_block_bytes=sizes)
    cards = tuple(CudaDevice(f'CUDA{i}', f'GPU {i}', 48 << 30, 44 << 30)
                  for i in range(len(split)))
    budget = HardwareBudget(80 << 30, 96 << 30, ram,
                            cuda_devices=cards, cuda_split_bytes=split)
    return profile, budget


@pytest.mark.parametrize('split', [(1, 1), (1, 2), (1, 2, 1), (1, 0, 1)])
def test_glm_spill_covers_each_native_gpu_range_not_only_early_blocks(split):
    from hermes_cli.local_runtime.context_policy import (
        WindowDecision, _cuda_layer_devices, _glm_cuda_spill_blocks, host_spill_bytes, launch_args)

    profile, budget = _glm_spill_fixture(split)
    requested = 17 << 30
    selected = _glm_cuda_spill_blocks(profile, requested, budget)
    native_devices = _cuda_layer_devices(46, split)
    for device, share in enumerate(split):
        quota = (requested * share + sum(split) - 1) // sum(split)
        released = sum(profile.ffn_block_bytes[i] for i in selected if native_devices[i] == device)
        assert released >= quota
        assert released < quota + (1 << 30) if share else released == 0
    assert selected == sorted(selected)  # Preserve front-to-back order inside each device range.
    pattern = r'blk\.({})\.ffn_.*\.weight=CPU'.format('|'.join(map(str, selected)))
    args = launch_args(profile, WindowDecision(98304, requested, True), budget=budget)
    assert args[args.index('-ot') + 1] == pattern
    actual_host = sum(profile.ffn_block_bytes[i] for i in selected)
    assert actual_host <= budget.ram_available_bytes
    assert host_spill_bytes(profile, requested, budget=budget) == actual_host


def test_glm_spill_refuses_unknown_or_insufficient_capacity_on_one_gpu():
    from dataclasses import replace
    from hermes_cli.local_runtime.context_policy import _glm_cuda_spill_blocks
    from hermes_cli.local_runtime.estimator import LayerKind

    profile, budget = _glm_spill_fixture()
    unknown = replace(profile, ffn_block_bytes={i: size for i, size in profile.ffn_block_bytes.items()
                                              if i != 24})
    with pytest.raises(ValueError, match='CUDA1.*sizes are unknown'):
        _glm_cuda_spill_blocks(unknown, 17 << 30, budget)
    no_recurrent_on_second = replace(profile, layers=[
        layer if i < 24 else (LayerKind.FULL, 0) for i, layer in enumerate(profile.layers)])
    with pytest.raises(ValueError, match='CUDA1.*cannot cover'):
        _glm_cuda_spill_blocks(no_recurrent_on_second, 17 << 30, budget)


def test_glm_spill_rounding_counts_actual_host_bytes_and_preserves_other_models():
    from dataclasses import replace
    from hermes_cli.local_runtime.context_policy import _glm_cuda_spill_blocks, spill_overrides

    profile, budget = _glm_spill_fixture(ram=17 << 30)
    # Whole-block selection exceeds 17 GiB, even though the aggregate planned spill was only 17.
    with pytest.raises(ValueError, match='exceeds available host RAM'):
        _glm_cuda_spill_blocks(profile, 17 << 30, budget)
    other = replace(profile, architecture='qwen35moe')
    assert spill_overrides(other, 17 << 30, budget=budget) == [
        '-ot', r'blk\.\d+\.ffn_.*_exps\.weight=CPU']


def test_glm_spill_rejects_invalid_split_instead_of_emitting_partial_placement():
    from dataclasses import replace
    from hermes_cli.local_runtime.context_policy import _glm_cuda_spill_blocks

    profile, budget = _glm_spill_fixture()
    for split in [(0, 0), (1,), (-1, 1)]:
        with pytest.raises(ValueError, match='GLM CUDA placement'):
            _glm_cuda_spill_blocks(profile, 17 << 30, replace(budget, cuda_split_bytes=split))


@pytest.fixture
def glm_placement_preset(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.local_runtime.context_policy import RUNTIME_OVERHEAD_BYTES, host_spill_bytes

    profile, budget = _glm_spill_fixture()
    path = tmp_path / 'placement-model.gguf'
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    header = SimpleNamespace(sampling_defaults={}, architecture='glm5-next')
    monkeypatch.setattr(presets, 'read_gguf_header', lambda _path: header)
    monkeypatch.setattr(presets, 'profile_from_gguf', lambda _header: profile)
    monkeypatch.setattr(glm, 'preset_keys', lambda *_args: {})
    plan = presets.plan_launch(profile, budget, fixed_overhead=RUNTIME_OVERHEAD_BYTES,
                               requested_window=98304)
    rounded = host_spill_bytes(profile, plan.decision.spill_bytes, budget=budget)
    assert rounded > plan.decision.spill_bytes > 0
    return path, profile, budget, plan, rounded


def test_glm_preset_refuses_rounded_spill_that_exceeds_live_ram(glm_placement_preset):
    from dataclasses import replace

    path, _profile, budget, plan, rounded = glm_placement_preset
    live = replace(budget, ram_available_bytes=(plan.decision.spill_bytes + rounded) // 2)
    result = presets.preset_for_model(path, budget, set(), requested_window=98304,
                                      live=live, engine_tag='b11706')
    assert result.keys is None
    assert 'placement refused' in result.refusal
    assert 'available host RAM' in result.refusal
    assert result.placement['tensor-split'] == '1,1'


def test_glm_preset_reserves_rounding_for_cache_and_keeps_admitted_gpu_ratios(
        glm_placement_preset, monkeypatch):
    from dataclasses import replace
    from hermes_cli.local_runtime import hardware
    from hermes_cli.local_runtime.context_policy import spill_overrides

    path, profile, budget, plan, rounded = glm_placement_preset
    # A launch reading can have different free-memory ratios; it must not change tensor placement.
    live = replace(budget, ram_available_bytes=32 << 30, commit_room_bytes=200 << 30,
                   cuda_split_bytes=(999, 1))
    seen = []
    monkeypatch.setattr(hardware, 'prompt_cache_bytes',
                        lambda cache_budget, *_args: seen.append(cache_budget) or 0)
    result = presets.preset_for_model(path, budget, set(), requested_window=98304,
                                      live=live, residents=2, engine_tag='b11706')
    assert result.refusal is None
    assert result.keys['tensor-split'] == '1,1'
    assert result.keys['override-tensor'] == spill_overrides(
        profile, plan.decision.spill_bytes, budget=budget)[1]
    assert seen[0].ram_available_bytes == live.ram_available_bytes - 2 * (
        rounded - plan.decision.spill_bytes)
    assert seen[0].commit_room_bytes == live.commit_room_bytes
