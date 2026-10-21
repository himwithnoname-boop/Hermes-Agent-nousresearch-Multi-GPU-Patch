"""Tensor-table residency evidence must conserve bytes and never price incomplete shards as zero."""
from dataclasses import replace
import struct

import pytest

from hermes_cli.local_runtime.estimator import (
    LayerKind, ModelProfile, gpu_weight_residency, profile_from_gguf)
from hermes_cli.local_runtime.gguf import GGUFHeader, read_gguf_header


def _string(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def _write_table(path, tensors, metadata=None):
    metadata = metadata or {}
    data = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata)))
    for name, value in metadata.items():
        data += _string(name)
        data += (struct.pack('<I', 8) + _string(value) if isinstance(value, str)
                 else struct.pack('<II', 4, value))
    for name, nbytes in tensors:
        assert nbytes % 4 == 0
        data += _string(name) + struct.pack('<IQIQ', 1, nbytes // 4, 0, 0)
    # No tensor payload is written. Successful reads therefore require only the tensor table.
    path.write_bytes(data)


_META = {'general.architecture': 'glm5-next', 'glm5-next.block_count': 5,
         'glm5-next.nextn_predict_layers': 1, 'glm5-next.context_length': 65536,
         'glm5-next.attention.head_count_kv': 1}
_TENSORS = [('token_embd.weight', 400), ('output.weight', 120), ('output_norm.weight', 8),
            ('blk.0.attn_q.weight', 40), ('blk.0.ffn_up.weight', 240),
            ('blk.1.attn_q.weight', 40), ('blk.1.ffn_up.weight', 160),
            ('blk.2.attn_q.weight', 40), ('blk.2.ffn_up.weight', 320),
            ('blk.3.attn_q.weight', 40), ('blk.3.ffn_up.weight', 640),
            ('blk.4.attn_q.weight', 40), ('blk.4.ffn_up.weight', 280),
            ('blk.4.nextn.eh_proj.weight', 80)]


@pytest.fixture
def glm_profile(tmp_path):
    path = tmp_path / 'model.gguf'
    _write_table(path, _TENSORS, _META)
    header = read_gguf_header(path)
    assert header.tensor_table_complete
    assert header.tensor_bytes_by_name == dict(_TENSORS)
    profile = profile_from_gguf(header)
    assert profile.tensor_bytes_by_name is not header.tensor_bytes_by_name
    assert profile.nextn_predict_layers == 1
    return profile


def test_weight_placement_counts_host_input_gpu_output_and_inactive_nextn(glm_profile):
    # Five metadata blocks + output: default equal two-card split is 0,1,2 / 3,4,output.
    result = gpu_weight_residency(glm_profile, [0, 0, 0, 1, 1, 1], 2)
    assert result.gpu_bytes == (840, 808)
    assert result.host_bytes == 400
    assert result.inactive_bytes == 400
    assert result.cpu_ffn_bytes == (0, 0)
    assert sum(result.gpu_bytes) + result.host_bytes + result.inactive_bytes == glm_profile.weights_bytes
    # Loading MTP changes active tensors, never the full metadata layer/output split positions.
    mtp = gpu_weight_residency(glm_profile, [0, 0, 0, 1, 1, 1], 2, load_nextn=True)
    assert mtp.gpu_bytes == (840, 1208)
    assert mtp.host_bytes == 400 and mtp.inactive_bytes == 0


def test_cpu_ffn_overrides_release_each_original_device_and_keep_attention(glm_profile):
    result = gpu_weight_residency(glm_profile, [0, 0, 0, 1, 1, 1], 2, cpu_ffn_blocks=[0, 3])
    assert result.gpu_bytes == (600, 168)
    assert result.host_bytes == 1280
    assert result.cpu_ffn_bytes == (240, 640)
    assert result.inactive_bytes == 400
    assert sum(result.gpu_bytes) + result.host_bytes + result.inactive_bytes == glm_profile.weights_bytes


def test_three_card_partial_offload_does_not_credit_already_host_weights(glm_profile):
    result = gpu_weight_residency(glm_profile, [0, -1, 1, 1, 2, 2], 3, cpu_ffn_blocks=[1, 3])
    assert result.gpu_bytes == (280, 400, 128)
    assert result.host_bytes == 1240
    assert result.cpu_ffn_bytes == (0, 640, 0)
    assert result.inactive_bytes == 400


def test_input_and_output_scales_follow_their_own_tensor_layer(glm_profile):
    tensors = {**glm_profile.tensor_bytes_by_name, 'token_embd.scale': 4,
               'output.scale': 8, 'output_norm.bias': 12}
    profile = replace(glm_profile, tensor_bytes_by_name=tensors, weights_bytes=sum(tensors.values()))
    result = gpu_weight_residency(profile, [0, 0, 0, 1, 1, 1], 2)
    assert result.gpu_bytes == (840, 828)
    assert result.host_bytes == 404


@pytest.mark.parametrize('changes', [
    {'tensor_bytes_by_name': None}, {'tensor_table_complete': False}, {'tensor_bytes_by_name': {}},
    {'architecture': 'qwen35moe'}, {'nextn_predict_layers': 6}, {'nextn_predict_layers': -1},
    {'weights_bytes': 1}, {'lazy_bytes': 1},
])
def test_unknown_or_inconsistent_evidence_cannot_become_residency(glm_profile, changes):
    assert gpu_weight_residency(replace(glm_profile, **changes), [0, 0, 0, 1, 1, 1], 2) is None


@pytest.mark.parametrize('name,size', [('new_global.weight', 4), ('blk.5.ffn_up.weight', 4),
                                       ('blk.-1.attn.weight', 4), ('output.scale', 0)])
def test_unclassified_or_invalid_tensor_table_is_unknown(glm_profile, name, size):
    tensors = {**glm_profile.tensor_bytes_by_name, name: size}
    profile = replace(glm_profile, tensor_bytes_by_name=tensors, weights_bytes=sum(tensors.values()))
    assert gpu_weight_residency(profile, [0, 0, 0, 1, 1, 1], 2) is None


@pytest.mark.parametrize('missing', ['token_embd.weight', 'output.weight', 'output_norm.weight'])
def test_missing_required_glm_globals_do_not_invent_tied_weights(glm_profile, missing):
    tensors = {name: size for name, size in glm_profile.tensor_bytes_by_name.items() if name != missing}
    profile = replace(glm_profile, tensor_bytes_by_name=tensors, weights_bytes=sum(tensors.values()))
    assert gpu_weight_residency(profile, [0, 0, 0, 1, 1, 1], 2) is None


@pytest.mark.parametrize('mapping,count,blocks', [
    ([0, 0], 2, []), ([0, 0, 0, 1, 1, 2], 2, []),
    ([0, 0, 0, 1, 1, -2], 2, []), ([0, 0, 0, 1, 1, 1], 0, []),
    ([0, 0, 0, 1, 1, 1], 2, [-1]), ([0, 0, 0, 1, 1, 1], 2, [5]),
])
def test_invalid_native_mapping_or_cpu_blocks_remain_unknown(glm_profile, mapping, count, blocks):
    assert gpu_weight_residency(glm_profile, mapping, count, cpu_ffn_blocks=blocks) is None


def test_legacy_constructor_positions_and_unknown_header_remain_compatible():
    header = GGUFHeader('old', 3, _META, 0, 0)
    profile = ModelProfile('old', 0, 0, 65536, [(LayerKind.RECURRENT, 0)])
    assert header.tensor_bytes_by_name is None and not header.tensor_table_complete
    assert profile.tensor_bytes_by_name is None and not profile.tensor_table_complete
    assert gpu_weight_residency(profile, [0, 0], 1) is None


def test_complete_split_combines_tables_but_missing_or_unreadable_part_is_unknown(tmp_path):
    first = tmp_path / 'split-00001-of-00003.gguf'
    second = tmp_path / 'split-00002-of-00003.gguf'
    third = tmp_path / 'split-00003-of-00003.gguf'
    _write_table(first, [], {**_META, 'split.count': 3, 'split.tensors.count': len(_TENSORS)})
    _write_table(second, _TENSORS[:7], {'split.count': 3})
    _write_table(third, _TENSORS[7:], {'split.count': 3})
    complete = read_gguf_header(first)
    assert complete.tensor_table_complete and complete.tensor_bytes_by_name == dict(_TENSORS)
    assert gpu_weight_residency(profile_from_gguf(complete), [0, 0, 0, 1, 1, 1], 2) is not None
    third.unlink()
    partial = read_gguf_header(first)
    assert not partial.tensor_table_complete
    assert partial.tensor_bytes > 0  # Existing coarse reader still prices the available subset.
    _write_table(third, _TENSORS[7:], {'split.count': 3})
    third.write_bytes(third.read_bytes()[:-1])
    assert not read_gguf_header(first).tensor_table_complete
    second.unlink()
    assert not read_gguf_header(first).tensor_table_complete


def test_isolated_stub_and_declared_tensor_count_mismatch_are_unknown(tmp_path):
    stub = tmp_path / 'stub-00001-of-00003.gguf'
    _write_table(stub, [], _META)
    header = read_gguf_header(stub)
    assert header.tensor_bytes_by_name == {} and not header.tensor_table_complete
    standalone = tmp_path / 'standalone.gguf'
    _write_table(standalone, _TENSORS, {**_META, 'split.count': 2})
    assert not read_gguf_header(standalone).tensor_table_complete
    _write_table(standalone, _TENSORS, {**_META, 'split.tensors.count': 999})
    assert not read_gguf_header(standalone).tensor_table_complete


def test_duplicate_names_and_inconsistent_shard_counts_do_not_claim_complete_evidence(tmp_path):
    path = tmp_path / 'duplicate.gguf'
    _write_table(path, _TENSORS + [_TENSORS[0]], _META)
    assert not read_gguf_header(path).tensor_table_complete
    first = tmp_path / 'split-00001-of-00002.gguf'
    second = tmp_path / 'split-00002-of-00002.gguf'
    _write_table(first, _TENSORS, {**_META, 'split.count': 2})
    _write_table(second, [_TENSORS[0]], {'split.count': 2})
    assert not read_gguf_header(first).tensor_table_complete
    _write_table(second, [], {'split.count': 3})
    assert not read_gguf_header(first).tensor_table_complete


def test_truncated_last_offset_never_counts_as_a_complete_table(tmp_path):
    path = tmp_path / 'truncated.gguf'
    _write_table(path, _TENSORS, _META)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='truncated GGUF header'):
        read_gguf_header(path)


def test_unused_nextn_ffns_cannot_be_credited_as_gpu_spill(glm_profile):
    inactive = gpu_weight_residency(glm_profile, [0, 0, 0, 1, 1, 1], 2, cpu_ffn_blocks=[4])
    assert inactive.gpu_bytes == (840, 808)
    assert inactive.host_bytes == 400 and inactive.inactive_bytes == 400
    assert inactive.cpu_ffn_bytes == (0, 0)
    active = gpu_weight_residency(glm_profile, [0, 0, 0, 1, 1, 1], 2,
                                  cpu_ffn_blocks=[4], load_nextn=True)
    assert active.gpu_bytes == (840, 928)
    assert active.host_bytes == 680 and active.cpu_ffn_bytes == (0, 280)


def test_zero_share_device_holds_no_weight_and_output_can_remain_on_cpu(glm_profile):
    result = gpu_weight_residency(glm_profile, [0, 0, 0, 2, 2, 2], 3)
    assert result.gpu_bytes == (840, 0, 808)
    cpu_output = gpu_weight_residency(glm_profile, [0, 0, 0, 2, 2, -1], 3)
    assert cpu_output.gpu_bytes == (840, 0, 680)
    assert cpu_output.host_bytes == 528


def test_entire_missing_trunk_layer_does_not_look_like_a_zero_byte_layer(glm_profile):
    tensors = {name: size for name, size in glm_profile.tensor_bytes_by_name.items()
               if not name.startswith('blk.1.')}
    profile = replace(glm_profile, tensor_bytes_by_name=tensors, weights_bytes=sum(tensors.values()))
    assert gpu_weight_residency(profile, [0, 0, 0, 1, 1, 1], 2) is None


def test_shard_number_metadata_cannot_disagree_with_complete_filename_set(tmp_path):
    first = tmp_path / 'split-00001-of-00002.gguf'
    second = tmp_path / 'split-00002-of-00002.gguf'
    _write_table(first, _TENSORS[:7], {**_META, 'split.count': 2, 'split.no': 0})
    _write_table(second, _TENSORS[7:], {'split.count': 2, 'split.no': 0})
    assert not read_gguf_header(first).tensor_table_complete
    _write_table(second, _TENSORS[7:], {'split.count': 2, 'split.no': 1})
    assert read_gguf_header(first).tensor_table_complete
