"""Hardware labels expose the allocator's planning capacity independently of live use."""
import pytest

from hermes_cli.local_runtime import hardware, multigpu
from hermes_cli.web_routers import local_models

G = 1 << 30


@pytest.mark.parametrize('sizes', [(48, 48), (48, 48, 32), (96, 48, 48)])
def test_per_card_usable_capacity_is_planning_budget_in_visible_order(monkeypatch, sizes):
    cards = tuple(multigpu.CudaDevice(f'CUDA{i}', f'physical card {len(sizes) - i}', total * G,
                                     (i + 1) * G, used=(total - i - 1) * G)
                  for i, total in enumerate(sizes))
    monkeypatch.setattr(hardware, 'probe_budget', lambda *, planning=False:
                        multigpu.cuda_budget(cards, 64 * G, planning=planning))
    monkeypatch.setattr(hardware, '_ram_bytes', lambda: (64 * G, 50 * G))
    monkeypatch.setattr(local_models, '_nvidia_smi_facts', dict)
    data = local_models.local_models_hardware()
    expected = multigpu.cuda_budget(cards, 64 * G, planning=True)
    assert len(data['gpus']) == len(cards)
    assert [gpu['name'] for gpu in data['gpus']] == [card.description for card in cards]
    assert [gpu['usable_bytes'] for gpu in data['gpus']] == list(expected.cuda_split_bytes)
    assert data['vram_usable_bytes'] == sum(gpu['usable_bytes'] for gpu in data['gpus'])
    assert data['vram_total_bytes'] == sum(card.total for card in cards)
    assert data['vram_used_bytes'] == sum(card.used for card in cards)
    assert all(gpu['usable_bytes'] > gpu['free_bytes'] for gpu in data['gpus'])