"""Model-specific reasoning controls follow literal GGUF template evidence, not names."""
from pathlib import Path
import struct

import pytest

from hermes_cli.local_runtime import bootstrap, endpoint, gguf, reasoning


TEMPLATE = """
{%- if enable_thinking is undefined or enable_thinking is true %}
{%- set resolved = reasoning_effort|default('xhigh') %}
{%- if resolved == 'high' %}{%- set resolved = 'xhigh' %}{%- endif %}
{%- if resolved not in ('xhigh', 'medium', 'low') %}
{{- raise_exception('Unsupported effort') }}
{%- endif %}
{%- endif %}
{%- if add_generation_prompt %}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\\n\\n</think>\\n\\n' }}
{%- else %}{{- '<think>\\n' }}{%- endif %}
{%- endif %}
"""


def _write_model(path: Path, template: str):
    def string(value):
        data = value.encode('utf-8')
        return struct.pack('<Q', len(data)) + data

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
                     + string('tokenizer.chat_template') + struct.pack('<I', 8) + string(template))


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, 'models_dir', lambda: tmp_path)
    monkeypatch.setattr(reasoning, '_cache', reasoning.OrderedDict())
    return tmp_path


def test_real_literal_guard_preserves_defaults_aliases_and_thinking_off():
    caps = reasoning.capabilities_from_template(TEMPLATE)
    assert caps.efforts == ('low', 'medium', 'xhigh')
    assert caps.default_effort == 'xhigh'
    assert dict(caps.aliases) == {'high': 'xhigh'}
    assert caps.supports_disable
    with pytest.raises(TypeError):
        caps.aliases['high'] = 'medium'


@pytest.mark.parametrize('template', [
    '{{ reasoning_effort }}',
    "{# reasoning_effort not in ('low', 'high') #}",
    TEMPLATE.replace("('xhigh', 'medium', 'low')", 'allowed_efforts'),
    TEMPLATE.replace("raise_exception('Unsupported effort')", "'Allowed'"),
    TEMPLATE.replace("default('xhigh')", "default('max')"),
    TEMPLATE.replace("('xhigh', 'medium', 'low')", "('xhigh', 'medium', 'bespoke')"),
    TEMPLATE.replace("{%- set resolved = 'xhigh' %}", "{%- set resolved = dynamic_effort %}"),
    '{% if broken',
])
def test_unproven_templates_do_not_invent_capabilities(template):
    assert reasoning.capabilities_from_template(template) is None


def test_off_requires_a_proven_closed_empty_thinking_branch():
    caps = reasoning.capabilities_from_template(TEMPLATE.replace('</think>', '</different>'))
    assert caps.efforts == ('low', 'medium', 'xhigh') and not caps.supports_disable
    assert reasoning.capabilities_from_template('x' * (reasoning._TEMPLATE_LIMIT + 1)) is None


def test_each_selected_model_uses_its_own_header_and_cache_invalidates(models, monkeypatch):
    first = models / 'local-a.gguf'
    second = models / 'local-b.gguf'
    _write_model(first, TEMPLATE)
    other = TEMPLATE.replace("'xhigh'", "'high'").replace("{%- if resolved == 'high' %}{%- set resolved = 'high' %}{%- endif %}", '')
    _write_model(second, other)
    calls = []
    original = gguf._read_part
    monkeypatch.setattr(gguf, '_read_part', lambda path: calls.append(path) or original(path))
    assert reasoning.model_reasoning_capabilities('local-a').efforts == ('low', 'medium', 'xhigh')
    assert reasoning.model_reasoning_capabilities('local-a').efforts == ('low', 'medium', 'xhigh')
    assert calls == [first]
    assert reasoning.model_reasoning_capabilities('local-b').efforts == ('low', 'medium', 'high')
    _write_model(first, other + '\n')
    assert reasoning.model_reasoning_capabilities('local-a').efforts == ('low', 'medium', 'high')
    assert calls == [first, second, first]
    first.unlink()
    assert reasoning.model_reasoning_capabilities('local-a') is None


def test_incomplete_splits_assets_and_unknown_names_are_not_model_capabilities(models):
    first = models / 'split-00001-of-00002.gguf'
    _write_model(first, TEMPLATE)
    _write_model(models / 'assets' / 'projector.gguf', TEMPLATE)
    assert reasoning.model_reasoning_capabilities('split') is None
    assert reasoning.model_reasoning_capabilities('projector') is None
    assert reasoning.model_reasoning_capabilities('../split') is None
    _write_model(models / 'split-00002-of-00002.gguf', '{{ ignored }}')
    assert reasoning.model_reasoning_capabilities('split').default_effort == 'xhigh'


def test_managed_endpoint_ownership_gates_wire_capabilities(models, monkeypatch):
    _write_model(models / 'selected.gguf', TEMPLATE)
    own = 'http://127.0.0.1:8123/v1'
    monkeypatch.setattr(endpoint, '_state_endpoint', lambda: {'base_url': own})
    assert reasoning.managed_reasoning_capabilities('selected', own + '/').efforts == ('low', 'medium', 'xhigh')
    assert reasoning.managed_reasoning_capabilities('selected', 'https://remote.invalid/v1') is None
    monkeypatch.setattr(endpoint, '_state_endpoint', lambda: None)
    assert reasoning.managed_reasoning_capabilities('selected', own) is None


def test_probe_failure_does_not_break_remote_provider(models, monkeypatch):
    def fail():
        raise OSError('unavailable')

    monkeypatch.setattr(endpoint, '_state_endpoint', fail)
    assert reasoning.managed_reasoning_capabilities('selected', 'https://remote.invalid/v1') is None


def test_template_calls_are_never_executed(models):
    # A valid literal guard may coexist with arbitrary Jinja calls. Detection parses only.
    template = TEMPLATE + '{{ caller_that_does_not_exist() }}'
    _write_model(models / 'parse-only.gguf', template)
    assert reasoning.model_reasoning_capabilities('parse-only').default_effort == 'xhigh'

def test_unknown_staged_template_is_distinct_from_unowned_or_missing(models, monkeypatch):
    _write_model(models / 'unknown.gguf', '{{ messages }}')
    caps = reasoning.model_reasoning_capabilities('unknown')
    assert caps is not None and caps.efforts == ()
    assert caps.default_effort is None and not caps.supports_disable
    own = 'http://127.0.0.1:8123/v1'
    monkeypatch.setattr(endpoint, '_state_endpoint', lambda: {'base_url': own})
    assert reasoning.managed_reasoning_capabilities('unknown', own) is caps
    assert reasoning.managed_reasoning_capabilities('missing', own) is None
    assert reasoning.managed_reasoning_capabilities('unknown', 'https://remote.invalid/v1') is None
    (models / 'unknown.gguf').write_bytes(b'broken GGUF')
    assert reasoning.model_reasoning_capabilities('unknown').efforts == ()


def test_later_reassignment_cannot_claim_the_original_effort_vocabulary():
    modified = TEMPLATE.replace("{{- raise_exception('Unsupported effort') }}\n{%- endif %}",
                                "{{- raise_exception('Unsupported effort') }}\n{%- endif %}{% set resolved = 'max' %}")
    assert reasoning.capabilities_from_template(modified) is None


def test_uncalled_macro_does_not_advertise_a_vocabulary_or_thinking_toggle():
    assert reasoning.capabilities_from_template('{% macro unused() %}' + TEMPLATE + '{% endmacro %}') is None


def test_header_cache_has_a_bound_and_rechecks_evicted_models(models, monkeypatch):
    calls = []
    original = gguf._read_part
    monkeypatch.setattr(gguf, '_read_part', lambda path: calls.append(path) or original(path))
    for index in range(reasoning._CACHE_LIMIT + 1):
        model = f'model-{index}'
        _write_model(models / f'{model}.gguf', TEMPLATE)
        assert reasoning.model_reasoning_capabilities(model).default_effort == 'xhigh'
    assert len(reasoning._cache) == reasoning._CACHE_LIMIT
    before = len(calls)
    assert reasoning.model_reasoning_capabilities('model-0').default_effort == 'xhigh'
    assert len(calls) == before + 1


def test_literal_guard_whitespace_does_not_change_capabilities():
    template = TEMPLATE.replace('{%-', '{%').replace('{{-', '{{')
    assert reasoning.capabilities_from_template(template) == reasoning.capabilities_from_template(TEMPLATE)


@pytest.mark.parametrize('guard', ['false', 'enable_thinking', 'unproven_context'])
def test_off_in_an_unreachable_or_unproven_branch_is_not_advertised(guard):
    template = TEMPLATE.replace('if add_generation_prompt', f'if {guard}')
    caps = reasoning.capabilities_from_template(template)
    assert caps.efforts == ('low', 'medium', 'xhigh') and not caps.supports_disable
