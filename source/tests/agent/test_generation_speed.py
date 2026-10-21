"""Generation speed must use one completion's accepted tokens and decode time, not GPU sums."""
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

from agent.generation_speed import engine_sample, read_speed, record_speed


def response(tokens=100, millis=2000):
    return NS(timings={'predicted_n': tokens, 'predicted_ms': millis,
                       'prompt_ms': 18000, 'draft_n': 900, 'draft_n_accepted': 80})


def test_generation_speed_excludes_prefill_wait_and_rejected_drafts():
    agent = NS(gpu_count=3)
    record_speed(agent, response(), 100, 20)
    assert read_speed(agent) == (50, 'generation')
    # Changing the number of GPUs cannot multiply a single completion's output.
    agent.gpu_count = 7
    assert read_speed(agent) == (50, 'generation')


def test_rolling_speed_is_time_weighted_and_keeps_only_ten_calls():
    agent = NS()
    record_speed(agent, response(1, 1000000), 1, 2000)
    for _ in range(9):
        record_speed(agent, response(100, 2000), 100, 20)
    record_speed(agent, response(1000, 10000), 1000, 20)
    assert len(agent._api_speed_history) == 10
    speed, source = read_speed(agent)
    assert speed == pytest.approx(1900 / 28) and source == 'generation'


@pytest.mark.parametrize('tokens,millis', [(100, 0), (100, -1), (True, 2000), (1.5, 2000),
                                          (100, float('nan')), (100, float('inf')), ('100', 2000), (10**1000, 2000)])
def test_invalid_engine_timing_falls_back_to_request_time(tokens, millis):
    agent = NS()
    record_speed(agent, response(tokens, millis), 100, 20)
    assert read_speed(agent) == (5, 'request')


def test_switching_timing_sources_never_mixes_durations():
    agent = NS()
    record_speed(agent, response(), 100, 20)
    record_speed(agent, NS(), 100, 20)
    assert read_speed(agent) == (5, 'request')
    record_speed(agent, response(80, 2000), 80, 20)
    assert read_speed(agent) == (40, 'generation')


def test_invalid_request_time_does_not_show_stale_generation_rate():
    agent = NS()
    record_speed(agent, response(), 100, 20)
    record_speed(agent, NS(), 100, float('nan'))
    assert read_speed(agent) == (None, None)


def test_sdk_extra_fields_and_nonstream_response_are_supported():
    item = NS(model_extra={'timings': {'predicted_n': 200, 'predicted_ms': 4000}})
    assert engine_sample(item) == (200, 4)
    assert engine_sample({'timings': {'predicted_n': 200, 'predicted_ms': 4000}}) == (200, 4)


def test_gateway_and_embedded_usage_report_generation_but_keep_request_latency():
    from gateway.session_usage import tui_usage, usage_stats
    from tui_gateway.server import _get_usage
    agent = NS(_api_latency_history=deque([20]), _api_output_history=deque([100]), model='example')
    record_speed(agent, response(), 100, 20)
    stats = usage_stats(agent)
    assert stats['tokens_per_s'] == 50 and stats['avg_latency_s'] == 20
    assert stats['tps_source'] == 'generation'
    assert tui_usage(stats)['avg_tps'] == 50 and tui_usage(stats)['tps_source'] == 'generation'
    embedded = _get_usage(agent)
    assert embedded['avg_tps'] == 50 and embedded['avg_latency_s'] == 20
    assert embedded['tps_source'] == 'generation'


def test_real_usage_accounting_records_response_timings(tmp_path, monkeypatch, caplog):
    from tests.agent.test_turn_usage_log_line import _agent, _usage, _line
    from gateway.session_usage import usage_stats
    agent = _agent(tmp_path, monkeypatch)
    try:
        resp = response()
        resp.usage = _usage(0, 0, 100)
        resp.usage.completion_tokens = 100
        _line(agent, caplog, resp)  # The real accounting seam receives a different request latency.
        assert usage_stats(agent)['tokens_per_s'] == 50
        assert list(agent._api_latency_history) == [0.2]
    finally:
        agent.close()


from tests.agent.test_first_chunk_at_hook import _make_stream_chunk

pytest_plugins = ["tests.agent.test_first_chunk_at_hook"]


@patch('run_agent.AIAgent._create_request_openai_client')
@patch('run_agent.AIAgent._close_request_openai_client')
def test_choiceless_final_stream_timings_survive_assembly(_close, create, agent):
    first = _make_stream_chunk(content='hello', model='example')
    last = _make_stream_chunk(content=None, finish_reason='stop', model='example')
    timing = NS(choices=[], usage=NS(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
                timings=response().timings, model='example', id='example-completion')
    client = MagicMock()
    client.chat.completions.create.return_value = iter([first, last, timing])
    create.return_value = client
    result = agent._interruptible_streaming_api_call({})
    assert engine_sample(result) == (100, 2)
    assert result.choices[0].message.content == 'hello'

def test_nonterminal_timing_frames_are_not_presented_as_complete_generation():
    from agent.generation_speed import attach_stream_timings, capture_stream_timings
    from hermes_constants import PARTIAL_STREAM_STUB_ID
    partial = NS(choices=[NS(finish_reason=None)], usage=None, timings=response().timings)
    assert capture_stream_timings(partial, None) is None
    incomplete = NS(id=PARTIAL_STREAM_STUB_ID)
    attach_stream_timings(incomplete, response().timings)
    assert engine_sample(incomplete) is None
    cut = NS(id='completion')
    attach_stream_timings(cut, response().timings, runaway=True)
    assert engine_sample(cut) is None
