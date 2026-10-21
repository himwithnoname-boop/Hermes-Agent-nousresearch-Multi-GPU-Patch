"""Generation throughput: accepted engine tokens/time, with explicit request fallback."""
from __future__ import annotations

from collections import deque
import math


def _field(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    extra = getattr(obj, "model_extra", None)
    return value if value is not None else (extra.get(name) if isinstance(extra, dict) else None)


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 <= value <= 1e100 and math.isfinite(value) and (value > 0 if positive else value >= 0) else None


def engine_sample(response):
    """llama.cpp timings are cumulative for this completion, not per GPU or draft proposal."""
    timings = _field(response, "timings")
    tokens = _number(_field(timings, "predicted_n"), positive=True)
    millis = _number(_field(timings, "predicted_ms"), positive=True)
    if tokens is None or millis is None or not tokens.is_integer():
        return None
    seconds = millis / 1000
    return (tokens, seconds) if seconds > 0 and tokens / seconds < 1e6 else None


def record_speed(agent, response, output_tokens, api_duration):
    """One atomic paired sample per completed call; keep request latency separate."""
    sample = engine_sample(response)
    kind = "generation" if sample is not None else "request"
    if sample is None:
        sample = (_number(output_tokens), _number(api_duration, positive=True))
    history = getattr(agent, "_api_speed_history", None)
    if history is None:
        history = agent._api_speed_history = deque(maxlen=10)
    history.append((*sample, kind))


def read_speed(agent):
    """Duration-weighted rate; never mix engine decode time with full request time."""
    history = getattr(agent, "_api_speed_history", None)
    if history:
        newest = history[-1][2]
        samples = []
        for tokens, seconds, kind in reversed(history):
            if kind != newest:
                break
            if tokens is None or seconds is None:
                return None, None
            samples.append((tokens, seconds))
    else:
        latencies = list(getattr(agent, "_api_latency_history", None) or [])
        outputs = list(getattr(agent, "_api_output_history", None) or [])
        n = min(len(latencies), len(outputs))
        samples = [(_number(o), _number(s, positive=True)) for o, s in zip(outputs[-n:], latencies[-n:])] if n else []
        newest = "request"
    if not samples or any(t is None or s is None for t, s in samples):
        return None, None
    duration = sum(s for _, s in samples)
    rate = sum(t for t, _ in samples) / duration if duration > 0 else 0
    return (rate, newest) if 0 < rate < 1e6 else (None, None)


def capture_stream_timings(chunk, previous):
    """Retain the final cumulative timing frame, including a choiceless usage frame."""
    terminal = _field(chunk, "usage") or any(
        _field(choice, "finish_reason") for choice in (_field(chunk, "choices") or ()))
    return (_field(chunk, "timings") or previous) if terminal else previous


def attach_stream_timings(response, timings, *, runaway=False):
    """Incomplete/cancelled streams must not acquire a completed generation sample."""
    from hermes_constants import PARTIAL_STREAM_STUB_ID

    if not runaway and getattr(response, "id", None) != PARTIAL_STREAM_STUB_ID and timings is not None:
        response.timings = timings
