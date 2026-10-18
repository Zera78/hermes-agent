"""Measurement evidence captured after a context engine accepts provider usage."""
from dataclasses import dataclass
from time import monotonic_ns
from uuid import uuid4

from agent.context_breakdown import context_usage_fields


@dataclass(frozen=True)
class ContextMeasurement:
    identity: str
    observed_ns: int
    engine: object
    provider: object
    model: object
    compression_count: object
    used: int
    maximum: int
    source: str


def capture_context_measurement(agent, prompt_tokens):
    """Publish one immutable reading; usage-less responses leave it untouched.

    The identity belongs to a positive provider measurement, not to a request or
    a change in token count. Engine identity, route and values travel together.
    """
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        return
    engine = agent.context_compressor
    values = context_usage_fields(engine)
    if not values:
        return
    agent.context_measurement = ContextMeasurement(
        uuid4().hex, monotonic_ns(), engine, getattr(agent, 'provider', None), getattr(agent, 'model', None),
        getattr(engine, 'compression_count', None), values['context_used'],
        values['context_max'], values['context_source'])
