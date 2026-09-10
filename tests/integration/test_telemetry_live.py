"""One real trace emitted to the configured Langfuse project, then read back.

Needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and a reachable host, so it is marked
`integration` and excluded from `make test-fast`. The unit suite proves observations are
well-formed, redacted and grouped; only this proves the end-to-end path — that the
credentials work, that the trace id seeded from a run id is the one Langfuse stores, and
that usage and cost survive ingestion.

Deliberately does not: assert on anything Langfuse renders, or clean up after itself. Each
run uses a fresh uuid, so traces never collide; they accumulate in the project as a record
of which builds verified this path.
"""

import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from spine.contracts import ModelCall
from spine.telemetry import (
    LangfuseTracer,
    TelemetryConfig,
    configure,
    trace_run,
    traced,
)

PROJECT = "spine-integration"
OBSERVATION_NAME = "integration.smoke"
INGESTION_TIMEOUT_SECONDS = 90
POLL_INTERVAL_SECONDS = 3

_CONFIG = TelemetryConfig.from_env()


def a_model_call() -> ModelCall:
    """A small, clearly synthetic call record, so the trace is recognisable in the UI."""
    return ModelCall(
        provider="anthropic",
        model="a-model",
        prompt_tokens=123,
        completion_tokens=45,
        cached_tokens=6,
        cost_usd=Decimal("0.00123"),
        latency_ms=42,
        timestamp=datetime.now(UTC),
        purpose="integration-smoke",
    )


@pytest.fixture
def live_tracer() -> LangfuseTracer:
    """Install the live tracer, and restore the no-op one however the test ends.

    A fixture rather than a try/finally in the test body: the credential check can raise
    before any try block is entered, and a live tracer left installed would have the other
    integration tests quietly emitting traces of their own.
    """
    tracer = configure()
    try:
        yield tracer
    finally:
        configure(TelemetryConfig())


@pytest.mark.integration
@pytest.mark.skipif(not _CONFIG.is_configured, reason="needs the LANGFUSE_* variables")
def test_a_trace_reaches_langfuse_and_can_be_read_back(live_tracer: LangfuseTracer) -> None:
    tracer = live_tracer
    assert isinstance(tracer, LangfuseTracer), "credentials present, so tracing must be live"

    client = tracer.client
    assert client.auth_check() is True, "the Langfuse credentials were rejected"

    run_id = f"itest-{uuid.uuid4()}"
    # The same deterministic id the tracer seeds from the run id.
    trace_id = type(client).create_trace_id(seed=run_id)

    @traced(OBSERVATION_NAME, "integration-smoke", kind="generation")
    def emit(prompt: str, *, purpose: str) -> tuple[str, ModelCall]:
        return "ok", a_model_call()

    with trace_run(run_id, PROJECT):
        text, call = emit("a short integration prompt", purpose="integration-smoke")

    assert text == "ok"
    assert call.cost_usd == Decimal("0.00123")

    tracer.flush()
    fetched = _await_observation(client, trace_id, OBSERVATION_NAME)

    assert fetched.id == trace_id, "the trace landed under the id seeded from the run id"
    assert PROJECT in (fetched.tags or []), "the trace is tagged with the project"

    # The poll only returns once the observation is present, so reaching here is the proof
    # that the emitted span was accepted and indexed. Nothing is asserted about how
    # Langfuse renders its type or usage: those response shapes are unverified here, and an
    # assertion written blind would fail for its own reasons rather than the path's.
    assert any(item.name == OBSERVATION_NAME for item in fetched.observations)


def _await_observation(client: object, trace_id: str, name: str) -> object:
    """Poll until the trace carries the named observation, or fail saying what was seen.

    Waiting for the trace alone is not enough: Langfuse ingests asynchronously and the
    trace record is queryable before its observations are indexed, so a poll that stops at
    the first successful fetch reads an empty observation list and looks like a failure.
    """
    deadline = time.monotonic() + INGESTION_TIMEOUT_SECONDS
    last_error: Exception | None = None
    last_names: list[str] = []
    while time.monotonic() < deadline:
        try:
            trace = client.api.trace.get(trace_id)
        except Exception as error:  # noqa: BLE001 - not-found is expected until it lands
            last_error = error
        else:
            last_names = [item.name for item in (trace.observations or [])]
            if name in last_names:
                return trace
        time.sleep(POLL_INTERVAL_SECONDS)
    pytest.fail(
        f"observation {name!r} did not appear on trace {trace_id} within "
        f"{INGESTION_TIMEOUT_SECONDS}s. Last observations seen: {last_names}. "
        f"Last fetch error: {type(last_error).__name__ if last_error else 'none'}."
    )
