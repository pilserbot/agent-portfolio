"""Unit tests for Langfuse tracing.

Nothing here reaches a network or needs a credential: the tests either exercise the no-op
tracer that a missing configuration produces, or inject a recording tracer and assert on
what the tracing boundary handed it.

Deliberately does not: talk to Langfuse. What the SDK does with a well-formed observation
is its business; what this module must get right is that observations are well-formed,
redacted, grouped, and never in the way.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import pytest

from spine.contracts import ModelCall, StepTrace
from spine.telemetry import (
    REDACTED,
    TEXT_LIMIT,
    Observation,
    RunContext,
    TelemetryConfig,
    Tracer,
    configure,
    current_run,
    get_tracer,
    redact,
    scrub_secrets,
    set_tracer,
    trace_run,
    traced,
    truncate,
)

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


class RecordedObservation(Observation):
    """Captures what the tracing boundary reported for one call."""

    def __init__(self, name: str, kind: str, run: RunContext | None, payload: object) -> None:
        """Store how the observation was opened."""
        self.name = name
        self.kind = kind
        self.run = run
        self.payload = payload
        self.output: object = None
        self.call: ModelCall | None = None
        self.step: StepTrace | None = None
        self.latency_ms: int | None = None
        self.error: BaseException | None = None
        self.finished = False

    def finish(
        self,
        *,
        output: object = None,
        call: ModelCall | None = None,
        step: StepTrace | None = None,
        latency_ms: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Record the reported result."""
        self.output = output
        self.call = call
        self.step = step
        self.latency_ms = latency_ms
        self.error = error
        self.finished = True


class Recorder(Tracer):
    """An enabled tracer that keeps everything in memory instead of sending it."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.runs: list[RunContext] = []
        self.observations: list[RecordedObservation] = []
        self.flushed = 0

    @property
    def is_enabled(self) -> bool:
        return True

    def run(self, context: RunContext):  # noqa: ANN201 - contextmanager from the base
        from contextlib import contextmanager

        @contextmanager
        def manager():  # noqa: ANN202
            self.runs.append(context)
            yield

        return manager()

    def observation(  # noqa: ANN201
        self,
        *,
        name: str,
        kind: Literal["span", "generation"],
        run: RunContext | None,
        payload: object,
    ):
        from contextlib import contextmanager

        @contextmanager
        def manager():  # noqa: ANN202
            recorded = RecordedObservation(name, kind, run, payload)
            self.observations.append(recorded)
            yield recorded

        return manager()

    def flush(self) -> None:
        self.flushed += 1

    @property
    def only(self) -> RecordedObservation:
        """The single observation recorded, asserting there was exactly one."""
        assert len(self.observations) == 1, f"expected 1 observation, got {len(self.observations)}"
        return self.observations[0]


@pytest.fixture
def recorder() -> Recorder:
    """Install a recording tracer for one test, restoring the no-op afterwards."""
    tracer = Recorder()
    set_tracer(tracer)
    yield tracer
    set_tracer(Tracer())


@pytest.fixture(autouse=True)
def _no_ambient_tracer() -> None:
    """Never let a configured tracer leak between tests."""
    set_tracer(Tracer())


def a_call(cost: str = "0.0031") -> ModelCall:
    return ModelCall(
        provider="anthropic",
        model="a-model",
        prompt_tokens=1000,
        completion_tokens=200,
        cached_tokens=50,
        cost_usd=Decimal(cost),
        latency_ms=120,
        timestamp=AT,
        purpose="extract",
    )


# --- no observability configured -------------------------------------------------------


def test_no_credentials_means_a_disabled_tracer() -> None:
    tracer = configure(TelemetryConfig.from_env({}))

    assert tracer.is_enabled is False


def test_partial_credentials_are_not_enough() -> None:
    config = TelemetryConfig.from_env({"LANGFUSE_PUBLIC_KEY": "pk-lf-x"})

    assert config.is_configured is False
    assert configure(config).is_enabled is False


def test_an_empty_variable_counts_as_absent() -> None:
    config = TelemetryConfig.from_env(
        {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "", "LANGFUSE_HOST": ""}
    )

    assert config.is_configured is False
    assert config.host is None


def test_the_host_is_optional() -> None:
    config = TelemetryConfig.from_env(
        {"LANGFUSE_PUBLIC_KEY": "pk-lf-x", "LANGFUSE_SECRET_KEY": "sk-lf-x"}
    )

    assert config.is_configured is True
    assert config.host is None


def test_a_traced_function_is_untouched_when_telemetry_is_off() -> None:
    @traced("work", "extract")
    def work(value: str) -> str:
        return value.upper()

    assert work("hello") == "HELLO"


def test_the_no_op_tracer_never_raises() -> None:
    tracer = configure(TelemetryConfig.from_env({}))

    with trace_run("run-1", "ri05"):
        pass
    tracer.flush()
    tracer.shutdown()


def test_an_exception_propagates_unchanged_with_telemetry_off() -> None:
    @traced("boom", "extract")
    def boom() -> None:
        raise ValueError("unchanged")

    with pytest.raises(ValueError, match="unchanged"):
        boom()


# --- what a traced call reports ---------------------------------------------------------


def test_inputs_and_outputs_are_recorded(recorder: Recorder) -> None:
    @traced("work", "extract")
    def work(prompt: str, *, tier: str = "large") -> str:
        return "done: " + prompt

    work("a tender", tier="small")

    observation = recorder.only
    assert observation.name == "work"
    assert observation.payload["args"] == ["a tender"]
    assert observation.payload["kwargs"] == {"tier": "small"}
    assert observation.output == "done: a tender"
    assert observation.finished


def test_latency_is_recorded(recorder: Recorder) -> None:
    @traced("work", "extract")
    def work() -> str:
        return "ok"

    work()

    assert recorder.only.latency_ms is not None
    assert recorder.only.latency_ms >= 0


def test_token_counts_and_cost_come_from_the_model_call(recorder: Recorder) -> None:
    call = a_call()

    @traced("router.complete", "model_call", kind="generation")
    def complete(prompt: str) -> tuple[str, ModelCall]:
        return "text", call

    complete("hi")

    observation = recorder.only
    assert observation.kind == "generation"
    assert observation.call is call
    assert observation.call.prompt_tokens == 1000
    assert observation.call.completion_tokens == 200
    assert observation.call.cached_tokens == 50
    assert observation.call.cost_usd == Decimal("0.0031")


def test_a_step_trace_in_a_node_update_is_recorded(recorder: Recorder) -> None:
    step = StepTrace(step_name="extract", started_at=AT, finished_at=AT, status="failed", error="x")

    @traced("node.extract", "extract")
    def node(state: object) -> dict[str, object]:
        return {"steps": [step], "visited": ["extract"]}

    node(object())

    assert recorder.only.step is step


def test_the_runtime_purpose_overrides_the_decorators_default(recorder: Recorder) -> None:
    @traced("router.complete", "default_purpose")
    def complete(prompt: str, *, purpose: str) -> str:
        return "ok"

    complete("hi", purpose="price_check")

    assert recorder.only.payload["purpose"] == "price_check"


def test_the_decorator_purpose_is_used_when_the_call_gives_none(recorder: Recorder) -> None:
    @traced("node.extract", "extraction")
    def node() -> str:
        return "ok"

    node()

    assert recorder.only.payload["purpose"] == "extraction"


def test_a_failure_is_recorded_and_still_raised(recorder: Recorder) -> None:
    @traced("boom", "extract")
    def boom() -> None:
        raise ValueError("still raised")

    with pytest.raises(ValueError, match="still raised"):
        boom()

    observation = recorder.only
    assert isinstance(observation.error, ValueError)
    assert observation.latency_ms is not None
    assert observation.finished


# --- grouping by run --------------------------------------------------------------------


def test_every_observation_of_a_run_carries_the_same_run_context(recorder: Recorder) -> None:
    @traced("step", "extract")
    def step(n: int) -> int:
        return n

    with trace_run("run-abc", "ri05"):
        step(1)
        step(2)
        step(3)

    assert len(recorder.observations) == 3
    assert {(o.run.run_id, o.run.project) for o in recorder.observations} == {("run-abc", "ri05")}
    assert recorder.runs == [RunContext(run_id="run-abc", project="ri05")]


def test_two_runs_do_not_share_a_context(recorder: Recorder) -> None:
    @traced("step", "extract")
    def step() -> None:
        return None

    with trace_run("run-a", "ri05"):
        step()
    with trace_run("run-b", "ri05"):
        step()

    assert [o.run.run_id for o in recorder.observations] == ["run-a", "run-b"]


def test_the_run_context_is_restored_afterwards() -> None:
    assert current_run() is None

    with trace_run("run-a", "ri05") as context:
        assert current_run() == context

    assert current_run() is None


def test_a_state_argument_supplies_the_run_when_there_is_no_ambient_context(
    recorder: Recorder,
) -> None:
    # LangGraph may run a node on a worker thread that never inherited the context var.
    class State:
        run_id = "run-from-state"
        project = "ri05"

    @traced("node.extract", "extract")
    def node(state: State) -> str:
        return "ok"

    node(State())

    assert recorder.only.run == RunContext(run_id="run-from-state", project="ri05")


def test_an_observation_outside_any_run_is_still_recorded(recorder: Recorder) -> None:
    @traced("loose", "extract")
    def loose() -> str:
        return "ok"

    loose()

    assert recorder.only.run is None


# --- redaction and truncation -------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "ANTHROPIC_API_KEY",
        "apiKey",
        "secret_key",
        "password",
        "authorization",
        "X-Auth-Token",
        "credential",
        "cookie",
    ],
)
def test_a_secret_shaped_key_is_redacted(key: str) -> None:
    assert redact({key: "sk-ant-super-secret"}) == {key: REDACTED}


def test_a_secret_nested_deep_is_still_redacted() -> None:
    payload = {"kwargs": {"headers": {"authorization": "Bearer abc123"}}}

    assert redact(payload)["kwargs"]["headers"]["authorization"] == REDACTED


def test_ordinary_values_survive_redaction() -> None:
    assert redact({"model": "a-model", "tier": "large", "n": 3}) == {
        "model": "a-model",
        "tier": "large",
        "n": 3,
    }


def test_a_secret_kwarg_never_reaches_the_tracer(recorder: Recorder) -> None:
    @traced("router.complete", "model_call")
    def complete(prompt: str, *, api_key: str) -> str:
        return "ok"

    complete("hi", api_key="sk-ant-do-not-send")

    payload = recorder.only.payload
    assert payload["kwargs"]["api_key"] == REDACTED
    assert "sk-ant-do-not-send" not in str(payload)


# --- secrets caught by shape, not just by key name -------------------------------------

ANTHROPIC_KEY = "sk-ant-api03-AbCdEf0123456789-_xyz"
LANGFUSE_SECRET = "sk-lf-1a2b3c4d-5e6f-7890-abcd-ef1234567890"
LANGFUSE_PUBLIC = "pk-lf-1a2b3c4d-5e6f-7890-abcd-ef1234567890"
DATABASE_URL = "postgresql://tender_user:s3cr3t-pw@db.example.com:5432/agent_portfolio"


@pytest.mark.parametrize(
    ("secret", "label"),
    [
        (ANTHROPIC_KEY, "anthropic"),
        (LANGFUSE_SECRET, "langfuse-secret"),
        (LANGFUSE_PUBLIC, "langfuse-public"),
        (DATABASE_URL, "database-url"),
    ],
    ids=lambda value: value if isinstance(value, str) and len(value) < 20 else "secret",
)
def test_a_secret_is_caught_as_a_whole_value(secret: str, label: str) -> None:
    assert scrub_secrets(secret) == REDACTED


@pytest.mark.parametrize(
    ("secret", "label"),
    [
        (ANTHROPIC_KEY, "anthropic"),
        (LANGFUSE_SECRET, "langfuse-secret"),
        (LANGFUSE_PUBLIC, "langfuse-public"),
        (DATABASE_URL, "database-url"),
    ],
    ids=lambda value: value if isinstance(value, str) and len(value) < 20 else "secret",
)
def test_a_secret_is_caught_inside_a_sentence(secret: str, label: str) -> None:
    sentence = f"The clause says to use {secret} when connecting, then stop."

    scrubbed = scrub_secrets(sentence)

    assert secret not in scrubbed
    assert REDACTED in scrubbed
    assert scrubbed.startswith("The clause says to use ")
    assert scrubbed.endswith(" when connecting, then stop.")


def test_several_secrets_in_one_string_are_all_caught() -> None:
    text = f"key={ANTHROPIC_KEY} pub={LANGFUSE_PUBLIC} db={DATABASE_URL}"

    scrubbed = scrub_secrets(text)

    for secret in (ANTHROPIC_KEY, LANGFUSE_PUBLIC, DATABASE_URL):
        assert secret not in scrubbed
    assert scrubbed.count(REDACTED) == 3


@pytest.mark.parametrize(
    "harmless",
    [
        "https://cloud.langfuse.com",
        "postgresql://localhost:5432/agent_portfolio",
        "claude-sonnet-5",
        "The tender closes on Friday at 17:00.",
        "ratio: 0.82",
    ],
)
def test_ordinary_text_is_not_mangled(harmless: str) -> None:
    # A host with no userinfo, a model name and plain prose must survive untouched.
    assert scrub_secrets(harmless) == harmless


def test_a_secret_passed_positionally_never_reaches_the_tracer(recorder: Recorder) -> None:
    # The gap this closes: no key name to match on, so only the value's shape gives it away.
    @traced("node.extract", "extract")
    def node(text: str) -> str:
        return text

    node(f"Authenticate with {ANTHROPIC_KEY} before extracting.")

    payload = str(recorder.only.payload)
    assert ANTHROPIC_KEY not in payload
    assert REDACTED in payload


def test_a_secret_in_a_return_value_never_reaches_the_tracer(recorder: Recorder) -> None:
    @traced("node.extract", "extract")
    def node() -> str:
        return f"connect to {DATABASE_URL}"

    node()

    assert DATABASE_URL not in str(recorder.only.output)


def test_a_secret_inside_a_nested_structure_is_caught(recorder: Recorder) -> None:
    @traced("node.extract", "extract")
    def node(payload: dict[str, object]) -> str:
        return "ok"

    node({"config": {"connection": DATABASE_URL, "notes": ["see " + ANTHROPIC_KEY]}})

    sent = str(recorder.only.payload)
    assert DATABASE_URL not in sent
    assert ANTHROPIC_KEY not in sent


def test_a_secret_is_scrubbed_before_truncation_not_after() -> None:
    # Truncating first would cut a key in half and ship the front of it.
    padding = "x" * (TEXT_LIMIT - 10)
    scrubbed = redact(padding + ANTHROPIC_KEY)

    assert "sk-ant-" not in scrubbed
    assert REDACTED in scrubbed


def test_long_text_is_truncated_with_a_visible_marker() -> None:
    result = truncate("x" * 5000)

    assert len(result) < 5000
    assert result.startswith("x" * TEXT_LIMIT)
    assert "truncated 1000 of 5000 characters" in result


def test_text_at_the_limit_is_left_alone() -> None:
    exact = "x" * TEXT_LIMIT

    assert truncate(exact) == exact
    assert "truncated" not in truncate(exact)


def test_a_long_document_argument_is_truncated_before_it_is_sent(recorder: Recorder) -> None:
    document = "clause " * 2000

    @traced("node.extract", "extract")
    def node(text: str) -> str:
        return text

    node(document)

    sent = recorder.only.payload["args"][0]
    assert len(sent) < len(document)
    assert "truncated" in sent
    assert len(recorder.only.output) < len(document)


def test_redaction_survives_a_pydantic_model() -> None:
    reduced = redact(a_call())

    assert reduced["model"] == "a-model"
    assert reduced["cost_usd"] == "0.0031"


def test_a_deeply_nested_structure_is_bounded_not_walked_forever() -> None:
    payload: dict[str, object] = {"level": 0}
    node = payload
    for depth in range(1, 20):
        child: dict[str, object] = {"level": depth}
        node["child"] = child
        node = child

    result = redact(payload)

    assert "depth limit" in str(result)


def test_a_self_referencing_structure_does_not_hang() -> None:
    payload: dict[str, object] = {}
    payload["self"] = payload

    assert "depth limit" in str(redact(payload))


# --- the tracer is replaceable --------------------------------------------------------------


def test_get_tracer_returns_whatever_was_installed(recorder: Recorder) -> None:
    assert get_tracer() is recorder


# --- wired through the router and the graph ---------------------------------------------


def test_a_router_call_is_traced_as_a_priced_generation(recorder: Recorder, tmp_path) -> None:  # noqa: ANN001
    from litellm.types.utils import ModelResponse, Usage

    from spine.router import DEFAULT_MODEL_LARGE, Router, RouterConfig

    def fake_completion(**kwargs: object) -> ModelResponse:
        return ModelResponse(
            model=DEFAULT_MODEL_LARGE,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            usage=Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
        )

    router = Router(
        RouterConfig(ledger_path=tmp_path / "ledger.json"), completion_fn=fake_completion
    )

    with trace_run("run-1", "ri05"):
        _text, call = router.complete("a tender", purpose="extract")

    observation = recorder.only
    assert observation.name == "router.complete"
    assert observation.kind == "generation", "so Langfuse prices it as a model call"
    assert observation.call == call
    assert observation.call.cost_usd > Decimal("0")
    assert observation.run == RunContext(run_id="run-1", project="ri05")
    assert observation.payload["purpose"] == "extract"


def test_a_refused_router_call_is_traced_as_a_failure(recorder: Recorder, tmp_path) -> None:  # noqa: ANN001
    from spine.router import Router, RouterConfig, SpendCapExceeded

    router = Router(
        RouterConfig(ledger_path=tmp_path / "ledger.json", daily_spend_cap_usd=Decimal("0")),
        completion_fn=lambda **_kwargs: None,
    )

    with pytest.raises(SpendCapExceeded):
        router.complete("hi", purpose="extract")

    assert isinstance(recorder.only.error, SpendCapExceeded)


def test_every_graph_node_is_traced_under_one_run(recorder: Recorder, tmp_path) -> None:  # noqa: ANN001
    from pydantic import Field

    from spine.graph import (
        END,
        START,
        CheckpointConfig,
        GraphState,
        build_graph,
        open_checkpointer,
        start,
    )

    class BidState(GraphState):
        visited: list[str] = Field(default_factory=list)

    def visit(label: str):  # noqa: ANN202
        def node(state: BidState) -> dict[str, object]:
            return {"visited": [*state.visited, label]}

        return node

    with open_checkpointer(CheckpointConfig(sqlite_path=tmp_path / "graph.sqlite")) as saver:
        graph = build_graph(
            BidState,
            {"extract": visit("extract"), "price": visit("price")},
            [(START, "extract"), ("extract", "price"), ("price", END)],
            checkpointer=saver,
        )
        start(graph, BidState(run_id="run-7", project="ri05", started_at=AT))

    assert [o.name for o in recorder.observations] == ["node.extract", "node.price"]
    assert {o.run.run_id for o in recorder.observations} == {"run-7"}
    assert {o.run.project for o in recorder.observations} == {"ri05"}
    assert all(o.step is None for o in recorder.observations), (
        "the node's own update carries no StepTrace; with_retry adds it outside the span"
    )


def test_each_retry_attempt_gets_its_own_observation(recorder: Recorder, tmp_path) -> None:  # noqa: ANN001
    from pydantic import Field

    from spine.graph import (
        END,
        START,
        CheckpointConfig,
        GraphState,
        RetryConfig,
        build_graph,
        open_checkpointer,
        start,
    )

    class BidState(GraphState):
        visited: list[str] = Field(default_factory=list)

    calls = {"n": 0}

    def flaky(state: BidState) -> dict[str, object]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return {"visited": ["ok"]}

    with open_checkpointer(CheckpointConfig(sqlite_path=tmp_path / "graph.sqlite")) as saver:
        graph = build_graph(
            BidState,
            {"flaky": flaky},
            [(START, "flaky"), ("flaky", END)],
            checkpointer=saver,
            retry=RetryConfig(max_attempts=3, initial_seconds=0.5),
            sleep_fn=lambda _s: None,
        )
        start(graph, BidState(run_id="run-8", project="ri05", started_at=AT))

    assert len(recorder.observations) == 3, "one observation per attempt, not one per node"
    assert [type(o.error).__name__ for o in recorder.observations] == [
        "TimeoutError",
        "TimeoutError",
        "NoneType",
    ]
