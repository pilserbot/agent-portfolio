"""Unit tests for the LangGraph foundation, on the SQLite fallback.

Every test runs against a file-backed SQLite checkpoint with no network and no environment
variables. The resume tests deliberately throw away the graph object and the connection and
rebuild both, which is what a run resuming days later in a fresh process actually does.

Deliberately does not: exercise Postgres. That is tests/integration/test_graph_postgres.py.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field, ValidationError

from spine.contracts import AgentRun
from spine.graph import (
    END,
    START,
    CheckpointConfig,
    GraphState,
    HumanDecision,
    NodeFn,
    RetryConfig,
    build_graph,
    human_review,
    open_checkpointer,
    pending_review,
    resume,
    run_record,
    start,
    to_agent_run,
)

AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


class BidState(GraphState):
    """A stand-in for a project's own state schema."""

    visited: list[str] = Field(default_factory=list)
    quote_usd: float | None = None


def a_state(**overrides: object) -> BidState:
    values: dict[str, object] = {
        "run_id": "run-1",
        "project": "ri05",
        "started_at": AT,
        **overrides,
    }
    return BidState(**values)


def visit(label: str) -> NodeFn:
    """Build a trivial node that appends its label."""

    def node(state: BidState) -> dict[str, object]:
        return {"visited": [*state.visited, label]}

    return node


def a_decision(item_id: str = "item-1", verdict: str = "approved") -> HumanDecision:
    return HumanDecision(
        item_id=item_id, verdict=verdict, decided_by="buyer@example.com", decided_at=AT
    )


def config_for(tmp_path: Path) -> CheckpointConfig:
    return CheckpointConfig(sqlite_path=tmp_path / "graph.sqlite")


def three_node_graph(checkpointer: BaseCheckpointSaver, **kwargs: object) -> CompiledStateGraph:
    return build_graph(
        BidState,
        {"extract": visit("extract"), "price": visit("price"), "draft": visit("draft")},
        [(START, "extract"), ("extract", "price"), ("price", "draft"), ("draft", END)],
        checkpointer=checkpointer,
        **kwargs,
    )


# --- a graph runs to completion --------------------------------------------------------


def test_a_three_node_graph_runs_to_completion(tmp_path: Path) -> None:
    with open_checkpointer(config_for(tmp_path)) as saver:
        final = start(three_node_graph(saver), a_state())

    assert final.visited == ["extract", "price", "draft"]
    assert [step.step_name for step in final.steps] == ["extract", "price", "draft"]
    assert all(step.status == "ok" for step in final.steps)


def test_the_returned_state_is_the_projects_own_schema(tmp_path: Path) -> None:
    with open_checkpointer(config_for(tmp_path)) as saver:
        final = start(three_node_graph(saver), a_state())

    assert isinstance(final, BidState)


def test_every_node_is_traced_without_the_node_doing_anything(tmp_path: Path) -> None:
    with open_checkpointer(config_for(tmp_path)) as saver:
        final = start(three_node_graph(saver), a_state())

    for step in final.steps:
        assert step.started_at <= step.finished_at
        assert step.metadata["attempts"] == 1


def test_the_run_record_is_assembled_from_the_traces(tmp_path: Path) -> None:
    with open_checkpointer(config_for(tmp_path)) as saver:
        final = start(three_node_graph(saver), a_state())

    record = to_agent_run(final, finished_at=AT)

    assert isinstance(record, AgentRun)
    assert record.status == "completed"
    assert len(record.steps) == 3
    assert record.total_cost_usd == 0


def test_a_state_schema_that_does_not_extend_graphstate_is_rejected(tmp_path: Path) -> None:
    from pydantic import BaseModel

    class Loose(BaseModel):
        x: int = 0

    with open_checkpointer(config_for(tmp_path)) as saver:
        with pytest.raises(TypeError, match="GraphState"):
            build_graph(Loose, {}, [], checkpointer=saver)


# --- human review interrupt ------------------------------------------------------------


def review_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    return build_graph(
        BidState,
        {
            "extract": visit("extract"),
            "human_review": human_review(threshold=0.7),
            "draft": visit("draft"),
        },
        [
            (START, "extract"),
            ("extract", "human_review"),
            ("human_review", "draft"),
            ("draft", END),
        ],
        checkpointer=checkpointer,
    )


def test_low_confidence_interrupts_and_persists_state(tmp_path: Path) -> None:
    with open_checkpointer(config_for(tmp_path)) as saver:
        graph = review_graph(saver)

        paused = start(graph, a_state(confidence=0.2, item_ids=["item-1"]))

        assert paused.visited == ["extract"], "the run stopped before drafting"
        request = pending_review(graph, "run-1")
        assert request is not None
        assert request.confidence == 0.2
        assert request.threshold == 0.7
        assert request.item_ids == ["item-1"]


def test_high_confidence_passes_straight_through(tmp_path: Path) -> None:
    with open_checkpointer(config_for(tmp_path)) as saver:
        graph = review_graph(saver)

        final = start(graph, a_state(confidence=0.9))

        assert final.visited == ["extract", "draft"]
        assert pending_review(graph, "run-1") is None


def test_a_paused_run_is_recorded_as_awaiting_review(tmp_path: Path) -> None:
    with open_checkpointer(config_for(tmp_path)) as saver:
        graph = review_graph(saver)
        start(graph, a_state(confidence=0.2))

        assert run_record(graph, "run-1").status == "awaiting_review"


# --- resume in a fresh process ---------------------------------------------------------


def test_resume_continues_in_a_new_process_equivalent(tmp_path: Path) -> None:
    settings = config_for(tmp_path)

    # Process one: run until the graph pauses, then drop every object it created.
    with open_checkpointer(settings) as saver:
        start(review_graph(saver), a_state(confidence=0.2, item_ids=["item-1"]))

    # Process two: a new connection, a new saver and a newly built graph. Nothing but the
    # checkpoint on disk survives from process one.
    with open_checkpointer(settings) as saver:
        graph = review_graph(saver)

        assert pending_review(graph, "run-1") is not None, "the pause survived the restart"
        final = resume(graph, "run-1", [a_decision()])

    assert final.visited == ["extract", "draft"], "the run continued past the interrupt"
    assert [d.item_id for d in final.decisions] == ["item-1"]
    assert final.decisions[0].verdict == "approved"
    assert final.decisions[0].decided_by == "buyer@example.com"


def test_the_traces_from_before_the_pause_survive_the_restart(tmp_path: Path) -> None:
    settings = config_for(tmp_path)
    with open_checkpointer(settings) as saver:
        start(review_graph(saver), a_state(confidence=0.2))

    with open_checkpointer(settings) as saver:
        graph = review_graph(saver)
        final = resume(graph, "run-1", [a_decision()])

    assert [step.step_name for step in final.steps] == ["extract", "human_review", "draft"]
    assert final.started_at == AT, "the tz-aware timestamp round-tripped through the checkpoint"
    assert final.project == "ri05"


def test_two_runs_on_one_checkpoint_file_stay_separate(tmp_path: Path) -> None:
    settings = config_for(tmp_path)
    with open_checkpointer(settings) as saver:
        graph = review_graph(saver)
        start(graph, a_state(run_id="run-a", confidence=0.2))
        start(graph, a_state(run_id="run-b", confidence=0.9))

        assert pending_review(graph, "run-a") is not None
        assert pending_review(graph, "run-b") is None


def test_a_resumed_run_records_as_completed(tmp_path: Path) -> None:
    settings = config_for(tmp_path)
    with open_checkpointer(settings) as saver:
        start(review_graph(saver), a_state(confidence=0.2))
    with open_checkpointer(settings) as saver:
        graph = review_graph(saver)
        resume(graph, "run-1", [a_decision()])

        assert run_record(graph, "run-1").status == "completed"


# --- retries and failure ----------------------------------------------------------------


class SimulatedTimeout(TimeoutError):
    """A transient node failure: the kind a second attempt could plausibly survive."""


def flaky(
    fail_times: int, counter: dict[str, int], error: type[Exception] = SimulatedTimeout
) -> NodeFn:
    """A node that fails a set number of times before succeeding."""

    def node(state: BidState) -> dict[str, object]:
        counter["calls"] = counter.get("calls", 0) + 1
        if counter["calls"] <= fail_times:
            raise error("transient")
        return {"visited": [*state.visited, "flaky"]}

    return node


def failing_graph(
    checkpointer: BaseCheckpointSaver, node: NodeFn, waits: list[float]
) -> CompiledStateGraph:
    return build_graph(
        BidState,
        {"before": visit("before"), "flaky": node, "after": visit("after")},
        [(START, "before"), ("before", "flaky"), ("flaky", "after"), ("after", END)],
        checkpointer=checkpointer,
        retry=RetryConfig(max_attempts=3, initial_seconds=0.5),
        sleep_fn=waits.append,
    )


def test_a_node_retries_and_then_succeeds(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    waits: list[float] = []
    with open_checkpointer(config_for(tmp_path)) as saver:
        final = start(failing_graph(saver, flaky(2, counter), waits), a_state())

    assert counter["calls"] == 3
    assert final.visited == ["before", "flaky", "after"]
    flaky_step = next(s for s in final.steps if s.step_name == "flaky")
    assert flaky_step.status == "ok"
    assert flaky_step.metadata["attempts"] == 3
    assert len(waits) == 2


def test_a_node_that_never_succeeds_records_failure_without_aborting(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    waits: list[float] = []
    with open_checkpointer(config_for(tmp_path)) as saver:
        final = start(failing_graph(saver, flaky(99, counter), waits), a_state())

    assert counter["calls"] == 3, "three attempts, then it gives up"
    flaky_step = next(s for s in final.steps if s.step_name == "flaky")
    assert flaky_step.status == "failed"
    assert "SimulatedTimeout: transient" in (flaky_step.error or "")
    # The run kept going: the node after the failure still ran.
    assert final.visited == ["before", "after"]
    assert [s.step_name for s in final.steps] == ["before", "flaky", "after"]


def test_a_failed_step_makes_the_run_completed_with_errors(tmp_path: Path) -> None:
    with open_checkpointer(config_for(tmp_path)) as saver:
        final = start(failing_graph(saver, flaky(99, {}), []), a_state())

    record = to_agent_run(final, finished_at=AT)

    assert record.status == "completed_with_errors"
    assert record.has_failed_steps
    assert [step.step_name for step in record.failed_steps] == ["flaky"]


def test_a_run_with_a_failed_step_cannot_be_marked_completed(tmp_path: Path) -> None:
    with open_checkpointer(config_for(tmp_path)) as saver:
        final = start(failing_graph(saver, flaky(99, {}), []), a_state())

    with pytest.raises(ValidationError, match="completed_with_errors"):
        to_agent_run(final, finished_at=AT, status="completed")


def test_backoff_waits_stay_inside_a_growing_window(tmp_path: Path) -> None:
    waits: list[float] = []
    with open_checkpointer(config_for(tmp_path)) as saver:
        start(failing_graph(saver, flaky(99, {}), waits), a_state())

    assert len(waits) == 2
    for attempt, wait in enumerate(waits):
        assert 0.0 <= wait <= 0.5 * 2**attempt


def test_an_interrupt_is_never_swallowed_by_the_retry_wrapper(tmp_path: Path) -> None:
    # human_review interrupts by raising; if the retry wrapper caught that as a failure,
    # the run would sail past the pause instead of waiting for a human.
    with open_checkpointer(config_for(tmp_path)) as saver:
        graph = build_graph(
            BidState,
            {"review": human_review(threshold=0.7), "draft": visit("draft")},
            [(START, "review"), ("review", "draft"), ("draft", END)],
            checkpointer=saver,
            retry=RetryConfig(max_attempts=3, initial_seconds=0.5),
            sleep_fn=lambda _s: None,
        )

        paused = start(graph, a_state(confidence=0.1))

        assert paused.visited == []
        assert pending_review(graph, "run-1") is not None
        assert [s.step_name for s in paused.steps] == []


# --- checkpoint configuration ------------------------------------------------------------


def test_sqlite_is_the_fallback_when_database_url_is_unset() -> None:
    settings = CheckpointConfig.from_env({})

    assert settings.database_url is None
    assert settings.sqlite_path.suffix == ".sqlite"


def test_postgres_is_selected_when_database_url_is_set() -> None:
    settings = CheckpointConfig.from_env({"DATABASE_URL": "postgresql://localhost/x"})

    assert settings.database_url == "postgresql://localhost/x"


def test_an_empty_database_url_falls_back_to_sqlite() -> None:
    assert CheckpointConfig.from_env({"DATABASE_URL": ""}).database_url is None


def test_the_checkpoint_file_is_created_on_demand(tmp_path: Path) -> None:
    settings = CheckpointConfig(sqlite_path=tmp_path / "deep" / "nested" / "graph.sqlite")

    with open_checkpointer(settings) as saver:
        start(three_node_graph(saver), a_state())

    assert settings.sqlite_path.exists()


def test_nodes_receive_the_projects_own_state_not_the_base(tmp_path: Path) -> None:
    # Regression: LangGraph infers a node's input schema from the wrapped function's
    # annotation. Without an explicit input_schema every node was handed a bare GraphState,
    # the project's fields vanished, and the AttributeError surfaced only as a step traced
    # `failed` — a silent wrong answer rather than a crash.
    seen: list[type] = []

    def inspecting_node(state: BidState) -> dict[str, object]:
        seen.append(type(state))
        return {"visited": [*state.visited, "seen"], "quote_usd": 12.5}

    with open_checkpointer(config_for(tmp_path)) as saver:
        graph = build_graph(
            BidState,
            {"inspect": inspecting_node},
            [(START, "inspect"), ("inspect", END)],
            checkpointer=saver,
        )
        final = start(graph, a_state())

    assert seen == [BidState], "the node must receive the project's schema"
    assert final.visited == ["seen"]
    assert final.quote_usd == 12.5
    assert all(step.status == "ok" for step in final.steps)


# --- retry classification ----------------------------------------------------------------


def raising(error: Exception, counter: dict[str, int]) -> NodeFn:
    """A node that always raises the given error, counting its attempts."""

    def node(state: BidState) -> dict[str, object]:
        counter["calls"] = counter.get("calls", 0) + 1
        raise error

    return node


def graph_with(
    checkpointer: BaseCheckpointSaver, node: NodeFn, waits: list[float], **kw: object
) -> CompiledStateGraph:
    return build_graph(
        BidState,
        {"before": visit("before"), "boom": node, "after": visit("after")},
        [(START, "before"), ("before", "boom"), ("boom", "after"), ("after", END)],
        checkpointer=checkpointer,
        retry=RetryConfig(max_attempts=3, initial_seconds=0.5),
        sleep_fn=waits.append,
        **kw,
    )


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("no such attribute"),
        TypeError("wrong type"),
        KeyError("missing"),
        IndexError("out of range"),
        NameError("undefined"),
        ImportError("no module"),
        AssertionError("invariant broken"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_a_programming_error_propagates_on_the_first_attempt(
    tmp_path: Path, error: Exception
) -> None:
    counter: dict[str, int] = {}
    waits: list[float] = []
    with open_checkpointer(config_for(tmp_path)) as saver:
        graph = graph_with(saver, raising(error, counter), waits)

        with pytest.raises(type(error)):
            start(graph, a_state())

        assert counter["calls"] == 1, "a bug must not be retried"
        assert waits == [], "and must not sleep between attempts"

        # No trace was filed for it, and the run did not sail on to the next node.
        state = _state_after_crash(graph)
        assert [s.step_name for s in state.steps] == ["before"]
        assert all(s.status == "ok" for s in state.steps)


def _state_after_crash(graph: CompiledStateGraph) -> BidState:
    """The state a crashed run left in its checkpoint."""
    snapshot = graph.get_state({"configurable": {"thread_id": "run-1"}})
    return BidState.model_validate(snapshot.values)


def test_a_simulated_timeout_still_retries(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    waits: list[float] = []
    with open_checkpointer(config_for(tmp_path)) as saver:
        graph = graph_with(saver, raising(SimulatedTimeout("slow"), counter), waits)
        final = start(graph, a_state())

    assert counter["calls"] == 3, "a timeout is worth another attempt"
    assert len(waits) == 2
    boom = next(s for s in final.steps if s.step_name == "boom")
    assert boom.status == "failed"
    assert boom.metadata["attempts"] == 3
    assert boom.metadata["retryable"] is True
    assert final.visited == ["before", "after"], "the run carried on"


def test_a_connection_error_still_retries(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    with open_checkpointer(config_for(tmp_path)) as saver:
        start(graph_with(saver, raising(ConnectionResetError("reset"), counter), []), a_state())

    assert counter["calls"] == 3


def test_a_rate_limit_still_retries(tmp_path: Path) -> None:
    import litellm

    counter: dict[str, int] = {}
    error = litellm.RateLimitError(message="slow down", llm_provider="anthropic", model="m")
    with open_checkpointer(config_for(tmp_path)) as saver:
        start(graph_with(saver, raising(error, counter), []), a_state())

    assert counter["calls"] == 3, "a provider rate limit is transient"


def test_an_unclassified_error_is_traced_without_retrying(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    waits: list[float] = []
    with open_checkpointer(config_for(tmp_path)) as saver:
        graph = graph_with(saver, raising(ValueError("bad data"), counter), waits)
        final = start(graph, a_state())

    assert counter["calls"] == 1, "nothing suggests a second attempt would differ"
    assert waits == []
    boom = next(s for s in final.steps if s.step_name == "boom")
    assert boom.status == "failed"
    assert boom.metadata["attempts"] == 1
    assert boom.metadata["retryable"] is False
    assert final.visited == ["before", "after"], "but the run still carried on"


def test_retry_on_opts_an_extra_type_into_retrying(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    with open_checkpointer(config_for(tmp_path)) as saver:
        start(
            graph_with(saver, raising(ValueError("flaky"), counter), [], retry_on=(ValueError,)),
            a_state(),
        )

    assert counter["calls"] == 3


def test_retry_on_cannot_re_enable_retrying_a_programming_error(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    with open_checkpointer(config_for(tmp_path)) as saver:
        graph = graph_with(saver, raising(KeyError("missing"), counter), [], retry_on=(KeyError,))

        with pytest.raises(KeyError):
            start(graph, a_state())

    assert counter["calls"] == 1, "NEVER_RETRY wins over an explicit retry_on"
