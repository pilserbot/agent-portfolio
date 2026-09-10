"""The LangGraph foundation every project in this portfolio builds its run on.

Provides a typed state base, a factory that compiles a graph from nodes and edges, durable
checkpointing (Postgres when DATABASE_URL is set, SQLite otherwise), a reusable human
review interrupt, and a retry wrapper that records a failed step instead of killing the
run. Every node is instrumented automatically, so an AgentRun can be assembled from the
state at any point.

The design target is a run that pauses for days — a bid waiting on a vendor quote — and
resumes in a process that shares no memory with the one that started it. Everything needed
to resume therefore lives in the checkpoint: the step traces, the human decisions and the
pending review request are all fields of the state, not attributes of some live object.

Deliberately does not: decide anything. No node here judges, scores or ranks; the graph
moves typed state between steps a project supplies. It also does not store the graph's
code — a resuming process must rebuild the same graph and hand it to `resume`, because a
checkpoint holds state, never behaviour.
"""

import operator
import random
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from spine.contracts import AgentRun, ModelCall, RunStatus, StepTrace

DEFAULT_SQLITE_PATH = Path(".checkpoints/graph.sqlite")
DEFAULT_REVIEW_THRESHOLD = 0.7

__all__ = [
    "END",
    "START",
    "CheckpointConfig",
    "GraphState",
    "HumanDecision",
    "NodeFn",
    "RetryConfig",
    "ReviewRequest",
    "build_graph",
    "human_review",
    "open_checkpointer",
    "pending_review",
    "resume",
    "run_record",
    "start",
    "to_agent_run",
    "with_retry",
]


class HumanDecision(BaseModel):
    """One reviewer's ruling on one item. A record of something that happened, so frozen."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    verdict: Literal["approved", "rejected", "amended"]
    decided_by: str
    decided_at: datetime
    note: str = ""


class ReviewRequest(BaseModel):
    """What the graph hands a human when it pauses, and what it persists while it waits."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    step_name: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    item_ids: list[str] = Field(default_factory=list)


class GraphState(BaseModel):
    """The state every project's schema extends.

    Not frozen: this is live working state that nodes advance, unlike the records in
    `spine.contracts`. What it carries is chosen so a checkpoint is sufficient to resume —
    the traces, the decisions and the open review request are all here rather than in some
    object that dies with the process.

    `steps` and `decisions` accumulate through a reducer, so parallel branches merge
    instead of overwriting each other.
    """

    run_id: str
    project: str
    started_at: datetime
    confidence: float | None = Field(
        default=None,
        description="Confidence in the work so far, 0..1. None means not yet assessed.",
    )
    steps: Annotated[list[StepTrace], operator.add] = Field(default_factory=list)
    decisions: Annotated[list[HumanDecision], operator.add] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)


class RetryConfig(BaseModel):
    """How many times to reattempt a node, and how long to wait between attempts."""

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=3, ge=1)
    initial_seconds: float = Field(default=0.5, gt=0)
    max_seconds: float = Field(default=30.0, gt=0)


class CheckpointConfig(BaseModel):
    """Where run state is durably stored between steps."""

    model_config = ConfigDict(frozen=True)

    database_url: str | None = None
    sqlite_path: Path = DEFAULT_SQLITE_PATH

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "CheckpointConfig":
        """Read the checkpoint destination from the environment.

        DATABASE_URL selects Postgres. With it unset the graph falls back to SQLite, so the
        sandbox and the unit suite work with no server and no network.
        """
        import os

        source = os.environ if env is None else env
        return cls(
            database_url=source.get("DATABASE_URL") or None,
            sqlite_path=Path(source.get("GRAPH_CHECKPOINT_PATH") or DEFAULT_SQLITE_PATH),
        )


# Types stored inside checkpointed state must be named explicitly, or a future LangGraph
# will refuse to deserialise them. A run that paused for days would then be unresumable
# after a routine upgrade, which is precisely the failure this foundation must not have.
CHECKPOINTED_TYPES: tuple[type, ...] = (
    StepTrace,
    ModelCall,
    HumanDecision,
    ReviewRequest,
)


def make_serde(extra_types: Sequence[type] = ()) -> JsonPlusSerializer:
    """Build a serializer that will still read today's checkpoints after an upgrade."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[*CHECKPOINTED_TYPES, *extra_types],
    )


@contextmanager
def open_checkpointer(
    config: CheckpointConfig | None = None,
    *,
    extra_types: Sequence[type] = (),
) -> Iterator[BaseCheckpointSaver]:
    """Open the configured checkpointer, creating its tables, and close it on exit.

    Postgres when DATABASE_URL is set; SQLite on a local file otherwise.
    """
    settings = config or CheckpointConfig.from_env()
    serde = make_serde(extra_types)

    if settings.database_url:
        # Imported lazily so the SQLite path needs no psycopg installed or reachable.
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection

        # TODO: verified against the documented usage but not exercised here; the live
        # coverage is tests/integration/test_graph_postgres.py.
        with Connection.connect(
            settings.database_url, autocommit=True, prepare_threshold=0
        ) as connection:
            saver = PostgresSaver(connection, serde=serde)
            saver.setup()
            yield saver
        return

    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(settings.sqlite_path), check_same_thread=False)
    try:
        saver = SqliteSaver(connection, serde=serde)
        saver.setup()
        yield saver
    finally:
        connection.close()


type NodeFn = Callable[..., Mapping[str, object] | None]


def _backoff_delay(attempt: int, retry: RetryConfig, rng: random.Random) -> float:
    """Full-jitter backoff: a random wait inside an exponentially growing window."""
    window = min(retry.initial_seconds * (2**attempt), retry.max_seconds)
    return rng.uniform(0.0, window)


def with_retry(
    fn: NodeFn,
    *,
    name: str,
    retry: RetryConfig | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    rng: random.Random | None = None,
) -> NodeFn:
    """Wrap a node so it retries, records a StepTrace, and never aborts the run.

    On success the step is traced `ok`. When every attempt fails the step is traced
    `failed` with the error, and the graph carries on to the next node — a step that could
    not be completed is a fact about the run, not a reason to lose the days of work behind
    it. A `GraphBubbleUp` (which is how an interrupt travels) is re-raised untouched.
    """
    policy = retry or RetryConfig()
    jitter = rng or random.Random()

    def node(state: GraphState) -> Mapping[str, object]:
        started_at = now_fn()
        last_error: Exception | None = None

        for attempt in range(policy.max_attempts):
            try:
                update = fn(state) or {}
            except GraphBubbleUp:
                # An interrupt is control flow, not a failure. Swallowing it here would
                # break human review outright.
                raise
            except Exception as error:  # noqa: BLE001 - a node may raise anything
                last_error = error
                if attempt < policy.max_attempts - 1:
                    sleep_fn(_backoff_delay(attempt, policy, jitter))
                continue
            trace = StepTrace(
                step_name=name,
                started_at=started_at,
                finished_at=now_fn(),
                status="ok",
                metadata={"attempts": attempt + 1},
            )
            return {**update, "steps": [trace]}

        trace = StepTrace(
            step_name=name,
            started_at=started_at,
            finished_at=now_fn(),
            status="failed",
            error=f"{type(last_error).__name__}: {last_error}",
            metadata={"attempts": policy.max_attempts},
        )
        return {"steps": [trace]}

    return node


def build_graph(
    state_schema: type[GraphState],
    nodes: Mapping[str, NodeFn],
    edges: Sequence[tuple[str, str]],
    *,
    checkpointer: BaseCheckpointSaver,
    retry: RetryConfig | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> CompiledStateGraph:
    """Compile a checkpointed graph whose every node is retried and traced.

    `state_schema` must extend `GraphState`, which is what guarantees a checkpoint carries
    the traces and decisions a fresh process needs. Use START and END in `edges` for the
    entry and exit.
    """
    if not (isinstance(state_schema, type) and issubclass(state_schema, GraphState)):
        raise TypeError(
            f"state_schema must be a GraphState subclass so that traces and decisions are "
            f"checkpointed; got {state_schema!r}."
        )

    builder: StateGraph = StateGraph(state_schema)
    for name, fn in nodes.items():
        # input_schema is essential, not decorative: without it LangGraph infers the node's
        # input from the wrapper's own annotation and hands nodes the GraphState base,
        # silently stripping every field the project added.
        builder.add_node(
            name,
            with_retry(fn, name=name, retry=retry, sleep_fn=sleep_fn, now_fn=now_fn),
            input_schema=state_schema,
        )
    for source, target in edges:
        builder.add_edge(source, target)
    return builder.compile(checkpointer=checkpointer)


def human_review(
    *,
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
    step_name: str = "human_review",
    reason: str = "confidence below threshold",
) -> NodeFn:
    """Build a node that pauses the run for a human when confidence is too low.

    Below the threshold the node raises an interrupt, which persists the state and returns
    control to the caller. The run stays paused indefinitely — days, if the answer takes
    days. `resume` feeds the decisions back, and they land in `state.decisions`.

    At or above the threshold the node is a no-op, so the same graph serves both paths.
    """

    def node(state: GraphState) -> Mapping[str, object]:
        confidence = state.confidence
        if confidence is not None and confidence >= threshold:
            return {}

        request = ReviewRequest(
            run_id=state.run_id,
            step_name=step_name,
            reason=reason,
            confidence=0.0 if confidence is None else confidence,
            threshold=threshold,
            item_ids=list(state.item_ids),
        )
        answer = interrupt(request.model_dump(mode="json"))
        decisions = [HumanDecision.model_validate(item) for item in answer]
        return {"decisions": decisions}

    return node


def _config(run_id: str) -> dict[str, dict[str, str]]:
    """The LangGraph config that binds an invocation to one run's checkpoint thread."""
    return {"configurable": {"thread_id": run_id}}


def _state_of[StateT: GraphState](graph: CompiledStateGraph, run_id: str) -> StateT:
    """Load a run's current state from the checkpoint, typed as the graph's schema."""
    snapshot = graph.get_state(_config(run_id))
    schema: type[StateT] = graph.builder.state_schema
    return schema.model_validate(snapshot.values)


def start[StateT: GraphState](graph: CompiledStateGraph, state: StateT) -> StateT:
    """Begin a run and return its state, whether it finished or paused for review."""
    graph.invoke(state, _config(state.run_id))
    return _state_of(graph, state.run_id)


def resume[StateT: GraphState](
    graph: CompiledStateGraph,
    run_id: str,
    decisions: Sequence[HumanDecision],
) -> StateT:
    """Continue a paused run with the human's decisions merged into its state.

    `graph` has to be supplied because a checkpoint stores state, not code: a fresh process
    rebuilds the same graph and passes it here. Everything else comes off the checkpoint.
    """
    payload = [decision.model_dump(mode="json") for decision in decisions]
    graph.invoke(Command(resume=payload), _config(run_id))
    return _state_of(graph, run_id)


def pending_review(graph: CompiledStateGraph, run_id: str) -> ReviewRequest | None:
    """The review a paused run is waiting on, or None when it is not waiting."""
    snapshot = graph.get_state(_config(run_id))
    for task in snapshot.tasks:
        for pending in task.interrupts:
            return ReviewRequest.model_validate(pending.value)
    return None


def run_record(
    graph: CompiledStateGraph,
    run_id: str,
    *,
    finished_at: datetime | None = None,
) -> AgentRun:
    """The run record for a checkpointed run, including whether it is awaiting review."""
    state = _state_of(graph, run_id)
    status: RunStatus | None = (
        "awaiting_review" if pending_review(graph, run_id) is not None else None
    )
    return to_agent_run(state, finished_at=finished_at, status=status)


def to_agent_run(
    state: GraphState,
    *,
    finished_at: datetime | None = None,
    status: RunStatus | None = None,
) -> AgentRun:
    """Assemble the immutable run record from the state accumulated so far.

    Status is derived when not given: `failed` if any step failed, else `completed`. A
    paused run cannot be recognised from state alone — the interrupt lives in the
    checkpoint, not the state — so use `run_record`, which can see it.
    """
    if status is None:
        status = "failed" if any(step.status == "failed" for step in state.steps) else "completed"
    return AgentRun(
        run_id=state.run_id,
        project=state.project,
        started_at=state.started_at,
        finished_at=finished_at or datetime.now(UTC),
        steps=list(state.steps),
        status=status,
    )
