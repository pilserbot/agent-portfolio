"""The graph foundation against a real Postgres checkpointer.

Needs DATABASE_URL and a reachable server, so it is marked `integration` and excluded from
`make test-fast`. It exists to cover what SQLite cannot prove: that `open_checkpointer`
selects Postgres, that `setup()` creates its tables, and above all that a run paused for a
human survives being resumed by a process holding a brand-new connection — the bid-waiting-
on-a-vendor-quote case.

Deliberately does not: assert timing or durability beyond one restart, or clean up the
checkpoint tables. Each test uses a unique run id, so runs never collide.
"""

import os
import uuid
from datetime import UTC, datetime

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from spine.graph import (
    END,
    START,
    CheckpointConfig,
    GraphState,
    HumanDecision,
    NodeFn,
    build_graph,
    human_review,
    open_checkpointer,
    pending_review,
    resume,
    run_record,
    start,
)

DATABASE_URL = os.environ.get("DATABASE_URL", "")


class BidState(GraphState):
    """A stand-in for a project's own state schema."""

    visited: list[str] = Field(default_factory=list)


def visit(label: str) -> NodeFn:
    """Build a trivial node that appends its label."""

    def node(state: BidState) -> dict[str, object]:
        return {"visited": [*state.visited, label]}

    return node


def review_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """A graph that pauses for a human between extraction and drafting."""
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


@pytest.mark.integration
@pytest.mark.skipif(not DATABASE_URL, reason="needs DATABASE_URL")
def test_a_run_pauses_and_resumes_across_a_fresh_postgres_connection() -> None:
    settings = CheckpointConfig(database_url=DATABASE_URL)
    run_id = f"itest-{uuid.uuid4()}"
    started_at = datetime.now(UTC)

    # Process one: run until the graph pauses on low confidence, then drop everything.
    with open_checkpointer(settings) as saver:
        paused = start(
            review_graph(saver),
            BidState(
                run_id=run_id,
                project="ri05",
                started_at=started_at,
                confidence=0.2,
                item_ids=["item-1"],
            ),
        )
        assert paused.visited == ["extract"]

    # Process two: a new connection, a new saver, a newly built graph. Only the rows in
    # Postgres carry over — this is the days-later resume.
    with open_checkpointer(settings) as saver:
        graph = review_graph(saver)

        request = pending_review(graph, run_id)
        assert request is not None, "the pause must survive in Postgres"
        assert request.confidence == 0.2
        assert request.item_ids == ["item-1"]
        assert run_record(graph, run_id).status == "awaiting_review"

        final = resume(
            graph,
            run_id,
            [
                HumanDecision(
                    item_id="item-1",
                    verdict="approved",
                    decided_by="buyer@example.com",
                    decided_at=datetime.now(UTC),
                )
            ],
        )

        assert final.visited == ["extract", "draft"]
        assert [d.verdict for d in final.decisions] == ["approved"]
        assert [s.step_name for s in final.steps] == ["extract", "human_review", "draft"]
        assert final.started_at == started_at, "tz-aware timestamps round-trip"
        assert run_record(graph, run_id).status == "completed"
