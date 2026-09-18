"""Live extraction against Kessler Point: a smoke test on every run, the full pass on demand.

Needs ANTHROPIC_API_KEY and a network, so everything here is marked `integration` and
excluded from `make test-fast`.

**Two tests, because they buy different things.**

- The **smoke test** runs on every pull request. One page, one structured call, a few cents:
  it proves the wire is connected — that the key resolves, that the router reaches the
  provider, and that a real response validates into the schema `req_core` asked for.
- The **full pass** reads all 301 clauses of Documents 4, 5 and 6 and measures: the anchor
  assertion on output nobody scripted, the gate against the Compliance Matrix, and what a
  real pass costs. It runs only when the pull request carries the `full-eval` label or on a
  push to `main`, through the same mechanism `ci.yml` already uses to decide between the
  full gold set and a 20-item sample.

Paying a quarter and nine minutes on every unrelated pull request buys neither. A smoke test
that ran the full corpus would not prove anything the one call does not, and a full pass on
every push would measure the same number over and over at everyone else's expense.

**The measurement is persisted before it is reported.** An earlier version of this test made
sixteen calls, spent real money, and then threw the result away because the reporting object
it was building raised on an unrelated field. So `persist_ledger` runs in a `finally` the
moment extraction returns — before any assertion, before any `AgentRun` or `CostReport`
exists — writing a JSON snapshot to a durable directory and printing the per-purpose token
totals and the cost to stdout. It reads the router's ledger directly with plain arithmetic
and constructs nothing that could fail. A partial run that died on call nine still records
the eight calls it paid for. A measurement that only survives the happy path is not a
measurement.

Each test writes its ledger to a temporary path, so a run never walks the project's real
daily spend cap toward its limit.

**What the prompt cache does on this path, measured in two halves rather than argued.**
An earlier version of this docstring claimed caching could not be engaged here and could not
be made to: `spine.router._build_messages` marks the whole prompt as one cacheable block,
the prompt is instructions **plus** that page's clause spans, so the block differs on every
call and nothing can ever hit it. Splitting the counter refuted that. Over the 14-call pass
the router recorded **57,066 tokens written and 47,473 read** — reads nearly equal to writes,
not the zero the argument required.

What was wrong was the unit. The shared *instruction* prefix genuinely cannot cache: it is
about 300 tokens and Claude Sonnet 5 will not cache a prefix shorter than 1024, silently
rather than with an error. But the block that is marked is the whole per-page prompt, and
that block IS re-sent verbatim whenever the same page goes to the model twice — which
happens on an `instructor` validation retry inside one `structured()` call, and again when a
second pass runs over the same corpus inside the five-minute TTL. In this run the first three
calls of the full pass recorded reads with **zero** writes, having been preceded by the
detection test and the smoke test sending those same pages.

So the reads are real, and they are not the saving the design was hoping for: they are the
0.1x discount on text that should not have been re-sent at all. The expensive thing on this
path is the re-sending, not the cache placement. Restructuring the prompt so the stable half
caches would still not qualify on length, and that half is 300 tokens out of ~5,000 a page.

The figures above are reported before they are interpreted, which is the whole point of
splitting the field: "caching is engaged" was read off a single counter that added a 1.25x
surcharge to a 0.1x discount, and no such counter can support a conclusion in either
direction. `describe_cache()` on the snapshot now names which of the two happened.

`per_call` exists because the totals hid the thing worth seeing, and the split rows settled
it: on every page whose prompt exceeds ~5k tokens the call records both a write and a read of
roughly the page's own size, and on every short page it records a write and no read. That is
the signature of the long pages being sent to the model more than once per `structured()`
call, which is what the ~1.4k of prompt this module builds against ~10.8k measured was
always pointing at. Summed figures cannot distinguish "every page cost what its text costs"
from "a few pages were sent more than once"; these rows can.

Deliberately does not: assert what the model said about any particular clause. How a clause
should be split is a judgement, and pinning one here would pin today's answer as the
contract. What is asserted is everything deterministic around it — the anchors, the lineage,
the scope, and that nothing was invented.
"""

import os
import uuid
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from ledger import LedgerSnapshot, artifact_dir, persist_ledger
from live_extraction import EXPECTED_EXTRACTION_CALLS, LiveExtraction

from req_core.anchors import verify
from req_core.clauses import clauses_on_page
from req_core.extraction import EXTRACT_PURPOSE, PageReading, build_prompt
from ri05_tender.extract.config import ITB_2_1_POLICY, KESSLER_POINT_CLAUSE_STYLE
from ri05_tender.extract.gate import (
    MATRIX_DOCUMENT_ID,
    compare,
    extraction_corpus,
    render_markdown,
    verify_scope,
)
from ri05_tender.tender.loader import load_tender
from spine.contracts import AgentRun, StepTrace
from spine.kpi import cost_report
from spine.router import Router, RouterConfig

if TYPE_CHECKING:
    # Names this module only ever writes in an annotation, so they cost nothing at runtime.
    # Every use of them below is quoted, one at a time, rather than reached by putting
    # `from __future__ import annotations` at the top of the file: that would stringify the
    # Pydantic field annotations too, leaving `Decimal` and `datetime` as ForwardRefs that
    # pydantic resolves through `sys.modules` — which works under pytest's importer and
    # fails under a loader that does not register the module. This file exists to stop a
    # paid-for measurement being lost to a construction that raised; it is not the place to
    # take on a second one.
    from req_core.anchors import AnchorReport
    from req_core.clauses import Clause
    from req_core.contracts import ExtractionResult
    from req_core.corpus import CorpusPage, SourceCorpus
    from ri05_tender.extract.gate import MatrixComparison
    from spine.kpi import CostReport

KESSLER_POINT = Path("data/tenders/kessler_point")

SMOKE_PURPOSE = "ri05.extract_smoke"

# What the deterministic layer finds, and what the package's own index claims for these three
# documents. Pinned so a live run that reads fewer clauses fails here rather than quietly
# reporting a smaller, better-looking gate.
EXPECTED_CLAUSES = 301

# The estimate the full pass was approved against: ~18k prompt and ~21k completion tokens
# over 16 calls at Sonnet 5's $2/$10 per million, so roughly $0.25. An order of magnitude
# above that is not a surprise to absorb — it is a bug in the batching, and it should fail.
COST_CEILING_USD = Decimal("3.00")

# CI sets this from the same expression that chooses the full gold set over a 20-item
# sample: a push to main, or a pull request labelled `full-eval`. Absent means smoke only.
FULL_EVAL_ENV = "RI05_FULL_EVAL"

needs_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"
)
needs_full_eval = pytest.mark.skipif(
    os.environ.get(FULL_EVAL_ENV, "").strip().lower() not in {"1", "true", "yes"},
    reason=(
        f"the full 301-clause pass runs only on a push to main or a pull request labelled "
        f"`full-eval` (set {FULL_EVAL_ENV}); the smoke test covers every other run"
    ),
)


def a_router(tmp_path: Path) -> Router:
    """A router whose ledger goes to a temporary path, never the project's real one."""
    return Router(
        RouterConfig.from_env().model_copy(update={"ledger_path": tmp_path / "ledger.json"})
    )


# --- the smoke test: one page, one call, every pull request ------------------------------


@pytest.mark.integration
@needs_key
def test_one_live_structured_call_returns_a_valid_typed_result(tmp_path: Path) -> None:
    """The wire is connected: the key resolves, the provider answers, the schema validates.

    One page and one call, chosen as the page of the corpus carrying the fewest clauses, so
    this costs cents and takes seconds. It proves nothing about extraction quality and is not
    meant to — that is what the full pass measures.
    """
    corpus = extraction_corpus(load_tender(KESSLER_POINT))
    page, clauses = _cheapest_page(corpus)
    router = a_router(tmp_path)

    try:
        reading, call = router.structured(
            build_prompt(clauses),
            PageReading,
            purpose=SMOKE_PURPOSE,
            tier="large",
        )
    finally:
        snapshot = persist_ledger(router, label="smoke")

    # A real response, validated into the schema req_core asked for.
    assert isinstance(reading, PageReading)
    assert reading.clauses, "the model returned no reading for a page that has clauses"

    # Every identifier it echoed is one that was really on the page. This is the smallest
    # honest check that the answer is about the input rather than about nothing.
    asked = {clause.identifier for clause in clauses}
    assert {item.identifier for item in reading.clauses} <= asked

    # The call was really made and really priced.
    assert call.purpose == SMOKE_PURPOSE
    assert call.was_billed
    assert call.prompt_tokens > 0 and call.completion_tokens > 0
    assert call.cost_usd > Decimal("0")

    assert snapshot.calls == 1
    assert snapshot.by_purpose[0].purpose == SMOKE_PURPOSE
    assert snapshot.cost_usd == call.cost_usd
    print(
        f"Smoke: 1 call over page {page.document_id} p{page.page_number} "
        f"({len(clauses)} clause(s)), ${snapshot.cost_usd:.6f}."
    )


def _cheapest_page(corpus: "SourceCorpus") -> "tuple[CorpusPage, list[Clause]]":
    """The page with the fewest clauses, so one call is as small as this tender allows.

    Ties broken by document then page, so the smoke test asks about the same page every run
    and a change in what it costs means the tender changed, not the pick.
    """
    candidates = [
        (page, found)
        for document in corpus.documents
        for page in document.pages
        if (found := clauses_on_page(page, style=KESSLER_POINT_CLAUSE_STYLE))
    ]
    assert candidates, "the corpus has no page carrying a clause"
    return min(
        candidates, key=lambda pair: (len(pair[1]), pair[0].document_id, pair[0].page_number)
    )


# --- the full pass: all 301 clauses, on the label or on main ------------------------------


@pytest.mark.integration
@needs_key
@needs_full_eval
def test_a_real_extraction_pass_anchors_every_requirement_and_clears_the_gate(
    live_extraction: LiveExtraction,
) -> None:
    """The gate, over the session's single extraction pass.

    The pass itself lives in the `live_extraction` fixture because the findings run needs
    exactly the same result, and extracting the same corpus twice in one job cost 14 calls
    and about ten minutes for an answer already in hand. The fixture persists its ledger
    before this test sees anything, so every assertion below is free to fail without costing
    the measurement.
    """
    package = live_extraction.package
    corpus = live_extraction.corpus
    result = live_extraction.result
    snapshot = live_extraction.snapshot
    started, finished = live_extraction.started_at, live_extraction.finished_at

    # The corpus physically cannot hold the matrix — `SourceCorpus` refuses it — so this is a
    # restatement of a guarantee already made, placed here because the live run is the one
    # where it would matter most.
    assert MATRIX_DOCUMENT_ID not in corpus.document_ids
    assert corpus.withheld == {MATRIX_DOCUMENT_ID}

    assert snapshot.calls == EXPECTED_EXTRACTION_CALLS, (
        f"extraction made {snapshot.calls} call(s) against {EXPECTED_EXTRACTION_CALLS} "
        f"expected — one per page carrying a clause. A different number means the batching "
        f"changed, and the cost of every labelled run changed with it."
    )

    verify_scope(result, corpus)

    # --- 1. the hard assertion, on output nobody scripted --------------------------------
    anchors = verify(result.requirements, corpus)
    assert anchors.ok, anchors.describe()
    assert anchors.checked == len(result.requirements)

    # --- what the deterministic layer must still be doing ---------------------------------
    assert result.clauses_read == EXPECTED_CLAUSES
    assert len(result.clause_ids) == EXPECTED_CLAUSES
    assert result.policy_name == ITB_2_1_POLICY.name
    # Every child names a parent that is really in the set, and no id was invented.
    by_id = {requirement.requirement_id: requirement for requirement in result.requirements}
    for requirement in result.requirements:
        if requirement.parent_id is not None:
            assert requirement.parent_id in by_id
            assert requirement.requirement_id.startswith(requirement.parent_id)
    assert len(by_id) == len(result.requirements), "requirement ids must be unique"

    # --- 2. the gate ---------------------------------------------------------------------
    comparison = compare(result, package)

    # --- 3. the cost, cross-checked against the snapshot already taken --------------------
    # `status` here is a RunStatus ("completed"), not a StepStatus ("ok"). The two enums are
    # deliberately disjoint; see the note on both types in spine.contracts.
    run = AgentRun(
        run_id=f"extract-{uuid.uuid4().hex[:8]}",
        project="ri05",
        started_at=started,
        finished_at=finished,
        steps=[
            StepTrace(
                step_name="req_core.extract",
                started_at=started,
                finished_at=finished,
                status="ok",
                model_calls=list(live_extraction.calls),
            )
        ],
        status="completed",
    )
    costs = cost_report(run)

    assert costs.basis is not None, costs.refusal
    assert costs.basis.is_measured, "a live pass must not be reporting replayed calls"
    assert costs.total_calls == snapshot.calls > 0
    assert {usage.purpose for usage in costs.by_purpose} == {EXTRACT_PURPOSE}
    # The two routes to the same number must agree, or one of them is wrong.
    assert costs.basis.total_usd == snapshot.cost_usd

    report = _render(result, comparison, run, costs, anchors, snapshot)
    print(report)
    (artifact_dir() / f"extraction_gate_{run.run_id}.md").write_text(report, encoding="utf-8")

    assert "extraction gate" in report
    assert costs.basis.total_usd < COST_CEILING_USD, (
        f"the pass cost ${costs.basis.total_usd} against a ceiling of ${COST_CEILING_USD}. "
        f"The estimate was ~$0.25 over 16 calls; an order of magnitude above that is a "
        f"batching bug, not a surprise to absorb."
    )


def _render(
    result: "ExtractionResult",
    comparison: "MatrixComparison",
    run: AgentRun,
    costs: "CostReport",
    anchors: "AnchorReport",
    snapshot: LedgerSnapshot,
) -> str:
    """The whole run as one block, for a human reading the test output."""
    return "\n".join(
        [
            "",
            render_markdown(comparison, result),
            "### Anchors",
            "",
            anchors.describe(),
            "",
            f"### Cost — run `{run.run_id}`",
            snapshot.describe(),
        ]
    )
