"""The live findings run: extract, detect, score, against the real tender and a real model.

Needs ANTHROPIC_API_KEY and a network, and runs only on a pull request labelled `full-eval`
— the same `RI05_FULL_EVAL` gate as the extraction pass, set in `ci.yml` from the same
expression that chooses the full gold set over a sample. There is no
smoke-test half here: `test_ri05_extraction` already proves the wire is connected on every
pull request, and a second call proving it again would be paying twice for one fact.

**Cost.** 88 calls: 44 for extraction and 44 for claims, one of each per page that carries
a clause across the ten documents the detection scope reads. The two passes ask different
questions of the same pages, so neither can be skipped; the extraction half is handed in by
the session fixture, which is why the job pays 88 and not 132.

The widening from three documents to ten is what these numbers are. 28 calls over 301
clauses cost $1.72 and took 17.5 minutes; 88 calls over 691 clauses project to about $4.00
and 41 minutes, which fits inside the job's 60-minute limit with room. There is no
parallelism here on purpose: 41 minutes fits, and if a real run overruns the fix will be
made against a measurement rather than a projection.

The ledger is persisted the moment each pass returns, before any assertion and before any
reporting object exists — see `ledger.persist_ledger` and the reason it is written that way.

Deliberately does not: assert a recall figure. What the detectors find against this gold set
is a measurement, and pinning today's number as a contract would turn every future
improvement into a failing test. What is asserted is that the run completes, that what it
asserts is what gets scored, and that the abstention count is reported beside the recall.
"""

import os
from pathlib import Path

import pytest
from ledger import artifact_dir, persist_ledger
from live_extraction import EXPECTED_EXTRACTION_CALLS, LiveExtraction

from ri05_tender.detect.config import CONFIDENCE_THRESHOLD
from ri05_tender.eval.findings_run import render_markdown, run
from spine.router import Router, RouterConfig

KESSLER_POINT = Path("data/tenders/kessler_point")

FULL_EVAL_ENV = "RI05_FULL_EVAL"

# Claims only: extraction arrives from the fixture already done. A different number is a
# batching bug or a lost hand-in, not a surprise to absorb — and either one costs money, so
# it is asserted rather than watched.
# Every clause the detection scope holds — ten documents, 44 clause-bearing pages. Pinned
# so a corpus that silently narrows fails here rather than reporting a better-looking recall
# over fewer items.
EXPECTED_CLAUSES = 691

EXPECTED_CALLS = 44
# What the whole labelled job spends on live calls: this run's claims plus the one shared
# extraction pass. Stated here because the saving is the point of the hand-in, and a figure
# nobody asserts is a figure that quietly goes back to 42.
EXPECTED_JOB_CALLS = EXPECTED_CALLS + EXPECTED_EXTRACTION_CALLS
# Headroom over the ~$4.00 the widened corpus projects to, for the whole job. Kept well
# above the projection so ordinary variance does not fail a build, and well below the point
# where a batching regression could hide inside it.
COST_CEILING_USD = 8.00

needs_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"
)
needs_full_eval = pytest.mark.skipif(
    os.environ.get(FULL_EVAL_ENV, "").strip().lower() not in {"1", "true", "yes"},
    reason=(
        f"the live findings run costs 44 calls and runs only on a pull request labelled "
        f"`full-eval` (set {FULL_EVAL_ENV})"
    ),
)


@pytest.mark.integration
@needs_key
@needs_full_eval
def test_a_live_findings_run_detects_scores_and_reports_its_abstentions(
    tmp_path: Path, live_extraction: LiveExtraction
) -> None:
    router = Router(
        RouterConfig.from_env().model_copy(update={"ledger_path": tmp_path / "ledger.json"})
    )

    def complete(prompt: str, schema: type, *, purpose: str):  # noqa: ANN202 - two schemas
        obj, _call = router.structured(prompt, schema, purpose=purpose, tier="large")
        return obj

    # Persisted the instant the run returns or raises. Every assertion below is free to fail
    # without costing the measurement, because the measurement is already on disk and in the
    # log by the time any of them run.
    try:
        result = run(
            KESSLER_POINT,
            complete,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            queue_root=artifact_dir(),
            # The extraction the gate test scores, handed in rather than paid for again.
            extraction=live_extraction.result,
        )
    finally:
        snapshot = persist_ledger(router, label="findings_full")

    report = render_markdown(result)
    print(report)
    (artifact_dir() / "findings_run.md").write_text(report, encoding="utf-8")

    # --- the run reached every stage ------------------------------------------------------
    assert result.extraction.clauses_read == EXPECTED_CLAUSES
    assert result.detection.clauses_examined == len(result.extraction.requirements)
    assert result.queue_path is not None and result.queue_path.exists()

    # --- what it asserts is what it scored -------------------------------------------------
    assert result.card.findings == result.asserted
    assert result.card.scored_items > 0

    # --- abstention is reported, not hidden ------------------------------------------------
    assert "abstained" in report
    assert report.index("abstained") < report.index("Recall")
    if result.abstained:
        assert result.routing.confidence < CONFIDENCE_THRESHOLD
        assert len(result.routing.item_ids) == result.abstained

    # --- cost --------------------------------------------------------------------------------
    assert snapshot.calls == EXPECTED_CALLS, (
        f"the run made {snapshot.calls} call(s) against {EXPECTED_CALLS} expected — claims "
        f"only, one per page. {EXPECTED_CALLS + EXPECTED_EXTRACTION_CALLS} would mean the "
        f"handed-in extraction was ignored and the corpus was extracted a second time; any "
        f"other number means the batching changed. Both cost money."
    )
    assert snapshot.calls + live_extraction.snapshot.calls == EXPECTED_JOB_CALLS, (
        f"the labelled job made {snapshot.calls + live_extraction.snapshot.calls} live "
        f"call(s) against {EXPECTED_JOB_CALLS} expected."
    )
    assert float(snapshot.cost_usd) < COST_CEILING_USD, (
        f"the run cost ${snapshot.cost_usd} against a ceiling of ${COST_CEILING_USD:.2f}."
    )
    # Claims and nothing else. `req_core.read_clauses` appearing here would mean this run
    # extracted the corpus itself despite being handed a result.
    assert {usage.purpose for usage in snapshot.by_purpose} == {"req_core.read_claims"}

    # --- the extraction really was shared, not re-run -------------------------------------
    assert result.extraction is live_extraction.result
