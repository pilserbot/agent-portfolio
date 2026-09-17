"""The live findings run: extract, detect, score, against the real tender and a real model.

Needs ANTHROPIC_API_KEY and a network, and runs only on a push to `main` or a pull request
labelled `full-eval` — the same `RI05_FULL_EVAL` gate as the extraction pass, set in
`ci.yml` from the same expression that chooses the full gold set over a sample. There is no
smoke-test half here: `test_ri05_extraction` already proves the wire is connected on every
pull request, and a second call proving it again would be paying twice for one fact.

**Cost.** 28 calls: 14 for extraction and 14 for claims. The two passes ask different
questions of the same pages, and a pipeline caching the first would pay for the second only.

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

from ri05_tender.detect.config import CONFIDENCE_THRESHOLD
from ri05_tender.eval.findings_run import render_markdown, run
from spine.router import Router, RouterConfig

KESSLER_POINT = Path("data/tenders/kessler_point")

FULL_EVAL_ENV = "RI05_FULL_EVAL"

# Extraction is 14 calls and claims is 14. A run materially above this is a batching bug,
# not a surprise to absorb.
EXPECTED_CALLS = 28
COST_CEILING_USD = 6.00

needs_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"
)
needs_full_eval = pytest.mark.skipif(
    os.environ.get(FULL_EVAL_ENV, "").strip().lower() not in {"1", "true", "yes"},
    reason=(
        f"the live findings run costs 28 calls and runs only on a push to main or a pull "
        f"request labelled `full-eval` (set {FULL_EVAL_ENV})"
    ),
)


@pytest.mark.integration
@needs_key
@needs_full_eval
def test_a_live_findings_run_detects_scores_and_reports_its_abstentions(tmp_path: Path) -> None:
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
        )
    finally:
        snapshot = persist_ledger(router, label="findings_full")

    report = render_markdown(result)
    print(report)
    (artifact_dir() / "findings_run.md").write_text(report, encoding="utf-8")

    # --- the run reached every stage ------------------------------------------------------
    assert result.extraction.clauses_read == 301
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
        f"the run made {snapshot.calls} call(s) against {EXPECTED_CALLS} expected — 14 for "
        f"extraction and 14 for claims. A different number means the batching changed, and "
        f"the cost of this step changed with it."
    )
    assert float(snapshot.cost_usd) < COST_CEILING_USD, (
        f"the run cost ${snapshot.cost_usd} against a ceiling of ${COST_CEILING_USD:.2f}."
    )
    assert {usage.purpose for usage in snapshot.by_purpose} == {
        "req_core.read_clauses",
        "req_core.read_claims",
    }
