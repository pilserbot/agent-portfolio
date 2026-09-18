"""The end-to-end findings runner, offline against the committed tender.

No network, no model, no key: claims and readings are scripted, which exercises every
decision the runner makes because every decision it makes is Python. What is pinned here is
the wiring — that the adapter reconciles the two finding vocabularies, that only what the run
is prepared to assert is scored, and that the report puts abstention in front of recall.

The severity policy and the ordinal catalogue are read from `detect.config` rather than
rebuilt, so a change there fails here rather than passing quietly with a stale copy.
"""

import re
from pathlib import Path

import pytest

from req_core.claims import ClauseClaims, Constraint, PageClaims
from req_core.detectors import DETECTORS
from req_core.extraction import EXTRACT_PURPOSE, ClauseReading, PageReading
from ri05_tender.detect.config import CONFIDENCE_THRESHOLD, ORDINAL_SCALES, SEVERITY_POLICY
from ri05_tender.eval.findings_run import (
    DETECTOR_TO_FINDING_TYPE,
    as_eval_finding,
    render_markdown,
    run,
)
from ri05_tender.eval.matcher import CLASS_TO_FINDING_TYPE

KESSLER_POINT = Path("data/tenders/kessler_point")


def scripted(prompt: str, schema: type, *, purpose: str):  # noqa: ANN201 - two schemas, one stub
    """Answer both passes from the prompt, with no model anywhere.

    Extraction gets every clause back unsplit; the claims pass reports one unbounded
    constraint per clause, which is the shape `missing_tolerance` asserts at 0.85. Enough to
    drive the whole runner and cheap enough to run in a unit suite.
    """
    identifiers = re.findall(r"^\[([^\]]+)\]", prompt, re.MULTILINE)
    if schema is PageReading:
        return PageReading(clauses=[ClauseReading(identifier=name) for name in identifiers])
    return PageClaims(
        clauses=[
            ClauseClaims(
                identifier=name,
                constraints=[
                    Constraint(
                        subject="the equipment",
                        attribute="performance",
                        operator="unbounded",
                        value="",
                    )
                ],
            )
            for name in identifiers
        ]
    )


@pytest.fixture(scope="module")
def result():  # noqa: ANN201 - a DetectionRun, named by the fixture
    """One scripted end-to-end run over the committed tender."""
    return run(KESSLER_POINT, scripted)


# --- configuration is complete and is this project's, not the library's ---------------------


def test_the_severity_policy_covers_every_detector_that_can_fire() -> None:
    # A detector with no severity would raise mid-run; catching it here is cheaper than
    # catching it after 28 model calls.
    for detector in DETECTORS:
        assert SEVERITY_POLICY.severity_for(detector)


def test_the_two_finding_vocabularies_reconcile() -> None:
    # Five detector names are already the matcher's finding types; one is not. If either
    # side renames anything, this fails rather than every finding silently going unmatched.
    assert set(DETECTOR_TO_FINDING_TYPE) == set(DETECTORS)
    unknown = set(DETECTOR_TO_FINDING_TYPE.values()) - set(CLASS_TO_FINDING_TYPE.values())
    assert not unknown, f"finding type(s) the class map does not know: {sorted(unknown)}"


def test_the_catalogue_says_a_scale_is_ordinal_and_never_which_end_is_better() -> None:
    assert ORDINAL_SCALES.is_ordinal("IP")
    assert ORDINAL_SCALES.is_ordinal("nema")
    assert not ORDINAL_SCALES.is_ordinal("metres")
    # The model reports a name; the catalogue answers one question about it and no more.
    assert set(type(ORDINAL_SCALES).model_fields) == {"name", "scales"}


# --- the adapter ------------------------------------------------------------------------------


def test_a_findings_statement_becomes_its_recovered_statement(result) -> None:  # noqa: ANN001
    # The statement is written by Python from the claims, so it is genuinely the engine's
    # own words rather than a span of the tender — which is what the matcher's recovery rule
    # requires. Checked against the real page text, not asserted.
    core = result.detection.findings[0]
    adapted = as_eval_finding(core)

    assert adapted.recovered_statement == core.statement
    assert adapted.refs == [core.requirement_id]
    assert adapted.finding_type == DETECTOR_TO_FINDING_TYPE[core.detector]


def test_a_split_child_is_scored_against_its_parent_clause() -> None:
    # The matrix and the gold set index what the document printed; a child is this package's
    # subdivision of it, so a finding about a child anchors to the parent's reference.
    from req_core.findings import Evidence
    from req_core.findings import Finding as CoreFinding
    from spine.contracts import EvidenceRef

    child = CoreFinding(
        finding_id="X-1.1/2::zero_margin::1",
        detector="zero_margin",
        requirement_id="X-1.1/2",
        severity="critical",
        statement="stated in the engine's own words",
        evidence=Evidence(summary="observed"),
        source=EvidenceRef(source_id="04_Spec", document="Spec", page=1, quote="q"),
        confidence=0.9,
        parent_id="X-1.1",
    )

    assert as_eval_finding(child).refs == ["X-1.1"]


# --- what the run produced ----------------------------------------------------------------------


def test_the_run_reaches_every_stage(result) -> None:  # noqa: ANN001
    assert result.tender_name == "kessler_point"
    assert result.extraction.clauses_read == 301
    assert result.detection.clauses_examined == len(result.extraction.requirements)
    assert result.card.scored_items > 0


def test_only_what_the_run_asserts_is_scored(result) -> None:  # noqa: ANN001
    # Abstained findings are not scored as hits and not as misses. They are scored as
    # nothing, which is what abstaining means.
    assert result.card.findings == result.asserted
    assert result.asserted + result.abstained == len(result.detection.findings) + len(
        result.detection.abstained
    )


def test_the_threshold_moves_what_is_asserted(result) -> None:  # noqa: ANN001
    # The same run at a threshold nothing can clear asserts nothing, and the abstention rate
    # goes to 1.0. The knob is real, not decorative.
    strict = run(KESSLER_POINT, scripted, confidence_threshold=0.99)

    assert strict.asserted == 0
    assert strict.detection.abstention_rate == 1.0
    assert result.asserted > 0


def test_the_report_puts_abstention_in_front_of_recall(result) -> None:  # noqa: ANN001
    rendered = render_markdown(result)

    assert rendered.index("abstained") < rendered.index("Recall")
    assert "recall of **detection**" in rendered
    assert str(CONFIDENCE_THRESHOLD) in rendered or f"{CONFIDENCE_THRESHOLD:.2f}" in rendered


def test_the_report_says_what_the_run_cannot_yet_produce(result) -> None:  # noqa: ANN001
    # Detectors do not emit clarification questions or price impacts yet, so an item whose
    # defect was found still scores OUTPUT_MISS. Saying so is the difference between a
    # measurement and a claim.
    rendered = render_markdown(result)

    assert "OUTPUT_MISS" in rendered
    assert "end-to-end recall" in rendered


def test_the_adjudication_queue_is_written_when_a_root_is_given(tmp_path: Path) -> None:
    written = run(KESSLER_POINT, scripted, queue_root=tmp_path)

    assert written.queue_path is not None
    assert written.queue_path.exists()
    assert written.queue_path.parent == tmp_path


def test_no_queue_is_written_when_no_root_is_given(result) -> None:  # noqa: ANN001
    assert result.queue_path is None


def test_the_review_routing_carries_the_lowest_withheld_confidence(result) -> None:  # noqa: ANN001
    routing = result.routing

    if result.abstained:
        assert routing.confidence == min(f.confidence for f in result.detection.abstained)
        assert len(routing.item_ids) == result.abstained
    else:
        assert routing.is_empty


def test_the_report_names_the_queue_when_one_was_written(tmp_path: Path) -> None:
    written = run(KESSLER_POINT, scripted, queue_root=tmp_path)

    rendered = render_markdown(written)

    assert "Adjudication queue written to" in rendered
    assert str(written.queue_path) in rendered


# --- a handed-in extraction -----------------------------------------------------------------
#
# The live suite used to extract this corpus twice in one job — once for the gate and once
# inside the findings run — for 42 model calls where 28 do. `run` now takes an
# `ExtractionResult`, and these tests pin both halves of that: the pass really is skipped,
# and a result from the wrong corpus is refused rather than silently scored.


class CountingCompletion:
    """The scripted stub, counting how many prompts of each kind it was asked."""

    def __init__(self) -> None:
        """Start with nothing asked."""
        self.purposes: list[str] = []

    def __call__(self, prompt: str, schema: type, *, purpose: str):  # noqa: ANN204 - two schemas
        """Record the purpose, then answer exactly as `scripted` does."""
        self.purposes.append(purpose)
        return scripted(prompt, schema, purpose=purpose)


def test_a_handed_in_extraction_is_used_and_not_recomputed() -> None:
    first = CountingCompletion()
    done = run(KESSLER_POINT, first)
    extraction_purposes = [p for p in first.purposes if p == EXTRACT_PURPOSE]
    assert extraction_purposes, "the unassisted run must extract for itself"

    second = CountingCompletion()
    reused = run(KESSLER_POINT, second, extraction=done.extraction)

    assert reused.extraction is done.extraction
    assert EXTRACT_PURPOSE not in second.purposes, (
        "a run handed an extraction must make no extraction call at all — that saving is the "
        "whole reason the parameter exists"
    )
    assert second.purposes, "the claims pass must still run"
    assert len(second.purposes) < len(first.purposes)


def test_a_handed_in_extraction_produces_the_same_detection() -> None:
    """Sharing the pass must not change the answer, or the saving would not be free."""
    done = run(KESSLER_POINT, scripted)
    reused = run(KESSLER_POINT, scripted, extraction=done.extraction)

    assert reused.detection.clauses_examined == done.detection.clauses_examined
    assert [f.finding_id for f in reused.detection.findings] == [
        f.finding_id for f in done.detection.findings
    ]
    assert reused.card.scored_items == done.card.scored_items


def test_an_extraction_of_a_different_corpus_is_refused() -> None:
    """The guard that makes the hand-in safe: wrong corpus in, error out, not a score."""
    done = run(KESSLER_POINT, scripted)
    foreign = done.extraction.model_copy(update={"corpus_name": "some-other-tender"})

    with pytest.raises(ValueError, match="some-other-tender"):
        run(KESSLER_POINT, scripted, extraction=foreign)
