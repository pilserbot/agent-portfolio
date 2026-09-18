"""Unit tests for the RI-05 scoring harness, against fixtures and the committed gold set.

No network, no model, no key. The pipeline this measures does not exist yet, so every
finding here is hand-built — which is the point: the measurement is defined before the
thing it measures, and these tests are that definition.

Every expected value is worked out by hand with the arithmetic written beside the
assertion, so a change to a formula shows up as a failing number rather than as a passing
rewrite.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ri05_tender.eval.adjudication import (
    AdjudicationEntry,
    AdjudicationError,
    AdjudicationQueue,
    load_queue,
    queue_from_findings,
    queue_path,
    write_queue,
)
from ri05_tender.eval.loader import GoldSetError, NoGoldSetError, load_gold, load_gold_file
from ri05_tender.eval.matcher import (
    CLASS_TO_FINDING_TYPE,
    OUTCOME_RANK,
    MatcherError,
    SourceText,
    allowed_finding_types,
    flatten,
    match_findings,
    normalise_ref,
    recovers_statement,
    source_text,
)
from ri05_tender.eval.metrics import SEVERITY_WEIGHTS, ScoreCard, score
from ri05_tender.eval.models import Finding, FindingOutputs, GoldItem, GoldReview
from ri05_tender.eval.report import render_markdown
from ri05_tender.tender.models import DocumentPage, TenderDocument, TenderPackage

KESSLER_POINT = Path("data/tenders/kessler_point")
AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


# --- fixtures, built by hand because nothing produces a Finding yet ------------------------


def a_gold_item(
    gold_id: str,
    *,
    refs: list[str] | None = None,
    classes: list[str] | None = None,
    finding_type: str = "displaced_requirement",
    severity: str = "major",
    tier: str = "cold_start",
    scored: bool = True,
    **expects: bool,
) -> GoldItem:
    payload: dict[str, object] = {
        "id": gold_id,
        "document": 1,
        "refs": refs or [f"ITB-{gold_id}"],
        "classes": classes or ["DISPLACED"],
        "finding_type": finding_type,
        "severity": severity,
        "tier": tier,
        "review": GoldReview(status="accepted_by_default"),
        "scored": scored,
    }
    payload.update(expects)
    return GoldItem.model_validate(payload)


def a_finding(
    finding_id: str,
    *,
    refs: list[str],
    finding_type: str = "displaced_requirement",
    severity: str = "major",
    confidence: float = 0.9,
    **outputs: object,
) -> Finding:
    return Finding(
        finding_id=finding_id,
        finding_type=finding_type,
        refs=refs,
        severity=severity,  # type: ignore[arg-type]
        statement=f"statement for {finding_id}",
        confidence=confidence,
        outputs=FindingOutputs.model_validate(outputs),
    )


def a_gold_payload(**overrides: object) -> dict[str, object]:
    """A dict that validates as a GoldItem, for writing fixture files."""
    payload: dict[str, object] = {
        "id": "D1-01",
        "document": 1,
        "refs": ["ITB-9.2"],
        "classes": ["PROCESS"],
        "finding_type": "submission_constraint",
        "severity": "major",
        "tier": "cold_start",
        "review": {"status": "accepted_by_default"},
        "scored": True,
    }
    payload.update(overrides)
    return payload


def write_gold(path: Path, payloads: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(payload) for payload in payloads) + "\n", encoding="utf-8")
    return path


def a_package(root: Path, *, has_gold: bool) -> TenderPackage:
    gold = root / "gold" / "gold_set.jsonl"
    return TenderPackage(
        name=root.name,
        root_path=root,
        documents=[],
        has_gold=has_gold,
        gold_path=gold if has_gold else None,
    )


def scored_run(
    gold: list[GoldItem],
    findings: list[Finding],
    *,
    excluded: list[GoldItem] | None = None,
    true_new: list[str] | None = None,
    pending: list[str] | None = None,
) -> object:
    report = match_findings(findings, gold)
    return score(
        tender_name="fixture",
        gold=gold,
        excluded=excluded or [],
        finding_ids=[finding.finding_id for finding in findings],
        report=report,
        true_new=true_new or [],
        pending=pending or [],
    )


# --- reference normalisation -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("its-9.2", "ITS-9.2"),
        ("  ITB-9.2  ", "ITB-9.2"),
        ("ITB - 9.2", "ITB-9.2"),  # whitespace goes from inside the reference too
        ("ITB--9.2", "ITB-9.2"),  # a run of hyphens collapses to one
        ("ITB–9.2", "ITB-9.2"),  # an en dash from a PDF folds to a hyphen
        ("ITB—9.2", "ITB-9.2"),  # and an em dash
        ("", ""),
    ],
)
def test_a_reference_normalises_to_a_comparable_form(raw: str, expected: str) -> None:
    assert normalise_ref(raw) == expected


def test_hyphens_are_collapsed_not_removed() -> None:
    # Removing them would let TS-B.2 answer for TSB.2, which is a different clause.
    assert normalise_ref("TS-B.2") != normalise_ref("TSB.2")


# --- the class-to-type map -------------------------------------------------------------------


def test_every_class_of_a_multi_class_item_widens_the_allowed_types() -> None:
    # Reading only the first class would mark a correct contract_red_flag finding wrong.
    item = a_gold_item("G1", classes=["DISPLACED", "COMMERCIAL"])

    assert allowed_finding_types(item) == {"displaced_requirement", "contract_red_flag"}


def test_an_unmapped_class_stops_the_run_rather_than_never_matching() -> None:
    item = a_gold_item("G1", classes=["BRAND-NEW-CLASS"])

    with pytest.raises(MatcherError, match="BRAND-NEW-CLASS"):
        allowed_finding_types(item)


def test_the_map_covers_every_class_in_the_committed_gold_set() -> None:
    # If the answer key gains a class, this fails here rather than silently reading as a
    # system that stopped finding that kind of defect.
    lines = (KESSLER_POINT / "gold" / "gold_set.jsonl").read_text(encoding="utf-8").splitlines()
    classes = {name for line in lines if line.strip() for name in json.loads(line)["classes"]}

    assert classes <= set(CLASS_TO_FINDING_TYPE)


# --- the five outcomes -------------------------------------------------------------------------


def test_a_run_that_finds_everything_scores_full_recall_and_passes_the_gate() -> None:
    gold = [
        a_gold_item("G1", severity="no_bid", expects_no_bid=True),
        a_gold_item("G2", severity="critical"),
        a_gold_item("G3", severity="minor"),
    ]
    findings = [
        a_finding("F1", refs=["ITB-G1"], no_bid=True),
        a_finding("F2", refs=["ITB-G2"]),
        a_finding("F3", refs=["ITB-G3"]),
    ]

    card = scored_run(gold, findings)

    assert card.recall_overall == 1.0
    assert card.recall_weighted == 1.0
    assert card.no_bid_gate
    assert (card.matches, card.misses, card.partials, card.output_misses) == (3, 0, 0, 0)


def test_missing_one_no_bid_item_fails_the_gate_whatever_else_scored() -> None:
    # The gate is not a rate. Ninety other finds do not offset a bid going out on a tender
    # the system should have refused.
    gold = [a_gold_item("G1", severity="no_bid", expects_no_bid=True)] + [
        a_gold_item(f"G{n}", severity="minor") for n in range(2, 22)
    ]
    findings = [a_finding(f"F{n}", refs=[f"ITB-G{n}"]) for n in range(2, 22)]

    card = scored_run(gold, findings)

    # 20 of 21 found — a recall most people would call good.
    assert card.recall_overall == pytest.approx(20 / 21)
    assert not card.no_bid_gate
    assert (card.no_bid_found, card.no_bid_total) == (0, 1)
    assert card.passed_gate == "FAIL"


def test_an_anchor_only_match_is_partial_and_does_not_count_as_found() -> None:
    # Right clause, wrong kind of finding: a different failure from not looking.
    gold = [a_gold_item("G1", refs=["ITB-9.2"], classes=["PROCESS"])]
    findings = [a_finding("F1", refs=["ITB-9.2"], finding_type="displaced_requirement")]

    report = match_findings(findings, gold)

    assert [partial.finding_id for partial in report.partials] == ["F1"]
    assert report.partials[0].gold_ids == ["G1"]
    assert report.matches == []
    assert report.misses == ["G1"]
    assert report.unmatched == []


def test_a_match_missing_a_required_output_is_an_output_miss_and_not_found() -> None:
    # The defect was seen; the work it demanded was not done.
    gold = [a_gold_item("G1", refs=["ITB-9.2"], expects_clarification_question=True)]
    findings = [a_finding("F1", refs=["ITB-9.2"])]

    report = match_findings(findings, gold)

    assert report.matches == []
    assert [a.gold_id for a in report.output_misses] == ["G1"]
    assert report.output_misses[0].missing_outputs == ["clarification_question"]
    # OUTPUT_MISS and MISS are different buckets and the outcomes partition: the item *was*
    # matched, just incompletely, so it is not "matched by nothing". What matters is that it
    # does not count as found.
    assert report.misses == []
    assert scored_run(gold, findings).recall_overall == 0.0


def test_every_gold_item_lands_in_exactly_one_outcome() -> None:
    # The five outcomes partition. An item counted twice, or not at all, would make the
    # denominators disagree with each other.
    gold = [
        a_gold_item("G1", refs=["A-1"]),
        a_gold_item("G2", refs=["A-2"], expects_clarification_question=True),
        a_gold_item("G3", refs=["A-3"], classes=["PROCESS"], finding_type="submission_constraint"),
        a_gold_item("G4", refs=["A-4"]),
    ]
    findings = [
        a_finding("F1", refs=["A-1"]),
        a_finding("F2", refs=["A-2"]),
        a_finding("F3", refs=["A-3"]),
        a_finding("F5", refs=["Z-9"]),
    ]

    report = match_findings(findings, gold)
    placed = (
        {a.gold_id for a in report.matches}
        | {a.gold_id for a in report.output_misses}
        | set(report.misses)
    )

    assert placed == {"G1", "G2", "G3", "G4"}
    assert len(report.matches) + len(report.output_misses) + len(report.misses) == len(gold)


def test_every_finding_lands_in_exactly_one_outcome() -> None:
    gold = [
        a_gold_item("G1", refs=["A-1"]),
        a_gold_item("G2", refs=["A-2"], expects_clarification_question=True),
        a_gold_item("G3", refs=["A-3"], classes=["PROCESS"], finding_type="submission_constraint"),
    ]
    findings = [
        a_finding("F1", refs=["A-1"]),
        a_finding("F2", refs=["A-2"]),
        a_finding("F3", refs=["A-3"]),
        a_finding("F5", refs=["Z-9"]),
    ]

    report = match_findings(findings, gold)
    placed = (
        {a.finding_id for a in report.matches}
        | {a.finding_id for a in report.output_misses}
        | {partial.finding_id for partial in report.partials}
        | {duplicate.finding_id for duplicate in report.duplicates}
        | set(report.unmatched)
    )

    assert placed == {"F1", "F2", "F3", "F5"}
    assert (
        len(report.matches)
        + len(report.output_misses)
        + len(report.partials)
        + len(report.duplicates)
        + len(report.unmatched)
    ) == len(findings)


def test_the_finding_partition_holds_with_every_outcome_present_at_once() -> None:
    # One of each: MATCH, OUTPUT_MISS, PARTIAL, DUPLICATE, UNMATCHED.
    gold = [
        a_gold_item("G1", refs=["A-1"]),
        a_gold_item("G2", refs=["A-2"], expects_clarification_question=True),
        a_gold_item("G3", refs=["A-3"], classes=["PROCESS"], finding_type="submission_constraint"),
    ]
    findings = [
        a_finding("F1", refs=["A-1"], confidence=0.9),  # MATCH on G1
        a_finding("F2", refs=["A-2"]),  # OUTPUT_MISS on G2
        a_finding("F3", refs=["A-3"]),  # PARTIAL: anchors G3, wrong type
        a_finding("F4", refs=["A-1"], confidence=0.1),  # DUPLICATE of G1
        a_finding("F5", refs=["Z-9"]),  # UNMATCHED
    ]

    report = match_findings(findings, gold)

    assert [a.finding_id for a in report.matches] == ["F1"]
    assert [a.finding_id for a in report.output_misses] == ["F2"]
    assert [p.finding_id for p in report.partials] == ["F3"]
    assert [d.finding_id for d in report.duplicates] == ["F4"]
    assert report.unmatched == ["F5"]
    assert (
        len(report.matches)
        + len(report.output_misses)
        + len(report.partials)
        + len(report.duplicates)
        + len(report.unmatched)
    ) == len(findings)
    # And the gold side still partitions independently of it.
    assert len(report.matches) + len(report.output_misses) + len(report.misses) == len(gold)


def test_supplying_the_required_output_turns_the_output_miss_into_a_match() -> None:
    gold = [a_gold_item("G1", refs=["ITB-9.2"], expects_clarification_question=True)]
    findings = [a_finding("F1", refs=["ITB-9.2"], clarification_question="Which time zone?")]

    report = match_findings(findings, gold)

    assert [a.gold_id for a in report.matches] == ["G1"]
    assert report.output_misses == []


def test_a_blank_output_does_not_satisfy_a_required_one() -> None:
    # An empty clarification question is not a clarification question.
    gold = [a_gold_item("G1", refs=["ITB-9.2"], expects_clarification_question=True)]
    findings = [a_finding("F1", refs=["ITB-9.2"], clarification_question="   ")]

    assert match_findings(findings, gold).matches == []


def test_every_expects_flag_is_checked_against_the_matching_output() -> None:
    gold = [
        a_gold_item(
            "G1",
            refs=["ITB-9.2"],
            expects_clarification_question=True,
            expects_price_impact=True,
            expects_alternative=True,
            expects_split=True,
            expects_checklist_entry=True,
            expects_no_bid=True,
        )
    ]

    bare = match_findings([a_finding("F1", refs=["ITB-9.2"])], gold)
    full = match_findings(
        [
            a_finding(
                "F1",
                refs=["ITB-9.2"],
                clarification_question="q",
                price_impact_usd=1200.0,
                alternative="an equal-or-better part",
                split_children=["G1a", "G1b"],
                checklist_entry="chase before submission",
                no_bid=True,
            )
        ],
        gold,
    )

    assert bare.output_misses[0].missing_outputs == [
        "clarification_question",
        "price_impact_usd",
        "alternative",
        "split_children",
        "checklist_entry",
        "no_bid",
    ]
    assert [a.gold_id for a in full.matches] == ["G1"]


def test_a_finding_that_anchors_nothing_is_unmatched_not_wrong() -> None:
    gold = [a_gold_item("G1", refs=["ITB-9.2"])]
    findings = [a_finding("F1", refs=["SOW-4.4"])]

    report = match_findings(findings, gold)

    assert report.unmatched == ["F1"]
    assert report.partials == []
    assert report.misses == ["G1"]


# --- one-to-one assignment -----------------------------------------------------------------------


def test_one_finding_citing_two_gold_items_is_credited_to_exactly_one() -> None:
    gold = [a_gold_item("G1", refs=["ITB-9.2"]), a_gold_item("G2", refs=["ITB-9.3"])]
    findings = [a_finding("F1", refs=["ITB-9.2", "ITB-9.3"])]

    report = match_findings(findings, gold)

    assert len(report.matches) == 1
    assert len(report.misses) == 1
    assert report.matched_gold_ids | set(report.misses) == {"G1", "G2"}


def test_two_findings_cannot_both_be_credited_with_one_gold_item() -> None:
    gold = [a_gold_item("G1", refs=["ITB-9.2"])]
    findings = [
        a_finding("F1", refs=["ITB-9.2"], confidence=0.4),
        a_finding("F2", refs=["ITB-9.2"], confidence=0.9),
    ]

    report = match_findings(findings, gold)

    assert len(report.matches) == 1
    assert report.matches[0].finding_id == "F2"  # higher confidence wins
    # The loser anchored and typed onto an item somebody else now holds: a DUPLICATE. It
    # restates a known defect rather than proposing a new one, so no human is asked.
    assert [duplicate.finding_id for duplicate in report.duplicates] == ["F1"]
    assert report.partials == []
    assert report.unmatched == []


def test_the_higher_anchor_overlap_wins_before_confidence() -> None:
    gold = [a_gold_item("G1", refs=["ITB-9.2", "ITB-9.3"])]
    findings = [
        a_finding("F1", refs=["ITB-9.2", "ITB-9.3"], confidence=0.2),  # overlap 2
        a_finding("F2", refs=["ITB-9.2"], confidence=0.99),  # overlap 1
    ]

    assert match_findings(findings, gold).matches[0].finding_id == "F1"


def test_the_finding_id_breaks_a_tie_so_the_assignment_is_reproducible() -> None:
    gold = [a_gold_item("G1", refs=["ITB-9.2"])]
    findings = [
        a_finding("F2", refs=["ITB-9.2"], confidence=0.5),
        a_finding("F1", refs=["ITB-9.2"], confidence=0.5),
    ]

    assert match_findings(findings, gold).matches[0].finding_id == "F1"
    assert match_findings(list(reversed(findings)), gold).matches[0].finding_id == "F1"


def test_identical_input_always_produces_an_identical_assignment() -> None:
    gold = [a_gold_item(f"G{n}", refs=[f"ITB-{n}"]) for n in range(1, 6)]
    findings = [a_finding(f"F{n}", refs=[f"ITB-{n}"], confidence=0.5) for n in range(1, 6)]

    first = match_findings(findings, gold)
    second = match_findings(list(reversed(findings)), list(reversed(gold)))

    assert first == second


def test_completeness_outranks_overlap() -> None:
    """The whole point of the ordering: a complete answer takes the item.

    F1 anchors both references but omits the required output; F2 anchors one and supplies
    it. Under overlap-first, F1 would take G1 as an OUTPUT_MISS and the item would read as
    not found — the system answered correctly and would be scored as having failed. That is
    a defect in the instrument, not in the system under test.
    """
    gold = [a_gold_item("G1", refs=["A-1", "A-2"], expects_clarification_question=True)]
    findings = [
        a_finding("F1", refs=["A-1", "A-2"], confidence=0.9),  # overlap 2, no question
        a_finding("F2", refs=["A-1"], confidence=0.9, clarification_question="q"),  # overlap 1
    ]

    report = match_findings(findings, gold)

    assert [(a.finding_id, a.gold_id) for a in report.matches] == [("F2", "G1")]
    assert report.output_misses == []
    assert report.misses == []
    # F1 anchored and typed onto an item somebody else now holds: a DUPLICATE, not a
    # candidate new defect, so it is not sent to a human.
    assert [duplicate.finding_id for duplicate in report.duplicates] == ["F1"]
    assert report.unmatched == []


def test_the_old_pathology_no_longer_occurs_even_at_higher_confidence() -> None:
    # Same shape, with the incomplete finding also more confident. Completeness still wins:
    # it is rank 1 of the order, ahead of both overlap and confidence.
    gold = [a_gold_item("G1", refs=["A-1", "A-2"], expects_price_impact=True)]
    findings = [
        a_finding("F1", refs=["A-1", "A-2"], confidence=0.99),
        a_finding("F2", refs=["A-1"], confidence=0.10, price_impact_usd=4200.0),
    ]

    report = match_findings(findings, gold)

    assert [a.finding_id for a in report.matches] == ["F2"]
    assert scored_run(gold, findings).recall_overall == 1.0


def test_the_outcome_rank_is_completeness_first() -> None:
    # The order is data, so it can be read rather than inferred from behaviour.
    assert OUTCOME_RANK["match"] < OUTCOME_RANK["output_miss"] < OUTCOME_RANK["partial"]


def test_an_output_miss_still_wins_when_no_complete_answer_competes() -> None:
    # Completeness first does not mean an incomplete answer is discarded; it means it loses
    # only to a complete one.
    gold = [a_gold_item("G1", refs=["A-1"], expects_clarification_question=True)]
    findings = [a_finding("F1", refs=["A-1"], confidence=0.3)]

    report = match_findings(findings, gold)

    assert [a.finding_id for a in report.output_misses] == ["F1"]
    assert report.duplicates == []


def test_overlap_still_decides_between_two_equally_complete_answers() -> None:
    # Rank 2 of the order still does its job once rank 1 ties.
    gold = [a_gold_item("G1", refs=["A-1", "A-2"])]
    findings = [
        a_finding("F1", refs=["A-1"], confidence=0.99),
        a_finding("F2", refs=["A-1", "A-2"], confidence=0.10),
    ]

    assert [a.finding_id for a in match_findings(findings, gold).matches] == ["F2"]


# --- DUPLICATE --------------------------------------------------------------------


def test_a_second_finding_on_a_claimed_item_is_a_duplicate_not_unmatched() -> None:
    # It demonstrably refers to a known defect, so it is not a candidate new one.
    gold = [a_gold_item("G1", refs=["A-1"])]
    findings = [
        a_finding("F1", refs=["A-1"], confidence=0.9),
        a_finding("F2", refs=["A-1"], confidence=0.5),
    ]

    report = match_findings(findings, gold)

    assert [a.finding_id for a in report.matches] == ["F1"]
    assert [d.finding_id for d in report.duplicates] == ["F2"]
    assert report.duplicates[0].gold_ids == ["G1"]
    assert report.unmatched == []
    assert report.partials == []


def test_a_duplicate_counts_against_strict_precision() -> None:
    # Five variants of one finding is a real problem for whoever has to read them.
    gold = [a_gold_item("G1", refs=["A-1"])]
    findings = [
        a_finding("F1", refs=["A-1"], confidence=0.9),
        a_finding("F2", refs=["A-1"], confidence=0.5),
        a_finding("F3", refs=["A-1"], confidence=0.4),
    ]

    card = scored_run(gold, findings)

    # 1 match over (1 match + 0 unmatched + 2 duplicates) = 1/3.
    assert card.duplicates == 2
    assert card.precisions.strict == pytest.approx(1 / 3)
    assert card.precisions.strict_support == 3


def test_a_duplicate_never_enters_the_adjudication_queue() -> None:
    # Asking a human to rule on it spends the scarcest resource in the loop on a question
    # already answered.
    gold = [a_gold_item("G1", refs=["A-1"])]
    findings = [
        a_finding("F1", refs=["A-1"], confidence=0.9),
        a_finding("F2", refs=["A-1"], confidence=0.5),  # duplicate
        a_finding("F3", refs=["Z-9"]),  # genuinely unmatched
    ]
    report = match_findings(findings, gold)

    queue = queue_from_findings(findings, report.unmatched, tender_name="fixture")

    assert [d.finding_id for d in report.duplicates] == ["F2"]
    assert [entry.finding_id for entry in queue.entries] == ["F3"]


def test_a_finding_that_only_anchors_a_claimed_item_is_partial_not_duplicate() -> None:
    # DUPLICATE needs both anchor and type. Anchoring alone is still PARTIAL: it has not
    # shown that it refers to the same defect, only to the same clause.
    gold = [a_gold_item("G1", refs=["A-1"], classes=["DISPLACED"])]
    findings = [
        a_finding("F1", refs=["A-1"], confidence=0.9),
        a_finding("F2", refs=["A-1"], finding_type="contract_red_flag"),
    ]

    report = match_findings(findings, gold)

    assert [a.finding_id for a in report.matches] == ["F1"]
    assert [p.finding_id for p in report.partials] == ["F2"]
    assert report.duplicates == []


def test_duplicate_beats_partial_when_a_finding_is_both() -> None:
    # F2 anchors+types G1 (taken) and anchors-only G2. DUPLICATE is the more specific and
    # more useful label: it names a defect the run already reported.
    gold = [
        a_gold_item("G1", refs=["A-1"], classes=["DISPLACED"]),
        a_gold_item("G2", refs=["A-2"], classes=["PROCESS"], finding_type="submission_constraint"),
    ]
    findings = [
        a_finding("F1", refs=["A-1"], confidence=0.9),
        a_finding("F2", refs=["A-1", "A-2"], confidence=0.5),
    ]

    report = match_findings(findings, gold)

    assert [d.finding_id for d in report.duplicates] == ["F2"]
    assert report.partials == []


def test_a_duplicate_does_not_stop_a_gold_item_being_found() -> None:
    gold = [a_gold_item("G1", refs=["A-1"])]
    findings = [a_finding("F1", refs=["A-1"]), a_finding("F2", refs=["A-1"], confidence=0.1)]

    card = scored_run(gold, findings)

    assert card.recall_overall == 1.0
    assert card.misses == 0


# --- recall, weighted and broken out --------------------------------------------------------------


def hand_worked_set() -> list[GoldItem]:
    """Four items whose weights are 8 + 4 + 2 + 1 = 15."""
    return [
        a_gold_item("G1", severity="no_bid", tier="configured", expects_no_bid=True),
        a_gold_item("G2", severity="critical", classes=["COMMERCIAL"], finding_type="x"),
        a_gold_item("G3", severity="major"),
        a_gold_item("G4", severity="minor"),
    ]


def test_weighted_recall_matches_the_arithmetic_exactly() -> None:
    # Found G1 (8), G3 (2), G4 (1) = 11 of 8 + 4 + 2 + 1 = 15. Weighted 11/15 = 0.7333…,
    # while plain recall is 3/4 = 0.75. The two differ, so a swap between them shows.
    gold = hand_worked_set()
    findings = [
        a_finding("F1", refs=["ITB-G1"], no_bid=True),
        a_finding("F3", refs=["ITB-G3"]),
        a_finding("F4", refs=["ITB-G4"]),
    ]

    card = scored_run(gold, findings)

    assert card.recall_overall == pytest.approx(0.75)
    assert card.recall_weighted == pytest.approx(11 / 15)


def test_weighting_makes_the_same_hit_rate_worth_less_when_the_miss_is_severe() -> None:
    # Same 3-of-4 hit rate, but G1 (weight 8) missed instead of G2 (weight 4):
    # 4 + 2 + 1 = 7 of 15 = 0.4667 against 11/15 = 0.7333.
    gold = hand_worked_set()
    findings = [
        a_finding("F2", refs=["ITB-G2"], finding_type="contract_red_flag"),
        a_finding("F3", refs=["ITB-G3"]),
        a_finding("F4", refs=["ITB-G4"]),
    ]

    card = scored_run(gold, findings)

    assert card.recall_overall == pytest.approx(0.75)
    assert card.recall_weighted == pytest.approx(7 / 15)
    assert not card.no_bid_gate


def test_the_weights_are_the_ones_specified() -> None:
    assert SEVERITY_WEIGHTS == {"no_bid": 8, "critical": 4, "major": 2, "minor": 1}


def test_recall_is_broken_out_by_severity_tier_and_class() -> None:
    gold = hand_worked_set()
    findings = [a_finding("F1", refs=["ITB-G1"], no_bid=True), a_finding("F3", refs=["ITB-G3"])]

    card = scored_run(gold, findings)
    severity = {row.key: row for row in card.by_severity}
    tier = {row.key: row for row in card.by_tier}
    klass = {row.key: row for row in card.by_class}

    assert (severity["no_bid"].value, severity["no_bid"].found) == (1.0, 1)
    assert (severity["critical"].value, severity["critical"].found) == (0.0, 0)
    assert (severity["major"].value, severity["major"].found) == (1.0, 1)
    assert (severity["minor"].value, severity["minor"].found) == (0.0, 0)
    # G1 is the only configured item and it was found; the other three are cold_start,
    # of which G3 was found: 1/3.
    assert tier["configured"].value == 1.0
    assert tier["cold_start"].value == pytest.approx(1 / 3)
    # DISPLACED covers G1, G3, G4; two of the three were found.
    assert klass["DISPLACED"].value == pytest.approx(2 / 3)
    assert klass["COMMERCIAL"].value == 0.0


def test_the_two_recoveries_are_recall_over_their_own_class_and_nothing_else() -> None:
    gold = [
        a_gold_item("G1", classes=["DISPLACED"]),
        a_gold_item("G2", classes=["DISPLACED"]),
        a_gold_item("G3", classes=["UNSTATED"], finding_type="unstated_requirement"),
        a_gold_item("G4", classes=["UNSTATED"], finding_type="unstated_requirement"),
        a_gold_item("G5", classes=["PROCESS"], finding_type="submission_constraint"),
    ]
    findings = [
        a_finding("F1", refs=["ITB-G1"]),
        a_finding("F3", refs=["ITB-G3"], finding_type="unstated_requirement"),
        a_finding("F4", refs=["ITB-G4"], finding_type="unstated_requirement"),
        a_finding("F5", refs=["ITB-G5"], finding_type="submission_constraint"),
    ]

    card = scored_run(gold, findings)

    # 1 of the 2 DISPLACED items and 2 of the 2 UNSTATED ones — each over its own
    # denominator, neither over the four of them together.
    assert (card.displaced_recovery, card.displaced_total) == (pytest.approx(0.5), 2)
    assert (card.unstated_recovery, card.unstated_total) == (1.0, 2)
    assert card.recall_overall == pytest.approx(4 / 5)


def test_there_is_no_combined_recovery_figure_to_quote() -> None:
    # The two were one class, IMPLICIT, and one ratio over both hid that a "shall" grep
    # reaches every DISPLACED item and no UNSTATED one. There is no field to put the
    # average back into, and this is the test that says so on purpose.
    fields = set(ScoreCard.model_fields)

    assert "unstated_recovery" in fields
    assert "displaced_recovery" in fields
    assert not {name for name in fields if "implicit" in name}
    assert not {name for name in fields if "recovery" in name} - {
        "unstated_recovery",
        "displaced_recovery",
    }


def test_a_group_with_no_items_reports_zero_over_zero_rather_than_raising() -> None:
    card = scored_run([], [])

    assert card.recall_overall == 0.0
    assert card.recall_weighted == 0.0
    assert card.unstated_recovery == 0.0
    assert card.displaced_recovery == 0.0


# --- precision, both of them ----------------------------------------------------------------------


def precision_fixture() -> tuple[list[GoldItem], list[Finding]]:
    """3 matches, 1 partial, 1 unmatched: five findings in total."""
    gold = [a_gold_item(f"G{n}", refs=[f"ITB-{n}"]) for n in range(1, 5)]
    findings = [
        a_finding("F1", refs=["ITB-1"]),
        a_finding("F2", refs=["ITB-2"]),
        a_finding("F3", refs=["ITB-3"]),
        a_finding("F4", refs=["ITB-4"], finding_type="contract_red_flag"),  # anchors, wrong type
        a_finding("F5", refs=["SOW-99"]),  # anchors nothing
    ]
    return gold, findings


def test_both_precisions_are_computed_and_differ() -> None:
    gold, findings = precision_fixture()

    card = scored_run(gold, findings)

    # strict = 3 matches / (3 matches + 1 unmatched) = 0.75
    assert card.precisions.strict == pytest.approx(0.75)
    assert card.precisions.strict_support == 4
    # adjudicated = (3 matches + 0 confirmed new) / 5 findings = 0.6
    assert card.precisions.adjudicated == pytest.approx(0.6)
    assert card.precisions.adjudicated_support == 5


def test_a_confirmed_new_finding_lifts_adjudicated_precision_only() -> None:
    gold, findings = precision_fixture()

    card = scored_run(gold, findings, true_new=["F5"])

    assert card.precisions.strict == pytest.approx(0.75)  # unchanged
    assert card.precisions.adjudicated == pytest.approx(0.8)  # (3 + 1) / 5
    assert card.precisions.true_new == 1


def test_neither_precision_can_be_reported_without_the_other() -> None:
    # Enforced by the type: `Precisions` has no default for either field, so half of it
    # cannot be constructed at all.
    from ri05_tender.eval.metrics import Precisions

    required = {name for name, field in Precisions.model_fields.items() if field.is_required()}

    assert {"strict", "adjudicated"} <= required


# --- scored=false ---------------------------------------------------------------------------------


def test_unscored_items_are_excluded_from_every_denominator(tmp_path: Path) -> None:
    path = write_gold(
        tmp_path / "gold" / "gold_set.jsonl",
        [
            a_gold_payload(id="G1", refs=["ITB-1"], classes=["DISPLACED"]),
            a_gold_payload(id="G2", refs=["ITB-2"], classes=["DISPLACED"], severity="no_bid"),
            a_gold_payload(
                id="G3", refs=["ITB-3"], classes=["DISPLACED"], severity="critical", scored=False
            ),
        ],
    )
    gold = load_gold_file(path)
    findings = [
        a_finding("F1", refs=["ITB-1"]),
        a_finding("F2", refs=["ITB-2"]),
    ]

    card = scored_run(gold.scored, findings, excluded=gold.excluded)

    assert (len(gold.scored), len(gold.excluded)) == (2, 1)
    # 2 of 2, not 2 of 3 — and the weighted denominator is 8 + 2 = 10, not 14.
    assert card.recall_overall == 1.0
    assert card.recall_weighted == 1.0
    assert card.scored_items == 2
    assert card.excluded_items == 1
    # The excluded item was critical; it must not appear in any breakdown.
    assert "critical" not in {row.key for row in card.by_severity}
    assert card.no_bid_gate


def test_the_committed_gold_set_loads_with_its_one_excluded_item() -> None:
    from ri05_tender.tender.loader import load_tender

    gold = load_gold(load_tender(KESSLER_POINT))

    assert len(gold.scored) == 186
    assert len(gold.excluded) == 1
    assert gold.excluded[0].id == "D1-14"
    assert not gold.excluded[0].scored


# --- what the loader refuses ----------------------------------------------------------------------


def test_scoring_a_tender_with_no_answer_key_raises_its_own_error(tmp_path: Path) -> None:
    # Not an empty run: a recall of zero over nothing reads like a result.
    package = a_package(tmp_path / "unlabelled", has_gold=False)

    with pytest.raises(NoGoldSetError) as error:
        load_gold(package)

    assert "unlabelled" in str(error.value)
    assert "no answer key" in str(error.value)


def test_the_no_gold_error_is_catchable_apart_from_a_broken_file(tmp_path: Path) -> None:
    package = a_package(tmp_path / "unlabelled", has_gold=False)

    with pytest.raises(GoldSetError):  # NoGoldSetError is one, so a broad catch still works
        load_gold(package)


def test_a_malformed_gold_line_names_its_line_number(tmp_path: Path) -> None:
    path = write_gold(
        tmp_path / "gold_set.jsonl",
        [a_gold_payload(id="G1"), a_gold_payload(id="G2")],
    )
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")

    with pytest.raises(GoldSetError) as error:
        load_gold_file(path)

    assert ":3:" in str(error.value)


def test_an_unknown_severity_fails_the_load_naming_the_item(tmp_path: Path) -> None:
    path = write_gold(
        tmp_path / "gold_set.jsonl",
        [a_gold_payload(id="G1"), a_gold_payload(id="G2", severity="showstopper")],
    )

    with pytest.raises(GoldSetError) as error:
        load_gold_file(path)

    assert "'G2'" in str(error.value)
    assert ":2:" in str(error.value)
    assert "severity" in str(error.value)


def test_an_unknown_tier_fails_the_load_naming_the_item(tmp_path: Path) -> None:
    path = write_gold(tmp_path / "gold_set.jsonl", [a_gold_payload(id="G7", tier="warm_start")])

    with pytest.raises(GoldSetError, match="'G7'"):
        load_gold_file(path)


def test_an_unknown_field_in_a_gold_line_fails_the_load(tmp_path: Path) -> None:
    # A key this model does not know would otherwise be dropped in silence.
    path = write_gold(tmp_path / "gold_set.jsonl", [a_gold_payload(id="G1", severuty="major")])

    with pytest.raises(GoldSetError, match="severuty"):
        load_gold_file(path)


def test_a_duplicate_gold_id_fails_the_load_naming_both_lines(tmp_path: Path) -> None:
    path = write_gold(
        tmp_path / "gold_set.jsonl", [a_gold_payload(id="G1"), a_gold_payload(id="G1")]
    )

    with pytest.raises(GoldSetError, match="duplicate gold id"):
        load_gold_file(path)


def test_a_missing_gold_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(GoldSetError, match="not found"):
        load_gold_file(tmp_path / "nowhere.jsonl")


# --- adjudication ---------------------------------------------------------------------------------


def test_unmatched_findings_are_queued_with_no_verdict(tmp_path: Path) -> None:
    gold, findings = precision_fixture()
    report = match_findings(findings, gold)

    queue = queue_from_findings(findings, report.unmatched, tender_name="fixture")
    path = write_queue(queue, root=tmp_path, at=AT)

    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [entry["finding_id"] for entry in written] == ["F5"]
    assert written[0]["verdict"] is None
    assert path.name == "fixture_20260102T030405Z.jsonl"


def test_the_queue_is_filed_under_the_tender_name_and_a_timestamp() -> None:
    assert queue_path("kessler_point", root=Path("evals/adjudication"), at=AT) == Path(
        "evals/adjudication/kessler_point_20260102T030405Z.jsonl"
    )


def test_a_human_verdict_is_read_back(tmp_path: Path) -> None:
    path = tmp_path / "fixture_20260102T030405Z.jsonl"
    path.write_text(
        "\n".join(
            [
                AdjudicationEntry(
                    finding_id="F5",
                    finding_type="x",
                    severity="major",
                    confidence=0.5,
                    verdict="true_new",
                    adjudicated_by="a reviewer",
                ).model_dump_json(),
                AdjudicationEntry(
                    finding_id="F6",
                    finding_type="x",
                    severity="minor",
                    confidence=0.5,
                    verdict="false_positive",
                ).model_dump_json(),
                AdjudicationEntry(
                    finding_id="F7", finding_type="x", severity="minor", confidence=0.5
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    queue = load_queue(path)

    assert queue.true_new_ids == ["F5"]
    assert queue.false_positive_ids == ["F6"]
    assert queue.pending_ids == ["F7"]


def test_a_confirmed_new_finding_gets_a_u_prefixed_id() -> None:
    confirmed = AdjudicationEntry(
        finding_id="F5", finding_type="x", severity="major", confidence=0.5, verdict="true_new"
    )
    rejected = confirmed.model_copy(update={"verdict": "false_positive"})

    assert confirmed.adjudicated_id == "U-F5"
    # Not confirmed, so it has not become anything and keeps its own id.
    assert rejected.adjudicated_id == "F5"
    assert AdjudicationQueue(tender_name="t", entries=[confirmed]).new_finding_ids == ["U-F5"]


def test_an_unrecognised_verdict_is_refused(tmp_path: Path) -> None:
    # "probably fine" is not a verdict; only a human writing one of two words counts.
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps(
            {
                "finding_id": "F5",
                "finding_type": "x",
                "severity": "major",
                "confidence": 0.5,
                "verdict": "probably fine",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AdjudicationError, match="verdict"):
        load_queue(path)


def test_a_queue_round_trips_through_disk(tmp_path: Path) -> None:
    gold, findings = precision_fixture()
    report = match_findings(findings, gold)
    written = write_queue(
        queue_from_findings(findings, report.unmatched, tender_name="fixture"),
        root=tmp_path,
        at=AT,
    )

    reloaded = load_queue(written)

    assert [entry.finding_id for entry in reloaded.entries] == ["F5"]
    assert reloaded.pending_ids == ["F5"]
    assert reloaded.tender_name == "fixture"


def test_an_empty_queue_writes_an_empty_file(tmp_path: Path) -> None:
    path = write_queue(AdjudicationQueue(tender_name="fixture"), root=tmp_path, at=AT)

    assert path.read_text(encoding="utf-8") == ""
    assert len(load_queue(path)) == 0


# --- the report -----------------------------------------------------------------------------------


def test_the_report_states_the_scored_count_and_what_was_excluded() -> None:
    gold = hand_worked_set()
    card = scored_run(gold, [a_finding("F1", refs=["ITB-G1"], no_bid=True)], excluded=gold[:1])

    markdown = render_markdown(card)

    assert "**4 scored gold item(s)**, 1 excluded (`scored: false`)" in markdown


def test_the_report_says_none_excluded_when_none_were() -> None:
    card = scored_run(hand_worked_set(), [])

    assert "none excluded" in render_markdown(card)


def test_the_report_shows_the_gate_as_an_explicit_pass_or_fail() -> None:
    gold = hand_worked_set()
    passed = render_markdown(scored_run(gold, [a_finding("F1", refs=["ITB-G1"], no_bid=True)]))
    failed = render_markdown(scored_run(gold, []))

    assert "### No-bid gate" in passed
    assert "**PASS** — 1 of 1 scored `no_bid` item(s) matched." in passed
    assert "**FAIL** — 0 of 1 scored `no_bid` item(s) matched." in failed
    assert "not a rate" in failed


def test_the_report_carries_both_precisions_and_all_three_breakdowns() -> None:
    gold, findings = precision_fixture()

    markdown = render_markdown(scored_run(gold, findings))

    assert "Precision (strict)" in markdown
    assert "Precision (adjudicated)" in markdown
    assert "### Recall by severity" in markdown
    assert "### Recall by tier" in markdown
    assert "### Recall by class" in markdown


def test_the_report_counts_the_outcomes_that_are_neither_found_nor_a_clean_miss() -> None:
    gold, findings = precision_fixture()

    markdown = render_markdown(scored_run(gold, findings, pending=["F5"]))

    assert "| PARTIAL | 1 |" in markdown
    assert "| OUTPUT_MISS | 0 |" in markdown
    assert "| Pending adjudication | 1 |" in markdown


def test_a_gateless_tender_says_the_gate_asserted_nothing() -> None:
    # Passing a gate that had nothing to check is not evidence of anything.
    card = scored_run([a_gold_item("G1", severity="minor")], [])

    assert card.no_bid_gate
    assert "without asserting" in render_markdown(card)


# --- the loaders' remaining edges ------------------------------------------------------


def test_a_gold_set_exposes_its_items_by_id_and_across_both_lists(tmp_path: Path) -> None:
    path = write_gold(
        tmp_path / "gold_set.jsonl",
        [a_gold_payload(id="G2"), a_gold_payload(id="G1", scored=False)],
    )

    gold = load_gold_file(path)

    assert [item.id for item in gold.all_items] == ["G1", "G2"]
    assert len(gold) == 1  # only the scored ones count
    assert gold.item("G2") is not None
    assert gold.item("G1") is None  # excluded, so not reachable as a scored item
    assert gold.item("nope") is None


def test_blank_lines_in_a_gold_set_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "gold_set.jsonl"
    path.write_text(
        json.dumps(a_gold_payload(id="G1")) + "\n\n   \n" + json.dumps(a_gold_payload(id="G2")),
        encoding="utf-8",
    )

    assert [item.id for item in load_gold_file(path).scored] == ["G1", "G2"]


def test_a_gold_line_that_is_not_an_object_fails_with_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "gold_set.jsonl"
    path.write_text(json.dumps(a_gold_payload(id="G1")) + "\n[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(GoldSetError, match=r":2:.*expected a JSON object"):
        load_gold_file(path)


def test_a_gold_path_that_is_a_directory_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "gold_set.jsonl").mkdir()

    with pytest.raises(GoldSetError, match="is a directory"):
        load_gold_file(tmp_path / "gold_set.jsonl")


def test_a_missing_adjudication_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(AdjudicationError, match="not found"):
        load_queue(tmp_path / "nowhere.jsonl")


def test_an_adjudication_path_that_is_a_directory_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "queue.jsonl").mkdir()

    with pytest.raises(AdjudicationError, match="is a directory"):
        load_queue(tmp_path / "queue.jsonl")


def test_blank_lines_in_an_adjudication_file_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    entry = AdjudicationEntry(finding_id="F1", finding_type="x", severity="minor", confidence=0.1)
    path.write_text(entry.model_dump_json() + "\n\n   \n", encoding="utf-8")

    assert [item.finding_id for item in load_queue(path).entries] == ["F1"]


def test_a_malformed_adjudication_line_names_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    entry = AdjudicationEntry(finding_id="F1", finding_type="x", severity="minor", confidence=0.1)
    path.write_text(entry.model_dump_json() + "\n{not json\n", encoding="utf-8")

    with pytest.raises(AdjudicationError, match=":2:"):
        load_queue(path)


def test_a_report_table_with_no_rows_says_none_rather_than_rendering_empty() -> None:
    # A bare header reads like a rendering bug; "_none_" reads like an answer.
    card = scored_run([], [])

    assert "_none_" in render_markdown(card)


def test_the_report_shows_the_duplicate_count() -> None:
    gold = [a_gold_item("G1", refs=["A-1"])]
    findings = [
        a_finding("F1", refs=["A-1"], confidence=0.9),
        a_finding("F2", refs=["A-1"], confidence=0.5),
    ]

    markdown = render_markdown(scored_run(gold, findings))

    assert "| DUPLICATE | 1 |" in markdown
    assert "never adjudicated" in markdown


def test_the_greedy_rule_is_the_definition_not_an_approximation() -> None:
    """Pins the documented trade-off: greedy really can credit fewer items than an optimum.

    G1 cites A-1; G2 cites A-1 and A-2. F1 cites both, so it overlaps G2 by two and G1 by
    one; F2 cites only A-2, so it overlaps G2 by one and G1 not at all.

    Greedy takes the highest-ranked pair first — F1 to G2, overlap 2 — which spends F1 and
    G2 and leaves F2 with nothing it can answer. One match, G1 missed, F2 a duplicate. An
    optimal assignment would pair F1 with G1 and F2 with G2: **two** matches, recall 100%
    instead of 50%.

    That is accepted, because the rule is greedy *by definition*: the score is one anybody
    can re-derive by hand from the stated order, rather than one whose value depends on
    which solver ran. If this test ever needs changing, the definition changed with it.
    """
    gold = [a_gold_item("G1", refs=["A-1"]), a_gold_item("G2", refs=["A-1", "A-2"])]
    findings = [
        a_finding("F1", refs=["A-1", "A-2"], confidence=0.9),
        a_finding("F2", refs=["A-2"], confidence=0.9),
    ]

    report = match_findings(findings, gold)

    assert [(a.finding_id, a.gold_id) for a in report.matches] == [("F1", "G2")]
    assert report.misses == ["G1"]
    assert [d.finding_id for d in report.duplicates] == ["F2"]
    assert scored_run(gold, findings).recall_overall == pytest.approx(0.5)


# --- a recovered statement must be the finding's own words -----------------------------------

TENDER_LINE = "TS-1.1 All equipment shall be new, unused, and of current manufacture."


def a_source(*lines: str) -> SourceText:
    """A SourceText over pages built from the given lines, one page each."""
    return source_text(
        TenderPackage(
            name="fixture",
            root_path=Path("data/tenders/fixture"),
            documents=[
                TenderDocument(
                    document_id=f"{n:02d}_doc",
                    filename=f"{n:02d}_doc.pdf",
                    media_type="pdf",
                    title=f"Document {n}",
                    pages=[DocumentPage(document_id=f"{n:02d}_doc", page_number=1, text=line)],
                    sha256=f"{n:064d}",
                )
                for n, line in enumerate(lines or (TENDER_LINE,), start=1)
            ],
            has_gold=False,
            gold_path=None,
        )
    )


def a_recovery_item(gold_id: str = "G1", **overrides: object) -> GoldItem:
    return a_gold_item(gold_id, expects_recovered_statement=True, **overrides)  # type: ignore[arg-type]


def recovery_run(item: GoldItem, finding: Finding, source: SourceText | None = None) -> object:
    report = match_findings([finding], [item], source=source or a_source())
    return report


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  All   equipment\n shall be new. ", "all equipment shall be new."),
        ("ALL EQUIPMENT", "all equipment"),
        ("a\tb\nc", "a b c"),
        ("", ""),
    ],
)
def test_flatten_collapses_whitespace_and_case_and_nothing_else(raw: str, expected: str) -> None:
    # Punctuation and word forms are left alone on purpose: touching them would start
    # deciding how close a paraphrase has to be, which is a judgement, not a check.
    assert flatten(raw) == expected


def test_a_quote_is_recognised_through_rewrapping_and_recapitalisation() -> None:
    source = a_source(TENDER_LINE)

    assert source.quotes("All equipment shall be new, unused, and of current manufacture.")
    assert source.quotes("ALL  EQUIPMENT\n   SHALL BE NEW,\nunused, and of current manufacture.")
    assert not source.quotes("Every item supplied has to be of a currently made model.")


def test_a_finding_that_states_the_obligation_in_its_own_words_matches() -> None:
    item = a_recovery_item()
    finding = a_finding("F1", refs=["ITB-G1"])
    finding = finding.model_copy(
        update={
            "recovered_statement": "Catalogue items near end of life are disqualified on age alone."
        }
    )

    report = recovery_run(item, finding)

    assert [a.gold_id for a in report.matches] == ["G1"]
    assert not report.output_misses


def test_a_finding_that_only_quotes_the_source_line_scores_output_miss() -> None:
    # The whole of the claim. A regex can copy a line; copying is retrieval, and the item
    # asks for the obligation stated, which is not the same act.
    item = a_recovery_item()
    finding = a_finding("F1", refs=["ITB-G1"]).model_copy(
        update={"recovered_statement": TENDER_LINE}
    )

    report = recovery_run(item, finding)

    assert not report.matches
    assert [(a.gold_id, a.missing_outputs) for a in report.output_misses] == [
        ("G1", ["recovered_statement"])
    ]


def test_a_quote_lifted_from_the_middle_of_a_page_is_still_a_quote() -> None:
    # Containment, not equality: quoting half a clause is quoting.
    item = a_recovery_item()
    finding = a_finding("F1", refs=["ITB-G1"]).model_copy(
        update={"recovered_statement": "shall be new, unused"}
    )

    report = recovery_run(item, finding)

    assert not report.matches


@pytest.mark.parametrize("claimed", [None, "", "   ", "\n\t "])
def test_an_absent_or_blank_recovered_statement_is_not_a_recovery(claimed: str | None) -> None:
    item = a_recovery_item()
    finding = a_finding("F1", refs=["ITB-G1"]).model_copy(update={"recovered_statement": claimed})

    report = recovery_run(item, finding)

    assert not report.matches
    assert report.output_misses[0].missing_outputs == ["recovered_statement"]


def test_recovers_statement_is_the_two_conditions_and_nothing_else() -> None:
    source = a_source(TENDER_LINE)
    quoting = a_finding("F1", refs=["ITB-G1"]).model_copy(
        update={"recovered_statement": TENDER_LINE}
    )
    own_words = quoting.model_copy(
        update={"recovered_statement": "New kit only, nothing near EOL."}
    )
    silent = quoting.model_copy(update={"recovered_statement": None})

    assert not recovers_statement(quoting, source)
    assert recovers_statement(own_words, source)
    assert not recovers_statement(silent, source)


def test_the_recovery_rule_does_not_reach_an_item_that_did_not_ask_for_one() -> None:
    item = a_gold_item("G1")  # expects_recovered_statement defaults to False
    finding = a_finding("F1", refs=["ITB-G1"]).model_copy(
        update={"recovered_statement": TENDER_LINE}
    )

    report = recovery_run(item, finding)

    assert [a.gold_id for a in report.matches] == ["G1"]


def test_scoring_an_item_that_expects_a_recovery_without_a_source_raises() -> None:
    # Not a silently skipped check. A skipped check is how the claim that a keyword rule
    # recovers nothing survived a year while being false.
    gold = [a_recovery_item("G1"), a_recovery_item("G2"), a_gold_item("G3")]

    with pytest.raises(MatcherError) as caught:
        match_findings([a_finding("F1", refs=["ITB-G1"])], gold)

    message = str(caught.value)
    assert "G1" in message and "G2" in message
    assert "2 gold item(s)" in message


def test_no_source_is_fine_when_nothing_asks_for_a_recovery() -> None:
    gold = [a_gold_item("G1")]

    report = match_findings([a_finding("F1", refs=["ITB-G1"])], gold)

    assert [a.gold_id for a in report.matches] == ["G1"]


def test_source_text_reads_every_page_of_every_document() -> None:
    source = a_source("first page text", "second page text", "third page text")

    assert len(source.pages) == 3
    assert source.quotes("SECOND page   text")


def test_the_recovery_requirement_rides_with_the_other_expected_outputs() -> None:
    item = a_recovery_item("G1", expects_clarification_question=True)

    assert item.expected_outputs == ["clarification_question", "recovered_statement"]


# --- the ceiling behind the recall figure ---------------------------------------------------
#
# The first full scored run read 0 / 186. Most of that denominator was never reachable: only
# the items whose classes map to a finding type some registered detector emits can be
# answered at all, and of those, the ones demanding an output nothing produces cannot reach
# MATCH however well the clause is read. These tests pin both denominators, because a recall
# figure whose ceiling is not stated beside it is not interpretable.


EMITTING = frozenset({"missing_tolerance", "atomicity_split", "unverifiable_requirement"})


def test_only_items_a_registered_detector_could_answer_are_addressable() -> None:
    gold = [
        a_gold_item("G1", classes=["TOLERANCE"], finding_type="missing_tolerance"),
        a_gold_item("G2", classes=["COMPOUND"], finding_type="atomicity_split"),
        # Nothing emits contract_red_flag, so this one is out of the addressable set and
        # still in recall's denominator — which is the whole distinction.
        a_gold_item("G3", classes=["COMMERCIAL"], finding_type="contract_red_flag"),
    ]
    report = match_findings([], gold, source=None)
    card = score(
        tender_name="t",
        gold=gold,
        finding_ids=[],
        report=report,
        emitted_finding_types=EMITTING,
    )

    assert card.addressable is not None
    assert card.addressable.scored_total == 3
    assert card.addressable.total == 2
    assert [item.gold_id for item in card.addressable.items] == ["G1", "G2"]


def test_an_item_demanding_an_output_nothing_emits_is_outside_the_ceiling() -> None:
    gold = [
        a_gold_item("G1", classes=["TOLERANCE"], finding_type="missing_tolerance"),
        a_gold_item(
            "G2",
            classes=["COMPOUND"],
            finding_type="atomicity_split",
            expects_split=True,
        ),
    ]
    report = match_findings([], gold, source=None)
    card = score(
        tender_name="t",
        gold=gold,
        finding_ids=[],
        report=report,
        emitted_finding_types=EMITTING,
        emitted_outputs=frozenset(),
    )
    assert card.addressable is not None

    assert card.addressable.total == 2
    assert card.addressable.reachable_total == 1, "G2 demands split_children, which nothing emits"
    assert card.addressable.blocked_on_outputs == 1
    rows = {item.gold_id: item for item in card.addressable.items}
    assert rows["G1"].reachable and not rows["G1"].unemitted_demands
    assert not rows["G2"].reachable
    assert rows["G2"].unemitted_demands == ["split_children"]


def test_emitting_the_output_moves_the_item_inside_the_ceiling() -> None:
    """The same gold set, the same detectors, one output now produced. The ceiling moves."""
    gold = [
        a_gold_item("G2", classes=["COMPOUND"], finding_type="atomicity_split", expects_split=True)
    ]
    report = match_findings([], gold, source=None)
    card = score(
        tender_name="t",
        gold=gold,
        finding_ids=[],
        report=report,
        emitted_finding_types=EMITTING,
        emitted_outputs=frozenset({"split_children"}),
    )

    assert card.addressable is not None
    assert card.addressable.reachable_total == 1
    assert card.addressable.blocked_on_outputs == 0


def test_each_addressable_item_says_what_blocked_it() -> None:
    """The four outcomes, each with the reason that distinguishes it from the others."""
    gold = [
        a_gold_item("MATCHED", refs=["A-1"], classes=["TOLERANCE"]),
        a_gold_item("NO_OUTPUT", refs=["A-2"], classes=["COMPOUND"], expects_split=True),
        a_gold_item("WRONG_TYPE", refs=["A-3"], classes=["TOLERANCE"]),
        a_gold_item("UNTOUCHED", refs=["A-4"], classes=["TOLERANCE"]),
    ]
    findings = [
        a_finding("f1", refs=["A-1"], finding_type="missing_tolerance"),
        a_finding("f2", refs=["A-2"], finding_type="atomicity_split"),
        # Cites the right clause, says the wrong kind of thing about it.
        a_finding("f3", refs=["A-3"], finding_type="atomicity_split"),
    ]
    report = match_findings(findings, gold, source=None)
    card = score(
        tender_name="t",
        gold=gold,
        finding_ids=[f.finding_id for f in findings],
        report=report,
        emitted_finding_types=EMITTING,
    )

    assert card.addressable is not None
    rows = {item.gold_id: item for item in card.addressable.items}

    assert rows["MATCHED"].outcome == "match"
    assert rows["MATCHED"].blocked_by == ""

    assert rows["NO_OUTPUT"].outcome == "output_miss"
    assert "split_children" in rows["NO_OUTPUT"].blocked_by

    assert rows["WRONG_TYPE"].outcome == "partial"
    assert "atomicity_split" in rows["WRONG_TYPE"].blocked_by

    assert rows["UNTOUCHED"].outcome == "miss"
    assert "no finding cited" in rows["UNTOUCHED"].blocked_by

    assert card.addressable.found == 1
    assert card.addressable.recall == pytest.approx(0.25)


def test_the_ceiling_is_printed_beside_the_recall_it_bounds() -> None:
    gold = [
        a_gold_item("G1", classes=["TOLERANCE"]),
        a_gold_item("G2", classes=["COMPOUND"], expects_split=True),
        a_gold_item("G3", classes=["COMMERCIAL"], finding_type="contract_red_flag"),
    ]
    report = match_findings([], gold, source=None)
    card = score(
        tender_name="t",
        gold=gold,
        finding_ids=[],
        report=report,
        emitted_finding_types=EMITTING,
    )
    rendered = render_markdown(card)

    assert "Recall over addressable items" in rendered
    assert "0 / 2" in rendered
    assert "Every addressable item" in rendered
    # The per-item rows, and the items that are not addressable staying out of them.
    assert "`G1`" in rendered and "`G2`" in rendered
    assert "`G3`" not in rendered
    # The ceiling is stated before the table, in words a reader can act on.
    assert rendered.index("addressable") < rendered.index("Every addressable item")


def test_a_card_with_no_ceiling_stated_prints_none() -> None:
    """The baseline scores without a detector registry; it must not gain an invented ceiling."""
    gold = [a_gold_item("G1", classes=["TOLERANCE"])]
    card = score(
        tender_name="t", gold=gold, finding_ids=[], report=match_findings([], gold, source=None)
    )

    assert card.addressable is None
    assert "addressable" not in render_markdown(card)


def test_the_real_gold_set_has_the_ceiling_the_detectors_can_reach() -> None:
    """Pinned against the committed gold set and the registered detector map.

    Not a target — a measurement of the gap between what the gold set plants and what the
    six single-clause detectors can answer. It moves when a detector or an output is added,
    and it should fail here when it does, so the number in the report is never stale.
    """
    from ri05_tender.eval.findings_run import DETECTOR_TO_FINDING_TYPE, EMITTED_OUTPUTS
    from ri05_tender.tender.loader import load_tender

    gold = load_gold(load_tender(KESSLER_POINT))
    report = match_findings([], gold.scored, source=source_text(load_tender(KESSLER_POINT)))
    card = score(
        tender_name="kessler_point",
        gold=gold.scored,
        finding_ids=[],
        report=report,
        emitted_finding_types=frozenset(DETECTOR_TO_FINDING_TYPE.values()),
        emitted_outputs=EMITTED_OUTPUTS,
    )

    assert card.addressable is not None
    assert card.addressable.scored_total == 186
    assert card.addressable.total == 32, "items the six detector families can speak to at all"
    # Ten, not five: `split_children` is now carried through from the splitter's own lineage,
    # so the five items demanding only a split joined the five demanding nothing. This number
    # moves with `EMITTED_OUTPUTS` and the two are changed together.
    assert card.addressable.reachable_total == 10, "items demanding no output nothing emits"
    assert card.addressable.blocked_on_outputs == 22
    assert [item.gold_id for item in card.addressable.items if item.reachable] == [
        "D1-05",
        "D1-19",
        "D2-12",
        "D2-13",
        "D4-03",
        "D4-04",
        "D4-06",
        "D4-23",
        "D4-24",
        "D4-37",
    ]


def test_the_split_children_an_atomicity_finding_holds_reach_the_scorer() -> None:
    """Capability, now that the plumbing is there.

    `req_core` populates `Finding.children` on every atomicity_split finding and refuses one
    with fewer than two. `as_eval_finding` now passes them to `FindingOutputs.split_children`,
    so an item whose only demand is a split can reach MATCH. This and `EMITTED_OUTPUTS` and
    the pinned ceiling above are one fact in three places, and they move together.
    """
    from req_core.findings import Evidence
    from req_core.findings import Finding as CoreFinding
    from ri05_tender.eval.findings_run import EMITTED_OUTPUTS, as_eval_finding
    from spine.contracts import EvidenceRef

    core = CoreFinding(
        finding_id="C-1::atomicity_split::1",
        detector="atomicity_split",
        requirement_id="C-1",
        severity="major",
        statement="The clause states three separable obligations.",
        evidence=Evidence(summary="three obligations"),
        source=EvidenceRef(source_id="s", document="d", page=1, clause="C-1", quote="shall"),
        confidence=0.9,
        children=["C-1.1", "C-1.2", "C-1.3"],
    )

    converted = as_eval_finding(core)
    assert converted.outputs.split_children == ["C-1.1", "C-1.2", "C-1.3"]
    assert converted.outputs.present() == {"split_children"}
    assert "split_children" in EMITTED_OUTPUTS


def test_a_finding_from_any_other_detector_carries_no_split_children() -> None:
    """No branch on the detector name is needed, and this is why: children are always empty.

    `req_core.findings.Finding` refuses children on anything but an atomicity_split, so the
    adapter can pass them through unconditionally without ever inventing a split.
    """
    from req_core.findings import Evidence
    from req_core.findings import Finding as CoreFinding
    from ri05_tender.eval.findings_run import as_eval_finding
    from spine.contracts import EvidenceRef

    core = CoreFinding(
        finding_id="C-2::missing_tolerance::1",
        detector="missing_tolerance",
        requirement_id="C-2",
        severity="major",
        statement="A property is constrained and never bounded.",
        evidence=Evidence(summary="unbounded"),
        source=EvidenceRef(source_id="s", document="d", page=1, clause="C-2", quote="shall"),
        confidence=0.85,
    )

    assert as_eval_finding(core).outputs.split_children == []
    assert as_eval_finding(core).outputs.present() == set()


def test_an_item_demanding_only_a_split_can_now_be_matched() -> None:
    """End to end through the matcher: the plumbing is what turns OUTPUT_MISS into MATCH."""
    gold = [a_gold_item("G", refs=["A-1"], classes=["COMPOUND"], expects_split=True)]
    with_children = a_finding(
        "f1", refs=["A-1"], finding_type="atomicity_split", split_children=["A-1.1", "A-1.2"]
    )
    without = a_finding("f2", refs=["A-1"], finding_type="atomicity_split")

    assert match_findings([with_children], gold, source=None).matches
    assert not match_findings([without], gold, source=None).matches
    assert match_findings([without], gold, source=None).output_misses


# --- what the confidence threshold cost -----------------------------------------------------


def test_a_withheld_finding_that_would_have_matched_shows_as_a_delta() -> None:
    """The measurement that separates "cannot find it" from "found it and declined"."""
    gold = [
        a_gold_item("FOUND", refs=["A-1"], classes=["TOLERANCE"]),
        a_gold_item("WITHHELD", refs=["A-2"], classes=["TOLERANCE"]),
    ]
    asserted = [a_finding("f1", refs=["A-1"], finding_type="missing_tolerance")]
    abstained = [a_finding("f2", refs=["A-2"], finding_type="missing_tolerance", confidence=0.5)]

    report = match_findings(asserted, gold, source=None)
    if_asserted = match_findings(asserted + abstained, gold, source=None)
    card = score(
        tender_name="t",
        gold=gold,
        finding_ids=[f.finding_id for f in asserted],
        report=report,
        if_asserted=if_asserted,
        abstained_findings=len(abstained),
    )

    assert card.abstention is not None
    assert card.recall_overall == pytest.approx(0.5), "the run's own recall is unchanged"
    assert card.abstention.overall.value == pytest.approx(0.5)
    assert card.abstention.overall.value_if_asserted == pytest.approx(1.0)
    assert card.abstention.overall.delta == pytest.approx(0.5)
    assert card.abstention.overall.recovered == 1
    assert card.abstention.abstained_findings == 1
    assert card.abstention.abstention_rate == pytest.approx(0.5)


def test_a_zero_delta_says_the_detectors_did_not_find_it() -> None:
    """Withheld findings that match nothing cost nothing, and the report must say so."""
    gold = [a_gold_item("G", refs=["A-1"], classes=["TOLERANCE"])]
    abstained = [a_finding("f2", refs=["ELSEWHERE-9"], finding_type="missing_tolerance")]

    report = match_findings([], gold, source=None)
    card = score(
        tender_name="t",
        gold=gold,
        finding_ids=[],
        report=report,
        if_asserted=match_findings(abstained, gold, source=None),
        abstained_findings=len(abstained),
    )

    assert card.abstention is not None
    assert card.abstention.overall.delta == pytest.approx(0.0)
    assert card.abstention.overall.recovered == 0
    assert "cost this run nothing in recall" in render_markdown(card)


def test_the_delta_is_reported_per_class_so_each_zero_can_be_read() -> None:
    gold = [
        a_gold_item("T1", refs=["A-1"], classes=["TOLERANCE"]),
        a_gold_item("M1", refs=["A-2"], classes=["MODALITY"]),
    ]
    abstained = [
        a_finding("f2", refs=["A-2"], finding_type="modality_inconsistency", confidence=0.55)
    ]
    card = score(
        tender_name="t",
        gold=gold,
        finding_ids=[],
        report=match_findings([], gold, source=None),
        if_asserted=match_findings(abstained, gold, source=None),
        abstained_findings=1,
    )

    assert card.abstention is not None
    by_class = {pair.key: pair for pair in card.abstention.by_class}
    assert by_class["MODALITY"].delta == pytest.approx(1.0), "found, and declined to say so"
    assert by_class["TOLERANCE"].delta == pytest.approx(0.0), "not found at all"

    rendered = render_markdown(card)
    assert "If asserted" in rendered and "Δ" in rendered
    assert "What abstention cost" in rendered
    assert "Shapes the threshold makes unreportable" in rendered


def test_a_card_with_no_abstention_report_prints_no_delta_columns() -> None:
    """The baseline measures no abstention; it must not gain a column implying it cost zero."""
    gold = [a_gold_item("G1", classes=["TOLERANCE"])]
    card = score(
        tender_name="t", gold=gold, finding_ids=[], report=match_findings([], gold, source=None)
    )

    assert card.abstention is None
    rendered = render_markdown(card)
    assert "If asserted" not in rendered
    assert "What abstention cost" not in rendered


def test_precision_carries_no_if_asserted_reading() -> None:
    """Scoring withheld findings moves precision's denominator too, so the cells stay empty."""
    gold = [a_gold_item("G", refs=["A-1"], classes=["TOLERANCE"])]
    card = score(
        tender_name="t",
        gold=gold,
        finding_ids=[],
        report=match_findings([], gold, source=None),
        if_asserted=match_findings([], gold, source=None),
    )
    rendered = render_markdown(card)

    strict = next(line for line in rendered.splitlines() if "Precision (strict)" in line)
    assert strict.rstrip().endswith("— | — |"), strict
