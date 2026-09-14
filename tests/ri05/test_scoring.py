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
    MatcherError,
    allowed_finding_types,
    match_findings,
    normalise_ref,
)
from ri05_tender.eval.metrics import SEVERITY_WEIGHTS, score
from ri05_tender.eval.models import Finding, FindingOutputs, GoldItem, GoldReview
from ri05_tender.eval.report import render_markdown
from ri05_tender.tender.models import TenderPackage

KESSLER_POINT = Path("data/tenders/kessler_point")
AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


# --- fixtures, built by hand because nothing produces a Finding yet ------------------------


def a_gold_item(
    gold_id: str,
    *,
    refs: list[str] | None = None,
    classes: list[str] | None = None,
    finding_type: str = "implicit_requirement",
    severity: str = "major",
    tier: str = "cold_start",
    scored: bool = True,
    **expects: bool,
) -> GoldItem:
    payload: dict[str, object] = {
        "id": gold_id,
        "document": 1,
        "refs": refs or [f"ITB-{gold_id}"],
        "classes": classes or ["IMPLICIT"],
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
    finding_type: str = "implicit_requirement",
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
    item = a_gold_item("G1", classes=["IMPLICIT", "COMMERCIAL"])

    assert allowed_finding_types(item) == {"implicit_requirement", "contract_red_flag"}


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
    findings = [a_finding("F1", refs=["ITB-9.2"], finding_type="implicit_requirement")]

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
        | set(report.unmatched)
    )

    assert placed == {"F1", "F2", "F3", "F5"}
    assert (
        len(report.matches)
        + len(report.output_misses)
        + len(report.partials)
        + len(report.unmatched)
    ) == len(findings)


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
    # The loser anchored a gold item, so it is a PARTIAL rather than an UNMATCHED finding.
    assert [partial.finding_id for partial in report.partials] == ["F1"]


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


def test_the_tiebreak_can_prefer_an_output_miss_over_a_match() -> None:
    """Pins a consequence of the specified tiebreak, so a change to it is visible.

    The rule is overlap, then confidence, then finding id — it says nothing about
    preferring a complete answer. So a finding that anchors more references but omits a
    required output takes the item, and the finding that fully answered it is left over as
    a PARTIAL. Implemented as specified rather than quietly improved; if the intended rule
    is "prefer a MATCH first", this test is where that change lands.
    """
    gold = [a_gold_item("G1", refs=["A-1", "A-2"], expects_clarification_question=True)]
    findings = [
        a_finding("F1", refs=["A-1", "A-2"], confidence=0.9),  # overlap 2, no question
        a_finding("F2", refs=["A-1"], confidence=0.9, clarification_question="q"),  # overlap 1
    ]

    report = match_findings(findings, gold)

    assert [a.finding_id for a in report.output_misses] == ["F1"]
    assert report.matches == []
    assert [partial.finding_id for partial in report.partials] == ["F2"]


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
    # IMPLICIT covers G1, G3, G4; two of the three were found.
    assert klass["IMPLICIT"].value == pytest.approx(2 / 3)
    assert klass["COMMERCIAL"].value == 0.0


def test_implicit_recovery_is_recall_over_the_implicit_items_only() -> None:
    gold = [
        a_gold_item("G1", classes=["IMPLICIT"]),
        a_gold_item("G2", classes=["IMPLICIT"]),
        a_gold_item("G3", classes=["PROCESS"], finding_type="submission_constraint"),
    ]
    findings = [
        a_finding("F1", refs=["ITB-G1"]),
        a_finding("F3", refs=["ITB-G3"], finding_type="submission_constraint"),
    ]

    card = scored_run(gold, findings)

    # 1 of the 2 IMPLICIT items, not 2 of 3 overall.
    assert card.implicit_recovery == pytest.approx(0.5)
    assert card.implicit_total == 2
    assert card.recall_overall == pytest.approx(2 / 3)


def test_a_group_with_no_items_reports_zero_over_zero_rather_than_raising() -> None:
    card = scored_run([], [])

    assert card.recall_overall == 0.0
    assert card.recall_weighted == 0.0
    assert card.implicit_recovery == 0.0


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
            a_gold_payload(id="G1", refs=["ITB-1"], classes=["IMPLICIT"]),
            a_gold_payload(id="G2", refs=["ITB-2"], classes=["IMPLICIT"], severity="no_bid"),
            a_gold_payload(
                id="G3", refs=["ITB-3"], classes=["IMPLICIT"], severity="critical", scored=False
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
