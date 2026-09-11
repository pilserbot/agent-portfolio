"""Unit tests for the deterministic checkers.

No network, no model. Every expected score is worked out by hand in the test that asserts
it, so a change to a formula shows up as a failing number rather than a passing rewrite.
"""

from decimal import Decimal

import pytest
from pydantic import BaseModel, Field

from spine.contracts import EvidenceRef, Verdict
from spine.eval.checkers import (
    Claim,
    citation_resolves,
    exact_match,
    numeric_within_tolerance,
    reference_integrity,
    schema_valid,
    set_f1,
)

ITEM = "item-1"


class Person(BaseModel):
    """A schema for the schema_valid tests."""

    name: str
    age: int = Field(ge=0)


def a_reference(source_id: str = "src-1", quote: str = "a quoted passage") -> EvidenceRef:
    return EvidenceRef(
        source_id=source_id, document="tender.pdf", page=4, clause="3.2.1", quote=quote
    )


# --- every checker returns a well-formed Verdict -------------------------------------------


def test_every_checker_returns_a_verdict() -> None:
    results = [
        schema_valid({"name": "Ada", "age": 36}, Person, item_id=ITEM),
        reference_integrity([a_reference()], item_id=ITEM),
        citation_resolves(
            Claim(claim_id="c", text="t", evidence=[a_reference()]), {"src-1"}, item_id=ITEM
        ),
        exact_match("a", "a", item_id=ITEM),
        set_f1({"a"}, {"a"}, item_id=ITEM),
        numeric_within_tolerance(1, 1, item_id=ITEM),
    ]

    for verdict in results:
        assert isinstance(verdict, Verdict)
        assert verdict.item_id == ITEM
        assert 0.0 <= verdict.score <= 1.0
        assert verdict.rationale
        assert verdict.judge_model is None, "no checker here consults a model"


# --- schema_valid ----------------------------------------------------------------------------


def test_schema_valid_accepts_a_mapping() -> None:
    verdict = schema_valid({"name": "Ada", "age": 36}, Person, item_id=ITEM)

    assert verdict.passed and verdict.score == 1.0


def test_schema_valid_accepts_a_json_string() -> None:
    assert schema_valid('{"name": "Ada", "age": 36}', Person, item_id=ITEM).passed


def test_schema_valid_accepts_an_instance() -> None:
    assert schema_valid(Person(name="Ada", age=36), Person, item_id=ITEM).passed


def test_schema_valid_rejects_a_missing_field_and_names_it() -> None:
    verdict = schema_valid({"name": "Ada"}, Person, item_id=ITEM)

    assert not verdict.passed
    assert verdict.score == 0.0
    assert "age" in verdict.rationale


def test_schema_valid_rejects_a_constraint_violation() -> None:
    verdict = schema_valid({"name": "Ada", "age": -1}, Person, item_id=ITEM)

    assert not verdict.passed
    assert "age" in verdict.rationale


def test_schema_valid_rejects_unparsable_json() -> None:
    verdict = schema_valid("{not json", Person, item_id=ITEM)

    assert not verdict.passed and verdict.score == 0.0


# --- reference_integrity -----------------------------------------------------------------------


def test_reference_integrity_passes_when_all_resolve() -> None:
    verdict = reference_integrity([a_reference("s1"), a_reference("s2")], item_id=ITEM)

    assert verdict.passed and verdict.score == 1.0


def test_reference_integrity_scores_the_resolved_fraction() -> None:
    # 3 of 4 resolve: one is None, so the score is 0.75.
    verdict = reference_integrity(
        [a_reference("a"), a_reference("b"), a_reference("c"), None], item_id=ITEM
    )

    assert not verdict.passed
    assert verdict.score == 0.75


def test_a_reference_with_a_blank_quote_does_not_resolve() -> None:
    # It points at a page but shows nothing, which is a citation rather than evidence.
    verdict = reference_integrity([a_reference(quote="   ")], item_id=ITEM)

    assert not verdict.passed and verdict.score == 0.0


def test_no_references_at_all_does_not_pass() -> None:
    verdict = reference_integrity([], item_id=ITEM)

    assert not verdict.passed
    assert verdict.score == 0.0
    assert "no references" in verdict.rationale


# --- citation_resolves --------------------------------------------------------------------------


def test_citation_resolves_when_every_source_is_known() -> None:
    claim = Claim(claim_id="c1", text="t", evidence=[a_reference("s1"), a_reference("s2")])

    verdict = citation_resolves(claim, {"s1", "s2", "s3"}, item_id=ITEM)

    assert verdict.passed and verdict.score == 1.0


def test_a_dangling_citation_fails_and_is_named() -> None:
    claim = Claim(claim_id="c1", text="t", evidence=[a_reference("s1"), a_reference("ghost")])

    verdict = citation_resolves(claim, {"s1"}, item_id=ITEM)

    assert not verdict.passed
    assert verdict.score == 0.5, "one of two citations resolved"
    assert "ghost" in verdict.rationale


def test_a_claim_citing_nothing_does_not_pass() -> None:
    verdict = citation_resolves(Claim(claim_id="c1", text="t"), {"s1"}, item_id=ITEM)

    assert not verdict.passed
    assert verdict.score == 0.0
    assert "cites no source" in verdict.rationale


def test_an_empty_source_set_makes_every_citation_dangle() -> None:
    claim = Claim(claim_id="c1", text="t", evidence=[a_reference("s1")])

    assert citation_resolves(claim, set(), item_id=ITEM).score == 0.0


# --- exact_match ------------------------------------------------------------------------------


def test_exact_match_passes_on_identical_strings() -> None:
    verdict = exact_match("monthly reporting", "monthly reporting", item_id=ITEM)

    assert verdict.passed and verdict.score == 1.0


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("yes", "Yes"),
        ("a b", "a  b"),
        ("total", " total"),
        ("cap", "cap."),
    ],
)
def test_exact_match_is_exact(expected: str, actual: str) -> None:
    # No case folding, no whitespace collapsing, no punctuation stripping: a caller wanting
    # a looser comparison normalises at the call site where the looseness is visible.
    assert not exact_match(expected, actual, item_id=ITEM).passed


def test_exact_match_compares_structures_deterministically() -> None:
    # Key order must not decide the answer.
    assert exact_match({"a": 1, "b": 2}, {"b": 2, "a": 1}, item_id=ITEM).passed


def test_exact_match_distinguishes_different_structures() -> None:
    assert not exact_match({"a": 1}, {"a": 2}, item_id=ITEM).passed


# --- set_f1 -------------------------------------------------------------------------------------


def test_set_f1_is_one_on_an_exact_match() -> None:
    verdict = set_f1({"iso-9001", "iso-27001"}, {"iso-27001", "iso-9001"}, item_id=ITEM)

    assert verdict.passed and verdict.score == 1.0


def test_set_f1_on_a_partial_overlap() -> None:
    # |E|=3, |A|=3, overlap=2 -> P=2/3, R=2/3, F1=2*(2/3)*(2/3)/(4/3)=2/3.
    verdict = set_f1({"a", "b", "c"}, {"b", "c", "d"}, item_id=ITEM)

    assert verdict.score == pytest.approx(2 / 3)
    assert not verdict.passed


def test_set_f1_when_the_answer_is_a_superset() -> None:
    # |E|=2, |A|=4, overlap=2 -> P=0.5, R=1.0, F1=2*0.5*1/1.5=2/3.
    verdict = set_f1({"a", "b"}, {"a", "b", "c", "d"}, item_id=ITEM)

    assert verdict.score == pytest.approx(2 / 3)
    assert "unexpected: c, d" in verdict.rationale


def test_set_f1_when_the_answer_is_a_subset() -> None:
    # |E|=4, |A|=2, overlap=2 -> P=1.0, R=0.5, F1=2/3.
    verdict = set_f1({"a", "b", "c", "d"}, {"a", "b"}, item_id=ITEM)

    assert verdict.score == pytest.approx(2 / 3)
    assert "missing: c, d" in verdict.rationale


def test_set_f1_with_no_overlap_is_zero() -> None:
    assert set_f1({"a"}, {"b"}, item_id=ITEM).score == 0.0


def test_two_empty_sets_agree_completely() -> None:
    verdict = set_f1(set(), set(), item_id=ITEM)

    assert verdict.passed and verdict.score == 1.0


def test_an_empty_answer_against_a_non_empty_expectation_is_zero() -> None:
    assert set_f1({"a"}, set(), item_id=ITEM).score == 0.0


def test_an_answer_where_nothing_was_expected_is_zero() -> None:
    assert set_f1(set(), {"a"}, item_id=ITEM).score == 0.0


def test_set_f1_accepts_any_iterable_and_ignores_duplicates() -> None:
    assert set_f1(["a", "a", "b"], ("b", "a"), item_id=ITEM).passed


# --- numeric_within_tolerance ---------------------------------------------------------------------


def test_a_number_inside_an_absolute_tolerance_passes() -> None:
    verdict = numeric_within_tolerance(100, 102, item_id=ITEM, tolerance=5)

    assert verdict.passed and verdict.score == 1.0


def test_a_number_outside_an_absolute_tolerance_fails() -> None:
    verdict = numeric_within_tolerance(100, 106, item_id=ITEM, tolerance=5)

    assert not verdict.passed
    assert "outside" in verdict.rationale


def test_a_difference_exactly_on_the_tolerance_passes() -> None:
    assert numeric_within_tolerance(100, 105, item_id=ITEM, tolerance=5).passed


def test_a_relative_tolerance_scales_with_the_expected_value() -> None:
    # 10% of 1000 is 100, so 1090 is inside and 1101 is not.
    assert numeric_within_tolerance(1000, 1090, item_id=ITEM, tolerance=0.1, relative=True).passed
    assert not numeric_within_tolerance(
        1000, 1101, item_id=ITEM, tolerance=0.1, relative=True
    ).passed


def test_a_relative_tolerance_on_zero_falls_back_to_equality() -> None:
    # 10% of nothing is nothing; dividing by the expected value would be worse.
    assert numeric_within_tolerance(0, 0, item_id=ITEM, tolerance=0.1, relative=True).passed
    assert not numeric_within_tolerance(0, 1, item_id=ITEM, tolerance=0.1, relative=True).passed


def test_money_compares_exactly_rather_than_by_binary_float() -> None:
    # 0.1 + 0.2 != 0.3 as binary floats; as decimals the sum is exact.
    total = Decimal("0.1") + Decimal("0.2")

    assert numeric_within_tolerance(Decimal("0.30"), total, item_id=ITEM, tolerance=0).passed


def test_decimal_strings_compare_exactly() -> None:
    assert numeric_within_tolerance("1250000.00", "1250000", item_id=ITEM, tolerance=0).passed


def test_a_zero_tolerance_demands_equality() -> None:
    assert not numeric_within_tolerance(100, 100.01, item_id=ITEM, tolerance=0).passed


def test_a_negative_difference_is_measured_by_magnitude() -> None:
    assert numeric_within_tolerance(100, 96, item_id=ITEM, tolerance=5).passed


@pytest.mark.parametrize("bad", ["not a number", "", None, float("nan"), float("inf")])
def test_a_non_number_fails_rather_than_raising(bad: object) -> None:
    verdict = numeric_within_tolerance(100, bad, item_id=ITEM, tolerance=5)

    assert not verdict.passed
    assert verdict.score == 0.0


def test_a_boolean_is_not_accepted_as_a_number() -> None:
    # True == 1 in Python; letting it through would score a label as a quantity.
    assert not numeric_within_tolerance(1, True, item_id=ITEM, tolerance=0).passed


def test_exact_match_renders_a_scalar_that_is_not_a_string() -> None:
    assert exact_match(42, "42", item_id=ITEM).passed
    assert not exact_match(42, "43", item_id=ITEM).passed


def test_exact_match_renders_a_decimal_inside_a_structure() -> None:
    # Decimal is not JSON-serialisable; it must render deterministically rather than raise.
    verdict = exact_match({"cap": Decimal("2.5")}, {"cap": 2.5}, item_id=ITEM)

    assert verdict.passed
    assert "2.5" in verdict.expected


def test_exact_match_renders_a_list_inside_a_structure() -> None:
    verdict = exact_match({"formats": ["pdf", "csv"]}, {"formats": ["pdf", "csv"]}, item_id=ITEM)

    assert verdict.passed
    # Order inside a list is significant; only mapping key order is normalised.
    assert not exact_match({"f": ["a", "b"]}, {"f": ["b", "a"]}, item_id=ITEM).passed
