"""Every detector, offline, including the cases where each must NOT fire.

No network, no model, no key: claims are handed in directly, which is what the `claims=`
argument on `engine.detect` exists for. That is not a shortcut around the model — it is the
architecture. The model's only job is to produce `ClauseClaims`; everything a detector
decides is decided from those records by Python, so a test that supplies the records
exercises the whole of the decision.

**Half of these tests are negative on purpose.** A detector that fires on everything has
perfect recall and is worthless — it moves the work of deciding onto the reader, which is
the work it was built to do. So each detector has its firing case and at least one case that
looks like it should fire and must not: a bounded constraint for `missing_tolerance`, a
clause with a stated test for `unverifiable`, an inequality for `zero_margin`, an ordinary
number for `ordinal_trap`, a single-modality clause for `modality_inconsistency`, an
unsplit clause for `atomicity_split`.

Also pinned: that no detector can set its own severity or mint its own id, that confidence
is computed rather than read off a model's answer, and that abstention is reported beside
the findings rather than instead of them.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from req_core.claims import ClauseClaims, Constraint, Obligation, PageClaims
from req_core.contracts import Requirement
from req_core.corpus import CorpusDocument, CorpusPage, SourceCorpus
from req_core.detectors import DETECTORS, DetectorContext, OrdinalScales, run_detectors
from req_core.detectors.base import FindingDraft, read_number, read_range
from req_core.engine import detect, finding_id_for, read_claims, route_for_review
from req_core.findings import DetectionReport, Evidence, Finding, SeverityPolicy
from req_core.policy import Modality, ModalityPolicy
from spine.contracts import EvidenceRef

# --- fixtures -----------------------------------------------------------------------------

PAGE_TEXT = "X-1.1 The Contractor shall provide a thing.\nX-1.2 The system should be nice."

HOUSE_STYLE = ModalityPolicy(
    name="house style",
    source="clause 2.1",
    mandatory=frozenset({"shall", "must"}),
    advisory=frozenset({"may"}),
    optional=frozenset({"will", "should"}),
)

UNSOURCED_STYLE = HOUSE_STYLE.model_copy(update={"source": "", "name": "assumed"})

SCALES = OrdinalScales(name="test catalogue", scales=frozenset({"ip", "nema"}))

SEVERITIES = SeverityPolicy(
    name="test policy",
    severities={name: "major" for name in DETECTORS},
)


def a_requirement(
    identifier: str = "X-1.1",
    *,
    text: str = "The Contractor shall provide a thing.",
    parent_id: str | None = None,
    is_atomic: bool = True,
) -> Requirement:
    return Requirement(
        requirement_id=identifier,
        text=text,
        modality=Modality.MANDATORY,
        source=EvidenceRef(
            source_id="04_Spec", document="Spec", page=1, clause=identifier, quote=text
        ),
        parent_id=parent_id,
        is_atomic=is_atomic,
    )


def a_context(
    *,
    requirement: Requirement | None = None,
    claims: ClauseClaims | None = None,
    siblings: tuple[str, ...] = (),
    modality: ModalityPolicy = HOUSE_STYLE,
) -> DetectorContext:
    return DetectorContext(
        requirement=requirement or a_requirement(),
        claims=claims,
        siblings=siblings,
        modality=modality,
        ordinal_scales=SCALES,
    )


def some_claims(
    *,
    constraints: list[Constraint] | None = None,
    obligations: list[str] | None = None,
    states_test_method: bool = False,
    conditions: list[str] | None = None,
) -> ClauseClaims:
    return ClauseClaims(
        identifier="X-1.1",
        constraints=constraints or [],
        obligations=[Obligation(text=item) for item in (obligations or [])],
        states_test_method=states_test_method,
        test_method="a stated test" if states_test_method else "",
        measurement_conditions=conditions or [],
    )


def a_constraint(**overrides: object) -> Constraint:
    payload: dict[str, object] = {
        "subject": "the pump",
        "attribute": "flow rate",
        "operator": "gte",
        "value": "7",
        "unit": "L/s",
    }
    payload.update(overrides)
    return Constraint.model_validate(payload)


def fired(context: DetectorContext, name: str) -> list[FindingDraft]:
    """What one named detector concluded about one clause."""
    return run_detectors(context, only=frozenset({name}))


def a_corpus() -> SourceCorpus:
    return SourceCorpus(
        name="fixture",
        documents=[
            CorpusDocument(
                document_id="04_Spec",
                pages=[CorpusPage(document_id="04_Spec", page_number=1, text=PAGE_TEXT)],
            )
        ],
    )


# --- the claims schema cannot carry a verdict ------------------------------------------------


@pytest.mark.parametrize(
    "field", ["severity", "confidence", "is_defective", "finding_type", "risk"]
)
def test_the_model_cannot_return_a_verdict_because_the_schema_has_nowhere_to_put_one(
    field: str,
) -> None:
    # The architectural rule, enforced by a type rather than by a prompt. Rewording the
    # instructions cannot erode this; adding a field would have to be a deliberate change.
    assert field not in ClauseClaims.model_fields
    assert field not in Constraint.model_fields
    with pytest.raises(ValidationError):
        ClauseClaims.model_validate({"identifier": "X-1.1", field: "critical"})


# --- missing_tolerance ------------------------------------------------------------------------


def test_missing_tolerance_fires_on_a_property_required_without_any_bound() -> None:
    claims = some_claims(
        constraints=[
            a_constraint(operator="unbounded", value="", unit="", attribute="illumination")
        ]
    )

    drafts = fired(a_context(claims=claims), "missing_tolerance")

    assert len(drafts) == 1
    assert drafts[0].confidence == 0.85
    assert "illumination" in drafts[0].statement


def test_missing_tolerance_is_less_sure_when_the_clause_states_a_test() -> None:
    # A stated test may carry the bound the clause omits, so the same shape is worth less.
    claims = some_claims(
        constraints=[a_constraint(operator="unbounded", value="")], states_test_method=True
    )

    assert fired(a_context(claims=claims), "missing_tolerance")[0].confidence == 0.65


def test_missing_tolerance_fires_on_a_bound_with_no_measurement_condition() -> None:
    claims = some_claims(constraints=[a_constraint()])

    drafts = fired(a_context(claims=claims), "missing_tolerance")

    assert len(drafts) == 1
    assert drafts[0].confidence == 0.7
    assert "measurement condition" in drafts[0].statement


def test_missing_tolerance_does_not_fire_on_a_bound_that_states_its_conditions() -> None:
    # The negative case that matters: a complete constraint is not a defect.
    claims = some_claims(constraints=[a_constraint()], conditions=["at the inlet, at 20 degrees"])

    assert not fired(a_context(claims=claims), "missing_tolerance")


def test_missing_tolerance_does_not_fire_on_a_clause_with_no_constraints() -> None:
    # A clause that is not about a quantity is not a clause missing one.
    assert not fired(a_context(claims=some_claims()), "missing_tolerance")


def test_missing_tolerance_does_not_fire_on_a_membership_test() -> None:
    # `one_of` has nothing to measure, so the absence of a measurement condition says
    # nothing about it.
    claims = some_claims(constraints=[a_constraint(operator="one_of", value="type A or type B")])

    assert not fired(a_context(claims=claims), "missing_tolerance")


def test_missing_tolerance_says_nothing_when_no_claims_were_read() -> None:
    # A missing reading becomes silence, never a clean bill of health.
    assert not fired(a_context(claims=None), "missing_tolerance")


# --- modality_inconsistency ---------------------------------------------------------------------


def test_modality_inconsistency_fires_on_a_clause_mixing_two_modalities() -> None:
    requirement = a_requirement(
        text="The Contractor shall provide a spares list and may substitute equivalents."
    )

    drafts = fired(a_context(requirement=requirement), "modality_inconsistency")

    assert len(drafts) == 1
    assert drafts[0].confidence == 0.8
    assert "mixes" in drafts[0].statement


def test_modality_inconsistency_is_worth_less_when_the_convention_is_assumed() -> None:
    # The contradiction is real either way; the convention it is judged against is not.
    requirement = a_requirement(text="The Contractor shall do it and may decline.")

    sourced = fired(a_context(requirement=requirement), "modality_inconsistency")
    assumed = fired(
        a_context(requirement=requirement, modality=UNSOURCED_STYLE), "modality_inconsistency"
    )

    assert sourced[0].confidence == 0.8
    assert assumed[0].confidence == 0.6


def test_modality_inconsistency_reads_the_convention_it_is_given() -> None:
    """The case the whole design exists for, on one sentence and two conventions.

    Under the house style `will` is optional; under an ordinary reading it is mandatory. The
    same clause is a finding under one and silence under the other, and no model is asked.
    """
    requirement = a_requirement(text="The Contractor will provide a spares list.")
    claims = some_claims(obligations=["Provide a spares list."])

    under_house = fired(a_context(requirement=requirement, claims=claims), "modality_inconsistency")

    ordinary = ModalityPolicy(
        name="ordinary",
        mandatory=frozenset({"shall", "will", "must"}),
        advisory=frozenset({"should"}),
        optional=frozenset({"may"}),
    )
    under_ordinary = fired(
        a_context(requirement=requirement, claims=claims, modality=ordinary),
        "modality_inconsistency",
    )

    assert len(under_house) == 1
    assert "optional" in under_house[0].statement
    assert not under_ordinary


def test_modality_inconsistency_does_not_fire_on_a_single_modality_clause() -> None:
    requirement = a_requirement(text="The Contractor shall provide a spares list and shall fit it.")

    assert not fired(a_context(requirement=requirement), "modality_inconsistency")


def test_modality_inconsistency_does_not_fire_on_a_clause_with_no_modal_verb() -> None:
    # A heading or a definition is not a defective requirement.
    requirement = a_requirement(text="This section describes the pumping systems.")

    assert not fired(a_context(requirement=requirement), "modality_inconsistency")


def test_modality_inconsistency_needs_no_model_call() -> None:
    # Claims are None and the detector still works: the convention is a word list and the
    # clause is text, so there is nothing to ask.
    requirement = a_requirement(text="The Contractor shall do it and may decline.")

    assert fired(a_context(requirement=requirement, claims=None), "modality_inconsistency")


# --- atomicity_split ------------------------------------------------------------------------------


def test_atomicity_split_fires_when_extraction_separated_the_clause() -> None:
    drafts = fired(a_context(siblings=("X-1.1/1", "X-1.1/2", "X-1.1/3")), "atomicity_split")

    assert len(drafts) == 1
    assert drafts[0].children == ["X-1.1/1", "X-1.1/2", "X-1.1/3"]
    assert drafts[0].confidence == 0.8


@pytest.mark.parametrize(("count", "confidence"), [(2, 0.7), (3, 0.8), (4, 0.9), (7, 0.9)])
def test_atomicity_confidence_rises_with_how_many_obligations_came_out(
    count: int, confidence: float
) -> None:
    siblings = tuple(f"X-1.1/{n}" for n in range(1, count + 1))

    assert fired(a_context(siblings=siblings), "atomicity_split")[0].confidence == confidence


def test_atomicity_split_does_not_fire_on_an_unsplit_clause() -> None:
    assert not fired(a_context(siblings=()), "atomicity_split")
    assert not fired(a_context(siblings=("X-1.1/1",)), "atomicity_split")


def test_atomicity_split_does_not_fire_on_a_child_of_a_split() -> None:
    # A child is not itself compound. Firing here would count one defect once per obligation.
    child = a_requirement("X-1.1/1", parent_id="X-1.1")

    assert not fired(a_context(requirement=child, siblings=("X-1.1/2",)), "atomicity_split")


def test_atomicity_split_needs_no_model_call() -> None:
    # It reads lineage extraction already produced rather than paying twice for the answer.
    assert fired(a_context(claims=None, siblings=("a", "b")), "atomicity_split")


# --- unverifiable --------------------------------------------------------------------------------


def test_unverifiable_fires_on_an_obligation_with_nothing_to_show_compliance_by() -> None:
    claims = some_claims(obligations=["Workmanship shall be of the highest standard."])

    drafts = fired(a_context(claims=claims), "unverifiable")

    assert len(drafts) == 1
    assert drafts[0].confidence == 0.8


def test_unverifiable_does_not_fire_when_the_clause_states_a_test() -> None:
    claims = some_claims(obligations=["Do the thing."], states_test_method=True)

    assert not fired(a_context(claims=claims), "unverifiable")


def test_unverifiable_does_not_fire_when_something_is_bounded_to_measure() -> None:
    # Measurable is demonstrable. Whether the bound is usable is missing_tolerance's
    # question, and answering it here too would report one defect under two names.
    claims = some_claims(obligations=["Do the thing."], constraints=[a_constraint()])

    assert not fired(a_context(claims=claims), "unverifiable")


def test_unverifiable_is_less_sure_when_the_constraints_are_all_unbounded() -> None:
    claims = some_claims(
        obligations=["Do the thing."], constraints=[a_constraint(operator="unbounded", value="")]
    )

    assert fired(a_context(claims=claims), "unverifiable")[0].confidence == 0.7


def test_unverifiable_abstains_on_an_advisory_clause() -> None:
    # An advisory clause with no test is usually guidance. 0.45 is below the default
    # threshold on purpose, so this shape is withheld rather than asserted.
    requirement = a_requirement(text="The Contractor should aim for a tidy installation.")
    claims = some_claims(obligations=["Aim for a tidy installation."])

    assert (
        fired(a_context(requirement=requirement, claims=claims), "unverifiable")[0].confidence
        == 0.45
    )


def test_unverifiable_does_not_fire_on_a_clause_that_obliges_nobody() -> None:
    assert not fired(a_context(claims=some_claims()), "unverifiable")


# --- ordinal_trap --------------------------------------------------------------------------------


def test_ordinal_trap_fires_on_an_ordering_comparison_over_a_catalogued_scale() -> None:
    claims = some_claims(
        constraints=[
            a_constraint(attribute="enclosure rating", operator="gte", value="54", scale="IP")
        ]
    )

    drafts = fired(a_context(claims=claims), "ordinal_trap")

    assert len(drafts) == 1
    assert drafts[0].confidence == 0.85
    assert "ordered categories" in drafts[0].statement


def test_ordinal_trap_does_not_fire_on_a_scale_the_catalogue_does_not_name() -> None:
    # An unknown scale is unknown. Guessing it is ordinal would produce findings about
    # ordinary numbers.
    claims = some_claims(
        constraints=[a_constraint(operator="gte", value="7", scale="some in-house index")]
    )

    assert not fired(a_context(claims=claims), "ordinal_trap")


def test_ordinal_trap_does_not_fire_on_an_ordinary_quantity() -> None:
    assert not fired(a_context(claims=some_claims(constraints=[a_constraint()])), "ordinal_trap")


def test_ordinal_trap_does_not_fire_on_equality_over_an_ordinal_scale() -> None:
    # "shall be IP66" names a category. Nothing is being compared arithmetically.
    claims = some_claims(constraints=[a_constraint(operator="eq", value="66", scale="IP", unit="")])

    assert not fired(a_context(claims=claims), "ordinal_trap")


def test_ordinal_trap_is_less_sure_when_a_test_method_may_resolve_the_axis() -> None:
    claims = some_claims(
        constraints=[a_constraint(operator="gte", value="54", scale="ip")], states_test_method=True
    )

    assert fired(a_context(claims=claims), "ordinal_trap")[0].confidence == 0.7


def test_the_catalogue_is_configuration_and_matching_ignores_case() -> None:
    claims = some_claims(constraints=[a_constraint(operator="gte", value="4", scale="NEMA")])
    empty = OrdinalScales(name="empty", scales=frozenset())

    assert fired(a_context(claims=claims), "ordinal_trap")
    context = a_context(claims=claims).model_copy(update={"ordinal_scales": empty})
    assert not run_detectors(context, only=frozenset({"ordinal_trap"}))


# --- zero_margin ---------------------------------------------------------------------------------


def test_zero_margin_fires_on_equality_over_a_physical_quantity() -> None:
    claims = some_claims(constraints=[a_constraint(operator="eq", value="7", unit="L/s")])

    drafts = fired(a_context(claims=claims), "zero_margin")

    assert len(drafts) == 1
    assert drafts[0].confidence == 0.75


def test_zero_margin_fires_on_a_range_of_zero_width() -> None:
    claims = some_claims(
        constraints=[a_constraint(operator="between", value="50 to 50", unit="dB")]
    )

    drafts = fired(a_context(claims=claims), "zero_margin")

    assert len(drafts) == 1
    assert drafts[0].confidence == 0.85
    assert "zero width" in drafts[0].statement


def test_zero_margin_does_not_fire_on_a_real_range() -> None:
    claims = some_claims(
        constraints=[a_constraint(operator="between", value="40 to 60", unit="dB")]
    )

    assert not fired(a_context(claims=claims), "zero_margin")


def test_zero_margin_does_not_fire_on_an_inequality() -> None:
    # `>= 7` has margin on one side by construction.
    assert not fired(a_context(claims=some_claims(constraints=[a_constraint()])), "zero_margin")


def test_zero_margin_abstains_on_an_exact_count() -> None:
    # "exactly two redundant controllers" is a design, not a defect. 0.4 is below the
    # default threshold, so this is withheld for a human rather than asserted.
    claims = some_claims(
        constraints=[a_constraint(operator="eq", value="2", unit="", attribute="controllers")]
    )

    assert fired(a_context(claims=claims), "zero_margin")[0].confidence == 0.4


def test_zero_margin_fires_at_a_percentage_endpoint() -> None:
    claims = some_claims(
        constraints=[a_constraint(operator="eq", value="100", unit="%", attribute="coverage")]
    )

    drafts = fired(a_context(claims=claims), "zero_margin")

    assert drafts[0].confidence == 0.75
    assert "no headroom" in drafts[0].evidence.summary


# --- number reading ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("80", 80.0), ("eighty (80)", 80.0), ("1,200", 1200.0), ("-40", -40.0), ("3.5", 3.5)],
)
def test_read_number_takes_the_figure_a_specification_writes(text: str, expected: float) -> None:
    assert read_number(text) == expected


@pytest.mark.parametrize("text", ["adequate", "as required", "to the Engineer's satisfaction", ""])
def test_read_number_declines_rather_than_inventing_a_bound(text: str) -> None:
    # A detector that turned "adequate" into a figure would manufacture the very bound whose
    # absence it reports.
    assert read_number(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("50 to 50", (50.0, 50.0)), ("40 to 60", (40.0, 60.0)), ("60 to 40", (40.0, 60.0))],
)
def test_read_range_takes_both_ends(text: str, expected: tuple[float, float]) -> None:
    assert read_range(text) == expected


@pytest.mark.parametrize("text", ["50", "", "two numbers 4 here 5 but no range word"])
def test_read_range_declines_when_two_ends_are_not_written(text: str) -> None:
    assert read_range(text) is None


# --- the frame: severity, ids, confidence, abstention --------------------------------------------


def test_a_detector_cannot_set_its_own_severity_or_mint_its_own_id() -> None:
    # Both are stamped by the engine. A detector wanting to call its own finding critical
    # would have to change `FindingDraft` to do it.
    assert "severity" not in FindingDraft.model_fields
    assert "finding_id" not in FindingDraft.model_fields


def test_severity_comes_from_the_callers_policy() -> None:
    claims = {"X-1.1": some_claims(constraints=[a_constraint(operator="unbounded", value="")])}
    mild = SeverityPolicy(name="mild", severities={name: "minor" for name in DETECTORS})
    harsh = SeverityPolicy(name="harsh", severities={name: "no_bid" for name in DETECTORS})

    def run(policy: SeverityPolicy) -> DetectionReport:
        return detect(
            [a_requirement()],
            a_corpus(),
            _never_called,
            modality=HOUSE_STYLE,
            severity=policy,
            ordinal_scales=SCALES,
            claims=claims,
        )

    assert {f.severity for f in run(mild).findings} == {"minor"}
    assert {f.severity for f in run(harsh).findings} == {"no_bid"}


def test_a_policy_missing_a_detector_refuses_rather_than_defaulting() -> None:
    # Silently defaulting would make an unweighted finding count as the mildest thing
    # in the report.
    incomplete = SeverityPolicy(name="partial", severities={"unverifiable": "critical"})

    with pytest.raises(KeyError, match="says nothing about"):
        incomplete.severity_for("zero_margin")


def test_finding_ids_are_deterministic() -> None:
    # Two runs over the same input produce the same ids, so a diff between reports is about
    # the findings rather than about their names.
    assert finding_id_for("X-1.1", "zero_margin", 1) == "X-1.1::zero_margin::1"


def test_findings_below_the_threshold_are_withheld_not_reported() -> None:
    # An exact count abstains at 0.4; an unbounded property asserts at 0.85.
    claims = {
        "X-1.1": some_claims(constraints=[a_constraint(operator="eq", value="2", unit="")]),
        "X-1.2": some_claims(constraints=[a_constraint(operator="unbounded", value="", unit="")]),
    }
    requirements = [a_requirement("X-1.1"), a_requirement("X-1.2")]

    report = detect(
        requirements,
        a_corpus(),
        _never_called,
        modality=HOUSE_STYLE,
        severity=SEVERITIES,
        ordinal_scales=SCALES,
        claims=claims,
        confidence_threshold=0.6,
    )

    # X-1.2 requires a property with no bound: asserted at 0.85. X-1.1 pins a count exactly
    # with no unit, which TWO detectors read — zero_margin at 0.4 and missing_tolerance at
    # 0.5 — and both are withheld. Two withheld drafts from one clause is what the run
    # produced; asserting a list of one here would have been asserting a tidier result.
    assert [f.requirement_id for f in report.findings] == ["X-1.2"]
    assert {f.requirement_id for f in report.abstained} == {"X-1.1"}
    assert {f.detector for f in report.abstained} == {"zero_margin", "missing_tolerance"}
    assert report.abstention_rate == pytest.approx(2 / 3)


def test_the_summary_reports_abstention_beside_the_findings() -> None:
    # Recall quoted without this number beside it is recall bought by guessing.
    report = DetectionReport(
        corpus_name="x",
        confidence_threshold=0.6,
        severity_policy="p",
        clauses_examined=2,
        findings=[_a_finding("a", 0.9)],
        abstained=[_a_finding("b", 0.4)],
    )

    summary = report.render_summary()

    assert "1 finding(s)" in summary
    assert "1 abstained (50.0%)" in summary
    assert "declined to assert" in summary


def test_an_empty_report_says_none_rather_than_printing_an_empty_table() -> None:
    empty = DetectionReport(
        corpus_name="x", confidence_threshold=0.6, severity_policy="p", clauses_examined=0
    )

    assert "_none_" in empty.render_summary()
    assert empty.abstention_rate == 0.0


def test_review_routing_takes_the_lowest_confidence_not_the_average() -> None:
    # An average would let one genuinely unsure finding ride past the threshold on the backs
    # of several nearly confident ones.
    report = DetectionReport(
        corpus_name="x",
        confidence_threshold=0.6,
        severity_policy="p",
        clauses_examined=3,
        abstained=[_a_finding("a", 0.55), _a_finding("b", 0.2), _a_finding("c", 0.59)],
    )

    routing = route_for_review(report)

    assert routing.confidence == 0.2
    assert routing.item_ids == ["a", "b", "c"]
    assert not routing.is_empty


def test_review_routing_is_a_no_op_when_nothing_was_withheld() -> None:
    report = DetectionReport(
        corpus_name="x", confidence_threshold=0.6, severity_policy="p", clauses_examined=1
    )

    routing = route_for_review(report)

    assert routing.is_empty
    assert routing.confidence == 1.0  # clears any threshold, so the interrupt does not pause


def test_a_finding_carrying_children_that_is_not_a_split_is_refused() -> None:
    with pytest.raises(ValidationError, match="Only"):
        _a_finding("x", 0.9).model_copy(update={"children": ["a", "b"]}).model_validate(
            _a_finding("x", 0.9).model_dump() | {"children": ["a", "b"]}
        )


def test_a_split_finding_with_fewer_than_two_children_is_refused() -> None:
    payload = _a_finding("x", 0.9).model_dump() | {"detector": "atomicity_split", "children": ["a"]}

    with pytest.raises(ValidationError, match="not compound"):
        Finding.model_validate(payload)


# --- the registry and the claims pass ------------------------------------------------------------


def test_asking_for_an_unknown_detector_raises_rather_than_running_nothing() -> None:
    # A run that silently skipped the detector somebody asked for would report a clean
    # result for work it never did.
    with pytest.raises(KeyError, match="no such detector"):
        run_detectors(a_context(), only=frozenset({"telepathy"}))


def test_every_detector_has_both_a_firing_and_a_non_firing_test() -> None:
    """A detector that fires on everything scores well and is worthless.

    Read from the registry rather than listed, so adding a detector fails this until it has
    both halves: the case it must catch, and a case it must leave alone.
    """
    names = [
        line.split("(")[0].removeprefix("def ")
        for line in Path(__file__).read_text(encoding="utf-8").splitlines()
        if line.startswith("def test_")
    ]

    for detector in DETECTORS:
        stem = detector.removesuffix("_split").removesuffix("_trap").removesuffix("_inconsistency")
        mentioning = [name for name in names if stem in name]
        holds = [n for n in mentioning if "does_not" in n or "abstains" in n]

        assert mentioning, f"{detector!r} has no test of the case it must catch"
        assert holds, (
            f"{detector!r} has no test of a case it must NOT fire on. A detector with only "
            f"positive tests is one nobody has checked for over-firing."
        )


def test_claims_are_read_one_call_per_page_not_one_per_clause() -> None:
    # 301 clauses over 14 pages is the difference between 14 calls and 301.
    calls: list[str] = []

    def counting(prompt: str, schema: type, *, purpose: str) -> PageClaims:
        calls.append(purpose)
        return PageClaims(
            clauses=[ClauseClaims(identifier="X-1.1"), ClauseClaims(identifier="X-1.2")]
        )

    readings = read_claims(a_corpus(), counting)

    assert len(calls) == 1  # one page, two clauses
    assert set(readings) == {"X-1.1", "X-1.2"}


def test_a_reading_for_a_clause_not_on_the_page_is_dropped() -> None:
    def inventing(prompt: str, schema: type, *, purpose: str) -> PageClaims:
        return PageClaims(clauses=[ClauseClaims(identifier="X-9.9")])

    assert read_claims(a_corpus(), inventing) == {}


def test_a_page_with_no_clauses_costs_no_call() -> None:
    calls: list[str] = []

    def counting(prompt: str, schema: type, *, purpose: str) -> PageClaims:
        calls.append(purpose)
        return PageClaims()

    corpus = SourceCorpus(
        name="x",
        documents=[
            CorpusDocument(
                document_id="04_Spec",
                pages=[CorpusPage(document_id="04_Spec", page_number=1, text="prose, unnumbered")],
            )
        ],
    )
    read_claims(corpus, counting)

    assert not calls


def _never_called(prompt: str, schema: type, *, purpose: str) -> PageClaims:
    raise AssertionError("claims were supplied; no model call should be made")


def _a_finding(finding_id: str, confidence: float) -> Finding:
    return Finding(
        finding_id=finding_id,
        detector="zero_margin",
        requirement_id="X-1.1",
        severity="major",
        statement="something",
        evidence=Evidence(summary="observed"),
        source=EvidenceRef(source_id="04_Spec", document="Spec", page=1, quote="q"),
        confidence=confidence,
    )


# --- the remaining branches ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["missing_tolerance", "unverifiable", "ordinal_trap", "zero_margin"]
)
def test_a_detector_that_needs_claims_says_nothing_when_none_were_read(name: str) -> None:
    # A missing reading becomes silence, never a clean bill of health. Every claims-driven
    # detector has to behave this way or a page the model failed on would read as clean.
    assert not fired(a_context(claims=None), name)


def test_missing_tolerance_records_the_conditions_that_do_exist() -> None:
    # When one constraint is unbounded and the clause states conditions for another, the
    # conditions are recorded on the evidence: they are what a reviewer checks first when
    # deciding whether the detector fired fairly.
    claims = some_claims(
        constraints=[a_constraint(operator="unbounded", value="", unit="")],
        conditions=["at the inlet", "at 20 degrees"],
    )

    draft = fired(a_context(claims=claims), "missing_tolerance")[0]

    assert any("conditions stated elsewhere" in line for line in draft.evidence.observations)
    assert any("at the inlet" in line for line in draft.evidence.observations)


def test_a_range_whose_value_cannot_be_read_as_two_ends_is_left_alone() -> None:
    # "between the limits given at Annex B" is a range in words. Inventing ends for it would
    # be inventing the defect.
    claims = some_claims(
        constraints=[a_constraint(operator="between", value="the limits at Annex B", unit="")]
    )

    assert not fired(a_context(claims=claims), "zero_margin")


def test_lineage_reaches_the_detector_from_the_requirement_set() -> None:
    # The engine builds `siblings` from parent_id across the whole set, which is how
    # atomicity_split sees a split it did not make.
    parent = a_requirement("X-1.1")
    children = [
        a_requirement("X-1.1/1", parent_id="X-1.1"),
        a_requirement("X-1.1/2", parent_id="X-1.1"),
    ]

    report = detect(
        [parent, *children],
        a_corpus(),
        _never_called,
        modality=HOUSE_STYLE,
        severity=SEVERITIES,
        ordinal_scales=SCALES,
        claims={},
    )
    splits = report.by_detector("atomicity_split")

    assert [f.requirement_id for f in splits] == ["X-1.1"]
    assert splits[0].children == ["X-1.1/1", "X-1.1/2"]


def test_by_detector_selects_only_the_reported_findings() -> None:
    report = DetectionReport(
        corpus_name="x",
        confidence_threshold=0.6,
        severity_policy="p",
        clauses_examined=1,
        findings=[_a_finding("a", 0.9)],
        abstained=[_a_finding("b", 0.1)],
    )

    assert [f.finding_id for f in report.by_detector("zero_margin")] == ["a"]
    assert report.by_detector("unverifiable") == []


@pytest.mark.parametrize(("confidence", "expected"), [(0.9, True), (0.6, True), (0.59, False)])
def test_is_confident_reads_the_default_threshold(confidence: float, expected: bool) -> None:
    assert _a_finding("x", confidence).is_confident is expected
