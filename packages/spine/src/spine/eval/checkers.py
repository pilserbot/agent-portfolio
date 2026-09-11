"""Deterministic checks that turn one answer into one Verdict, with no model involved.

Every check here is ordinary Python over typed structures: parsing, set arithmetic,
comparison against a tolerance. That is the point. A language model may produce the answer
being checked, but nothing in this module asks a model whether that answer was right, and
`judge_model` on the Verdicts returned here is always None.

Each function returns a `Verdict` carrying `passed`, a `score` in 0..1 and a rationale
stating what was compared, so a failing item explains itself without a rerun.

Deliberately does not: aggregate. One check, one item, one Verdict; combining them into
precision, F1 or calibration is `spine.eval.metrics`' work. It also does not normalise
inputs on your behalf — `exact_match` is exact, and a caller wanting case-insensitive
comparison lowercases before calling.
"""

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spine.contracts import EvidenceRef, Verdict

__all__ = [
    "Claim",
    "citation_resolves",
    "exact_match",
    "numeric_within_tolerance",
    "reference_integrity",
    "schema_valid",
    "set_f1",
]


class Claim(BaseModel):
    """One assertion a system made, and the evidence it cited for it."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @property
    def cited_source_ids(self) -> list[str]:
        """The source ids this claim points at, in citation order."""
        return [reference.source_id for reference in self.evidence]


def _verdict(
    *,
    item_id: str,
    expected: str,
    actual: str,
    passed: bool,
    score: float,
    rationale: str,
) -> Verdict:
    """Build a Verdict, clamping the score into range so a checker cannot emit a bad one."""
    return Verdict(
        item_id=item_id,
        expected=expected,
        actual=actual,
        passed=passed,
        score=min(1.0, max(0.0, score)),
        rationale=rationale,
        judge_model=None,
    )


def schema_valid(output: object, model: type[BaseModel], *, item_id: str) -> Verdict:
    """Check that an output parses into a Pydantic model.

    Accepts a JSON string, a mapping, or an already-built model instance. The rationale
    names the failing fields, which is what makes a schema failure actionable.
    """
    if isinstance(output, model):
        return _verdict(
            item_id=item_id,
            expected=model.__name__,
            actual=model.__name__,
            passed=True,
            score=1.0,
            rationale=f"already an instance of {model.__name__}.",
        )
    try:
        if isinstance(output, str):
            model.model_validate_json(output)
        else:
            model.model_validate(output)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc']) or '(root)'}: {problem['msg']}"
            for problem in error.errors()
        )
        return _verdict(
            item_id=item_id,
            expected=model.__name__,
            actual=type(output).__name__,
            passed=False,
            score=0.0,
            rationale=f"does not parse as {model.__name__}: {problems}",
        )
    return _verdict(
        item_id=item_id,
        expected=model.__name__,
        actual=model.__name__,
        passed=True,
        score=1.0,
        rationale=f"parses as {model.__name__}.",
    )


def _is_resolved(reference: EvidenceRef | None) -> bool:
    """Whether a reference actually points somewhere with something quoted.

    EvidenceRef already refuses to exist without a locator, so what is left to check is
    that it carries a quote: a citation to a page with no text is not evidence.
    """
    return reference is not None and bool(reference.quote.strip())


def reference_integrity(items: Sequence[EvidenceRef | None], *, item_id: str) -> Verdict:
    """Check that every item resolves to a non-empty EvidenceRef.

    The score is the fraction that resolve, so a partly-cited answer is distinguishable
    from an uncited one. An empty sequence has nothing to resolve and does not pass: an
    answer that cites nothing has not shown its work.
    """
    total = len(items)
    if total == 0:
        return _verdict(
            item_id=item_id,
            expected="at least one resolved reference",
            actual="no references",
            passed=False,
            score=0.0,
            rationale="no references were supplied.",
        )
    resolved = sum(1 for reference in items if _is_resolved(reference))
    unresolved = total - resolved
    return _verdict(
        item_id=item_id,
        expected=f"{total} resolved reference(s)",
        actual=f"{resolved} resolved reference(s)",
        passed=unresolved == 0,
        score=resolved / total,
        rationale=(
            f"{resolved} of {total} reference(s) resolved."
            if unresolved
            else f"all {total} reference(s) resolved."
        ),
    )


def citation_resolves(claim: Claim, sources: Collection[str], *, item_id: str) -> Verdict:
    """Check that every source a claim cites exists in the source set.

    The score is the fraction of citations that resolve. A claim citing nothing does not
    pass: an unsupported assertion is the failure this check exists to catch.
    """
    cited = claim.cited_source_ids
    known = set(sources)
    if not cited:
        return _verdict(
            item_id=item_id,
            expected="at least one citation",
            actual="no citations",
            passed=False,
            score=0.0,
            rationale=f"claim {claim.claim_id!r} cites no source.",
        )
    dangling = [source_id for source_id in cited if source_id not in known]
    resolved = len(cited) - len(dangling)
    return _verdict(
        item_id=item_id,
        expected=f"{len(cited)} citation(s) resolving into the source set",
        actual=f"{resolved} resolved, {len(dangling)} dangling",
        passed=not dangling,
        score=resolved / len(cited),
        rationale=(
            f"unknown source id(s): {', '.join(sorted(dangling))}."
            if dangling
            else f"all {len(cited)} citation(s) resolve."
        ),
    )


def exact_match(expected: object, actual: object, *, item_id: str) -> Verdict:
    """Check that two values are exactly equal as strings.

    Exact means exact: no case folding, no whitespace collapsing, no punctuation stripping.
    A caller wanting a looser comparison normalises before calling, so that the loosening is
    visible at the call site rather than hidden here.
    """
    expected_text = _as_text(expected)
    actual_text = _as_text(actual)
    same = expected_text == actual_text
    return _verdict(
        item_id=item_id,
        expected=expected_text,
        actual=actual_text,
        passed=same,
        score=1.0 if same else 0.0,
        rationale="exact match." if same else "differs from the expected value.",
    )


def _as_text(value: object) -> str:
    """Render a value for comparison and for the Verdict's own fields."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False)
    return str(value)


def _plain(value: object) -> object:
    """Convert to something json.dumps accepts, deterministically."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    return value


def set_f1(expected: Iterable[str], actual: Iterable[str], *, item_id: str) -> Verdict:
    """Score the overlap of two sets by F1, passing only on an exact set match.

    Two empty sets agree completely and score 1.0. One empty and one not scores 0.0, since
    there is no overlap to reward.
    """
    expected_set = set(expected)
    actual_set = set(actual)
    overlap = len(expected_set & actual_set)

    if not expected_set and not actual_set:
        score = 1.0
    elif overlap == 0:
        score = 0.0
    else:
        precision = overlap / len(actual_set)
        recall = overlap / len(expected_set)
        score = 2 * precision * recall / (precision + recall)

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    notes = []
    if missing:
        notes.append(f"missing: {', '.join(missing)}")
    if extra:
        notes.append(f"unexpected: {', '.join(extra)}")

    return _verdict(
        item_id=item_id,
        expected=", ".join(sorted(expected_set)) or "(empty set)",
        actual=", ".join(sorted(actual_set)) or "(empty set)",
        passed=expected_set == actual_set,
        score=score,
        rationale="; ".join(notes) if notes else "sets match exactly.",
    )


def numeric_within_tolerance(
    expected: float | int | Decimal | str,
    actual: float | int | Decimal | str,
    *,
    item_id: str,
    tolerance: float | Decimal = 0,
    relative: bool = False,
) -> Verdict:
    """Check that a number is within a tolerance of the expected one.

    The tolerance is absolute by default. With `relative=True` it is a fraction of the
    expected value, and an expected value of zero then has no meaningful relative
    tolerance, so the comparison falls back to exact equality rather than dividing by zero.

    Values are compared as Decimals: a tolerance check on money that fails through binary
    float drift would be a bug in the harness rather than the system under test.
    """
    try:
        expected_value = _as_decimal(expected)
        actual_value = _as_decimal(actual)
    except (InvalidOperation, ValueError, TypeError):
        return _verdict(
            item_id=item_id,
            expected=str(expected),
            actual=str(actual),
            passed=False,
            score=0.0,
            rationale="not a number.",
        )

    allowed = _as_decimal(tolerance)
    if relative:
        allowed = abs(expected_value) * allowed
    difference = abs(expected_value - actual_value)
    within = difference <= allowed

    kind = "relative" if relative else "absolute"
    return _verdict(
        item_id=item_id,
        expected=str(expected_value),
        actual=str(actual_value),
        passed=within,
        score=1.0 if within else 0.0,
        rationale=(
            f"differs by {difference}, within the {kind} tolerance of {allowed}."
            if within
            else f"differs by {difference}, outside the {kind} tolerance of {allowed}."
        ),
    )


def _as_decimal(value: float | int | Decimal | str) -> Decimal:
    """Convert to Decimal without inheriting binary-float noise, rejecting non-finites."""
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, bool):
        raise TypeError("a boolean is not a number here")
    elif isinstance(value, int):
        decimal_value = Decimal(value)
    else:
        decimal_value = Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"not a finite number: {value}")
    return decimal_value
