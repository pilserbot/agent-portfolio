"""Measure a judge against human labels, and say plainly when it is not fit for use.

`python -m spine.eval.calibrate --rubric example --labels data/gold/example_human.jsonl`
runs the judge over items a person has already labelled and writes
`evals/calibration/<rubric>_<date>.md` with raw agreement, Cohen's kappa, the confusion
matrix, and the items where judge and human disagreed most.

The point of this tool is to be able to say the instrument is bad. A judge whose kappa falls
below FITNESS_THRESHOLD is reported as NOT FIT FOR USE, in those words, at the top of the
report. An uncalibrated judge is a guess with a number attached, and a judge calibrated and
found wanting must not be quietly used anyway.

Raw agreement is reported but never used for the verdict: on a skewed set a judge that
answers the same way every time can look 80% right while carrying no information at all.
Kappa is what corrects for that, which is why it, and not agreement, decides fitness.

Deliberately does not: tune anything. It measures and reports; it does not adjust a
threshold, drop a hard item, or rerun a disagreement until it agrees. It also does not
decide that synthetic labels are good enough — a labels file whose provenance is not human
produces a report stamped as a smoke test, not a calibration.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spine.eval.judge import Judge, JudgeResult, Rubric, RubricError, load_rubric

FITNESS_THRESHOLD = 0.6
DEFAULT_REPORT_ROOT = Path("evals/calibration")
NOT_FIT = "NOT FIT FOR USE"
DEFAULT_DISAGREEMENTS_SHOWN = 10

__all__ = [
    "DEFAULT_DISAGREEMENTS_SHOWN",
    "DEFAULT_REPORT_ROOT",
    "FITNESS_THRESHOLD",
    "NOT_FIT",
    "AgreementResult",
    "CalibrationError",
    "CalibrationReport",
    "ConfusionMatrix",
    "Disagreement",
    "HumanLabel",
    "JudgeFactory",
    "KappaResult",
    "calibrate",
    "cohens_kappa",
    "confusion_matrix",
    "load_labels",
    "main",
    "raw_agreement",
    "render_report",
]


class CalibrationError(Exception):
    """The calibration could not be run as asked."""


class HumanLabel(BaseModel):
    """One item a person judged, and their ruling.

    `provenance` is required and not defaulted. A labels file is only worth anything if it
    is clear who produced it, and a fixture that forgot to say so would be indistinguishable
    from real work.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(min_length=1)
    expected: str
    output: str
    human_passed: bool
    provenance: Literal["human", "synthetic-fixture"]
    human_note: str = ""
    tags: list[str] = Field(default_factory=list)


def load_labels(path: Path) -> list[HumanLabel]:
    """Load human labels from JSONL, failing loudly on a bad line with its number."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise CalibrationError(f"labels file not found: {path}") from error
    except IsADirectoryError as error:
        raise CalibrationError(f"labels path is a directory: {path}") from error

    labels: list[HumanLabel] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise CalibrationError(f"{path}:{number}: not valid JSON: {error.msg}") from error
        try:
            label = HumanLabel.model_validate(payload)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in problem['loc']) or '(root)'}: {problem['msg']}"
                for problem in error.errors()
            )
            raise CalibrationError(f"{path}:{number}: {problems}") from error
        if label.item_id in seen:
            raise CalibrationError(
                f"{path}:{number}: duplicate item_id {label.item_id!r}, "
                f"first seen on line {seen[label.item_id]}"
            )
        seen[label.item_id] = number
        labels.append(label)

    if not labels:
        raise CalibrationError(f"{path}: no labels found")
    return labels


class AgreementResult(BaseModel):
    """How often the two raters said the same thing."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, le=1.0)
    agreed: int = Field(ge=0)
    total: int = Field(ge=0)


class KappaResult(BaseModel):
    """Cohen's kappa, and the two probabilities behind it."""

    model_config = ConfigDict(frozen=True)

    value: float
    observed_agreement: float = Field(ge=0.0, le=1.0)
    expected_agreement: float = Field(ge=0.0, le=1.0)
    total: int = Field(ge=0)
    degenerate: bool = Field(
        default=False,
        description="True when chance agreement is 1.0, so kappa has no defined value.",
    )

    @property
    def is_fit(self) -> bool:
        """Whether the judge clears the fitness threshold."""
        return self.value >= FITNESS_THRESHOLD


class ConfusionMatrix(BaseModel):
    """Counts of human label against judge label."""

    model_config = ConfigDict(frozen=True)

    labels: list[str] = Field(default_factory=list)
    counts: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="counts[human_label][judge_label]"
    )
    total: int = Field(ge=0)

    def count(self, human: str, judge: str) -> int:
        """One cell of the matrix."""
        return self.counts.get(human, {}).get(judge, 0)


class Disagreement(BaseModel):
    """One item the judge and the human ruled differently on."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    human_passed: bool
    judge_passed: bool
    judge_score: float = Field(ge=0.0, le=1.0)
    distance: float = Field(
        ge=0.0, description="How far the judge's score sat from the human's ruling."
    )
    human_note: str = ""
    judge_rationale: str = ""


class CalibrationReport(BaseModel):
    """Everything one calibration run measured."""

    model_config = ConfigDict(frozen=True)

    rubric_label: str
    labels_path: Path
    measured_at: datetime
    agreement: AgreementResult
    kappa: KappaResult
    matrix: ConfusionMatrix
    disagreements: list[Disagreement] = Field(default_factory=list)
    provenances: list[str] = Field(default_factory=list)
    samples_per_item: int = Field(ge=1)

    @property
    def is_fit(self) -> bool:
        """Whether the judge may be used on the strength of this calibration."""
        return self.kappa.is_fit

    @property
    def is_smoke_test(self) -> bool:
        """Whether any label came from something other than a person."""
        return any(provenance != "human" for provenance in self.provenances)


def _label_of(passed: bool) -> str:
    """The label text used in the confusion matrix."""
    return "pass" if passed else "fail"


def raw_agreement(pairs: Sequence[tuple[bool, bool]]) -> AgreementResult:
    """The fraction of items both raters ruled the same way.

    Reported for context only. On a skewed set it flatters a judge that always answers the
    same way, which is exactly the failure kappa exists to expose.
    """
    total = len(pairs)
    agreed = sum(1 for human, judge in pairs if human == judge)
    return AgreementResult(value=agreed / total if total else 0.0, agreed=agreed, total=total)


def confusion_matrix(pairs: Sequence[tuple[bool, bool]]) -> ConfusionMatrix:
    """Count human rulings against judge rulings."""
    labels = ["pass", "fail"]
    counts = {human: {judge: 0 for judge in labels} for human in labels}
    for human, judge in pairs:
        counts[_label_of(human)][_label_of(judge)] += 1
    return ConfusionMatrix(labels=labels, counts=counts, total=len(pairs))


def cohens_kappa(pairs: Sequence[tuple[bool, bool]]) -> KappaResult:
    """Cohen's kappa for two raters over the same items.

    kappa = (observed agreement - chance agreement) / (1 - chance agreement), where chance
    agreement is the sum over labels of the product of the two raters' marginal rates.

    When both raters used exactly one label between them, chance agreement is 1.0 and kappa
    is 0/0. That is reported as degenerate rather than hidden: the value is 1.0 if they
    agreed on everything and 0.0 otherwise, and either way the set could not measure the
    judge, because a set with one class in it cannot.
    """
    total = len(pairs)
    if total == 0:
        return KappaResult(
            value=0.0, observed_agreement=0.0, expected_agreement=0.0, total=0, degenerate=True
        )

    observed = sum(1 for human, judge in pairs if human == judge) / total

    expected = 0.0
    for label in (True, False):
        human_rate = sum(1 for human, _ in pairs if human == label) / total
        judge_rate = sum(1 for _, judge in pairs if judge == label) / total
        expected += human_rate * judge_rate

    if expected >= 1.0:
        return KappaResult(
            value=1.0 if observed >= 1.0 else 0.0,
            observed_agreement=observed,
            expected_agreement=1.0,
            total=total,
            degenerate=True,
        )

    return KappaResult(
        value=(observed - expected) / (1 - expected),
        observed_agreement=observed,
        expected_agreement=expected,
        total=total,
    )


def _disagreements(
    labels: Sequence[HumanLabel], results: Sequence[JudgeResult], limit: int
) -> list[Disagreement]:
    """The items the raters differed on, worst first.

    Ranked by how far the judge's score sat from the human's ruling, so a confident wrong
    answer sorts above a borderline one: those are the items worth reading.
    """
    found: list[Disagreement] = []
    for label, result in zip(labels, results, strict=True):
        verdict = result.verdict
        if verdict.passed == label.human_passed:
            continue
        human_value = 1.0 if label.human_passed else 0.0
        found.append(
            Disagreement(
                item_id=label.item_id,
                human_passed=label.human_passed,
                judge_passed=verdict.passed,
                judge_score=verdict.score,
                distance=abs(verdict.score - human_value),
                human_note=label.human_note,
                judge_rationale=verdict.rationale,
            )
        )
    found.sort(key=lambda item: (-item.distance, item.item_id))
    return found[:limit]


def calibrate(
    judge: Judge,
    labels: Sequence[HumanLabel],
    *,
    labels_path: Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    limit: int = DEFAULT_DISAGREEMENTS_SHOWN,
) -> CalibrationReport:
    """Run the judge over labelled items and measure how well it matches the human."""
    if not labels:
        raise CalibrationError("no labels to calibrate against")

    results = [
        judge.judge(label.item_id, expected=label.expected, output=label.output) for label in labels
    ]
    pairs = [
        (label.human_passed, result.verdict.passed)
        for label, result in zip(labels, results, strict=True)
    ]

    return CalibrationReport(
        rubric_label=judge.rubric.label,
        labels_path=labels_path,
        measured_at=now(),
        agreement=raw_agreement(pairs),
        kappa=cohens_kappa(pairs),
        matrix=confusion_matrix(pairs),
        disagreements=_disagreements(labels, results, limit),
        provenances=sorted({label.provenance for label in labels}),
        samples_per_item=judge.samples,
    )


def render_report(report: CalibrationReport) -> str:
    """Render a calibration report as markdown, leading with the verdict."""
    kappa = report.kappa
    lines: list[str] = [f"# Judge calibration: {report.rubric_label}", ""]

    if report.is_fit:
        lines.append(
            f"**FIT FOR USE** — Cohen's kappa {kappa.value:.3f} (threshold {FITNESS_THRESHOLD})."
        )
    else:
        lines.append(
            f"**{NOT_FIT}** — Cohen's kappa {kappa.value:.3f} is below the "
            f"threshold of {FITNESS_THRESHOLD}."
        )
        lines.append("")
        lines.append(
            "Do not use this judge to gate anything. Its agreement with the human labels is "
            "not distinguishable enough from chance to carry a decision. Revise the rubric "
            "and re-calibrate, or score these items by deterministic checks instead."
        )

    if kappa.degenerate:
        lines += [
            "",
            "**This set could not measure the judge.** Chance agreement is 1.0, which happens "
            "when only one class is present in the labels or the judge answered the same way "
            "every time. Kappa has no defined value here; the number above is a convention, "
            "not a measurement.",
        ]

    if report.is_smoke_test:
        lines += [
            "",
            "---",
            "",
            f"**SMOKE TEST, NOT A CALIBRATION.** The labels in `{report.labels_path}` carry "
            f"provenance {', '.join(report.provenances)} — not all of them came from a person. "
            "Nothing in this report says whether the judge is actually fit for use. Replace "
            "the labels with human ones before relying on any figure here.",
        ]

    lines += [
        "",
        "## Measurement",
        "",
        f"- Labels: `{report.labels_path}` ({kappa.total} item(s))",
        f"- Rubric: `{report.rubric_label}`",
        f"- Samples per item: {report.samples_per_item}",
        f"- Measured at: {report.measured_at.isoformat()}",
        f"- Label provenance: {', '.join(report.provenances)}",
        "",
        "| Figure | Value |",
        "| --- | ---: |",
        f"| Cohen's kappa | {kappa.value:.3f} |",
        (
            f"| Raw agreement | {report.agreement.value:.3f} "
            f"({report.agreement.agreed}/{report.agreement.total}) |"
        ),
        f"| Chance agreement | {kappa.expected_agreement:.3f} |",
        "",
        (
            "Raw agreement is context, not the verdict: on a skewed set a judge that answers "
            "the same way every time can look accurate while carrying no information. Kappa "
            "corrects for that, and decides fitness here."
        ),
        "",
        "## Confusion matrix",
        "",
        "Rows are the human ruling, columns the judge's.",
        "",
        "| human \\ judge | pass | fail |",
        "| --- | ---: | ---: |",
        f"| pass | {report.matrix.count('pass', 'pass')} | {report.matrix.count('pass', 'fail')} |",
        f"| fail | {report.matrix.count('fail', 'pass')} | {report.matrix.count('fail', 'fail')} |",
        "",
    ]

    lines += ["## Largest disagreements", ""]
    if not report.disagreements:
        lines.append("None: the judge matched the human on every item.")
    else:
        lines.append(
            f"{len(report.disagreements)} shown, furthest first — how far the judge's score "
            "sat from the human's ruling, so a confident wrong answer sorts above a "
            "borderline one."
        )
        lines.append("")
        for item in report.disagreements:
            lines += [
                f"### {item.item_id}",
                "",
                f"- Human: **{_label_of(item.human_passed)}**"
                + (f" — {item.human_note}" if item.human_note else ""),
                f"- Judge: **{_label_of(item.judge_passed)}** at score {item.judge_score:.3f} "
                f"(distance {item.distance:.3f})",
            ]
            if item.judge_rationale:
                lines.append(f"- Judge's rationale: {item.judge_rationale}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def report_path(rubric_label: str, *, when: date, root: Path = DEFAULT_REPORT_ROOT) -> Path:
    """Where a report for one rubric on one day is written."""
    slug = rubric_label.replace("@", "_").replace("/", "-")
    return root / f"{slug}_{when.isoformat()}.md"


def _build_judge(rubric: Rubric, samples: int, tier: str) -> Judge:
    """Build the real, router-backed judge. Imported lazily: no provider SDK until needed."""
    from spine.router import Router

    return Judge(
        rubric,
        router=Router(),
        samples=samples,
        tier="small" if tier == "small" else "large",
    )


type JudgeFactory = Callable[[Rubric, int, str], Judge]


def main(argv: Sequence[str] | None = None, *, judge_factory: JudgeFactory = _build_judge) -> int:
    """Run the command-line entry point and return the process exit code.

    Exits non-zero when the judge is not fit for use: a calibration that fails should fail
    the pipeline that ran it, not merely leave a file behind.
    """
    parser = argparse.ArgumentParser(
        prog="python -m spine.eval.calibrate",
        description="Measure a judge against human labels and report whether it is usable.",
    )
    parser.add_argument("--rubric", required=True, help="Rubric name under evals/rubrics/.")
    parser.add_argument("--labels", required=True, type=Path, help="JSONL file of human labels.")
    parser.add_argument("--samples", type=int, default=1, help="Self-consistency samples per item.")
    parser.add_argument(
        "--tier", default="small", choices=("small", "large"), help="Router tier to judge on."
    )
    parser.add_argument(
        "--out-root", type=Path, default=DEFAULT_REPORT_ROOT, help="Directory for the report."
    )
    parser.add_argument(
        "--rubric-root",
        type=Path,
        default=None,
        help="Directory holding the rubric, if not evals/rubrics/.",
    )
    args = parser.parse_args(argv)

    try:
        rubric = (
            load_rubric(args.rubric)
            if args.rubric_root is None
            else load_rubric(args.rubric, root=args.rubric_root)
        )
        labels = load_labels(args.labels)
        report = calibrate(
            judge_factory(rubric, args.samples, args.tier), labels, labels_path=args.labels
        )
    except (RubricError, CalibrationError) as error:
        print(f"calibration failed: {error}")
        return 2

    destination = report_path(
        report.rubric_label, when=report.measured_at.date(), root=args.out_root
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(report), encoding="utf-8")

    print(f"Wrote {destination}")
    print(f"Cohen's kappa: {report.kappa.value:.3f}")
    if not report.is_fit:
        print(NOT_FIT)
    return 0 if report.is_fit else 1


if __name__ == "__main__":
    raise SystemExit(main())
