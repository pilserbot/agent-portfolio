"""Unit tests for the judge calibration harness.

Kappa is checked against hand-worked examples, and the not-fit-for-use threshold is
asserted on the rendered words, because the phrase is the deliverable: a calibration tool
that reports a bad instrument politely is not doing its job.

No network: the judge is always an injected function.
"""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from spine.contracts import ModelCall
from spine.eval.calibrate import (
    FITNESS_THRESHOLD,
    NOT_FIT,
    CalibrationError,
    HumanLabel,
    Judge,
    calibrate,
    cohens_kappa,
    confusion_matrix,
    load_labels,
    main,
    raw_agreement,
    render_report,
    report_path,
)
from spine.eval.judge import Criterion, JudgeRequest, JudgeResponse, Rubric

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "data" / "gold" / "example_human.jsonl"
AT = datetime(2026, 9, 11, tzinfo=UTC)


def a_rubric() -> Rubric:
    return Rubric(
        name="unit",
        version=1,
        instructions="Score each criterion.",
        pass_threshold=0.7,
        criteria=[Criterion(id="alpha", description="first", weight=1.0)],
    )


def a_label(item_id: str, human_passed: bool, provenance: str = "human") -> HumanLabel:
    return HumanLabel(
        item_id=item_id,
        expected="a requirement",
        output="an answer",
        human_passed=human_passed,
        provenance=provenance,
        human_note=f"note for {item_id}",
    )


def judge_that(decisions: dict[str, bool]) -> Callable[[JudgeRequest], object]:
    """A fake judge returning a confident pass or fail per item id."""

    def judge_fn(request: JudgeRequest) -> tuple[JudgeResponse, ModelCall | None]:
        verdict = decisions.get(request.item_id, False)
        # Score whatever criteria the request's rubric actually lists: hardcoding an id
        # would silently score zero against any other rubric.
        score = 1.0 if verdict else 0.0
        return (
            JudgeResponse(
                criterion_scores={criterion.id: score for criterion in request.rubric.criteria},
                rationale=f"judged {request.item_id}",
            ),
            None,
        )

    return judge_fn


# --- Cohen's kappa, against hand-worked examples --------------------------------------


def test_kappa_on_a_hand_worked_example() -> None:
    # N = 20: both pass 8, human pass / judge fail 2, human fail / judge pass 3, both fail 7.
    # observed  = (8 + 7) / 20 = 0.75
    # human pass rate = 10/20 = 0.5, judge pass rate = 11/20 = 0.55
    # chance    = 0.5*0.55 + 0.5*0.45 = 0.275 + 0.225 = 0.5
    # kappa     = (0.75 - 0.5) / (1 - 0.5) = 0.25 / 0.5 = 0.5
    pairs = [(True, True)] * 8 + [(True, False)] * 2 + [(False, True)] * 3 + [(False, False)] * 7

    result = cohens_kappa(pairs)

    assert result.observed_agreement == pytest.approx(0.75)
    assert result.expected_agreement == pytest.approx(0.5)
    assert result.value == pytest.approx(0.5)
    assert result.total == 20
    assert not result.degenerate


def test_kappa_on_a_second_hand_worked_example() -> None:
    # N = 20: both pass 9, one each way, both fail 9.
    # observed = 18/20 = 0.9; both pass rates are 10/20 = 0.5
    # chance   = 0.25 + 0.25 = 0.5; kappa = (0.9 - 0.5)/0.5 = 0.8
    pairs = [(True, True)] * 9 + [(True, False)] + [(False, True)] + [(False, False)] * 9

    result = cohens_kappa(pairs)

    assert result.value == pytest.approx(0.8)
    assert result.is_fit


def test_perfect_agreement_is_one() -> None:
    pairs = [(True, True)] * 5 + [(False, False)] * 5

    assert cohens_kappa(pairs).value == pytest.approx(1.0)


def test_chance_level_agreement_is_zero() -> None:
    # observed = 0.5; human pass 0.5, judge pass 0.5 -> chance = 0.5; kappa = 0.
    pairs = [(True, True), (True, False), (False, True), (False, False)]

    assert cohens_kappa(pairs).value == pytest.approx(0.0)


def test_systematic_disagreement_is_negative() -> None:
    # Judge inverts the human every time: observed 0, chance 0.5, kappa = -1.
    pairs = [(True, False)] * 5 + [(False, True)] * 5

    assert cohens_kappa(pairs).value == pytest.approx(-1.0)


def test_one_class_only_is_reported_as_degenerate() -> None:
    # Everything passes, so chance agreement is 1.0 and kappa is 0/0. The set cannot
    # measure the judge, and the report must say so rather than print a flattering 1.0.
    pairs = [(True, True)] * 10

    result = cohens_kappa(pairs)

    assert result.degenerate
    assert result.expected_agreement == pytest.approx(1.0)
    assert result.value == pytest.approx(1.0)


def test_a_judge_that_always_passes_scores_zero_against_a_mixed_human() -> None:
    # N = 10: human passes 8 and fails 2; the judge passes everything.
    # observed = 8/10 = 0.8. Judge pass rate is 1.0, fail rate 0.0, so
    # chance = 0.8*1.0 + 0.2*0.0 = 0.8, and kappa = (0.8 - 0.8)/(1 - 0.8) = 0.0.
    # Not degenerate: the human did distinguish, so the set could measure the judge — and
    # it measured no information at all, which is the finding.
    pairs = [(True, True)] * 8 + [(False, True)] * 2

    result = cohens_kappa(pairs)

    assert result.observed_agreement == pytest.approx(0.8)
    assert result.expected_agreement == pytest.approx(0.8)
    assert result.value == pytest.approx(0.0)
    assert not result.degenerate
    assert not result.is_fit


def test_degeneracy_needs_both_raters_to_use_one_label() -> None:
    # Only when neither rater distinguished is chance agreement 1.0 and kappa undefined.
    assert cohens_kappa([(True, True)] * 6).degenerate
    assert cohens_kappa([(False, False)] * 6).degenerate
    assert not cohens_kappa([(True, True)] * 5 + [(False, False)]).degenerate


def test_kappa_on_empty_input() -> None:
    result = cohens_kappa([])

    assert result.value == 0.0 and result.total == 0 and result.degenerate


# --- agreement and the confusion matrix -----------------------------------------------


def test_raw_agreement_counts_matching_rulings() -> None:
    pairs = [(True, True), (True, True), (True, False), (False, False)]

    result = raw_agreement(pairs)

    assert result.value == pytest.approx(0.75)
    assert (result.agreed, result.total) == (3, 4)


def test_agreement_flatters_a_judge_that_always_says_the_same_thing() -> None:
    # 18 of 20 pass; a judge that always says "pass" agrees 90% of the time and knows
    # nothing. This is exactly why kappa, not agreement, decides fitness.
    pairs = [(True, True)] * 18 + [(False, True)] * 2

    assert raw_agreement(pairs).value == pytest.approx(0.9)
    assert cohens_kappa(pairs).value == pytest.approx(0.0)
    assert not cohens_kappa(pairs).is_fit


def test_the_confusion_matrix_counts_every_cell() -> None:
    pairs = [(True, True)] * 8 + [(True, False)] * 2 + [(False, True)] * 3 + [(False, False)] * 7

    matrix = confusion_matrix(pairs)

    assert matrix.count("pass", "pass") == 8
    assert matrix.count("pass", "fail") == 2
    assert matrix.count("fail", "pass") == 3
    assert matrix.count("fail", "fail") == 7
    assert matrix.total == 20


# --- loading labels --------------------------------------------------------------------


def write_labels(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", "utf-8"
    )
    return path


def a_row(item_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "item_id": item_id,
        "expected": "a requirement",
        "output": "an answer",
        "human_passed": True,
        "provenance": "human",
    }
    row.update(overrides)
    return row


def test_labels_load(tmp_path: Path) -> None:
    path = write_labels(tmp_path / "l.jsonl", [a_row("a"), a_row("b", human_passed=False)])

    labels = load_labels(path)

    assert [label.item_id for label in labels] == ["a", "b"]
    assert labels[1].human_passed is False


def test_provenance_is_required(tmp_path: Path) -> None:
    # A labels file that does not say where it came from is indistinguishable from real work.
    row = a_row("a")
    del row["provenance"]
    path = write_labels(tmp_path / "l.jsonl", [row])

    with pytest.raises(CalibrationError, match="provenance"):
        load_labels(path)


def test_an_unknown_provenance_is_rejected(tmp_path: Path) -> None:
    path = write_labels(tmp_path / "l.jsonl", [a_row("a", provenance="vibes")])

    with pytest.raises(CalibrationError, match="provenance"):
        load_labels(path)


def test_a_bad_line_names_its_number(tmp_path: Path) -> None:
    path = write_labels(tmp_path / "l.jsonl", [a_row("a"), "{ not json"])

    with pytest.raises(CalibrationError, match=":2:"):
        load_labels(path)


def test_a_duplicate_item_id_is_rejected(tmp_path: Path) -> None:
    path = write_labels(tmp_path / "l.jsonl", [a_row("a"), a_row("a")])

    with pytest.raises(CalibrationError, match="duplicate"):
        load_labels(path)


def test_an_empty_labels_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(CalibrationError, match="no labels"):
        load_labels(path)


def test_a_missing_labels_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="not found"):
        load_labels(tmp_path / "absent.jsonl")


# --- the fixture is unmistakably a fixture ----------------------------------------------


def test_the_committed_fixture_loads_and_is_marked_synthetic() -> None:
    labels = load_labels(FIXTURE)

    assert len(labels) == 30
    assert {label.provenance for label in labels} == {"synthetic-fixture"}, (
        "every record must declare itself synthetic, so no report can mistake it for human"
    )


def test_the_fixture_is_balanced_enough_to_measure() -> None:
    # One class only would make kappa degenerate and the fixture useless as a smoke test.
    labels = load_labels(FIXTURE)
    passes = sum(1 for label in labels if label.human_passed)

    assert passes == 15
    assert len(labels) - passes == 15


def test_the_fixture_has_a_readme_warning_against_reuse() -> None:
    readme = FIXTURE.parent / "example_human.README.md"

    text = readme.read_text(encoding="utf-8")
    assert "NOT REAL HUMAN LABELS" in text
    assert "provenance" in text


# --- calibrating ------------------------------------------------------------------------


def test_a_calibration_measures_the_judge() -> None:
    labels = [a_label("a", True), a_label("b", True), a_label("c", False), a_label("d", False)]
    judge = Judge(a_rubric(), judge_fn=judge_that({"a": True, "b": True, "c": False, "d": False}))

    report = calibrate(judge, labels, labels_path=Path("labels.jsonl"), now=lambda: AT)

    assert report.kappa.value == pytest.approx(1.0)
    assert report.is_fit
    assert report.disagreements == []
    assert report.provenances == ["human"]
    assert not report.is_smoke_test


def test_disagreements_are_ranked_furthest_first() -> None:
    # The judge is confidently wrong on "a" and "b"; both are distance 1.0, so they sort by
    # item id, and correct items never appear.
    labels = [a_label("a", True), a_label("b", False), a_label("c", True)]
    judge = Judge(a_rubric(), judge_fn=judge_that({"a": False, "b": True, "c": True}))

    report = calibrate(judge, labels, labels_path=Path("l.jsonl"), now=lambda: AT)

    assert [d.item_id for d in report.disagreements] == ["a", "b"]
    assert all(d.distance == pytest.approx(1.0) for d in report.disagreements)
    assert report.disagreements[0].human_note == "note for a"
    assert "judged a" in report.disagreements[0].judge_rationale


def test_the_disagreement_list_is_capped() -> None:
    labels = [a_label(f"i-{n:02d}", True) for n in range(20)]
    judge = Judge(a_rubric(), judge_fn=judge_that({}))  # everything fails

    report = calibrate(judge, labels, labels_path=Path("l.jsonl"), now=lambda: AT, limit=3)

    assert len(report.disagreements) == 3


def test_calibrating_nothing_is_refused() -> None:
    with pytest.raises(CalibrationError, match="no labels"):
        calibrate(Judge(a_rubric(), judge_fn=judge_that({})), [], labels_path=Path("l.jsonl"))


def test_the_samples_per_item_is_recorded() -> None:
    judge = Judge(a_rubric(), samples=3, judge_fn=judge_that({"a": True}))

    report = calibrate(judge, [a_label("a", True)], labels_path=Path("l.jsonl"), now=lambda: AT)

    assert report.samples_per_item == 3


# --- the not-fit-for-use threshold ---------------------------------------------------------


def a_report_with_kappa_below_threshold() -> object:
    # The hand-worked 0.5 case: 8 / 2 / 3 / 7 over 20 items.
    labels = (
        [a_label(f"tp-{n}", True) for n in range(8)]
        + [a_label(f"fn-{n}", True) for n in range(2)]
        + [a_label(f"fp-{n}", False) for n in range(3)]
        + [a_label(f"tn-{n}", False) for n in range(7)]
    )
    decisions = {f"tp-{n}": True for n in range(8)} | {f"fp-{n}": True for n in range(3)}
    judge = Judge(a_rubric(), judge_fn=judge_that(decisions))
    return calibrate(judge, labels, labels_path=Path("l.jsonl"), now=lambda: AT)


def test_a_kappa_below_the_threshold_is_not_fit() -> None:
    report = a_report_with_kappa_below_threshold()

    assert report.kappa.value == pytest.approx(0.5)
    assert 0.5 < FITNESS_THRESHOLD
    assert not report.is_fit


def test_the_report_says_not_fit_for_use_in_those_words() -> None:
    text = render_report(a_report_with_kappa_below_threshold())

    assert NOT_FIT == "NOT FIT FOR USE"
    assert "NOT FIT FOR USE" in text
    assert "Do not use this judge to gate anything" in text


def test_the_not_fit_verdict_leads_the_report() -> None:
    # It must be the first thing a reader sees, not a footnote.
    lines = [line for line in render_report(a_report_with_kappa_below_threshold()).splitlines()]

    assert lines[0].startswith("# Judge calibration")
    assert "NOT FIT FOR USE" in lines[2]


def test_a_fit_judge_is_reported_as_fit() -> None:
    labels = [a_label("a", True), a_label("b", False)]
    judge = Judge(a_rubric(), judge_fn=judge_that({"a": True}))

    text = render_report(calibrate(judge, labels, labels_path=Path("l.jsonl"), now=lambda: AT))

    assert "**FIT FOR USE**" in text
    assert NOT_FIT not in text


def test_exactly_at_the_threshold_is_fit() -> None:
    # 0.6 is not below 0.6.
    pairs = [(True, True)] * 8 + [(True, False)] * 2 + [(False, True)] * 2 + [(False, False)] * 8
    result = cohens_kappa(pairs)

    assert result.value == pytest.approx(0.6)
    assert result.is_fit


def test_a_degenerate_set_is_flagged_as_unable_to_measure() -> None:
    labels = [a_label(f"a-{n}", True) for n in range(5)]
    judge = Judge(a_rubric(), judge_fn=judge_that({f"a-{n}": True for n in range(5)}))

    text = render_report(calibrate(judge, labels, labels_path=Path("l.jsonl"), now=lambda: AT))

    assert "could not measure the judge" in text


def test_synthetic_labels_stamp_the_report_as_a_smoke_test() -> None:
    labels = [a_label("a", True, provenance="synthetic-fixture"), a_label("b", False)]
    judge = Judge(a_rubric(), judge_fn=judge_that({"a": True}))

    report = calibrate(judge, labels, labels_path=Path("l.jsonl"), now=lambda: AT)
    text = render_report(report)

    assert report.is_smoke_test
    assert "SMOKE TEST, NOT A CALIBRATION" in text


def test_the_report_shows_the_matrix_and_both_figures() -> None:
    text = render_report(a_report_with_kappa_below_threshold())

    assert "## Confusion matrix" in text
    assert "| pass | 8 | 2 |" in text
    assert "| fail | 3 | 7 |" in text
    assert "Cohen's kappa | 0.500" in text
    assert "Raw agreement | 0.750 (15/20)" in text
    assert "## Largest disagreements" in text


# --- the command line ----------------------------------------------------------------------


def test_the_report_path_is_rubric_and_date_stamped() -> None:
    path = report_path("example@v1", when=date(2026, 9, 11), root=Path("out"))

    assert path == Path("out/example_v1_2026-09-11.md")


def test_the_cli_reports_a_missing_rubric_without_crashing(tmp_path: Path) -> None:
    code = main(
        [
            "--rubric",
            "absent",
            "--labels",
            str(FIXTURE),
            "--rubric-root",
            str(tmp_path),
            "--out-root",
            str(tmp_path),
        ]
    )

    assert code == 2


def test_the_cli_reports_missing_labels_without_crashing(tmp_path: Path) -> None:
    code = main(
        [
            "--rubric",
            "example",
            "--labels",
            str(tmp_path / "absent.jsonl"),
            "--rubric-root",
            str(REPO_ROOT / "evals" / "rubrics"),
            "--out-root",
            str(tmp_path),
        ]
    )

    assert code == 2


# --- the command line writes a report and fails when the judge is unfit -----------------


def a_judge_factory(decisions: dict[str, bool]) -> Callable[[Rubric, int, str], Judge]:
    """Build the fake judge the CLI will use, ignoring the tier it asks for."""

    def factory(rubric: Rubric, samples: int, tier: str) -> Judge:
        return Judge(rubric, samples=samples, judge_fn=judge_that(decisions))

    return factory


def run_cli(tmp_path: Path, decisions: dict[str, bool]) -> tuple[int, Path]:
    code = main(
        [
            "--rubric",
            "example",
            "--labels",
            str(FIXTURE),
            "--rubric-root",
            str(REPO_ROOT / "evals" / "rubrics"),
            "--out-root",
            str(tmp_path),
        ],
        judge_factory=a_judge_factory(decisions),
    )
    written = sorted(tmp_path.glob("*.md"))
    return code, written[0] if written else tmp_path / "missing.md"


def test_the_cli_writes_a_dated_report(tmp_path: Path) -> None:
    labels = load_labels(FIXTURE)
    perfect = {label.item_id: label.human_passed for label in labels}

    code, report = run_cli(tmp_path, perfect)

    assert report.exists()
    assert report.name.startswith("example_v1_")
    assert report.name.endswith(".md")
    text = report.read_text(encoding="utf-8")
    assert NOT_FIT not in text, "a perfect judge is fit"
    assert "**FIT FOR USE**" in text
    assert code == 0


def test_the_cli_exits_non_zero_when_the_judge_is_not_fit(tmp_path: Path) -> None:
    # A judge that passes everything: agreement 0.5, kappa 0.0 against the balanced fixture.
    labels = load_labels(FIXTURE)
    always_pass = {label.item_id: True for label in labels}

    code, report = run_cli(tmp_path, always_pass)

    assert code == 1, "a failed calibration must fail the pipeline that ran it"
    assert NOT_FIT in report.read_text(encoding="utf-8")


def test_the_cli_report_is_stamped_as_a_smoke_test(tmp_path: Path) -> None:
    # The committed fixture is synthetic, so no report built from it may look authoritative.
    labels = load_labels(FIXTURE)
    perfect = {label.item_id: label.human_passed for label in labels}

    _code, report = run_cli(tmp_path, perfect)

    assert "SMOKE TEST, NOT A CALIBRATION" in report.read_text(encoding="utf-8")


def test_the_router_path_is_used_when_no_judge_fn_is_given() -> None:
    # A stand-in router with just the one method the protocol needs: this exercises the
    # branch that would otherwise only run against a live provider.
    prompts: list[str] = []

    class StandInRouter:
        def structured(
            self, prompt: str, schema: type, *, purpose: str, tier: str
        ) -> tuple[JudgeResponse, ModelCall | None]:
            prompts.append(prompt)
            assert schema is JudgeResponse
            assert tier == "small", "a judge runs on the cheap tier by default"
            assert purpose == "judge:unit@v1"
            return JudgeResponse(criterion_scores={"alpha": 1.0}, rationale="via router"), None

    result = Judge(a_rubric(), router=StandInRouter()).judge("i1", expected="e", output="o")

    assert result.verdict.passed
    assert "via router" in result.verdict.rationale
    assert "unit@v1" in prompts[0]
