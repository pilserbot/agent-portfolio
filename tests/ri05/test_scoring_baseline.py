"""The keyword baseline, pinned — and what its number actually says.

An implicit-recovery claim is meaningless without a floor, and the floor has to be measured
on the same inputs by the same code, in CI, not written into a README. These tests pin it.

**The measured figure is not what was expected, and that is the finding.** The expectation
was that a "shall" rule recovers none of the IMPLICIT items, because an implicit obligation
is by definition not written as a shall statement. It recovers 20 of 72. The assertion below
records the real number and the message says what has to be done about it, because a test
rewritten to expect 0.0 would be a test that lies.

No network, no model, no key.
"""

import json
from datetime import date
from pathlib import Path

import pytest
from openpyxl import Workbook

from ri05_tender.eval.baseline import (
    BASELINE_NAME,
    KEYWORD,
    REFERENCE,
    BaselineRecord,
    baseline_findings,
    baseline_path,
    main,
    measure_baseline,
    write_record,
)
from ri05_tender.tender.loader import load_tender
from ri05_tender.tender.models import TenderPackage

KESSLER_POINT = Path("data/tenders/kessler_point")
COMMITTED = Path("evals/results/kessler_point_keyword_baseline.json")

# Measured, not chosen. Every one of these is re-derived by the tests below from the
# committed tender, so a change to the extraction or the gold set fails here.
EXPECTED_HITS = 861
EXPECTED_IMPLICIT_ITEMS = 72
EXPECTED_IMPLICIT_RECOVERED = 20

# What the expectation was. Kept in the file because the gap between it and the measurement
# is the whole point of this module.
INTENDED_IMPLICIT_RECALL = 0.0


def a_tiny_tender(root: Path) -> Path:
    """A real tender folder small enough that the CLI tests do not read 48 PDF pages.

    Two worksheet rows, one of them a "shall" statement citing a clause the answer key
    plants. Enough for the command to have something to measure and compare.
    """
    folder = root / "tiny_point"
    (folder / "documents").mkdir(parents=True)
    book = Workbook()
    book.remove(book.active)
    sheet = book.create_sheet(title="Requirements")
    sheet.append(["TS-9.1 The Contractor shall provide a plan."])
    sheet.append(["TS-9.2 Pricing is indicative."])
    book.save(folder / "documents" / "01_Spec.xlsx")

    (folder / "gold").mkdir(parents=True)
    (folder / "gold" / "gold_set.jsonl").write_text(
        json.dumps(
            {
                "id": "T-01",
                "document": 1,
                "refs": ["TS-9.1"],
                "classes": ["IMPLICIT"],
                "finding_type": "implicit_requirement",
                "severity": "minor",
                "tier": "cold_start",
                "review": {"status": "accepted_by_default"},
                "scored": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return folder


@pytest.fixture(scope="module")
def package() -> TenderPackage:
    """The committed tender, loaded once: reading 48 real PDF pages is the slow part."""
    return load_tender(KESSLER_POINT)


@pytest.fixture(scope="module")
def record() -> BaselineRecord:
    """The baseline, measured once from the committed tender."""
    return measure_baseline(KESSLER_POINT, on=date(2026, 1, 1))


# --- the headline number ----------------------------------------------------------------


def test_the_keyword_baseline_does_not_recover_zero_implicit_items(
    record: BaselineRecord,
) -> None:
    """The intended invariant was 0.0. It is not, and the number is recorded rather than wished.

    Why this matters: `implicit_recovery` is the metric RI-05's value rests on. If a rule
    anyone could write in a minute clears 27.8% of it, then either those items are not
    implicit, or the matcher is crediting findings that identified nothing. Both are true
    here — see the two tests below — and both must be settled before an uplift figure is
    published against this denominator.
    """
    assert record.implicit_recovered == EXPECTED_IMPLICIT_RECOVERED, (
        f"The keyword baseline recovers {record.implicit_recovered} of "
        f"{record.implicit_items} IMPLICIT items ({record.implicit_recall:.1%}), where "
        f"{EXPECTED_IMPLICIT_RECOVERED} was measured when this was written. The intended "
        f"value is {INTENDED_IMPLICIT_RECALL:.1f} — a 'shall' rule "
        f"should recover nothing, because an implicit obligation is one the package does "
        f"not state. It does not, so either the IMPLICIT labelling is wrong for the items "
        f"named in {COMMITTED}, or the matcher credits a finding for citing the clause an "
        f"obligation hides in without identifying the obligation. Settle that before "
        f"publishing any implicit-uplift figure: the denominator is not what it claims. "
        f"Recovered: {', '.join(record.recovered_ids)}"
    )
    assert record.implicit_recall > INTENDED_IMPLICIT_RECALL


def test_the_baseline_is_measured_on_the_committed_tender(record: BaselineRecord) -> None:
    assert record.tender_name == "kessler_point"
    assert record.hits == EXPECTED_HITS
    assert record.implicit_items == EXPECTED_IMPLICIT_ITEMS
    assert record.implicit_recall == pytest.approx(
        EXPECTED_IMPLICIT_RECOVERED / EXPECTED_IMPLICIT_ITEMS
    )


def test_the_committed_record_matches_a_fresh_measurement(record: BaselineRecord) -> None:
    # The file in the repo is the floor an uplift figure divides by. If it drifts from what
    # the code measures, the published uplift is against a number nobody can reproduce.
    committed = BaselineRecord.model_validate_json(COMMITTED.read_text(encoding="utf-8"))

    assert committed.measured() == record.measured()


# --- why it is not zero: the two causes, each shown ---------------------------------------


def test_some_recovered_items_are_stated_as_shall_clauses_in_the_package(
    package: TenderPackage,
) -> None:
    """Cause one: the IMPLICIT label is wrong for at least some items.

    D4-25's gold statement is that no equipment may be "scheduled for end of production or
    end of support within five (5) years". TS-1.1 says, in the tender, that equipment
    "shall not be scheduled for end of production or end of support within five (5)". The
    obligation is written out. Whatever else it is, it is not implicit.
    """
    spec = package.document("04_Technical_Specification")
    assert spec is not None

    stated = [line for line in spec.text.splitlines() if line.startswith("TS-1.1")]

    assert stated, "TS-1.1 not found; the gold set or the extraction has moved"
    assert KEYWORD in stated[0].lower()
    assert "end of production" in stated[0]


def test_the_matcher_credits_a_finding_that_only_quoted_the_clause(
    package: TenderPackage,
) -> None:
    """Cause two: `finding_type` is asserted, not earned.

    Every baseline finding claims to be an `implicit_requirement` and carries nothing but
    the clause text. The matcher checks refs and type; it cannot check that the finding
    identified the implication. So a rule that understood nothing is credited, and the same
    would be true of a pipeline that paraphrased clauses back.
    """
    findings = baseline_findings(package)

    assert all(finding.finding_type == "implicit_requirement" for finding in findings)
    # No output is produced by any of them, so any item requiring one is an OUTPUT_MISS
    # rather than a match — the credit above comes from items requiring nothing.
    assert all(finding.outputs.present() == set() for finding in findings)


# --- the rule itself -----------------------------------------------------------------------


def test_one_finding_per_line_containing_the_keyword(package: TenderPackage) -> None:
    findings = baseline_findings(package)
    lines = sum(
        1
        for document in package.documents
        for page in document.pages
        for line in page.text.splitlines()
        if KEYWORD in line.lower()
    )

    assert len(findings) == lines == EXPECTED_HITS


def test_findings_are_identified_by_where_they_came_from(package: TenderPackage) -> None:
    # Document, page and line, so a baseline hit can be looked at.
    first = baseline_findings(package)[0]

    assert first.finding_id.startswith("KW-")
    assert "-p" in first.finding_id and "-l" in first.finding_id


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("TS-B.5 All cameras shall support HTTPS", ["TS-B.5"]),
        ("B.01–B.11 vs TS-B.1 shall apply", ["TS-B.1"]),
        ("The Contractor shall comply", []),
        ("CS-1.4 and GCC-13.1 shall both apply", ["CS-1.4", "GCC-13.1"]),
    ],
)
def test_the_reference_pattern_takes_what_the_line_cites(line: str, expected: list[str]) -> None:
    assert REFERENCE.findall(line) == expected


def test_the_rule_is_deterministic(package: TenderPackage) -> None:
    assert baseline_findings(package) == baseline_findings(package)


# --- the record and its CLI ------------------------------------------------------------------


def test_the_record_names_the_rule_that_produced_it(record: BaselineRecord) -> None:
    # A floor with no note of how it was measured is not a floor anybody can re-derive.
    assert record.baseline == BASELINE_NAME
    assert record.keyword == KEYWORD


def test_the_record_names_the_items_it_recovered(record: BaselineRecord) -> None:
    # Named, not counted: each is a question about whether that item is implicit at all.
    assert len(record.recovered_ids) == EXPECTED_IMPLICIT_RECOVERED
    assert "D4-25" in record.recovered_ids
    assert record.recovered_ids == sorted(record.recovered_ids)


def test_the_date_is_the_only_field_a_rerun_may_change(record: BaselineRecord) -> None:
    later = record.model_copy(update={"measured_on": date(2027, 6, 30)})

    assert later.measured() == record.measured()
    assert later != record


def test_writing_and_reading_the_record_round_trips(tmp_path: Path, record: BaselineRecord) -> None:
    path = write_record(record, root=tmp_path)

    assert path == baseline_path("kessler_point", root=tmp_path)
    assert BaselineRecord.model_validate_json(path.read_text(encoding="utf-8")) == record


def test_the_path_carries_no_timestamp() -> None:
    # One baseline per tender, replaced in place: the uplift always divides by the current
    # floor rather than by whichever old file somebody reached for.
    assert baseline_path("kessler_point", root=Path("evals/results")) == COMMITTED


def test_the_cli_writes_the_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = a_tiny_tender(tmp_path)

    exit_code = main([str(folder), "--results-root", str(tmp_path)])

    written = json.loads(
        (tmp_path / "tiny_point_keyword_baseline.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert written["hits"] == 1  # one line carries the keyword
    assert written["implicit_recovered"] == 1
    assert "implicit recall" in capsys.readouterr().out


def test_the_cli_check_passes_against_the_committed_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # This is the CI step. It reads the repository's own file.
    exit_code = main([str(KESSLER_POINT), "--check"])

    assert exit_code == 0
    assert "is current" in capsys.readouterr().out


def test_the_cli_check_fails_when_a_measured_figure_moved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = a_tiny_tender(tmp_path)
    main([str(folder), "--results-root", str(tmp_path)])
    stale = BaselineRecord.model_validate_json(
        (tmp_path / "tiny_point_keyword_baseline.json").read_text(encoding="utf-8")
    )
    write_record(stale.model_copy(update={"hits": 99}), root=tmp_path)

    exit_code = main([str(folder), "--results-root", str(tmp_path), "--check"])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "has moved" in out
    assert "hits: committed 99" in out


def test_the_cli_check_ignores_the_date(tmp_path: Path) -> None:
    # A re-run on another day is not a diff, or the CI step would fail every midnight.
    folder = a_tiny_tender(tmp_path)
    main([str(folder), "--results-root", str(tmp_path)])
    written = tmp_path / "tiny_point_keyword_baseline.json"
    stale = BaselineRecord.model_validate_json(written.read_text(encoding="utf-8"))
    write_record(stale.model_copy(update={"measured_on": date(2020, 1, 1)}), root=tmp_path)

    assert main([str(folder), "--results-root", str(tmp_path), "--check"]) == 0


def test_the_cli_check_says_so_when_nothing_is_committed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # And says it without first reading the tender: there is nothing to compare against.
    exit_code = main([str(KESSLER_POINT), "--results-root", str(tmp_path), "--check"])

    assert exit_code == 1
    assert "Run without --check" in capsys.readouterr().out
