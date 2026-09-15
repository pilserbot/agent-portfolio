"""The keyword baseline, pinned — and what its number says now that the class was split.

A recovery claim is meaningless without a floor, and the floor has to be measured on the
same inputs by the same code, in CI, not written into a README. These tests pin it.

**This module is why the gold set was re-labelled.** The written claim was that a "shall"
rule recovers none of the 72 IMPLICIT items, because an implicit obligation is one the
package does not state. Measured, it recovered 20 — because 40 of those 72 obligations are
plain "shall" clauses, merely displaced into drawing notes, annexes and federal-provisions
text. One class was doing two jobs. At rev 3 it became two: UNSTATED (32) and DISPLACED
(40), reported separately and never summed.

Both floors now measure 0.0, and the two tests that matter here are the ones establishing
that neither zero is an accident:

- DISPLACED is 0.0 **because of the recovery rule**, not because the class was renamed
  out from under the rule. The keyword findings still reach 37 DISPLACED items on anchor
  and type; relax `expects_recovered_statement` and exactly the same 20 match as before
  the split. The rule is doing the work.
- UNSTATED is 0.0 **structurally**. No keyword finding anchors a single UNSTATED item —
  not even as a PARTIAL — because there is no "shall" sentence anywhere for one.

A zero that no test can distinguish from a broken harness is not evidence of anything.

No network, no model, no key.
"""

import json
from datetime import date
from pathlib import Path

import pytest
from openpyxl import Workbook

from ri05_tender.eval.baseline import (
    BASELINE_FINDING_TYPE,
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
from ri05_tender.eval.loader import load_gold
from ri05_tender.eval.matcher import (
    DISPLACED_CLASS,
    UNSTATED_CLASS,
    match_findings,
    source_text,
)
from ri05_tender.tender.loader import load_tender
from ri05_tender.tender.models import TenderPackage

KESSLER_POINT = Path("data/tenders/kessler_point")
COMMITTED = Path("evals/results/kessler_point_keyword_baseline.json")

# Measured, not chosen. Every one is re-derived below from the committed tender, so a change
# to the extraction, the matcher or the gold set fails here rather than moving a claim.
EXPECTED_HITS = 861
EXPECTED_UNSTATED_ITEMS = 32
EXPECTED_DISPLACED_ITEMS = 40
EXPECTED_UNSTATED_RECOVERED = 0
EXPECTED_DISPLACED_RECOVERED = 0

# What the rule reaches before the recovery rule refuses it: 37 DISPLACED items on anchor
# and type, of which 20 would be credited as matches. 20 is the figure that was published as
# implicit recovery before the split, and it is kept here because the two zeroes above only
# mean something beside it.
DISPLACED_REACHED_AS_OUTPUT_MISS = 37
DISPLACED_MATCHED_WITHOUT_THE_RECOVERY_RULE = 20


def a_tiny_tender(root: Path) -> Path:
    """A real tender folder small enough that the CLI tests do not read 48 PDF pages.

    Two worksheet rows, one of them a "shall" statement citing a clause the answer key
    plants. The planted item is DISPLACED and does **not** ask for a recovered statement, so
    the command has a non-zero figure to report and a moved figure to detect — which the
    committed tender, measuring 0.0 on both classes, could not give it.
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
                "classes": [DISPLACED_CLASS],
                "finding_type": BASELINE_FINDING_TYPE,
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


# --- the two headline numbers -------------------------------------------------------------


def test_the_keyword_baseline_recovers_nothing_on_either_class(record: BaselineRecord) -> None:
    """Both floors are 0.0. The two tests after this one are what make that mean something."""
    assert record.unstated_recovered == EXPECTED_UNSTATED_RECOVERED, (
        f"The keyword baseline recovers {record.unstated_recovered} of "
        f"{record.unstated_items} UNSTATED items. It should reach none: an UNSTATED "
        f"obligation has no requirement sentence anywhere in the package, so a rule that "
        f"searches sentences cannot produce a candidate for one. A non-zero figure here "
        f"means the UNSTATED labelling is wrong for {', '.join(record.unstated_recovered_ids)} "
        f"— settle that before publishing any unstated-uplift figure."
    )
    assert record.displaced_recovered == EXPECTED_DISPLACED_RECOVERED, (
        f"The keyword baseline recovers {record.displaced_recovered} of "
        f"{record.displaced_items} DISPLACED items "
        f"({', '.join(record.displaced_recovered_ids)}). A DISPLACED obligation IS written "
        f"out, so the rule reaches it — the recovery rule is what refuses it, because the "
        f"rule can only quote. A non-zero figure means a quote is being credited as a "
        f"recovery again, which is the exact error the split was made to stop."
    )
    assert record.unstated_recall == 0.0
    assert record.displaced_recall == 0.0


def test_the_displaced_zero_is_the_recovery_rule_and_not_a_type_mismatch(
    package: TenderPackage,
) -> None:
    """Relax the one rule and the old number comes back. That is what makes the zero earned.

    Renaming IMPLICIT to DISPLACED could have produced 0.0 on its own, by leaving the
    baseline claiming a finding type nothing maps to any more. It did not: the findings
    still anchor and type onto 37 DISPLACED items, and with `expects_recovered_statement`
    switched off, exactly the 20 that were published as implicit recovery are credited.
    """
    gold = load_gold(package)
    findings = baseline_findings(package)
    source = source_text(package)

    enforced = match_findings(findings, gold.scored, source=source)
    displaced = {item.id for item in gold.scored if DISPLACED_CLASS in item.classes}

    reached = {a.gold_id for a in enforced.output_misses} & displaced
    assert len(reached) == DISPLACED_REACHED_AS_OUTPUT_MISS
    assert all(
        "recovered_statement" in a.missing_outputs
        for a in enforced.output_misses
        if a.gold_id in reached
    )

    relaxed_gold = [
        item.model_copy(update={"expects_recovered_statement": False}) for item in gold.scored
    ]
    relaxed = match_findings(findings, relaxed_gold, source=source)

    assert len(relaxed.matched_gold_ids & displaced) == DISPLACED_MATCHED_WITHOUT_THE_RECOVERY_RULE
    assert "D4-25" in relaxed.matched_gold_ids


def test_the_unstated_zero_is_structural_not_merely_unearned(package: TenderPackage) -> None:
    """No keyword finding even anchors an UNSTATED item, so the class is out of reach by shape.

    Not "the rule tried and failed the recovery test" — the rule cannot produce a candidate
    at all. An UNSTATED obligation lives in a Bill of Quantities line or a Pricing Schedule
    row, and there is no sentence to grep.
    """
    gold = load_gold(package)
    report = match_findings(baseline_findings(package), gold.scored, source=source_text(package))
    unstated = {item.id for item in gold.scored if UNSTATED_CLASS in item.classes}

    anchored = {gold_id for partial in report.partials for gold_id in partial.gold_ids}
    reached = {a.gold_id for a in report.output_misses} | report.matched_gold_ids

    assert not unstated & anchored
    assert not unstated & reached
    assert len(unstated) == EXPECTED_UNSTATED_ITEMS


def test_the_baseline_is_measured_on_the_committed_tender(record: BaselineRecord) -> None:
    assert record.tender_name == "kessler_point"
    assert record.hits == EXPECTED_HITS
    assert record.unstated_items == EXPECTED_UNSTATED_ITEMS
    assert record.displaced_items == EXPECTED_DISPLACED_ITEMS
    # 32 + 40 is the 72 the single IMPLICIT class used to carry. Asserted here, and nowhere
    # in the code: the classes are counted separately and never added.
    assert record.unstated_items + record.displaced_items == 72


def test_the_committed_record_matches_a_fresh_measurement(record: BaselineRecord) -> None:
    # The file in the repo is the floor an uplift figure divides by. If it drifts from what
    # the code measures, the published uplift is against a number nobody can reproduce.
    committed = BaselineRecord.model_validate_json(COMMITTED.read_text(encoding="utf-8"))

    assert committed.measured() == record.measured()


# --- why the split was needed, still shown from the package -------------------------------


def test_a_displaced_item_is_written_out_as_a_shall_clause_in_the_package(
    package: TenderPackage,
) -> None:
    """The evidence the DISPLACED class exists for, on the item that exposed the problem.

    D4-25's gold statement is that no equipment may be "scheduled for end of production or
    end of support within five (5) years". TS-1.1 says, in the tender, that equipment
    "shall not be scheduled for end of production or end of support within five (5)". The
    obligation is written out — displaced into the Technical Specification's opening general
    clause, not absent. Calling that implicit is what made the old claim false.
    """
    spec = package.document("04_Technical_Specification")
    assert spec is not None

    stated = [line for line in spec.text.splitlines() if line.startswith("TS-1.1")]

    assert stated, "TS-1.1 not found; the gold set or the extraction has moved"
    assert KEYWORD in stated[0].lower()
    assert "end of production" in stated[0]


def test_every_baseline_finding_quotes_rather_than_recovers(package: TenderPackage) -> None:
    """The second cause, now enforced rather than merely observed.

    `finding_type` is still asserted by the rule rather than earned — it claims every hit is
    a displaced requirement. What has changed is that the claim alone no longer earns credit:
    each finding's `recovered_statement` is the source line verbatim, and the matcher
    refuses a verbatim span of the tender.
    """
    findings = baseline_findings(package)
    source = source_text(package)

    assert all(finding.finding_type == BASELINE_FINDING_TYPE for finding in findings)
    assert all(finding.recovered_statement == finding.statement for finding in findings)
    assert all(source.quotes(finding.recovered_statement or "") for finding in findings)


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


def test_the_rule_claims_the_strongest_type_it_honestly_can() -> None:
    # A "shall" grep is a detector of requirements written as "shall" clauses, which is the
    # DISPLACED class. Claiming `unstated_requirement` would be claiming to have read a
    # sentence that does not exist.
    assert BASELINE_FINDING_TYPE == "displaced_requirement"


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
    assert record.finding_type == BASELINE_FINDING_TYPE


def test_the_record_keeps_a_named_list_per_class(record: BaselineRecord) -> None:
    # Named, not counted: each id would be a question about that item's class. Both lists
    # are empty today, and each is reported beside its own denominator.
    assert record.unstated_recovered_ids == []
    assert record.displaced_recovered_ids == []
    assert len(record.unstated_recovered_ids) == record.unstated_recovered
    assert len(record.displaced_recovered_ids) == record.displaced_recovered


def test_the_record_carries_no_combined_recovery_figure() -> None:
    fields = set(BaselineRecord.model_fields)

    assert not {name for name in fields if "implicit" in name}
    assert {"unstated_recall", "displaced_recall"} <= fields


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
    out = capsys.readouterr().out
    assert exit_code == 0
    assert written["hits"] == 1  # one line carries the keyword
    assert written["displaced_recovered"] == 1
    assert written["unstated_items"] == 0
    assert "UNSTATED recall" in out and "DISPLACED recall" in out


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
