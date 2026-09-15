"""The keyword baseline: what a one-line rule recovers, measured rather than asserted.

A recovery figure means nothing on its own. "The system recovers 68% of the obligations the
package does not state plainly" is only interesting against what a rule anyone could write
in a minute recovers, and that comparison has to be **measured in CI on the same inputs**,
not written into a README once and left there.

This module is also the reason the gold set was re-labelled at rev 3. Run against the old
IMPLICIT class it recovered 20 of 72 items, against a written claim that it would recover
none — because 40 of those 72 obligations are plain "shall" clauses, merely displaced into
drawing notes, annexes and federal-provisions text. Two figures are reported now, over the
two classes that replaced it, and they are never added together.

So this module runs the crudest possible rule — every line containing the word "shall" —
and scores it through exactly the same machinery the real pipeline will be scored through:

- the same tender loader, so the baseline reads the same PDFs and worksheets through the
  same extraction code. A baseline that read different input would prove nothing.
- the same `matcher`, with no special case anywhere for the fact that these findings came
  from a regex. A baseline scored by a kinder rule is not a baseline. In particular it is
  held to the same `expects_recovered_statement` rule as anything else, and a regex can only
  quote, so it fails that rule on the merits rather than by exemption.

Invoked as::

    python -m ri05_tender.eval.baseline data/tenders/<folder>          # write the record
    python -m ri05_tender.eval.baseline data/tenders/<folder> --check  # CI: has it drifted?

`--check` recomputes and compares against the committed file on the measured figures only,
ignoring the date, so a re-run on another day is not a diff. It exits non-zero when a
number has moved, which means either the extraction changed or the gold set did — both
worth knowing before an uplift figure is published off the old denominator.

Deliberately does not: call a model, reach a network, or try to be good. It is meant to be
beaten. Making the rule cleverer would raise the floor the real system is measured against
and flatter nothing but this file. It also does not decide whether the number it produces
is acceptable — see `tests/ri05/test_scoring_baseline.py`, where the figures are pinned and
what they imply is written down.
"""

import argparse
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ri05_tender.eval.loader import load_gold
from ri05_tender.eval.matcher import (
    DISPLACED_CLASS,
    UNSTATED_CLASS,
    match_findings,
    source_text,
)
from ri05_tender.eval.metrics import score
from ri05_tender.eval.models import Finding
from ri05_tender.tender.loader import load_tender
from ri05_tender.tender.models import TenderPackage

DEFAULT_RESULTS_ROOT = Path("evals/results")
KEYWORD = "shall"
BASELINE_NAME = "keyword_shall"

# What the rule claims each hit is. A "shall" grep is, by construction, a detector of
# requirements written as "shall" clauses — which is exactly the DISPLACED class. Claiming
# `unstated_requirement` instead would be claiming to have read a sentence that does not
# exist, so the rule makes the strongest claim it honestly can and no more.
BASELINE_FINDING_TYPE = "displaced_requirement"

# A clause reference as these documents print it: a short uppercase prefix, a hyphen, then a
# dotted number, optionally with a letter section — TS-B.25, ITB-9.2, CS-1.4, GN-02. Kept
# deliberately plain: a cleverer extractor would be a better baseline, and the point of a
# baseline is to be the floor.
REFERENCE = re.compile(r"\b[A-Z]{1,4}-[A-Z]?\.?\d+(?:\.\d+)*\b")

__all__ = [
    "BASELINE_FINDING_TYPE",
    "BASELINE_NAME",
    "DEFAULT_RESULTS_ROOT",
    "KEYWORD",
    "REFERENCE",
    "BaselineRecord",
    "baseline_findings",
    "baseline_path",
    "main",
    "measure_baseline",
    "write_record",
]


class BaselineRecord(BaseModel):
    """What the keyword rule recovered, on one tender, on one day."""

    model_config = ConfigDict(frozen=True)

    tender_name: str
    baseline: str = Field(default=BASELINE_NAME, description="Which rule produced this.")
    keyword: str = Field(default=KEYWORD)
    measured_on: date
    finding_type: str = Field(default=BASELINE_FINDING_TYPE, description="What each hit claims.")
    hits: int = Field(ge=0, description="Lines containing the keyword; one finding each.")

    unstated_items: int = Field(ge=0, description="Scored items carrying the UNSTATED class.")
    unstated_recovered: int = Field(ge=0)
    unstated_recall: float = Field(
        ge=0.0, le=1.0, description="The floor an UNSTATED-recovery claim must clear."
    )
    unstated_recovered_ids: list[str] = Field(default_factory=list)

    displaced_items: int = Field(ge=0, description="Scored items carrying the DISPLACED class.")
    displaced_recovered: int = Field(ge=0)
    displaced_recall: float = Field(
        ge=0.0, le=1.0, description="The floor a DISPLACED-recovery claim must clear."
    )
    displaced_recovered_ids: list[str] = Field(
        default_factory=list,
        description="Which DISPLACED items the rule reached. Named, not counted: this is the "
        "half of the old IMPLICIT class a keyword rule has any claim on, so every id here is "
        "a question to answer before an uplift figure is published.",
    )

    recall_overall: float = Field(ge=0.0, le=1.0)
    precision_strict: float = Field(ge=0.0, le=1.0)

    def measured(self) -> dict[str, object]:
        """Everything except the date, which is the part a re-run is allowed to change."""
        return self.model_dump(exclude={"measured_on"})


def baseline_findings(package: TenderPackage) -> list[Finding]:
    """One finding per line containing the keyword, citing whatever references that line names.

    `recovered_statement` is the source line itself, because quoting is the most a regex can
    do — it has no way to say an obligation in other words. The matcher rejects a verbatim
    quote, so every hit fails the recovery test on the merits. That is the point: the claim
    "a keyword rule cannot recover an obligation" is now something the harness establishes
    rather than something a document asserts, and the previous assertion was false.
    """
    findings: list[Finding] = []
    for document in package.documents:
        for page in document.pages:
            for number, line in enumerate(page.text.splitlines(), start=1):
                if KEYWORD not in line.lower():
                    continue
                findings.append(
                    Finding(
                        finding_id=f"KW-{document.document_id}-p{page.page_number}-l{number:04d}",
                        finding_type=BASELINE_FINDING_TYPE,
                        refs=REFERENCE.findall(line),
                        severity="minor",
                        statement=line.strip(),
                        recovered_statement=line.strip(),
                        confidence=0.5,
                    )
                )
    return findings


def measure_baseline(folder: Path, *, on: date | None = None) -> BaselineRecord:
    """Run the keyword rule over a tender folder and score it like any other run."""
    package = load_tender(folder)
    gold = load_gold(package)
    findings = baseline_findings(package)

    report = match_findings(findings, gold.scored, source=source_text(package))
    card = score(
        tender_name=package.name,
        gold=gold.scored,
        excluded=gold.excluded,
        finding_ids=[finding.finding_id for finding in findings],
        report=report,
    )

    def reached(class_name: str) -> list[str]:
        ids = {item.id for item in gold.scored if class_name in item.classes}
        return sorted(ids & report.matched_gold_ids)

    unstated, displaced = reached(UNSTATED_CLASS), reached(DISPLACED_CLASS)

    return BaselineRecord(
        tender_name=package.name,
        measured_on=on or datetime.now(UTC).date(),
        hits=len(findings),
        unstated_items=card.unstated_total,
        unstated_recovered=len(unstated),
        unstated_recall=card.unstated_recovery,
        unstated_recovered_ids=unstated,
        displaced_items=card.displaced_total,
        displaced_recovered=len(displaced),
        displaced_recall=card.displaced_recovery,
        displaced_recovered_ids=displaced,
        recall_overall=card.recall_overall,
        precision_strict=card.precisions.strict,
    )


def baseline_path(tender_name: str, *, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    """Where a tender's baseline record lives.

    No timestamp in the name, unlike an evaluation result: there is one keyword baseline per
    tender and a new measurement replaces it, so the uplift figure always divides by the
    current floor rather than by whichever old file somebody reached for.
    """
    return root / f"{tender_name}_keyword_baseline.json"


def write_record(record: BaselineRecord, *, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    """Write the record and return where it went."""
    path = baseline_path(record.tender_name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _drift(fresh: BaselineRecord, committed: BaselineRecord) -> list[str]:
    """Every measured field that moved, described for a CI log."""
    was, now = committed.measured(), fresh.measured()
    return [
        f"{key}: committed {was[key]!r}, measured {now[key]!r}"
        for key in now
        if was[key] != now[key]
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry point and return the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m ri05_tender.eval.baseline",
        description="Measure what a plain keyword rule recovers, as the floor for any uplift.",
    )
    parser.add_argument("folder", type=Path, help="The tender folder, as the loader takes it.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Where the record is written or read from.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute and compare against the committed record instead of writing it. "
        "Exits non-zero if any measured figure moved. The date is ignored.",
    )
    args = parser.parse_args(argv)

    # Under --check, look for the record before measuring: reading a tender takes seconds,
    # and there is nothing to compare a fresh measurement against if no file exists.
    if args.check:
        expected = baseline_path(Path(args.folder).name, root=args.results_root)
        if not expected.is_file():
            print(f"No committed baseline at {expected}. Run without --check to write one.")
            return 1

    fresh = measure_baseline(args.folder)
    path = baseline_path(fresh.tender_name, root=args.results_root)

    if not args.check:
        write_record(fresh, root=args.results_root)
        print(f"Wrote {path}: {_summary(fresh)}")
        return 0

    committed = BaselineRecord.model_validate_json(path.read_text(encoding="utf-8"))
    drift = _drift(fresh, committed)
    if drift:
        print(f"The keyword baseline for {fresh.tender_name!r} has moved since {path} was written:")
        for line in drift:
            print(f"  {line}")
        print(
            "Either the extraction changed or the gold set did. Re-run without --check and "
            "commit the new record before publishing an uplift figure against the old floor."
        )
        return 1

    print(f"{path} is current: {_summary(fresh)}")
    return 0


def _summary(record: BaselineRecord) -> str:
    """One line naming both floors, never their sum."""
    return (
        f"{record.hits} keyword hit(s), "
        f"UNSTATED recall {record.unstated_recall:.4f} "
        f"({record.unstated_recovered} of {record.unstated_items}), "
        f"DISPLACED recall {record.displaced_recall:.4f} "
        f"({record.displaced_recovered} of {record.displaced_items})."
    )


if __name__ == "__main__":
    raise SystemExit(main())
