"""Putting unmatched findings in front of a human, because the gold set is not the truth.

The answer key records what was **planted** in a tender, not every defect the package
contains. A finding that matches nothing may be a false positive — or it may be a real
problem nobody thought to label. Counting all of them as wrong would punish the pipeline
for reading more carefully than the person who wrote the key.

So an unmatched finding is not scored against the system until somebody rules on it.
`write_queue` files them at
``evals/adjudication/<tender_name>_<timestamp>.jsonl`` with ``verdict: null``, and
`load_queue` reads back what a human wrote in the ``verdict`` field: ``true_new`` or
``false_positive``. A confirmed new finding is given a ``U-`` prefixed id — ``U-`` for
unplanted — so it can never be mistaken for a gold item that was in the key all along, and
it feeds `precision_adjudicated`.

Anything still ``null`` is pending and is reported as such. Pending is not a quiet pass:
`precision_strict` counts every unmatched finding against the system regardless, and both
precisions are always shown together, so an unreviewed queue cannot flatter a run.

Deliberately does not: decide a verdict. Nothing here infers, guesses or asks a model
whether an unmatched finding is real — the whole point of the file is that a person writes
that word. It also never edits the gold set: a confirmed new finding is recorded here, and
promoting it into the answer key is a separate, deliberate act.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ri05_tender.eval.models import Finding

DEFAULT_ADJUDICATION_ROOT = Path("evals/adjudication")
NEW_FINDING_PREFIX = "U-"

Verdict = Literal["true_new", "false_positive"]

__all__ = [
    "DEFAULT_ADJUDICATION_ROOT",
    "NEW_FINDING_PREFIX",
    "AdjudicationEntry",
    "AdjudicationError",
    "AdjudicationQueue",
    "Verdict",
    "load_queue",
    "queue_from_findings",
    "queue_path",
    "write_queue",
]


class AdjudicationError(Exception):
    """An adjudication file could not be read."""


class AdjudicationEntry(BaseModel):
    """One unmatched finding, and what a human made of it.

    `verdict` is None until somebody writes one. The fields around it are copied from the
    finding rather than referenced, so the file can be read and ruled on without the run
    that produced it still being around.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str = Field(min_length=1)
    finding_type: str
    refs: list[str] = Field(default_factory=list)
    severity: str
    statement: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: Verdict | None = Field(
        default=None, description='Written by a human: "true_new" or "false_positive".'
    )
    adjudicated_by: str | None = None
    note: str = ""

    @property
    def is_true_new(self) -> bool:
        """Whether a human confirmed this as a real defect the answer key had not planted."""
        return self.verdict == "true_new"

    @property
    def is_pending(self) -> bool:
        """Whether nobody has ruled on it yet."""
        return self.verdict is None

    @property
    def adjudicated_id(self) -> str:
        """The id a confirmed new finding is known by: the finding id under a `U-` prefix.

        Prefixed so it can never be confused with a gold id from the answer key. A finding
        that was not confirmed keeps its own id, because it has not become anything.
        """
        return f"{NEW_FINDING_PREFIX}{self.finding_id}" if self.is_true_new else self.finding_id


class AdjudicationQueue(BaseModel):
    """Every unmatched finding from one run, and the verdicts recorded against them."""

    model_config = ConfigDict(frozen=True)

    tender_name: str
    source_path: Path | None = None
    entries: list[AdjudicationEntry] = Field(default_factory=list)

    @property
    def true_new_ids(self) -> list[str]:
        """The finding ids a human confirmed as real, for `precision_adjudicated`."""
        return [entry.finding_id for entry in self.entries if entry.is_true_new]

    @property
    def false_positive_ids(self) -> list[str]:
        """The finding ids a human ruled were not defects."""
        return [entry.finding_id for entry in self.entries if entry.verdict == "false_positive"]

    @property
    def pending_ids(self) -> list[str]:
        """The finding ids nobody has ruled on."""
        return [entry.finding_id for entry in self.entries if entry.is_pending]

    @property
    def new_finding_ids(self) -> list[str]:
        """The confirmed-new findings under their `U-` prefixed ids."""
        return [entry.adjudicated_id for entry in self.entries if entry.is_true_new]

    def __len__(self) -> int:
        """How many findings are in the queue."""
        return len(self.entries)


def queue_from_findings(
    findings: Sequence[Finding], unmatched_ids: Sequence[str], *, tender_name: str
) -> AdjudicationQueue:
    """Build a queue from the findings a match report left unmatched, in finding-id order."""
    wanted = set(unmatched_ids)
    entries = [
        AdjudicationEntry(
            finding_id=finding.finding_id,
            finding_type=finding.finding_type,
            refs=list(finding.refs),
            severity=finding.severity,
            statement=finding.statement,
            confidence=finding.confidence,
        )
        for finding in sorted(findings, key=lambda f: f.finding_id)
        if finding.finding_id in wanted
    ]
    return AdjudicationQueue(tender_name=tender_name, entries=entries)


def queue_path(
    tender_name: str,
    *,
    root: Path = DEFAULT_ADJUDICATION_ROOT,
    at: datetime | None = None,
) -> Path:
    """Where one run's queue is filed: `<root>/<tender_name>_<timestamp>.jsonl`."""
    stamp = (at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{tender_name}_{stamp}.jsonl"


def write_queue(
    queue: AdjudicationQueue,
    *,
    root: Path = DEFAULT_ADJUDICATION_ROOT,
    at: datetime | None = None,
) -> Path:
    """Write the queue as JSONL and return where it went.

    One entry per line, so a reviewer can rule on them in any editor and a diff shows
    exactly which verdicts changed.
    """
    path = queue_path(queue.tender_name, root=root, at=at)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [entry.model_dump_json() for entry in queue.entries]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def load_queue(path: Path, *, tender_name: str | None = None) -> AdjudicationQueue:
    """Read a queue back, including whatever verdicts a human has written into it."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise AdjudicationError(f"adjudication file not found: {path}") from error
    except IsADirectoryError as error:
        raise AdjudicationError(f"adjudication path is a directory: {path}") from error

    entries: list[AdjudicationEntry] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(AdjudicationEntry.model_validate_json(line))
        except ValidationError as error:
            # One handler, not two: pydantic raises ValidationError for malformed JSON as
            # well as for a bad field, and ValidationError is itself a ValueError, so a
            # second `except ValueError` below it could never run. Checked, not assumed.
            problems = "; ".join(
                f"{'.'.join(str(part) for part in problem['loc']) or '(root)'}: {problem['msg']}"
                for problem in error.errors()
            )
            raise AdjudicationError(f"{path}:{number}: {problems}") from error

    return AdjudicationQueue(
        tender_name=tender_name or path.stem.rsplit("_", 1)[0],
        source_path=path,
        entries=entries,
    )
