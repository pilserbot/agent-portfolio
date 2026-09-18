"""End to end: load, extract, detect, score, queue, report.

One function, `run`, doing the six things in order and returning everything each of them
produced. It is the only place in the project where the extraction pass, the findings engine
and the scoring harness meet, and it is deliberately thin — every decision it makes is
already made somewhere else, and what is left is plumbing plus one adapter.

**The adapter is the interesting part.** `req_core.findings.Finding` is domain-agnostic and
knows nothing about this project's matcher; `ri05_tender.eval.models.Finding` is what the
matcher scores. `as_eval_finding` converts, and `DETECTOR_TO_FINDING_TYPE` is where the two
vocabularies are reconciled — five detector names already match the class map's finding
types and one, `unverifiable`, is called `unverifiable_requirement` there. That mismatch
lives here rather than in either package, because it is a fact about this application's
scoring vocabulary and neither side should have to know about the other.

**What this run cannot yet produce.** Detectors emit findings; they do not yet emit the
artefacts a gold item's `expects_*` flags demand — a clarification question, a price impact,
an alternative. So an item requiring one scores OUTPUT_MISS rather than MATCH even when the
defect was found. That is the honest result and the report says so: the recall figure here
is recall of *detection*, and the outputs are the next step's work. Reading it as end-to-end
recall would flatter the second half of a job that has not been done.

**Cost.** 28 calls over the three specification documents: 14 to extract and 14 to read
claims, one of each per page that carries a clause. The two passes ask different questions
of the same text, so neither can be skipped — but the *extraction* half can be handed in.
A caller that already holds an `ExtractionResult` for this corpus passes it as `extraction`
and this run makes 14 calls, not 28. That is how the live suite stays at 28 calls in total
rather than 42: the extraction test's pass is the one both tests use.

**Why this lives in `eval` and not beside the detectors.** It scores against the answer key,
and the gold-leakage guard forbids any module outside `eval` from so much as importing the
package that reads it. That guard caught this file in the wrong place, which is the guard
working: the detectors are pipeline code and must never be able to reach the answers; this
is harness code whose job is to compare the two. Keeping the boundary where the guard draws
it is worth more than keeping the runner next to its configuration.

Deliberately does not: decide severity (that is `detect.config`), decide what a defect is
(that is `req_core.detectors`), hold a model client, or interpret its own score. It writes
the adjudication queue and renders the markdown; what the numbers mean is a human's.
"""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from req_core.contracts import ExtractionResult
from req_core.detectors import OrdinalScales
from req_core.engine import ReviewRouting, detect, route_for_review
from req_core.extraction import StructuredCompletion, extract_requirements
from req_core.findings import DetectionReport, SeverityPolicy
from req_core.findings import Finding as CoreFinding
from ri05_tender.detect.config import CONFIDENCE_THRESHOLD, ORDINAL_SCALES, SEVERITY_POLICY
from ri05_tender.eval.adjudication import queue_from_findings, write_queue
from ri05_tender.eval.loader import load_gold
from ri05_tender.eval.matcher import MatchReport, match_findings, source_text
from ri05_tender.eval.metrics import ScoreCard, score
from ri05_tender.eval.models import Finding as EvalFinding
from ri05_tender.eval.report import render_markdown as render_score_card
from ri05_tender.extract.config import ITB_2_1_POLICY, KESSLER_POINT_CLAUSE_STYLE
from ri05_tender.extract.gate import extraction_corpus
from ri05_tender.tender.loader import load_tender
from ri05_tender.tender.models import TenderPackage

__all__ = [
    "DETECTOR_TO_FINDING_TYPE",
    "DetectionRun",
    "as_eval_finding",
    "run",
]

# Where the two vocabularies differ. Five detector names are already the finding types the
# class map uses; `unverifiable` is `unverifiable_requirement` there. Stated once, here,
# rather than renaming a detector to suit one application's scoring harness.
DETECTOR_TO_FINDING_TYPE: dict[str, str] = {
    "missing_tolerance": "missing_tolerance",
    "modality_inconsistency": "modality_inconsistency",
    "atomicity_split": "atomicity_split",
    "unverifiable": "unverifiable_requirement",
    "ordinal_trap": "ordinal_trap",
    "zero_margin": "zero_margin",
}


def as_eval_finding(finding: CoreFinding) -> EvalFinding:
    """Convert a detector's finding into the record this project's matcher scores.

    `recovered_statement` is the finding's own statement, which is written by Python from
    the claims rather than quoted from the page — so it satisfies the matcher's rule that a
    recovered obligation must not be a verbatim span of the tender. It is not a way around
    that rule: the statement really is the engine's own words.
    """
    return EvalFinding(
        finding_id=finding.finding_id,
        finding_type=DETECTOR_TO_FINDING_TYPE[finding.detector],
        refs=[finding.parent_id or finding.requirement_id],
        severity=finding.severity,
        statement=finding.statement,
        recovered_statement=finding.statement,
        evidence=[finding.source],
        confidence=finding.confidence,
    )


class DetectionRun(BaseModel):
    """Everything one end-to-end run produced, including what it declined to assert."""

    model_config = ConfigDict(frozen=True)

    tender_name: str
    extraction: ExtractionResult
    detection: DetectionReport
    match: MatchReport
    card: ScoreCard
    routing: ReviewRouting
    queue_path: Path | None = Field(
        default=None, description="Where the adjudication queue was written, if it was."
    )

    @property
    def asserted(self) -> int:
        """Findings reported as fact."""
        return len(self.detection.findings)

    @property
    def abstained(self) -> int:
        """Findings withheld and routed to a human."""
        return len(self.detection.abstained)


def run(
    folder: Path,
    complete: StructuredCompletion,
    *,
    severity: SeverityPolicy = SEVERITY_POLICY,
    ordinal_scales: OrdinalScales = ORDINAL_SCALES,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    queue_root: Path | None = None,
    at: datetime | None = None,
    extraction: ExtractionResult | None = None,
) -> DetectionRun:
    """Load a tender, extract it, detect, score, queue the unmatched and return the lot.

    `extraction` may be supplied by a caller that already has one for this corpus, and then
    no extraction call is made at all. That is not an optimisation bolted on: extraction is
    deterministic given the corpus and the policy, so a second pass over the same three
    documents buys nothing and costs 14 calls, about $0.89 and roughly ten minutes. A
    supplied result is checked against the corpus it is about, because silently detecting
    over requirements extracted from a different package would be worse than paying twice.
    """
    package: TenderPackage = load_tender(folder)
    corpus = extraction_corpus(package)

    if extraction is None:
        extraction = extract_requirements(
            corpus,
            complete,
            policy=ITB_2_1_POLICY,
            style=KESSLER_POINT_CLAUSE_STYLE,
        )
    elif extraction.corpus_name != corpus.name:
        raise ValueError(
            f"the supplied extraction is of {extraction.corpus_name!r} and this run is over "
            f"{corpus.name!r}. Detecting over one package's requirements while scoring "
            f"against another's gold set would produce a number about nothing."
        )
    detection = detect(
        extraction.requirements,
        corpus,
        complete,
        modality=ITB_2_1_POLICY,
        severity=severity,
        ordinal_scales=ordinal_scales,
        style=KESSLER_POINT_CLAUSE_STYLE,
        confidence_threshold=confidence_threshold,
    )

    # Only what the run is prepared to assert is scored. The abstained set is not scored as
    # a miss and not scored as a hit — it is scored as nothing, which is what abstaining
    # means, and it is reported beside the recall so the trade is visible.
    findings = [as_eval_finding(finding) for finding in detection.findings]
    gold = load_gold(package)
    report = match_findings(findings, gold.scored, source=source_text(package))
    card = score(
        tender_name=package.name,
        gold=gold.scored,
        excluded=gold.excluded,
        finding_ids=[finding.finding_id for finding in findings],
        report=report,
        pending=report.unmatched,
    )

    written: Path | None = None
    if queue_root is not None:
        queue = queue_from_findings(findings, report.unmatched, tender_name=package.name)
        written = write_queue(queue, root=queue_root, at=at or datetime.now(UTC))

    return DetectionRun(
        tender_name=package.name,
        extraction=extraction,
        detection=detection,
        match=report,
        card=card,
        routing=route_for_review(detection),
        queue_path=written,
    )


def render_markdown(result: DetectionRun) -> str:
    """The run as a human reads it: what was found, what was withheld, and the score.

    Abstention comes before recall on the page, because a recall figure read without it is
    read as better than it is.
    """
    routing = result.routing
    lines = [
        f"# RI-05 findings — {result.tender_name}",
        "",
        f"Extracted {result.extraction.clauses_read} clause(s); "
        f"{len(result.extraction.requirements)} requirement(s) after splitting.",
        "",
        "## Detection",
        "",
        result.detection.render_summary(),
        "",
        "### Routed to human review",
        "",
        (
            f"**{len(routing.item_ids)}** finding(s) were not asserted. The interrupt is "
            f"entered at a confidence of {routing.confidence:.2f} — the lowest in the set, "
            f"not the average, so one unsure finding cannot ride in on several confident ones."
            if not routing.is_empty
            else "_Nothing was withheld. Every finding cleared the threshold._"
        ),
        "",
        "## Score",
        "",
        "_Recall here is recall of **detection**. Detectors do not yet emit the clarification "
        "questions, price impacts and alternatives that a gold item's `expects_*` flags "
        "demand, so an item whose defect was found but whose output was not produced scores "
        "OUTPUT_MISS. Reading this as end-to-end recall would credit work that has not been "
        "done._",
        "",
        render_score_card(result.card),
    ]
    if result.queue_path is not None:
        lines += [f"Adjudication queue written to `{result.queue_path}`.", ""]
    return "\n".join(lines)
