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

**Scope.** This run reads `scope.DETECTION_DOCUMENT_IDS` — every document the segmenter
finds a clause in, ten of the thirteen. It used to call `extraction_corpus` with no
argument and inherit the gate's three, which was a set chosen to make a comparison against
the Compliance Matrix meaningful and never chosen for detection at all. Every recall figure
that run produced was bounded by a decision nobody had made for it, and nothing said so.

**Cost.** 88 calls: 44 to extract and 44 to read claims, one of each per clause-bearing
page. The two passes ask different questions of the same text, so neither can be skipped —
but the *extraction* half can be handed in. A caller that already holds an
`ExtractionResult` covering this corpus passes it as `extraction` and this run makes 44
calls, not 88. That hand-in is checked against the corpus's documents rather than its name:
both scopes over this package carry the package's name, so a name check passes a
three-document extraction into a ten-document run, and every clause in the seven documents
it never read would be reported as carrying no defect.

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

from req_core.clauses import segment
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
from ri05_tender.eval.metrics import ScoreCard, ceiling_chain, score
from ri05_tender.eval.models import Finding as EvalFinding
from ri05_tender.eval.models import FindingOutputs
from ri05_tender.eval.report import render_markdown as render_score_card
from ri05_tender.extract.config import ITB_2_1_POLICY, KESSLER_POINT_CLAUSE_STYLE
from ri05_tender.scope import detection_corpus, document_numbers
from ri05_tender.tender.loader import load_tender
from ri05_tender.tender.models import TenderPackage

__all__ = [
    "DETECTOR_TO_FINDING_TYPE",
    "EMITTED_OUTPUTS",
    "UNMATCHED_EXAMPLES_PER_DETECTOR",
    "DetectionRun",
    "as_eval_finding",
    "run",
]

# How many unmatched findings per detector the report quotes in full. Enough to judge
# whether a detector that fired two hundred times is reading real defects or firing on
# everything; few enough that the report stays a page somebody reads.
UNMATCHED_EXAMPLES_PER_DETECTOR = 5

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


# Which of a gold item's `expects_*` artefacts the detectors actually carry. `split_children`
# only: `as_eval_finding` passes an `atomicity_split` finding's children through, and nothing
# produces a clarification question, a price impact, an alternative, a checklist entry or a
# no-bid recommendation yet.
#
# This was empty until the children were plumbed through, and the emptiness was measured
# before it was fixed: five addressable gold items demand `split_children` and nothing else,
# and they scored OUTPUT_MISS not because no detector could see the compound clause but
# because the adapter dropped the lineage the splitter had already produced. The ceiling in
# `metrics.Addressability` moves with this constant, which is why they change together.
EMITTED_OUTPUTS: frozenset[str] = frozenset({"split_children"})


def as_eval_finding(finding: CoreFinding) -> EvalFinding:
    """Convert a detector's finding into the record this project's matcher scores.

    `split_children` carries the lineage straight through. The splitter already produced
    the requirement ids a compound clause was separated into and the detector already holds
    them; dropping them here was the whole reason five gold items demanding only a split
    scored OUTPUT_MISS.

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
        # The split lineage the detector already holds. `children` is populated only on an
        # `atomicity_split` finding — `req_core.findings.Finding` refuses children on any
        # other detector and refuses a split carrying fewer than two — so this needs no
        # branch on the detector name: everywhere else it is already empty.
        outputs=FindingOutputs(split_children=list(finding.children)),
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
    corpus = detection_corpus(package)

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
    elif not corpus.document_ids <= extraction.documents_seen:
        # The name check above is not enough and was not, once: both scopes over this
        # package are called by the package's name, so a three-document extraction passed
        # in here sailed through and detection ran over a tenth of the requirements it was
        # about to be scored on. What has to agree is the documents, so that is what is
        # compared.
        missing = ", ".join(sorted(corpus.document_ids - extraction.documents_seen))
        raise ValueError(
            f"the supplied extraction never read {missing}, which this run's corpus holds. "
            f"Detecting over requirements from a narrower pass would report every clause in "
            f"the documents it skipped as carrying no defect."
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
    withheld = [as_eval_finding(finding) for finding in detection.abstained]
    gold = load_gold(package)
    source = source_text(package)
    report = match_findings(findings, gold.scored, source=source)

    # The same scoring over the withheld findings as well. This is not a second opinion
    # about quality and nothing it produces is credited: it exists so the delta between the
    # two says what the threshold cost. A detector that located a planted defect and was
    # withheld scores identically to one that never saw it, and only this separates them.
    if_asserted = match_findings(findings + withheld, gold.scored, source=source)

    # The chain from every planted defect to what this configuration could match, built
    # here because only this function knows both halves: which documents the corpus holds
    # and which clause identifiers the segmenter produced from them. `metrics` is given the
    # finished chain rather than the raw collections — it must not have to guess at a
    # corpus, since a ceiling guessed wrong is wrong in the flattering direction.
    chain = ceiling_chain(
        gold.scored,
        corpus_documents=document_numbers(corpus.document_ids),
        clause_identifiers=[
            clause.identifier for clause in segment(corpus, style=KESSLER_POINT_CLAUSE_STYLE)
        ],
        emitted_finding_types=frozenset(DETECTOR_TO_FINDING_TYPE.values()),
        emitted_outputs=EMITTED_OUTPUTS,
        # Items a withheld finding would have matched and an asserted one did not: found,
        # and not said. Taken from the two reports rather than from the confidences, so it
        # is what the matcher actually did and not what a threshold implies it did.
        withheld_gold_ids=if_asserted.matched_gold_ids - report.matched_gold_ids,
    )

    card = score(
        tender_name=package.name,
        gold=gold.scored,
        excluded=gold.excluded,
        finding_ids=[finding.finding_id for finding in findings],
        report=report,
        pending=report.unmatched,
        # The ceiling, stated from what is registered rather than assumed. Both come from
        # this module because this module is what wires the detectors to the scorer.
        emitted_finding_types=frozenset(DETECTOR_TO_FINDING_TYPE.values()),
        emitted_outputs=EMITTED_OUTPUTS,
        if_asserted=if_asserted,
        abstained_findings=len(withheld),
        ceiling=chain,
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
    lines += _unmatched_section(result)
    if result.queue_path is not None:
        lines += [f"Adjudication queue written to `{result.queue_path}`.", ""]
    return "\n".join(lines)


def _unmatched_section(result: DetectionRun) -> list[str]:
    """The findings that cite nothing in the answer key, grouped and sampled.

    The first scored run reported 410 findings and 357 of them matched nothing. That number
    means one of two very different things — the detectors are surfacing real defects the
    gold set never planted, or they are firing on everything — and the aggregate cannot tell
    them apart. Only the clause text and the claim made about it can, and that is a reading a
    human does.

    So this prints the material for that reading and draws no conclusion from it. Nothing
    here adjudicates: an unmatched finding is neither credited nor counted wrong by being
    quoted. It is in the report rather than only in the adjudication queue because the queue
    is a file in a CI artifact, and a measurement nobody can open is one nobody reads.
    """
    unmatched = set(result.match.unmatched)
    if not unmatched:
        return ["## Unmatched findings", "", "_Every finding cited something in the key._", ""]

    by_detector: dict[str, list[CoreFinding]] = {}
    by_document: dict[str, int] = {}
    for finding in result.detection.findings:
        if finding.finding_id not in unmatched:
            continue
        by_detector.setdefault(finding.detector, []).append(finding)
        by_document[finding.source.document] = by_document.get(finding.source.document, 0) + 1

    lines = [
        "## Unmatched findings",
        "",
        f"**{len(unmatched)}** of {len(result.detection.findings)} reported finding(s) cite "
        f"nothing in the answer key. They are counted against strict precision and left out "
        f"of adjudicated precision, and nothing below rules on any of them.",
        "",
        "| Detector | Unmatched |",
        "|---|---:|",
        *(
            f"| `{name}` | {len(found)} |"
            for name, found in sorted(by_detector.items(), key=lambda pair: -len(pair[1]))
        ),
        "",
        "| Document | Unmatched |",
        "|---|---:|",
        *(
            f"| `{name}` | {count} |"
            for name, count in sorted(by_document.items(), key=lambda pair: -pair[1])
        ),
        "",
        f"### {UNMATCHED_EXAMPLES_PER_DETECTOR} example(s) per detector",
        "",
        "_The clause as the tender writes it, and what the detector said about it. Read them "
        "together: a detector firing on a clause that really does state an unbounded quantity "
        "is finding something the gold set did not plant, and one firing on a clause that "
        "bounds everything it names is firing on everything._",
        "",
    ]
    for name, found in sorted(by_detector.items()):
        lines += [f"**`{name}`** — {len(found)} unmatched", ""]
        for finding in found[:UNMATCHED_EXAMPLES_PER_DETECTOR]:
            lines += [
                f"- `{finding.requirement_id}` (confidence {finding.confidence:.2f})",
                f"  - clause: _{finding.source.quote.strip()}_",
                f"  - claimed: {finding.statement}",
            ]
        lines.append("")
    return lines
