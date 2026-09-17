"""The frame: one model call per page for claims, then every decision in Python.

`detect(requirements, corpus, complete, ...)` is the whole entry point. It reads claims for
each page in one structured call, runs every detector over each clause, stamps severity from
the caller's policy, and sorts what came back into what it will assert and what it will not.

**Cost.** Batched by page exactly as extraction is: 14 pages carry clauses in the corpus this
was built against, so detection is 14 calls. Per clause it would be 301, which is why nobody
should write it that way. Note that an end-to-end run costs 28 — extraction's 14 and these
14 — because the two passes ask different questions and a pipeline that cached the first
would pay only once.

**The architectural rule, concretely.** The model is asked what the clause says
(`req_core.claims`). Python decides whether that constitutes a defect
(`req_core.detectors`), what it is worth (`SeverityPolicy`, supplied by the caller), how
sure to be (each detector's own table) and whether to assert it at all (the threshold here).
No call in this module returns a verdict, and no type it passes to a model has a field for
one. A page whose claims cannot be read produces clauses with `claims=None`, and the
detectors that need claims decline rather than guessing — a missing reading becomes silence,
never a clean bill of health.

**Abstention.** A finding below `confidence_threshold` is not emitted as fact. It goes to
`DetectionReport.abstained`, and `route_for_review` hands the set to the human-review
interrupt already in `spine.graph`. The report carries both counts, and `render_summary`
prints both, because recall with an abstention rate beside it and recall without one are
different claims and only the first is honest.

Deliberately does not: decide what a human should look at first, write a file, score
anything against an answer key, or hold a model client. The completion function arrives as
an argument, as it does in `extraction`, so this package still holds no key and no retry
policy.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from req_core.claims import CLAIMS_PURPOSE, ClauseClaims, PageClaims
from req_core.clauses import DEFAULT_CLAUSE_STYLE, Clause, ClauseStyle, clauses_on_page
from req_core.contracts import Requirement
from req_core.corpus import SourceCorpus
from req_core.detectors import DetectorContext, OrdinalScales, run_detectors
from req_core.extraction import StructuredCompletion
from req_core.findings import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DetectionReport,
    Finding,
    SeverityPolicy,
)
from req_core.policy import ModalityPolicy

__all__ = [
    "CLAIMS_INSTRUCTIONS",
    "ReviewRouting",
    "build_claims_prompt",
    "detect",
    "read_claims",
    "finding_id_for",
    "route_for_review",
]

CLAIMS_INSTRUCTIONS = """\
You are reading numbered clauses from a formal document. For each clause, report what it \
SAYS. You are not being asked whether it is any good.

1. CONSTRAINTS. For every property the clause constrains, give subject, attribute, \
operator, value and unit. Copy values exactly as written — do not convert units, round, or \
compute. Use operator `unbounded` when the clause requires a property without bounding it \
at all ("adequate illumination"): that is the correct answer, not a gap in yours. If the \
value sits on a named rating scale, give the scale's NAME in `scale` and nothing more — do \
not say which end of it is better.

2. TEST METHOD. Set `states_test_method` true if the clause states any means by which \
compliance could be demonstrated: a test, a calculation, a certificate, an inspection, or a \
named standard to test against. Quote or closely paraphrase it in `test_method`. Whether \
that means is adequate is not being asked.

3. MEASUREMENT CONDITIONS. List the conditions any measurement is to be taken under, as \
written — "at 25 °C", "at the far edge of the detection zone". Empty if the clause states \
none.

4. OBLIGATIONS. List the separable things the clause requires somebody to do, each as one \
self-contained sentence, with the actor as named.

Copy each `identifier` exactly as given. Do not invent a clause that is not listed below. \
Do not judge whether a clause is defective, risky, unclear, important or compliant: nothing \
you said about that would be used, because those decisions are made in code from what you \
report here.
"""


class ReviewRouting(BaseModel):
    """What goes to a human, in the shape `spine.graph.human_review` pauses on."""

    model_config = ConfigDict(frozen=True)

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The lowest confidence among the withheld findings, which is what the "
        "interrupt compares against its threshold. The worst case decides, not the average: "
        "an average would let a batch of confident findings carry an unsure one past review.",
    )
    item_ids: list[str] = Field(
        default_factory=list, description="The finding ids awaiting a ruling."
    )

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to review."""
        return not self.item_ids


def finding_id_for(requirement_id: str, detector: str, ordinal: int) -> str:
    """A finding's identifier, built by Python from what produced it.

    Deterministic, so two runs over the same input produce the same ids and a diff between
    reports is about the findings rather than about their names.
    """
    return f"{requirement_id}::{detector}::{ordinal}"


def build_claims_prompt(clauses: Sequence[Clause]) -> str:
    """The prompt for one page's clauses, over the spans the segmenter selected."""
    body = "\n\n".join(f"[{clause.identifier}]\n{clause.text}" for clause in clauses)
    return f"{CLAIMS_INSTRUCTIONS}\nCLAUSES\n\n{body}\n"


def read_claims(
    corpus: SourceCorpus,
    complete: StructuredCompletion,
    *,
    style: ClauseStyle = DEFAULT_CLAUSE_STYLE,
    purpose: str = CLAIMS_PURPOSE,
) -> dict[str, ClauseClaims]:
    """Claims for every clause in the corpus, one model call per page that has clauses.

    A page with no clauses costs nothing. A reading for an identifier that is not on the
    page is dropped: the model naming a clause the page does not contain is not evidence
    about a clause, it is evidence about the model.
    """
    found: dict[str, ClauseClaims] = {}
    for document in corpus.documents:
        for page in document.pages:
            clauses = clauses_on_page(page, style=style)
            if not clauses:
                continue
            reading = complete(build_claims_prompt(clauses), PageClaims, purpose=purpose)
            on_page = {clause.identifier for clause in clauses}
            for identifier, claims in reading.by_identifier().items():
                if identifier in on_page:
                    found.setdefault(identifier, claims)
    return found


def detect(
    requirements: Sequence[Requirement],
    corpus: SourceCorpus,
    complete: StructuredCompletion,
    *,
    modality: ModalityPolicy,
    severity: SeverityPolicy,
    ordinal_scales: OrdinalScales,
    style: ClauseStyle = DEFAULT_CLAUSE_STYLE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    only: frozenset[str] | None = None,
    claims: dict[str, ClauseClaims] | None = None,
) -> DetectionReport:
    """Run every single-clause detector over an extracted requirement set.

    `claims` may be supplied to skip the model entirely — which is how the offline tests
    exercise every detector, and how a caller that already read claims avoids paying twice.
    """
    readings = (
        claims
        if claims is not None
        else read_claims(corpus, complete, style=style, purpose=CLAIMS_PURPOSE)
    )

    children_of: dict[str, list[str]] = {}
    for requirement in requirements:
        if requirement.parent_id is not None:
            children_of.setdefault(requirement.parent_id, []).append(requirement.requirement_id)

    reported: list[Finding] = []
    withheld: list[Finding] = []
    for requirement in requirements:
        context = DetectorContext(
            requirement=requirement,
            claims=readings.get(requirement.requirement_id),
            siblings=tuple(children_of.get(requirement.requirement_id, ())),
            modality=modality,
            ordinal_scales=ordinal_scales,
        )
        for ordinal, draft in enumerate(run_detectors(context, only=only), start=1):
            finding = Finding(
                finding_id=finding_id_for(requirement.requirement_id, draft.detector, ordinal),
                detector=draft.detector,
                requirement_id=requirement.requirement_id,
                severity=severity.severity_for(draft.detector),
                statement=draft.statement,
                evidence=draft.evidence,
                source=requirement.source,
                confidence=draft.confidence,
                parent_id=requirement.parent_id,
                children=draft.children,
            )
            (reported if finding.confidence >= confidence_threshold else withheld).append(finding)

    return DetectionReport(
        corpus_name=corpus.name,
        confidence_threshold=confidence_threshold,
        severity_policy=severity.name,
        clauses_examined=len(requirements),
        findings=reported,
        abstained=withheld,
    )


def route_for_review(report: DetectionReport) -> ReviewRouting:
    """The withheld findings, in the shape the spine's human-review interrupt takes.

    `confidence` is the minimum across the withheld set rather than the mean. A mean would
    let one genuinely unsure finding ride past the threshold on the backs of several nearly
    confident ones, which is the opposite of what an abstention is for.
    """
    if not report.abstained:
        # Nothing withheld: a confidence of 1.0 clears any threshold, so the interrupt is a
        # no-op rather than a pause on an empty queue.
        return ReviewRouting(confidence=1.0, item_ids=[])
    return ReviewRouting(
        confidence=min(finding.confidence for finding in report.abstained),
        item_ids=[finding.finding_id for finding in report.abstained],
    )
