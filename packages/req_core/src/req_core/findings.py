"""What a detector produces, what it is worth, and what happens when it is unsure.

A `Finding` is a defect a detector concluded from typed claims. Every field on it was
decided by Python: the type, the severity, the confidence, and whether it exists at all.
The model contributed `ClauseClaims` and nothing else.

**Severity is configuration.** What a missed tolerance costs is a commercial question, and
the answer differs between a port authority and a hospital. `SeverityPolicy` maps detector
to severity and is supplied by the caller, so `req_core` never decides what anything is
worth. There is no default that quietly becomes a standard.

**Confidence is computed, not asked for.** Each detector derives it from how complete the
claim behind the finding was — see `Evidence`. A model cannot report a confidence through
`claims`, because `ClauseClaims` has no field for one. A number the model invented would be
an assessment of its own work, which is the one thing it should not be trusted with here.

**Abstention is a first-class outcome.** Below `confidence_threshold` a finding is not
emitted as fact. It goes to `abstained`, and from there to the human-review interrupt in
`spine.graph`. `DetectionReport` carries both counts side by side on purpose: recall bought
by never abstaining is not the same result as recall with an abstention rate beside it, and
a report that showed only the first would make the two look identical.

Deliberately does not: rank findings against each other, deduplicate across clauses, decide
what a human should look at first, or map onto any particular scoring harness's record
shape. An application that scores these adapts them at its own edge — see
`ri05_tender.eval.findings_run`, which converts to that project's `Finding` for its matcher.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from spine.contracts import EvidenceRef

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DetectionReport",
    "DetectorName",
    "Evidence",
    "Finding",
    "Severity",
    "SeverityPolicy",
]

# The detectors this package ships. A closed set: a finding whose type nothing recognises
# cannot be scored, routed or counted, so an unknown name fails the model rather than
# arriving in a report nobody can act on.
DetectorName = Literal[
    "missing_tolerance",
    "modality_inconsistency",
    "atomicity_split",
    "unverifiable",
    "ordinal_trap",
    "zero_margin",
]

Severity = Literal["no_bid", "critical", "major", "minor"]

# Below this a finding is routed to a human rather than reported. A default, not a standard:
# every caller should choose its own, and the one place it is chosen should be config.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class Evidence(BaseModel):
    """Why a detector concluded what it did, in terms a reader can check against the page.

    The point is that a reader can disagree. "Confidence 0.5" says nothing; "the clause
    constrains an attribute and states no bound, and no measurement condition was reported"
    says what was seen, and a reader who opens the page can say whether it is true.
    """

    model_config = ConfigDict(frozen=True)

    summary: str = Field(min_length=1, description="One sentence: what was observed.")
    observations: list[str] = Field(
        default_factory=list, description="The specific claims the conclusion rests on."
    )


class SeverityPolicy(BaseModel):
    """What each detector's findings are worth to this client.

    Supplied, never defaulted. `req_core` detects; what a defect costs is a commercial
    judgement belonging to whoever is bidding.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description='e.g. "kessler_point".')
    severities: dict[str, Severity] = Field(
        description="Detector name to severity. Every detector that can fire must appear."
    )

    def severity_for(self, detector: str) -> Severity:
        """What this client says a finding from that detector is worth."""
        if detector not in self.severities:
            raise KeyError(
                f"severity policy {self.name!r} says nothing about {detector!r}. Every "
                f"detector that can fire needs a severity, or its findings cannot be "
                f"weighted and would silently count as the mildest thing in the report."
            )
        return self.severities[detector]


class Finding(BaseModel):
    """One defect, concluded by Python from typed claims."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: str = Field(min_length=1)
    detector: DetectorName
    requirement_id: str = Field(min_length=1, description="The clause it was found in.")
    severity: Severity
    statement: str = Field(min_length=1, description="What is wrong, in the finding's own words.")
    evidence: Evidence
    source: EvidenceRef = Field(description="Where in the document, with the span quoted.")
    confidence: float = Field(ge=0.0, le=1.0)
    parent_id: str | None = Field(
        default=None,
        description="Set on a finding about a split child, naming the clause it came from.",
    )
    children: list[str] = Field(
        default_factory=list,
        description="For atomicity_split: the requirement ids of the obligations the clause "
        "was separated into. Lineage travels with the finding rather than being looked up.",
    )

    @property
    def is_confident(self) -> bool:
        """Whether this clears the default threshold. A report uses its own, not this.

        A plain property rather than a `computed_field` on purpose: a computed field is
        serialised, and this model is `extra="forbid"`, so a serialised derived value makes
        the record unable to validate its own `model_dump()`. A frozen record that cannot
        round-trip is a trap for anything that stores or ships one.
        """
        return self.confidence >= DEFAULT_CONFIDENCE_THRESHOLD

    @model_validator(mode="after")
    def _children_belong_to_a_split(self) -> "Finding":
        """Only a split finding names children, and a split with none found nothing."""
        if self.children and self.detector != "atomicity_split":
            raise ValueError(
                f"{self.finding_id!r} is a {self.detector} finding carrying children. Only "
                f"atomicity_split separates a clause, so children here would be lineage "
                f"nothing produced."
            )
        if self.detector == "atomicity_split" and len(self.children) < 2:
            raise ValueError(
                f"{self.finding_id!r} reports a split into {len(self.children)} obligation(s). "
                f"A clause separated into fewer than two was not compound, and reporting it "
                f"would be a finding about nothing."
            )
        return self


class DetectionReport(BaseModel):
    """Everything one detection pass concluded, and everything it declined to conclude.

    `findings` and `abstained` are both here, and `render_summary` prints both. A caller
    cannot report the first without the second being one attribute away, which is the
    intent: a system that never abstains scores better on recall and is worse.
    """

    model_config = ConfigDict(frozen=True)

    corpus_name: str = Field(min_length=1)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    severity_policy: str = Field(min_length=1)
    clauses_examined: int = Field(ge=0)
    findings: list[Finding] = Field(
        default_factory=list, description="At or above the threshold. Reported as fact."
    )
    abstained: list[Finding] = Field(
        default_factory=list,
        description="Below the threshold. NOT reported as fact — routed to a human through "
        "the review interrupt. Kept in full rather than counted, so a reviewer sees what was "
        "withheld and can say whether the threshold is set right.",
    )

    @computed_field
    @property
    def abstention_rate(self) -> float:
        """Share of everything detected that was withheld. 0.0 over nothing detected."""
        total = len(self.findings) + len(self.abstained)
        return len(self.abstained) / total if total else 0.0

    def by_detector(self, detector: str) -> list[Finding]:
        """The reported findings from one detector, in the order they were made."""
        return [finding for finding in self.findings if finding.detector == detector]

    def render_summary(self) -> str:
        """The counts a report leads with — always both, never one."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.detector] = counts.get(finding.detector, 0) + 1
        lines = [
            f"{len(self.findings)} finding(s) over {self.clauses_examined} clause(s); "
            f"{len(self.abstained)} abstained ({self.abstention_rate:.1%}) at a threshold of "
            f"{self.confidence_threshold:.2f}, severity per `{self.severity_policy}`.",
            "",
            "| Detector | Reported | Abstained |",
            "|---|---:|---:|",
        ]
        withheld: dict[str, int] = {}
        for finding in self.abstained:
            withheld[finding.detector] = withheld.get(finding.detector, 0) + 1
        for detector in sorted(set(counts) | set(withheld)):
            lines.append(
                f"| {detector} | {counts.get(detector, 0)} | {withheld.get(detector, 0)} |"
            )
        if not (counts or withheld):
            lines.append("| _none_ | 0 | 0 |")
        lines += [
            "",
            "_Abstentions are not failures and not successes. They are the findings this run "
            "declined to assert, and they go to a human. Recall quoted without this number "
            "beside it is recall bought by guessing._",
        ]
        return "\n".join(lines)
