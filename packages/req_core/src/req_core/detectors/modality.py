"""modality_inconsistency: a clause whose modal verb contradicts the document's own convention.

Every formal document that bothers to define its terms defines them somewhere near the
front, and then writes three hundred clauses without checking against that definition. The
defect this finds is a clause using a modal verb in a sense the document itself ruled out.

Two shapes:

- **Mixed modality in one obligation.** "The Contractor shall provide a spares list and may
  substitute equivalent parts" — mandatory and optional in one sentence, so what is actually
  required depends on which half a reader stops at.
- **A modality the document's convention makes meaningless.** Under a convention where
  `will` is optional, a clause written "the Contractor will provide" reads to its author as
  a requirement and to the convention as a preference. Nobody notices until the bid is
  scored.

**The convention is configuration, and is not assumed.** `ModalityPolicy` arrives from the
caller carrying the words and where they were stated. This matters more than it sounds:
the first document this ran against defines `will` as *optional* and `should` as the same,
which is the inverse of the ordinary English reading. Under a hardcoded map every clause
would still have come out with a modality — the wrong one, silently, and the strongest
distinction in the document reversed.

No model call. The policy is a word list and the clause is text; asking a model what a modal
verb means when the document has already said would be replacing a fact with an opinion.

Confidence, and why:

- **Mixed modality, policy quotes its source — 0.8.** The contradiction is in the text and
  the reading of it is the document's own.
- **Mixed modality, policy states no source — 0.6.** The contradiction is real; the
  convention it is judged against is an assumption, so the finding is worth less.
- **Non-mandatory modality on a clause carrying obligations — 0.55.** Deliberately below
  the default threshold: a `may` clause that also imposes a duty is often just prose, and
  this shape is worth a human's glance rather than a report's assertion.

Deliberately does not: fire on a clause with no modal verb (a heading or a definition is
not a defective requirement), decide which of two modalities the author meant, or rewrite
the clause.

Belongs in `req_core`: the mechanism is "the document defined its terms and a clause
disagrees", which is a property of formal documents rather than of tendering. The
convention itself is the only domain-specific part, and it arrives as an argument.
"""

from req_core.detectors.base import DetectorContext, FindingDraft
from req_core.findings import Evidence
from req_core.policy import Modality

__all__ = ["detect"]

DETECTOR = "modality_inconsistency"


def detect(context: DetectorContext) -> list[FindingDraft]:
    """Whether this clause's modal verbs contradict the convention it is read under."""
    policy = context.modality
    reading = policy.classify(context.requirement.text)
    if not reading.triggers:
        return []

    stated = f"per {policy.source}" if policy.source else f"per the {policy.name} policy"
    words = [trigger.word for trigger in reading.triggers]
    kinds = sorted({str(trigger.modality) for trigger in reading.triggers})

    if reading.is_mixed:
        confidence = 0.8 if policy.source else 0.6
        return [
            FindingDraft(
                detector=DETECTOR,
                statement=(
                    f"The clause mixes {' and '.join(kinds)} modality in one obligation "
                    f"({', '.join(repr(word) for word in words)}), so what it actually "
                    f"requires depends on which half the reader stops at."
                ),
                evidence=Evidence(
                    summary=f"Modal verbs of more than one kind in one clause, read {stated}.",
                    observations=[
                        f"modal verbs found, in order: {', '.join(words)}",
                        f"modalities they map to: {', '.join(kinds)}",
                        f"convention: {policy.name}"
                        + (f", stated at {policy.source}" if policy.source else ", not quoted"),
                    ],
                ),
                confidence=confidence,
            )
        ]

    claims = context.claims
    obligations = [item for item in (claims.obligations if claims else []) if item.text.strip()]
    if obligations and reading.modality is not Modality.MANDATORY:
        return [
            FindingDraft(
                detector=DETECTOR,
                statement=(
                    f"The clause places {len(obligations)} obligation(s) but its modal verb "
                    f"{reading.trigger_word!r} is {reading.modality} {stated}. The author "
                    f"probably meant to require this; the convention says otherwise."
                ),
                evidence=Evidence(
                    summary=(
                        f"An obligation carried by a verb the document's own convention "
                        f"treats as {reading.modality}."
                    ),
                    observations=[
                        f"modal verb: {reading.trigger_word!r} -> {reading.modality}",
                        f"obligations reported: {len(obligations)}",
                        f"convention: {policy.name}"
                        + (f", stated at {policy.source}" if policy.source else ", not quoted"),
                    ],
                ),
                confidence=0.55,
            )
        ]
    return []
