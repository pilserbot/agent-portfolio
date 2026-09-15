"""How this tender numbers its clauses, and what it says its modal verbs mean.

Both of these are `req_core` configuration objects. Neither is a default and neither is a
guess: each is quoted from the tender, with the clause it came from recorded on the object,
so a report can say *whose* convention produced a figure rather than implying there is only
one.

The modality policy is the case that justifies the design. ITB-2.1 prints a four-row table
saying what its modal verbs mean, and it does not mean what a reader assumes:

    shall   A mandatory requirement. Non-compliance may render the bid non-responsive.
    will    An optional requirement. Compliance is scored but is not a condition of
            responsiveness.
    may     A desirable feature. Compliance is scored as a differentiator.
    should  Treated as will for the purposes of this solicitation.

Read under `req_core.policy.DEFAULT_POLICY`, `will` is mandatory and `should` is advisory.
Read under the tender's own convention, `will` is **optional** and `should` is optional too.
A hardcoded mapping would have inverted the strongest distinction in the document and
reported it with a straight face, because every clause would still have come out with a
modality — just the wrong one.

Deliberately does not: extract anything, or decide that one convention is correct. It
records what this tender says about itself.
"""

from req_core.clauses import ClauseStyle
from req_core.policy import ModalityPolicy

__all__ = [
    "ITB_2_1_POLICY",
    "KESSLER_POINT_CLAUSE_STYLE",
]

# Kessler Point numbers every requirement with a document prefix: TS-B.25, IF-4.2, CS-10.3.
# `req_core`'s default style also accepts a bare dotted number (4.7.1), which in these
# documents matches section headings such as "5.1 General" — 27 of them, several colliding
# across documents. Narrowing the style here is how that is fixed: the segmenter is not
# taught about this tender, the tender states its own shape.
KESSLER_POINT_CLAUSE_STYLE = ClauseStyle(
    name="kessler_point (prefixed)",
    identifier_pattern=r"^([A-Z]{1,4}-[A-Z]?\.?\d+(?:\.\d+)*)(?=[ \t])",
)

ITB_2_1_POLICY = ModalityPolicy(
    name="kessler_point ITB-2.1",
    source="ITB-2.1, Document 1 page 1",
    mandatory=frozenset({"shall", "must"}),
    # "A desirable feature. Compliance is scored as a differentiator." Scored, so not
    # nothing; not a condition of responsiveness, so not mandatory.
    advisory=frozenset({"may"}),
    # "will": an optional requirement. "should": treated as will. Both sit here, and both
    # sit somewhere else under the default policy — which is the whole point of the object.
    optional=frozenset({"will", "should"}),
)
