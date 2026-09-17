"""The single-clause detectors, and the registry that runs them.

Every detector here needs one clause and nothing else. That is not a convention — it is what
`DetectorContext` can express: a detector is handed one `Requirement`, the typed claims for
it, and configuration. It cannot reach a document, another clause's text, or a model. A
detector that wanted two clauses would have to change the type to get them, which makes the
boundary between this half of the engine and the next one a thing you can see.

| Detector | Needs a model call | What Python decides |
|---|---|---|
| `missing_tolerance` | via claims | whether a constraint is bounded and conditioned |
| `modality_inconsistency` | **no** | the convention, applied as a word list |
| `atomicity_split` | **no** | reads lineage extraction already produced |
| `unverifiable` | via claims | whether anything could demonstrate compliance |
| `ordinal_trap` | via claims | whether the named scale is arithmetic |
| `zero_margin` | via claims | whether a constraint admits exactly one value |

Two of the six make no model call at all. That is worth noticing rather than treating as an
accident: a convention is a word list and lineage is a record, and asking a model about
either would be paying for an opinion where a fact was available.

**Why all six are here and none is in the application.** The placement test is whether a
detector needs a commercial or regulatory frame to decide anything. None of these does. An
unbounded quantity, a compound clause, an unverifiable obligation, a mis-compared rating and
a zero-margin constraint are the same defects in a tender, a safety case and a supply
agreement. Where domain knowledge genuinely enters — what a modal verb means, which rating
scales are ordinal, what a defect is worth — it enters as configuration the caller supplies,
so the knowledge lives with the client rather than in the library. The detectors that will
need a frame belong to the second half of this engine: a clause is only a contract red flag,
a regulatory misapplication or a budget breach relative to somebody's commercial position.

Deliberately does not: compare clauses, deduplicate findings, rank them, decide severity, or
emit anything. `engine` does the emitting and stamps severity from the caller's policy.
"""

from req_core.detectors import atomicity, margin, modality, ordinal, tolerance, verifiability
from req_core.detectors.base import (
    Detector,
    DetectorContext,
    FindingDraft,
    OrdinalScales,
    read_number,
    read_range,
)

__all__ = [
    "DETECTORS",
    "Detector",
    "DetectorContext",
    "FindingDraft",
    "OrdinalScales",
    "read_number",
    "read_range",
    "run_detectors",
]

# Name to function. The order is the order findings are produced in, so a report is stable
# between runs over the same input; nothing else depends on it.
DETECTORS: dict[str, Detector] = {
    tolerance.DETECTOR: tolerance.detect,
    modality.DETECTOR: modality.detect,
    atomicity.DETECTOR: atomicity.detect,
    verifiability.DETECTOR: verifiability.detect,
    ordinal.DETECTOR: ordinal.detect,
    margin.DETECTOR: margin.detect,
}


def run_detectors(
    context: DetectorContext, *, only: frozenset[str] | None = None
) -> list[FindingDraft]:
    """Run every detector over one clause and return what they concluded, in registry order.

    `only` narrows the set, for a caller measuring one detector at a time. An unknown name
    raises rather than being ignored: a run that silently skipped the detector somebody
    asked for would report a clean result for work it never did.
    """
    if only is not None:
        unknown = sorted(only - set(DETECTORS))
        if unknown:
            raise KeyError(
                f"no such detector(s): {', '.join(unknown)}. Known: {', '.join(sorted(DETECTORS))}."
            )

    drafts: list[FindingDraft] = []
    for name, detector in DETECTORS.items():
        if only is not None and name not in only:
            continue
        drafts.extend(detector(context))
    return drafts
