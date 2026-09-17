"""What a defect costs this client, and which rating scales this industry treats as ordered.

Three pieces of configuration, all of which `req_core` deliberately refuses to default:

- **`SEVERITY_POLICY`** — what each detector's findings are worth. Graded against this
  tender's own consequences rather than a general notion of badness. A compound clause is
  `major` because it costs a line in the response and a number in the price; an unverifiable
  obligation is `critical` because it cannot be closed out at handover and there is no
  remedy but argument; a zero-margin constraint is `critical` because nothing can satisfy it
  and the discovery arrives after signature.
- **`ORDINAL_SCALES`** — rating schemes whose values are ordered categories and not
  quantities. Ingress protection, enclosure type, fire resistance, network security level.
  The catalogue says only that arithmetic does not apply; it never says which end is better,
  because no detector needs to know and encoding it would be encoding an opinion.
- **`CONFIDENCE_THRESHOLD`** — below which a finding is not asserted but routed to a human.

Set deliberately low at 0.6. A presale engineer's scarcest resource is attention, and the
cost of the two mistakes is not symmetric: a defect asserted wrongly is read, investigated
and disbelieved, and the next one is trusted less; a defect routed to review is read once
and ruled on. The threshold is the price of the first mistake expressed as a number, and it
belongs in configuration where it can be argued about rather than inside a detector.

Deliberately does not: detect anything, or encode a single fact about the Kessler Point
package. Nothing here names a clause, a document or a value from this tender — see
`tests/test_no_domain_leakage.py`, which enforces that across every detector module and this
one, because a recall figure produced by detectors written against the answers is a number
about the developer rather than about the system.
"""

from req_core.detectors import OrdinalScales
from req_core.findings import SeverityPolicy

__all__ = [
    "CONFIDENCE_THRESHOLD",
    "ORDINAL_SCALES",
    "SEVERITY_POLICY",
]

CONFIDENCE_THRESHOLD = 0.6

SEVERITY_POLICY = SeverityPolicy(
    name="kessler_point",
    severities={
        # Two bidders read it differently, both comply, and the cheaper reading wins the bid.
        "missing_tolerance": "major",
        # The convention is stated in the document; a clause contradicting it is a question
        # to ask, not a bid-stopper.
        "modality_inconsistency": "minor",
        # One matrix row, one response and one price between several obligations.
        "atomicity_split": "major",
        # Cannot be demonstrated and cannot be refused. It surfaces at handover, with no
        # remedy but argument, which is the most expensive place to find anything.
        "unverifiable": "critical",
        # A comparison that is satisfiable by arithmetic and not in fact.
        "ordinal_trap": "major",
        # Nothing can hold it, so either the clause is unmeetable or a tolerance nobody
        # agreed is applied after signature.
        "zero_margin": "critical",
    },
)

ORDINAL_SCALES = OrdinalScales(
    name="kessler_point (physical and security ratings)",
    scales=frozenset(
        {
            "ip",  # ingress protection: two independent digits, not one number
            "nema",  # enclosure type: a category list, numbered
            "iec 62443 sl",  # security level
            "sl",
            "marsec",  # maritime security level
            "impact",  # impact resistance class
            "fire resistance",
            "insulation class",
            "cat",  # cabling category
            "category",
        }
    ),
)
