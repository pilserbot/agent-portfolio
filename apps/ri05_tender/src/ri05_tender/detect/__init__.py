"""This tender's frame for the findings engine: what things are worth, and what runs.

`req_core.detectors` decides what a defect is. This package decides what one COSTS, which
is a commercial judgement and therefore belongs to whoever is bidding rather than to a
library.

- `config` — the severity policy, the ordinal-scale catalogue, and the confidence threshold.

The runner that wires all of this to the scoring harness is `ri05_tender.eval.findings_run`,
not here: it reads the answer key, and the gold-leakage guard forbids any module outside
`eval` from importing the package that does.

No detector lives here, and the reason is the placement test rather than an oversight: every
single-clause detector in this step decides something true of any requirements document. The
detectors that will need this frame come in the second half — a clause is only a contract red
flag, a regulatory misapplication or a budget breach relative to somebody's position, and
those are the ones that will land in this package.
"""

__all__ = ["config"]
