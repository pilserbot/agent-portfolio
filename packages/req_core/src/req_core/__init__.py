"""Domain-agnostic requirement extraction.

Turns unstructured source text (a tender, an RFP, a specification, a contract) into typed
requirement structures that downstream code can reason about deterministically.

- `corpus` — the way in. A structural protocol for "documents with pages", a Pydantic
  corpus built from anything satisfying it, and a `withheld` set a corpus refuses to
  contradict, so "do not read that document" is a type rather than a rule to remember.
- `policy` — what a modal verb means, supplied as configuration. `shall` is easy; `will` is
  where a hardcoded mapping reads a document confidently and wrongly.
- `clauses` — where each clause starts and stops. A scan, in Python, and the source of every
  quoted span in the package.
- `extraction` — the only module that calls a model, for the only two things a model is used
  for: splitting a compound clause into atomic obligations, and identifying what it cites.
- `contracts` — `Requirement` and `ExtractionResult`, with lineage preserved through a split.
- `anchors` — the assertion the package rests on: every requirement resolves to a page that
  really contains its quoted span, whitespace normalised, every failure named.

Deliberately does not: encode any domain's vocabulary or scoring rules, judge whether a
requirement is met, or rank requirements by importance. Those judgements are the
applications' work, computed in Python over the structures produced here. It also holds no
client, no key and no retry policy: the model is reached through a callable a caller
supplies.
"""

__all__ = [
    "anchors",
    "clauses",
    "contracts",
    "corpus",
    "extraction",
    "policy",
]

__version__ = "0.1.0"
