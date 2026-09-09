"""Domain-agnostic requirement extraction.

Turns unstructured source text (a tender, an RFP, a specification, a contract) into typed
requirement structures that downstream code can reason about deterministically.

Deliberately does not: encode any domain's vocabulary or scoring rules, judge whether a
requirement is met, or rank requirements by importance. Those judgements are the
applications' work, computed in Python over the structures produced here.
"""

__version__ = "0.1.0"
