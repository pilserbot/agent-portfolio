"""Wiring the domain-agnostic extractor to this tender, and the gate that scores it.

`req_core` knows what a requirement is. This package knows what *this* client calls one:
how Kessler Point numbers its clauses, and the modality convention the tender states for
itself at ITB-2.1 — which is not the ordinary English one, and is exactly why `req_core`
takes the mapping as configuration.

- `config` — the clause style and the ITB-2.1 modality policy, with the tender's own words
  quoted beside each.
- `gate` — build the extraction corpus, run the pass, and compare what came out against the
  Compliance Matrix, which extraction is not allowed to read.

Deliberately does not: hold any extraction logic. Everything here is configuration and the
comparison; the reading is `req_core`'s and stays reusable by a project that has never heard
of a port.
"""

__all__ = ["config", "gate"]
