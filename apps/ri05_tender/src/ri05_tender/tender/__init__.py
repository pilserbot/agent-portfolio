"""The tender-loading layer: a folder of client-issued files becomes typed documents.

`ri05_tender.tender.models` is the shape a loaded tender takes — documents split into pages
that keep their real page numbers. `ri05_tender.tender.loader` is how any tender enters the
system: `load_tender(folder)`, given the folder itself.

No tender's name appears anywhere in this package. The layout is the contract, and a folder
with no answer key loads exactly as one with it — that is the production path.

Deliberately does not: call a model, reach a network, or interpret what it reads. It turns
files into text with the page numbers kept, and stops there. Nothing here opens a gold set.
"""

__all__ = ["loader", "models"]
