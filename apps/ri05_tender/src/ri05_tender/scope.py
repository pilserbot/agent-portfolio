"""Which documents each pass may read, named once so no caller has to infer it.

There are two scopes over the Kessler Point package and they answer different questions.
They were one default for a while, and the detection pass inherited a set of documents that
had been chosen for the gate — so detection read three documents out of thirteen and every
recall figure it produced was bounded by a decision nobody had made for it.

`GATE_DOCUMENT_IDS` — **what did extraction find, against what the Authority indexed?**
Documents 4, 5 and 6, which are the specifications the Compliance Matrix covers. The
comparison is only meaningful over exactly those: a clause from the annexes would show up as
"absent from the matrix" because the matrix was never about it.

`DETECTION_DOCUMENT_IDS` — **where in the readable package are the defects?** Everything the
segmenter can read: 1, 2, 3, 4, 5, 6, 10, 11, 12 and 13. The gold set plants defects in the
instructions, the scope of work and the annexes, and a detector that is never shown those
pages cannot be said to have missed anything in them.

Documents 7 and 8 are left out of both, and not because anything is being kept from
anybody: they are spreadsheets whose rows carry codes like `A.07`, which is not the clause
shape this package segments, so they yield zero clauses. Sending their worksheets would pay
for calls that return nothing.

Document 9 is **withheld** rather than left out, and the difference matters. It is the
Compliance Matrix — the Authority's own index of every requirement — and reading it during
extraction would let a pass produce the right clause list without reading a specification.
`SourceCorpus` refuses to be constructed holding a document it claims to have withheld, so
naming it here makes the mistake unrepresentable rather than merely discouraged.

Deliberately does not: read a document, call a model, or decide what any pass does with the
corpus it is handed. It answers "which documents", and nothing else.
"""

from collections.abc import Iterable, Sequence

from req_core.corpus import CorpusError, SourceCorpus, corpus_from
from ri05_tender.tender.models import TenderPackage

__all__ = [
    "DETECTION_DOCUMENT_IDS",
    "GATE_DOCUMENT_IDS",
    "MATRIX_DOCUMENT_ID",
    "NO_CLAUSE_DOCUMENT_IDS",
    "ScopeError",
    "detection_corpus",
    "document_numbers",
    "gate_corpus",
]


class ScopeError(Exception):
    """A corpus cannot be built for this package as it stands."""


# The one document withheld from every pass, and the only one named as such.
MATRIX_DOCUMENT_ID = "09_Compliance_Matrix"

# The gate's question: extraction against the Authority's own index. Only the three
# specifications the matrix covers, because the comparison means nothing outside them.
GATE_DOCUMENT_IDS: tuple[str, ...] = (
    "04_Technical_Specification",
    "05_Integration_and_Interface_Specification",
    "06_Cybersecurity_and_Information_Security",
)

# Detection's question: where in the package are the defects. Every document the segmenter
# reads — 691 clauses over 44 pages, against the gate's 301 over 14.
DETECTION_DOCUMENT_IDS: tuple[str, ...] = (
    "01_Instructions_to_Bidders",
    "02_Scope_of_Work",
    "03_General_Conditions_of_Contract",
    "04_Technical_Specification",
    "05_Integration_and_Interface_Specification",
    "06_Cybersecurity_and_Information_Security",
    "10_Annex_A_Site_and_Environmental_Conditions",
    "11_Annex_B_Schedule_of_Standards",
    "12_Annex_C_SSI_Handling",
    "13_Drawing_Register",
)

# Excluded from both scopes for a reason that is neither withholding nor oversight: the
# segmenter finds no clause in either, so a pass over them would make calls about nothing.
# Their gold items reference spreadsheet row codes, which no corpus of this package can
# anchor — see `tests/ri05/test_gold_refs_anchor.py`, which pins that.
NO_CLAUSE_DOCUMENT_IDS: tuple[str, ...] = (
    "07_Bill_of_Quantities",
    "08_Pricing_Schedule",
)


def document_numbers(document_ids: Iterable[str]) -> frozenset[int]:
    """The leading number of each document id, which is how the gold set names a document.

    `GoldItem.document` is an integer and a corpus holds filename stems, so somewhere the
    two have to be reconciled. Here, once, rather than at each call site: the stem's leading
    digits are the document's number in the package index, and a stem that does not start
    with digits is not one of this tender's documents.
    """
    found: set[int] = set()
    for document_id in document_ids:
        head = document_id.split("_", 1)[0]
        if not head.isdigit():
            raise ScopeError(
                f"{document_id!r} does not begin with a document number, so it cannot be "
                f"matched to the gold set, which refers to documents by number."
            )
        found.add(int(head))
    return frozenset(found)


def _corpus(
    package: TenderPackage, include: Sequence[str], withhold: Sequence[str], question: str
) -> SourceCorpus:
    """One scope's corpus.

    `TenderDocument` and `DocumentPage` already satisfy `req_core`'s `DocumentLike` and
    `PageLike` protocols structurally, so nothing is converted and `req_core` imports nothing
    from this package. That is the whole of the coupling between them.
    """
    try:
        return corpus_from(
            package.documents,
            name=package.name,
            include=include,
            withhold=withhold,
            titles={document.document_id: document.title for document in package.documents},
        )
    except CorpusError as error:
        raise ScopeError(
            f"cannot build the {question} corpus for {package.name!r}: {error}"
        ) from error


def gate_corpus(
    package: TenderPackage,
    *,
    include: Sequence[str] = GATE_DOCUMENT_IDS,
    withhold: Sequence[str] = (MATRIX_DOCUMENT_ID,),
) -> SourceCorpus:
    """The three specifications the Compliance Matrix covers, matrix withheld."""
    return _corpus(package, include, withhold, "gate")


def detection_corpus(
    package: TenderPackage,
    *,
    include: Sequence[str] = DETECTION_DOCUMENT_IDS,
    withhold: Sequence[str] = (MATRIX_DOCUMENT_ID,),
) -> SourceCorpus:
    """Every document the segmenter reads, matrix withheld.

    Wider than the gate's on purpose. The gold set plants defects across the package, and a
    recall figure over three documents is a figure about three documents whatever it is
    called.
    """
    return _corpus(package, include, withhold, "detection")
