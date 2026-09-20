"""The extraction gate: measured against the tender's own index, with no new answer key.

Writing a gold set for requirement extraction means a person transcribing three hundred
clause numbers, and the result is a measurement of that person's afternoon. This tender
already contains the index — Document 9, the Compliance Matrix, lists every requirement the
Authority believes it wrote — so the gate compares what extraction found against what the
Authority indexed, and the comparison costs nothing to maintain.

**Document 9 is not read during extraction, and that is enforced rather than remembered.**
`ri05_tender.scope` builds every corpus and names the matrix in `withheld`; `SourceCorpus`
refuses to be constructed holding a document it claims to have withheld, so the mistake is
not representable rather than merely discouraged. `verify_scope` then re-checks after the
fact that no requirement's anchor cites a document the corpus never held. The matrix is read
only by `matrix_clause_ids`, which takes the package directly and is called by `compare` —
after extraction has returned a frozen result it cannot feed back into.

**The comparison is scoped here, not by whichever corpus produced the result.** Extraction
now runs over the wider detection scope, so a result handed to `compare` carries clauses
from documents the matrix was never about. `compare` therefore restricts itself to
`GATE_DOCUMENT_IDS` explicitly — the three specifications the matrix covers — and says so in
`MatrixComparison.scope`. Reading the whole result would turn every annex clause into a row
"absent from the matrix", which is not a defect in anything.

**The comparison is a defect detector as much as a gate**, which is why it is worth building
this way rather than against a transcription. Each side means something different:

- *absent from the matrix* — a requirement in Documents 4/5/6 the bidder is never asked to
  respond to. Either extraction invented a clause, or the Authority's own index is short,
  and only one of those is extraction's problem.
- *absent from the documents* — a row the bidder must answer that points at a clause in no
  specification. Either extraction missed one, or the matrix indexes a requirement that was
  cut.

The gate does not decide which. It reports both lists in full, named, and a human reads
them — a number here would be a judgement dressed as a measurement.

Deliberately does not: call a model (that is `req_core.extraction`, reached through a
completion function this module never constructs), write a file, decide which documents a
pass reads (that is `ri05_tender.scope`), or convert either list into a score.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from req_core.contracts import ExtractionResult
from req_core.corpus import SourceCorpus
from ri05_tender.scope import GATE_DOCUMENT_IDS, MATRIX_DOCUMENT_ID, ScopeError, gate_corpus
from ri05_tender.tender.models import TenderPackage

__all__ = [
    "GATE_DOCUMENT_IDS",
    "MATRIX_DOCUMENT_ID",
    "GateError",
    "MatrixComparison",
    "gate_corpus",
    "matrix_clause_ids",
    "compare",
    "render_markdown",
    "verify_scope",
]

# A matrix row's first cell, when it is a clause reference. The sheet also carries header
# rows, the bidder block, and a literal "TS-x.x" example in the column guide; none of them
# match, because each has to be a prefix, a hyphen and a number and nothing else.
_ROW_IDENTIFIER = re.compile(r"^([A-Z]{1,4}-[A-Z]?\.?\d+(?:\.\d+)*)$")

_CELL_SEPARATOR = "\t"


class GateError(ScopeError):
    """The gate cannot be run against this package as it stands.

    A subclass of `ScopeError` because building the corpus is the first thing the gate does
    and a caller catching one thing for "this gate cannot run" should catch both.
    """


def verify_scope(result: ExtractionResult, corpus: SourceCorpus) -> None:
    """Re-check, after the fact, that nothing was extracted from outside the corpus.

    The corpus type already makes a withheld document unrepresentable. This is the second
    half of the same guarantee, on the way out rather than the way in: if any anchor cites a
    document the corpus never held, something read something it should not have.
    """
    strayed = sorted(result.documents_seen - corpus.document_ids)
    if strayed:
        raise GateError(
            f"extraction reports having seen {', '.join(strayed)}, which the corpus does not "
            f"contain. The withheld set was {', '.join(sorted(corpus.withheld)) or '(empty)'}."
        )
    cited = sorted(
        {requirement.source.source_id for requirement in result.requirements} - corpus.document_ids
    )
    if cited:
        raise GateError(
            f"requirement(s) cite document(s) {', '.join(cited)}, which extraction was not "
            f"given. An anchor into a document the corpus never held cannot be checked."
        )


def matrix_clause_ids(
    package: TenderPackage, *, document_id: str = MATRIX_DOCUMENT_ID
) -> list[str]:
    """Every clause reference the Compliance Matrix indexes, in sheet order.

    Read straight from the workbook the tender loader already produced — one row per
    requirement, the reference in the first cell. Deterministic, and deliberately in its own
    function so it is obvious at a call site that the matrix is being opened.
    """
    document = package.document(document_id)
    if document is None:
        raise GateError(
            f"{package.name!r} has no document {document_id!r}, so there is nothing to "
            f"compare the extraction against."
        )

    found: list[str] = []
    for page in document.pages:
        for line in page.text.splitlines():
            first = line.split(_CELL_SEPARATOR, 1)[0].strip()
            if _ROW_IDENTIFIER.match(first):
                found.append(first)
    return found


class MatrixComparison(BaseModel):
    """What extraction found, what the matrix indexes, and where the two disagree."""

    model_config = ConfigDict(frozen=True)

    tender_name: str
    scope: list[str] = Field(
        description="The documents this comparison was made over. On the record because the "
        "extraction behind it may have read more: a count whose scope is implicit is a "
        "count somebody will read against the wrong denominator."
    )
    extracted_count: int = Field(ge=0, description="Distinct clauses extracted from 4/5/6.")
    matrix_row_count: int = Field(ge=0, description="Rows in the matrix carrying a reference.")
    matrix_distinct_count: int = Field(ge=0, description="Distinct references among those rows.")

    absent_from_matrix: list[str] = Field(
        default_factory=list,
        description="Extracted from Documents 4/5/6 and NOT indexed by the matrix. Either "
        "extraction invented a clause, or the bidder is never asked to respond to a real one.",
    )
    absent_from_documents: list[str] = Field(
        default_factory=list,
        description="Indexed by the matrix and found in NO specification. Either extraction "
        "missed a clause, or the matrix indexes a requirement that was cut.",
    )
    duplicated_matrix_rows: list[str] = Field(
        default_factory=list,
        description="References the matrix lists more than once. A bidder answering the same "
        "requirement twice can answer it two different ways.",
    )

    @property
    def agreed(self) -> int:
        """Clauses both sides have."""
        return self.extracted_count - len(self.absent_from_matrix)

    @property
    def is_clean(self) -> bool:
        """Whether the two sides agree exactly.

        Not a pass mark. A disagreement here can be extraction's fault or the tender's, and
        this property says only that there is something to read, never who was wrong.
        """
        return not (
            self.absent_from_matrix or self.absent_from_documents or self.duplicated_matrix_rows
        )


def compare(
    result: ExtractionResult,
    package: TenderPackage,
    *,
    documents: Sequence[str] = GATE_DOCUMENT_IDS,
) -> MatrixComparison:
    """Compare a finished extraction against the matrix it was not allowed to read.

    Takes the frozen `ExtractionResult`, so the matrix cannot reach extraction even by
    accident: by the time this function has the answers, the reading is over.

    `documents` is the gate's own scope and defaults to the three specifications the matrix
    covers. It is a parameter rather than whatever the result happened to be extracted from,
    because extraction now runs over the wider detection scope: comparing an annex clause
    against an index that was never about annexes would report a disagreement that is not
    one. Extraction is per page, so this restriction produces exactly what a pass over only
    these three documents would have.

    Compares clause identifiers and not the derived split children, because the matrix
    indexes what the document printed and children are this package's own subdivision of it.
    """
    extracted = result.clause_ids_in(documents)
    rows = matrix_clause_ids(package)
    indexed = frozenset(rows)

    seen: dict[str, int] = {}
    for row in rows:
        seen[row] = seen.get(row, 0) + 1

    return MatrixComparison(
        tender_name=package.name,
        scope=list(documents),
        extracted_count=len(extracted),
        matrix_row_count=len(rows),
        matrix_distinct_count=len(indexed),
        absent_from_matrix=sorted(extracted - indexed),
        absent_from_documents=sorted(indexed - extracted),
        duplicated_matrix_rows=sorted(name for name, count in seen.items() if count > 1),
    )


def _listing(title: str, items: Sequence[str], meaning: str) -> list[str]:
    """One section of the report: a heading, what the list means, and the list."""
    lines = [f"### {title} — {len(items)}", "", meaning, ""]
    if not items:
        return [*lines, "_none_", ""]
    return [*lines, *(f"- `{item}`" for item in items), ""]


def render_markdown(comparison: MatrixComparison, result: ExtractionResult) -> str:
    """The gate as a human reads it: both lists in full, and no verdict."""
    lines = [
        f"## RI-05 extraction gate — {comparison.tender_name}",
        "",
        f"Read **{', '.join(sorted(result.documents_seen))}**. "
        f"Withheld **{', '.join(sorted(result.withheld)) or 'nothing'}**.",
        f"Compared over **{', '.join(comparison.scope)}** — the documents the matrix covers. "
        f"The extraction behind this may have read more; the rows below that are about the "
        f"comparison say so, and the rows about the pass say the pass.",
        f"Modality convention: `{result.policy_name}`.",
        "",
        "| | |",
        "|---|---:|",
        f"| Clauses segmented (whole pass) | {result.clauses_read} |",
        f"| Clauses extracted in the compared scope | {comparison.extracted_count} |",
        f"| Clauses truncated at a page break | {result.clauses_truncated} |",
        f"| Requirements (clauses + split children) | {len(result.requirements)} |",
        f"| Atomic | {result.atomic_count} |",
        f"| Clauses split | {result.split_count} |",
        f"| Readings discarded (clause not on the page) | {result.unmatched_readings} |",
        f"| Distinct references cited | {len(result.cited_references)} |",
        f"| Matrix rows carrying a reference | {comparison.matrix_row_count} |",
        f"| Distinct references in the matrix | {comparison.matrix_distinct_count} |",
        f"| Agreed by both sides | {comparison.agreed} |",
        "",
        *_listing(
            "In Documents 4/5/6, absent from the matrix",
            comparison.absent_from_matrix,
            "A requirement the bidder is never asked to respond to — unless extraction "
            "invented it. Check one against the page before believing either reading.",
        ),
        *_listing(
            "In the matrix, absent from Documents 4/5/6",
            comparison.absent_from_documents,
            "A row the bidder must answer that points at a clause in no specification — "
            "unless extraction missed it. Same rule: check before concluding.",
        ),
        *_listing(
            "Listed more than once in the matrix",
            comparison.duplicated_matrix_rows,
            "One requirement a bidder could answer two different ways.",
        ),
        "_This comparison is the extraction gate and a defect detector at once. It does not "
        "say which side is wrong, because it cannot: that is a reading, and it is a human's._",
        "",
    ]
    return "\n".join(lines)
