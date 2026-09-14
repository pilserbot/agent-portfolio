"""The typed shape of a tender package once it has been read off disk.

A tender arrives as a folder of files a client issued. These models are what it becomes:
documents, each split into pages that keep their real page numbers, so anything asserted
later can name the page it came from. That is the whole reason pages are modelled
separately rather than a document being one blob of text — an `EvidenceRef` citing
"page 4" has to mean page 4 of the PDF the client sent.

Everything here is frozen. A loaded tender is a record of files as they were at a moment,
and the sha256 on each document is what makes that checkable: a document whose bytes have
changed is a different document, whatever its filename still says.

Deliberately does not: read a file, know where tenders live on disk, or hold any knowledge
of a particular tender. Loading is `ri05_tender.tender.loader`'s business; these are the
structures it returns. It also does not interpret anything it carries — no requirement is
identified here, no clause parsed, no answer judged.
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The formats a tender is issued in. Extending this means teaching the loader to read the
# format too, which is why it is a closed set rather than "whatever has a suffix".
MediaType = Literal["pdf", "xlsx", "docx"]

_WHITESPACE = re.compile(r"\s+")

__all__ = [
    "DocumentPage",
    "MediaType",
    "TenderDocument",
    "TenderPackage",
]


class DocumentPage(BaseModel):
    """One page of one document, and the text on it.

    For a PDF this is a real printed page. For a workbook it is one worksheet, numbered by
    its position in the book. For a Word file there is only ever page 1 — see
    `TenderDocument.media_type` and the loader's docstring for why.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1, description="1-based. What a citation names.")
    text: str


class TenderDocument(BaseModel):
    """One file from the tender, read into pages."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(
        min_length=1,
        description='The filename stem, e.g. "04_Technical_Specification". Stable across a '
        "re-read, and what every citation is keyed on.",
    )
    filename: str = Field(min_length=1)
    media_type: MediaType
    title: str = Field(
        description="Best effort from the document itself — the first heading, or the first "
        "sheet name. Empty when the file offers nothing to go on. Never derived from the "
        "filename, so a page that disagrees with its filename can be seen to."
    )
    pages: list[DocumentPage] = Field(default_factory=list)
    sheet_names: list[str] = Field(
        default_factory=list, description="Worksheet names in book order. Empty for anything else."
    )
    sha256: str = Field(
        min_length=64, max_length=64, description="Of the file bytes, so a changed file is visible."
    )

    @property
    def page_count(self) -> int:
        """How many pages this document holds."""
        return len(self.pages)

    @property
    def text(self) -> str:
        """The whole document as one string, pages separated by a blank line.

        For searching and for a word count. Anything that must cite a location reads
        `pages` instead, because this has thrown the page numbers away.
        """
        return "\n\n".join(page.text for page in self.pages)

    def page(self, page_number: int) -> DocumentPage | None:
        """The page with this number, or None if the document has no such page."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def pages_containing(self, needle: str, *, case_sensitive: bool = True) -> list[int]:
        """The page numbers whose text contains `needle`, in order.

        The bridge from "the tender says X" to a citation that can be checked: this is how
        a clause reference becomes a page number an `EvidenceRef` can carry.
        """
        if case_sensitive:
            return [page.page_number for page in self.pages if needle in page.text]
        lowered = needle.lower()
        return [page.page_number for page in self.pages if lowered in page.text.lower()]

    @model_validator(mode="after")
    def _pages_belong_to_this_document(self) -> "TenderDocument":
        """Refuse pages filed under another document's id."""
        wrong = sorted(
            {page.document_id for page in self.pages if page.document_id != self.document_id}
        )
        if wrong:
            raise ValueError(
                f"document {self.document_id!r} holds pages belonging to {', '.join(wrong)}"
            )
        return self

    @model_validator(mode="after")
    def _pages_are_numbered_from_one_without_gaps(self) -> "TenderDocument":
        """Refuse page numbering that a citation could not be resolved against.

        A gap or a repeat would make "page 7" ambiguous, and the point of carrying page
        numbers at all is that they are not.
        """
        numbers = [page.page_number for page in self.pages]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(
                f"document {self.document_id!r} has page numbers {numbers}; they must run "
                f"1..{len(numbers)} without gaps or repeats."
            )
        return self

    @model_validator(mode="after")
    def _only_workbooks_have_sheets(self) -> "TenderDocument":
        """Refuse sheet names on something that is not a workbook, or a count that disagrees."""
        if self.sheet_names and self.media_type != "xlsx":
            raise ValueError(
                f"document {self.document_id!r} is a {self.media_type} but carries sheet names"
            )
        if self.media_type == "xlsx" and len(self.sheet_names) != len(self.pages):
            raise ValueError(
                f"workbook {self.document_id!r} has {len(self.sheet_names)} sheet name(s) and "
                f"{len(self.pages)} page(s); there is one page per sheet."
            )
        return self


class TenderPackage(BaseModel):
    """One tender as issued: its documents, and whether it came with an answer key."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="The tender folder's own name.")
    root_path: Path
    documents: list[TenderDocument] = Field(default_factory=list)
    has_gold: bool = Field(
        description="Whether a labelled answer key sits beside the documents. False is the "
        "production case: a real tender arrives without one."
    )
    gold_path: Path | None = Field(
        default=None, description="Where that answer key is. None when there is none."
    )

    @property
    def page_count(self) -> int:
        """Pages across every document."""
        return sum(document.page_count for document in self.documents)

    @property
    def document_ids(self) -> list[str]:
        """Every document id, in the order the package holds them."""
        return [document.document_id for document in self.documents]

    def document(self, document_id: str) -> TenderDocument | None:
        """One document by id, or None when the package does not hold it."""
        for document in self.documents:
            if document.document_id == document_id:
                return document
        return None

    def documents_of(self, media_type: MediaType) -> list[TenderDocument]:
        """Every document of one format, in package order."""
        return [document for document in self.documents if document.media_type == media_type]

    @model_validator(mode="after")
    def _document_ids_are_unique(self) -> "TenderPackage":
        """Refuse a package where two documents answer to one id.

        Two files differing only in extension would collide, and a citation naming that id
        could then mean either — so the package will not be built at all.
        """
        ids = self.document_ids
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate document id(s): {', '.join(duplicates)}")
        return self

    @model_validator(mode="after")
    def _gold_path_agrees_with_has_gold(self) -> "TenderPackage":
        """Refuse a package that claims a gold set it cannot point at, or the reverse."""
        if self.has_gold and self.gold_path is None:
            raise ValueError(f"tender {self.name!r} has_gold is True but gold_path is None")
        if not self.has_gold and self.gold_path is not None:
            raise ValueError(
                f"tender {self.name!r} has_gold is False but gold_path is {self.gold_path}"
            )
        return self


def first_line_of(text: str, *, limit: int = 200) -> str:
    """The first non-empty line of some text, with its whitespace collapsed.

    Used for a best-effort title. It takes what the document says rather than tidying it:
    if the first line is a running header, that is what a reader of page one sees too.
    """
    for line in text.splitlines():
        collapsed = _WHITESPACE.sub(" ", line).strip()
        if collapsed:
            return collapsed[:limit]
    return ""
