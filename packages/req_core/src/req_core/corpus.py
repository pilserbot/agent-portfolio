"""What a document looks like on the way in, and what a source may deliberately withhold.

Extraction needs almost nothing from a document: an id, a page number, and the text on that
page. This module says exactly that and no more, so a caller with its own document type does
not have to convert to somebody else's.

Two doors in:

- **A protocol.** `PageLike` and `DocumentLike` describe the shape structurally. Anything
  carrying `document_id` / `page_number` / `text` and `document_id` / `pages` satisfies them,
  including record types from packages this one has never heard of. No base class to inherit
  and no import to add on the caller's side.
- **A Pydantic model.** `corpus_from` turns anything satisfying the protocol into a
  `SourceCorpus`, which is what every other function here takes. The protocol is the doorway;
  the model is what crosses every module boundary after it, as this repository requires.

**Withholding is part of the contract, not a convention.** A corpus names the documents it
deliberately does not carry, and refuses to be built if one of them is present. That turns
"remember not to pass the answer sheet in" from a rule somebody has to keep into a type that
cannot represent the mistake. `documents_seen` on the way out then lets a caller prove, after
the fact, that nothing cited a document the corpus never held.

Deliberately does not: read a file, know what a page came from, or care what kind of source
it is. A tender, an RFP, a standard and a contract are all the same thing here — text with
addresses. Loading is the caller's; this module never touches a filesystem.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CorpusError",
    "CorpusDocument",
    "CorpusPage",
    "DocumentLike",
    "PageLike",
    "SourceCorpus",
    "corpus_from",
]


class CorpusError(Exception):
    """A corpus could not be built, or was built from something it should not contain."""


@runtime_checkable
class PageLike(Protocol):
    """One addressable page of text. Structural: no inheritance required."""

    @property
    def document_id(self) -> str:
        """Which document this page belongs to."""
        ...

    @property
    def page_number(self) -> int:
        """1-based page number, as a citation would name it."""
        ...

    @property
    def text(self) -> str:
        """The text on this page."""
        ...


@runtime_checkable
class DocumentLike(Protocol):
    """One document, as a sequence of pages. Structural: no inheritance required."""

    @property
    def document_id(self) -> str:
        """A stable identifier — a filename stem, an id, anything a citation can carry."""
        ...

    @property
    def pages(self) -> Sequence[PageLike]:
        """Its pages, in order."""
        ...


class CorpusPage(BaseModel):
    """One page of text, addressed by document and page number."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1, description="1-based. What a citation names.")
    text: str


class CorpusDocument(BaseModel):
    """One document of the corpus."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    title: str = Field(default="", description="Best effort; empty is fine and common.")
    pages: list[CorpusPage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _pages_belong_here(self) -> "CorpusDocument":
        """Reject a page filed under a different document than the one holding it."""
        stray = sorted({page.document_id for page in self.pages} - {self.document_id})
        if stray:
            raise ValueError(
                f"document {self.document_id!r} holds page(s) belonging to "
                f"{', '.join(stray)}. An anchor built from this would cite a document that "
                f"does not contain the text."
            )
        return self


class SourceCorpus(BaseModel):
    """The documents extraction is allowed to read, and the ones it deliberately may not.

    `withheld` is the load-bearing field. Naming a document there and also passing it in is
    refused at construction: the combination is how an evaluation quietly reads its own
    answers, and a type that cannot hold the mistake is worth more than a comment asking
    nobody to make it.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="What this selection is, for a report to say.")
    documents: list[CorpusDocument] = Field(min_length=1)
    withheld: frozenset[str] = Field(
        default_factory=frozenset,
        description="Document ids deliberately kept out of extraction's reach. Recorded "
        "rather than merely omitted, so a report can state what was withheld and a test can "
        "assert it — an absence nobody wrote down is indistinguishable from an oversight.",
    )

    @model_validator(mode="after")
    def _withheld_is_actually_withheld(self) -> "SourceCorpus":
        """Refuse a corpus that carries a document it claims to have withheld."""
        present = {document.document_id for document in self.documents}
        leaked = sorted(present & self.withheld)
        if leaked:
            raise ValueError(
                f"{', '.join(leaked)} is named in `withheld` and is also in `documents`. "
                f"Withholding is enforced here rather than left to the caller: build the "
                f"corpus from the documents extraction may read, and name the rest."
            )
        duplicate = sorted({name for name in present if _count(self.documents, name) > 1})
        if duplicate:
            raise ValueError(
                f"document id(s) {', '.join(duplicate)} appear more than once. Two documents "
                f"answering to one id would make every anchor built on it ambiguous."
            )
        return self

    @property
    def document_ids(self) -> frozenset[str]:
        """Every document id this corpus carries."""
        return frozenset(document.document_id for document in self.documents)

    @property
    def page_count(self) -> int:
        """How many pages there are to read."""
        return sum(len(document.pages) for document in self.documents)

    def page(self, document_id: str, page_number: int) -> CorpusPage | None:
        """One page by address, or None when the corpus has no such page."""
        for document in self.documents:
            if document.document_id != document_id:
                continue
            for page in document.pages:
                if page.page_number == page_number:
                    return page
        return None

    def title_of(self, document_id: str) -> str:
        """A document's title, falling back to its id so a citation is never blank."""
        for document in self.documents:
            if document.document_id == document_id:
                return document.title or document.document_id
        return document_id


def _count(documents: Sequence[CorpusDocument], document_id: str) -> int:
    """How many documents carry this id."""
    return sum(1 for document in documents if document.document_id == document_id)


def corpus_from(
    documents: Iterable[DocumentLike],
    *,
    name: str,
    include: Iterable[str] | None = None,
    withhold: Iterable[str] = (),
    titles: dict[str, str] | None = None,
) -> SourceCorpus:
    """Build a corpus from anything shaped like documents with pages.

    `include` selects; `withhold` records what is deliberately left out and is then enforced
    by `SourceCorpus`. Passing a document in `withhold` and also selecting it raises, which
    is the point of naming it.

    Raises `CorpusError` rather than a `ValidationError` so a caller has one thing to catch
    for "this corpus is not usable", whatever was wrong with it.
    """
    wanted = None if include is None else set(include)
    withheld = frozenset(withhold)
    titles = titles or {}

    built: list[CorpusDocument] = []
    for document in documents:
        if wanted is not None and document.document_id not in wanted:
            continue
        built.append(
            CorpusDocument(
                document_id=document.document_id,
                title=titles.get(document.document_id, ""),
                pages=[
                    CorpusPage(
                        document_id=page.document_id,
                        page_number=page.page_number,
                        text=page.text,
                    )
                    for page in document.pages
                ],
            )
        )

    if wanted is not None:
        missing = sorted(wanted - {document.document_id for document in built})
        if missing:
            raise CorpusError(
                f"asked for document(s) {', '.join(missing)}, which the source does not "
                f"contain. Extracting from a smaller corpus than intended would read as a "
                f"worse result rather than as a missing input."
            )

    try:
        return SourceCorpus(name=name, documents=built, withheld=withheld)
    except ValueError as error:
        raise CorpusError(str(error)) from error
