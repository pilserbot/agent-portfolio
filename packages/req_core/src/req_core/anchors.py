"""The assertion this package rests on: every anchor quotes text that is really there.

An extracted requirement is worth what its citation is worth. A requirement whose anchor
names a page that does not contain the quoted span is worse than no requirement at all — it
survives review, because checking it means opening the document, and nobody opens the
document for the one that looks fine.

So this is checked rather than hoped for, on every record, every run:

    100% of extracted requirements resolve to a source anchor whose page text actually
    contains the quoted span.

Two things make that a check and not a wish. `verify` compares with whitespace normalised —
a PDF wraps a clause across lines and the quote must not fail merely for being re-wrapped —
and it reports **every** failure, each naming the requirement id, the document and the page,
rather than raising on the first.

It is also made structurally hard to fail: a quote is always text `clauses` selected from the
page, never text a model produced. A model that paraphrases cannot put a paraphrase into an
anchor, because it is never asked for one. `verify` then catches what remains — a
normalisation bug, a page misnumbered, a corpus that does not hold the document a
requirement claims.

Deliberately does not: repair a bad anchor, drop the requirement carrying it, or downgrade a
failure to a warning. It reports, and the caller decides — which in every test in this
repository means failing the run.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from req_core.contracts import Requirement
from req_core.corpus import SourceCorpus

__all__ = [
    "AnchorFailure",
    "AnchorReport",
    "AnchorVerificationError",
    "flatten",
    "verify",
]

_WHITESPACE = re.compile(r"\s+")


class AnchorVerificationError(Exception):
    """One or more requirements carry an anchor that does not resolve."""


def flatten(text: str) -> str:
    """Put text into the one form containment is tested in.

    Whitespace collapsed to single spaces and the ends stripped — nothing else. A PDF breaks
    a clause across lines wherever the column ends, so a quote that is right would otherwise
    fail for being re-wrapped. Case and punctuation are left alone: a quote is supposed to be
    the document's own words, and loosening the comparison further would start excusing
    anchors that are not quotes at all.
    """
    return _WHITESPACE.sub(" ", text).strip()


class AnchorFailure(BaseModel):
    """One requirement whose anchor does not resolve, described well enough to go and look."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    document_id: str
    page: int | None
    reason: str
    quote_preview: str = Field(default="", description="The head of the quote that was not found.")

    def __str__(self) -> str:
        """The failure as a reviewer needs to read it."""
        where = f"{self.document_id} p{self.page}" if self.page is not None else self.document_id
        preview = f": {self.quote_preview!r}" if self.quote_preview else ""
        return f"{self.requirement_id} -> {where}: {self.reason}{preview}"


class AnchorReport(BaseModel):
    """What verification found over a whole set of requirements."""

    model_config = ConfigDict(frozen=True)

    checked: int = Field(ge=0)
    failures: list[AnchorFailure] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every anchor resolved."""
        return not self.failures

    @property
    def resolved(self) -> int:
        """How many anchors resolved."""
        return self.checked - len(self.failures)

    def describe(self) -> str:
        """The report as one message, naming every failure."""
        if self.ok:
            return f"{self.checked} anchor(s) checked, all resolve to their quoted page text."
        return (
            f"{len(self.failures)} of {self.checked} anchor(s) do not resolve to the page "
            f"they cite:\n" + "\n".join(f"  {failure}" for failure in self.failures)
        )

    def raise_for_failures(self) -> None:
        """Raise naming every failure, or return quietly when there are none."""
        if not self.ok:
            raise AnchorVerificationError(self.describe())


def verify(requirements: Sequence[Requirement], corpus: SourceCorpus) -> AnchorReport:
    """Check every requirement's anchor against the page it cites.

    Returns a report rather than raising, so a caller can print all of it. `raise_for_failures`
    turns it into an exception when that is what is wanted, which is what the tests do.
    """
    failures: list[AnchorFailure] = []

    for requirement in requirements:
        anchor = requirement.source
        document_id = anchor.source_id
        page_number = anchor.page

        if document_id not in corpus.document_ids:
            failures.append(
                AnchorFailure(
                    requirement_id=requirement.requirement_id,
                    document_id=document_id,
                    page=page_number,
                    reason="cites a document this corpus does not contain",
                )
            )
            continue

        if page_number is None:
            failures.append(
                AnchorFailure(
                    requirement_id=requirement.requirement_id,
                    document_id=document_id,
                    page=None,
                    reason="carries no page number, so the quote cannot be located",
                )
            )
            continue

        page = corpus.page(document_id, page_number)
        if page is None:
            failures.append(
                AnchorFailure(
                    requirement_id=requirement.requirement_id,
                    document_id=document_id,
                    page=page_number,
                    reason="cites a page the document does not have",
                )
            )
            continue

        quote = flatten(anchor.quote)
        if not quote:
            failures.append(
                AnchorFailure(
                    requirement_id=requirement.requirement_id,
                    document_id=document_id,
                    page=page_number,
                    reason="quotes nothing",
                )
            )
            continue

        if quote not in flatten(page.text):
            failures.append(
                AnchorFailure(
                    requirement_id=requirement.requirement_id,
                    document_id=document_id,
                    page=page_number,
                    reason="the quoted span does not appear on that page",
                    quote_preview=quote[:120],
                )
            )

    return AnchorReport(checked=len(requirements), failures=failures)
