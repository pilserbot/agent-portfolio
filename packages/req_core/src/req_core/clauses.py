"""Finding where each clause begins and ends, with no model involved.

A numbered document tells you where its requirements are: a clause starts at a line that
opens with an identifier and runs until the next one does. That is a scan, not a judgement,
so it is done here in Python and the result is exact — which matters, because every source
anchor in this package quotes text this module selected. A model never supplies a quote, so
a quote can never fail to be in the document.

**The identifier shape is configuration.** `ClauseStyle` carries the pattern. The default
matches the common engineering-document forms — ``ITB-9.2``, ``TS-B.25``, ``A-5.2``,
``GN-02``, ``4.7.1`` — and a source that numbers its clauses differently supplies its own
rather than asking this module to grow another special case.

**A clause is bounded to the page it starts on.** A clause that wraps past a page break
keeps only the part on its own page, and is flagged `truncated`. The alternative is an
anchor whose quoted span is not on the page it cites, which would make the one assertion
this package rests on unprovable. Losing the tail of a wrapped clause is visible and
countable; an anchor that cites the wrong page is neither.

Deliberately does not: call a model, decide whether a clause is a requirement, read
modality, or judge importance. It answers "where does each clause start and stop", and
`policy` and `extraction` take it from there.
"""

import re
from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field, field_validator

from req_core.corpus import CorpusPage, SourceCorpus

__all__ = [
    "DEFAULT_CLAUSE_STYLE",
    "Clause",
    "ClauseStyle",
    "clauses_on_page",
    "segment",
]

# A line that is page furniture rather than content: a running footer or header carrying a
# page number. Excluded from a clause's span so a quote does not trail off into boilerplate.
_FURNITURE = re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)

# Sentence-terminating punctuation. A clause whose last line ends in none of these probably
# continues onto the next page, which this module records rather than silently stitches.
_TERMINAL = (".", ":", ";", "?", "!", '"', "”", ")")


class ClauseStyle(BaseModel):
    """How this source numbers its clauses."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    identifier_pattern: str = Field(
        min_length=1,
        description="Regex matching a clause identifier at the very start of a line. Must "
        "define exactly one capturing group: the identifier as written.",
    )

    @field_validator("identifier_pattern")
    @classmethod
    def _one_group_and_compilable(cls, value: str) -> str:
        """Reject a pattern that will not compile, or that does not capture exactly one thing."""
        try:
            compiled = re.compile(value)
        except re.error as error:
            raise ValueError(f"identifier_pattern does not compile: {error}") from error
        if compiled.groups != 1:
            raise ValueError(
                f"identifier_pattern must define exactly one capturing group (the identifier "
                f"as written); this one defines {compiled.groups}."
            )
        return value

    @property
    def matcher(self) -> re.Pattern[str]:
        """The compiled pattern, anchored at the start of a line."""
        return re.compile(self.identifier_pattern)

    def identifier_at(self, line: str) -> str | None:
        """The clause identifier this line opens with, or None if it opens with none."""
        match = self.matcher.match(line)
        return match.group(1) if match else None


# Two shapes, either accepted: a lettered or numbered section joined by a hyphen
# (ITB-9.2, TS-B.25, A-5.2, GN-02), or a bare dotted number (4.7.1). Both must be followed
# by whitespace, so a sentence opening with a bare number is not mistaken for a clause.
DEFAULT_CLAUSE_STYLE = ClauseStyle(
    name="default",
    identifier_pattern=r"^([A-Z]{1,4}-[A-Z]?\.?\d+(?:\.\d+)*|\d+(?:\.\d+){1,3})(?=[ \t])",
)


class Clause(BaseModel):
    """One clause as the document prints it, located exactly."""

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(min_length=1, description="As written, e.g. TS-B.25.")
    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(
        min_length=1,
        description="The clause as it appears on the page, lines joined by a single newline. "
        "This exact string is what the source anchor quotes.",
    )
    truncated: bool = Field(
        default=False,
        description="The clause runs off the bottom of its page. Its text is what is on this "
        "page and no more — see the module docstring for why the tail is dropped rather than "
        "stitched on from the next page.",
    )

    @property
    def line_count(self) -> int:
        """How many lines of the page this clause occupies."""
        return len(self.text.splitlines())


def _is_furniture(line: str) -> bool:
    """Whether a line is a running header or footer rather than content."""
    return bool(_FURNITURE.search(line))


def clauses_on_page(page: CorpusPage, *, style: ClauseStyle = DEFAULT_CLAUSE_STYLE) -> list[Clause]:
    """Every clause beginning on one page, in reading order."""
    lines = page.text.splitlines()
    starts = [
        (index, identifier)
        for index, line in enumerate(lines)
        if (identifier := style.identifier_at(line)) is not None
    ]

    found: list[Clause] = []
    for position, (index, identifier) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        body = [line for line in lines[index:end] if not _is_furniture(line)]
        while body and not body[-1].strip():
            body.pop()
        if not body:  # pragma: no cover - the start line itself is never blank
            continue
        text = "\n".join(body)
        found.append(
            Clause(
                identifier=identifier,
                document_id=page.document_id,
                page_number=page.page_number,
                text=text,
                # Only the last clause on a page can run off it, and only if its final line
                # stops mid-sentence.
                truncated=(
                    position + 1 == len(starts) and not body[-1].rstrip().endswith(_TERMINAL)
                ),
            )
        )
    return found


def segment(corpus: SourceCorpus, *, style: ClauseStyle = DEFAULT_CLAUSE_STYLE) -> Iterator[Clause]:
    """Every clause in a corpus, document by document and page by page."""
    for document in corpus.documents:
        for page in document.pages:
            yield from clauses_on_page(page, style=style)
