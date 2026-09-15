"""Extraction pass: unstructured text to typed requirement structures.

This is the one place in the package where a language model is used, and it is used for
exactly two things:

1. **Splitting a compound clause into atomic obligations.** "The Contractor shall supply and
   install X, and shall commission it within 30 days" is three obligations wearing one
   number, and no regex tells them apart.
2. **Identifying what a clause cites.** A clause reference is a regex; ``ISO 9001``,
   ``33 CFR 105.275`` and ``drawing SEC-201`` are not, and the list of things that look like
   a standard has no end.

Everything else is deterministic Python, and the division is deliberate rather than
stylistic:

- **where a clause starts and stops** — `clauses`, a scan
- **what its modality is** — `policy`, a lookup against a configured word list
- **the source anchor** — built here from the span `clauses` selected. The model is never
  asked for a quote, so it cannot supply one that is not in the document.
- **the identifier of a split child** — `contracts.child_id`, so ids cannot collide, drift
  between runs, or renumber a clause the document numbered itself.
- **every count in the result** — computed from the records, not reported by the model.

Two model answers are taken as suggestions and checked, not obeyed. A reading for a clause
identifier that is not on the page is discarded and counted (`unmatched_readings`), because
a model naming a clause the page does not contain is exactly the failure the count exists to
make visible. And when a clause is not split, the record keeps the **document's** wording,
not the model's echo of it: a paraphrase of an atomic clause is a silent edit to the source.

The model is reached through `StructuredCompletion`, a protocol satisfied by anything that
turns a prompt and a schema into an instance of that schema — `spine.router.Router.structured`
with its `ModelCall` dropped, a stub in a test, a different provider later. This package
holds no client, no key and no retry policy.

Deliberately does not: assign statuses, scores, priorities or monetary figures to what it
extracts, decide whether a requirement matters, or let the model decide anything beyond the
shape of the text. Every downstream verdict is deterministic Python operating on the models
returned here.
"""

from collections.abc import Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from req_core.clauses import DEFAULT_CLAUSE_STYLE, Clause, ClauseStyle, clauses_on_page
from req_core.contracts import ExtractionResult, Requirement, child_id
from req_core.corpus import SourceCorpus
from req_core.policy import DEFAULT_POLICY, ModalityPolicy
from spine.contracts import EvidenceRef

__all__ = [
    "EXTRACT_PURPOSE",
    "ClauseReading",
    "PageReading",
    "StructuredCompletion",
    "build_prompt",
    "extract_requirements",
    "merge_reading",
]

EXTRACT_PURPOSE = "req_core.read_clauses"

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredCompletion(Protocol):
    """Anything that fills a Pydantic schema from a prompt.

    Deliberately the smallest surface that does the job. No client, no key, no tier, no
    retry: a caller supplies those by binding them into the callable it passes in, which is
    how this package stays testable offline and unattached to any one provider.
    """

    def __call__(self, prompt: str, schema: type[SchemaT], *, purpose: str) -> SchemaT:
        """Return an instance of `schema` built from `prompt`."""
        ...


class ClauseReading(BaseModel):
    """What the model was asked to say about one clause. Suggestions, not conclusions."""

    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(
        description="The clause identifier, copied exactly from the input. Not invented: a "
        "reading whose identifier is not on the page is discarded."
    )
    atomic_parts: list[str] = Field(
        default_factory=list,
        description="The separate obligations this clause states, each as one sentence. One "
        "entry (or none) means the clause is already atomic and its own wording is kept.",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Other clauses, standards, regulations and drawings the clause cites, as "
        "written.",
    )


class PageReading(BaseModel):
    """The model's reading of every clause on one page."""

    model_config = ConfigDict(extra="forbid")

    clauses: list[ClauseReading] = Field(default_factory=list)

    def by_identifier(self) -> dict[str, ClauseReading]:
        """The readings keyed by identifier, first occurrence winning."""
        found: dict[str, ClauseReading] = {}
        for reading in self.clauses:
            found.setdefault(reading.identifier.strip(), reading)
        return found


_INSTRUCTIONS = """\
You are reading numbered clauses from a formal document. For each clause you are given, do \
exactly two things and nothing else.

1. SPLIT. If the clause states more than one separable obligation, write each one as a \
single self-contained sentence in `atomic_parts`, preserving the clause's own terminology \
and any numbers, units or tolerances exactly. If the clause states exactly one obligation, \
return a single entry — or none at all, which means the same thing. Do not split a single \
obligation that merely has several conditions on it, and do not merge two clauses.

2. CITE. In `references`, list every OTHER clause, standard, regulation, drawing or document \
the clause refers to, written as the clause writes it (for example TS-B.13, ISO 9001, \
33 CFR 105.275, SEC-201, Document 4). Do not include the clause's own identifier. Do not \
include a reference the clause does not actually make.

Copy each `identifier` exactly as given. Do not invent a clause that is not listed below, \
do not rename one, and do not omit one. Do not judge whether a clause is important, whether \
it is mandatory, or whether anyone complies with it: you are not being asked, and nothing \
you say about it would be used.
"""


def build_prompt(clauses: Sequence[Clause]) -> str:
    """The prompt for one page's clauses.

    Built from the clause spans the segmenter selected, so the model sees exactly the text
    the anchors will quote and cannot be blamed for a boundary it did not draw.
    """
    body = "\n\n".join(f"[{clause.identifier}]\n{clause.text}" for clause in clauses)
    return f"{_INSTRUCTIONS}\nCLAUSES\n\n{body}\n"


def _clean_references(raw: Sequence[str], *, own_id: str) -> list[str]:
    """Normalise, de-duplicate and sort what the model listed.

    Order and duplicates are the model's; the set is what matters, so Python decides both.
    A self-citation is dropped rather than carried: `Requirement` refuses one, and a record
    that would not validate is a worse outcome than a reference nobody needed.
    """
    seen = {stripped for reference in raw if (stripped := reference.strip()) and stripped != own_id}
    return sorted(seen)


def _anchor(clause: Clause, corpus: SourceCorpus) -> EvidenceRef:
    """The citation for a clause: where it is, and the span the segmenter selected.

    The quote is always document text. Nothing a model produced reaches this field, which is
    what makes `anchors.verify` a check on this module's arithmetic rather than on a model's
    honesty.
    """
    return EvidenceRef(
        source_id=clause.document_id,
        document=corpus.title_of(clause.document_id),
        page=clause.page_number,
        clause=clause.identifier,
        quote=clause.text,
    )


def merge_reading(
    clause: Clause,
    reading: ClauseReading | None,
    *,
    corpus: SourceCorpus,
    policy: ModalityPolicy,
) -> list[Requirement]:
    """Turn one clause and the model's reading of it into requirement records.

    All of the deciding happens here, in Python. A missing reading is not an error: the
    clause becomes one atomic requirement citing nothing, which is what a clause with no
    reading honestly is.
    """
    anchor = _anchor(clause, corpus)
    references = _clean_references(reading.references if reading else (), own_id=clause.identifier)
    parts = [part.strip() for part in (reading.atomic_parts if reading else []) if part.strip()]

    if len(parts) <= 1:
        # Not split. The record keeps the document's wording, never the model's echo of it —
        # a paraphrase accepted here would be a silent edit to the source.
        modality = policy.classify(clause.text)
        return [
            Requirement(
                requirement_id=clause.identifier,
                text=clause.text,
                modality=modality.modality,
                modality_trigger=modality.trigger_word,
                source=anchor,
                references=references,
                parent_id=None,
                is_atomic=True,
            )
        ]

    parent_modality = policy.classify(clause.text)
    records = [
        Requirement(
            requirement_id=clause.identifier,
            text=clause.text,
            modality=parent_modality.modality,
            modality_trigger=parent_modality.trigger_word,
            source=anchor,
            references=references,
            parent_id=None,
            is_atomic=False,
        )
    ]
    for ordinal, part in enumerate(parts, start=1):
        child_modality = policy.classify(part)
        records.append(
            Requirement(
                requirement_id=child_id(clause.identifier, ordinal),
                text=part,
                modality=child_modality.modality,
                modality_trigger=child_modality.trigger_word,
                # The child's own words are nowhere in the document; the anchor quotes the
                # clause they came from, which is where a reader would go to check them.
                source=anchor,
                references=references,
                parent_id=clause.identifier,
                is_atomic=True,
            )
        )
    return records


def extract_requirements(
    corpus: SourceCorpus,
    complete: StructuredCompletion,
    *,
    policy: ModalityPolicy = DEFAULT_POLICY,
    style: ClauseStyle = DEFAULT_CLAUSE_STYLE,
    purpose: str = EXTRACT_PURPOSE,
) -> ExtractionResult:
    """Extract every requirement from a corpus, one model call per page that has clauses.

    A page with no clauses costs nothing: no call is made for it. The corpus's `withheld`
    set travels into the result, so what was kept out of extraction's reach is part of the
    record rather than a claim made about it afterwards.
    """
    requirements: list[Requirement] = []
    clauses_read = 0
    truncated = 0
    unmatched = 0

    for document in corpus.documents:
        for page in document.pages:
            clauses = clauses_on_page(page, style=style)
            if not clauses:
                continue
            clauses_read += len(clauses)
            truncated += sum(1 for clause in clauses if clause.truncated)

            reading = complete(build_prompt(clauses), PageReading, purpose=purpose)
            by_identifier = reading.by_identifier()
            unmatched += len(set(by_identifier) - {clause.identifier for clause in clauses})

            for clause in clauses:
                requirements.extend(
                    merge_reading(
                        clause,
                        by_identifier.get(clause.identifier),
                        corpus=corpus,
                        policy=policy,
                    )
                )

    return ExtractionResult(
        corpus_name=corpus.name,
        documents_seen=corpus.document_ids,
        withheld=corpus.withheld,
        policy_name=policy.name,
        requirements=requirements,
        clauses_read=clauses_read,
        clauses_truncated=truncated,
        unmatched_readings=unmatched,
    )
