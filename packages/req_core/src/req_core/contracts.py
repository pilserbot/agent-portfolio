"""What an extracted requirement is, and what one extraction pass produced.

`Requirement` is the unit everything downstream reasons about. It carries where it came
from, not merely what it says: `source` is `spine.contracts.EvidenceRef`, the portfolio's
one citation type, so a requirement extracted here resolves the same way as a finding made
anywhere else. Forking a second citation type would give the project two answers to "where
did this come from".

**Lineage, not replacement.** When a compound clause is split, the clause it came from stays
in the set with `is_atomic=False`, and each child carries `parent_id`. Nothing is thrown
away: a reader can still see the sentence the author actually wrote, and a count can be
taken over clauses or over atomic obligations without either being a guess.

**A child's quote is its parent's span.** A split child's `text` is a rewritten sentence and
therefore appears nowhere in the document; its `source.quote` is the parent clause verbatim,
because that is where it came from. This keeps every anchor in the set checkable against the
page — see `anchors.verify`, which is the assertion this package rests on.

Deliberately does not: hold a status, a score, a priority or a cost. Nothing here says
whether a requirement is met, important, expensive or risky. Those are the applications'
judgements, computed in Python over these records.
"""

from collections.abc import Collection

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from req_core.policy import Modality
from spine.contracts import EvidenceRef

__all__ = [
    "CHILD_SEPARATOR",
    "ExtractionResult",
    "Requirement",
    "child_id",
]

# How a split child's identifier is built from its parent's. Python builds it, never the
# model: an id the model invented could collide, drift between runs, or quietly renumber a
# clause the document numbered itself.
CHILD_SEPARATOR = "/"


def child_id(parent_id: str, ordinal: int) -> str:
    """The identifier of the nth atomic child of a clause, 1-based and stable."""
    if ordinal < 1:
        raise ValueError(f"child ordinal must be 1-based; got {ordinal}")
    return f"{parent_id}{CHILD_SEPARATOR}{ordinal}"


class Requirement(BaseModel):
    """One requirement, atomic or compound, anchored to the text it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requirement_id: str = Field(
        min_length=1,
        description="The clause reference as written, e.g. TS-B.25. A split child carries "
        "its parent's reference with an ordinal suffix, built by `child_id`.",
    )
    text: str = Field(min_length=1)
    modality: Modality
    modality_trigger: str | None = Field(
        default=None,
        description="The modal word the modality rests on, or None when the text has none. "
        "Kept so a reader can check the classification without re-running the policy.",
    )
    source: EvidenceRef
    references: list[str] = Field(
        default_factory=list,
        description="Other clauses, standards and drawings this requirement cites. Sorted "
        "and de-duplicated by Python; never includes the requirement's own identifier.",
    )
    parent_id: str | None = Field(
        default=None, description="Set when this is a split child, naming the clause it came from."
    )
    is_atomic: bool = Field(description="Whether this states exactly one obligation.")

    @computed_field
    @property
    def is_child(self) -> bool:
        """Whether this record was produced by splitting another."""
        return self.parent_id is not None

    @model_validator(mode="after")
    def _lineage_is_coherent(self) -> "Requirement":
        """Reject a record whose lineage contradicts itself."""
        if self.parent_id is not None and self.parent_id == self.requirement_id:
            raise ValueError(
                f"requirement {self.requirement_id!r} names itself as its parent, which would "
                f"make the lineage a cycle."
            )
        if self.parent_id is not None and not self.is_atomic:
            raise ValueError(
                f"requirement {self.requirement_id!r} is a split child and is also marked "
                f"non-atomic. Splitting exists to produce atomic children; a non-atomic one "
                f"means the split did not finish and the record would over-count obligations."
            )
        if self.requirement_id in self.references:
            raise ValueError(
                f"requirement {self.requirement_id!r} lists itself among its references. A "
                f"self-citation would make any dependency graph built on this cyclic."
            )
        return self


class ExtractionResult(BaseModel):
    """Everything one extraction pass produced, and what it was allowed to read.

    `documents_seen` and `withheld` travel with the requirements on purpose. A recall or
    coverage figure means nothing without them: the same number over a corpus that included
    the answer sheet and one that did not are different claims, and this is what lets a test
    tell them apart after the fact rather than on trust.
    """

    model_config = ConfigDict(frozen=True)

    corpus_name: str = Field(min_length=1)
    documents_seen: frozenset[str] = Field(default_factory=frozenset)
    withheld: frozenset[str] = Field(default_factory=frozenset)
    policy_name: str = Field(min_length=1, description="Which modality convention was applied.")
    requirements: list[Requirement] = Field(default_factory=list)
    clauses_read: int = Field(ge=0, description="Clauses the segmenter found, before splitting.")
    clauses_truncated: int = Field(
        default=0,
        ge=0,
        description="Clauses that ran off the bottom of their page and kept only the part on "
        "it. Reported because a silently shortened clause is a silently shortened quote.",
    )
    unmatched_readings: int = Field(
        default=0,
        ge=0,
        description="Readings the model returned for a clause identifier that was not on the "
        "page. Discarded, and counted here rather than dropped in silence: a model naming a "
        "clause the page does not contain is the failure this number exists to make visible.",
    )

    @computed_field
    @property
    def atomic_count(self) -> int:
        """How many records state exactly one obligation."""
        return sum(1 for requirement in self.requirements if requirement.is_atomic)

    @computed_field
    @property
    def split_count(self) -> int:
        """How many clauses were split into children."""
        return len({r.parent_id for r in self.requirements if r.parent_id is not None})

    @property
    def clause_ids(self) -> frozenset[str]:
        """Every clause identifier extracted, children excluded.

        Children are derived, not printed in the document, so a comparison against anything
        the document itself indexes has to be made over this and not over every record.
        """
        return frozenset(
            requirement.requirement_id
            for requirement in self.requirements
            if requirement.parent_id is None
        )

    def clause_ids_in(self, document_ids: Collection[str]) -> frozenset[str]:
        """Every clause identifier extracted from one subset of the documents read.

        A pass over a wide corpus answers questions about narrower ones too, and this is how
        it does so honestly: the clauses are selected by the document each was read from
        rather than by trusting that the corpus behind the result was the corpus the caller
        had in mind. Extraction is per page, so the set returned here is exactly what a pass
        over only those documents would have produced.
        """
        wanted = set(document_ids)
        return frozenset(
            requirement.requirement_id
            for requirement in self.requirements
            if requirement.parent_id is None and requirement.source.source_id in wanted
        )

    @property
    def cited_references(self) -> frozenset[str]:
        """Every reference any requirement cites."""
        return frozenset(
            reference for requirement in self.requirements for reference in requirement.references
        )
