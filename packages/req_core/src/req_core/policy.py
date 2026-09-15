"""What a modal verb means, stated as configuration rather than assumed.

Everyone knows "shall" is mandatory. Almost everyone assumes "will" is too. The Kessler
Point tender says, at ITB-2.1, that `will` is **optional** and `should` is treated the same
way — and it says so in a table on page 1 that a reader skims past. A hardcoded mapping
would have read that document confidently and wrongly, and the error would have been
invisible: every clause would still have got a modality, just the wrong one.

So the mapping is a `ModalityPolicy`: a named object a caller supplies. `DEFAULT_POLICY` is
the ordinary English reading and is a default, not a truth. When a source states its own
convention, that convention is written down as a policy and passed in.

**Assignment is deterministic Python.** No model is asked what a clause means. The policy
holds word sets; `classify` lowercases, finds whole-word matches, and reports every one it
found in the order they appear. The first decides, and the rest are kept: a clause carrying
both "shall" and "may" is a real thing in real documents, and the record says so rather than
quietly picking one.

Deliberately does not: rank requirements, decide whether a modality is important, infer a
convention from the text, or guess when no modal verb is present. A clause with no modal
verb is `Modality.NONE` — which is information, not a failure, and is how a heading or a
definition tells itself apart from an obligation.
"""

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DEFAULT_POLICY",
    "Modality",
    "ModalityPolicy",
    "ModalityReading",
    "ModalityTrigger",
    "PolicyError",
]


class PolicyError(Exception):
    """A modality policy is not usable as written."""


class Modality(StrEnum):
    """How binding a requirement is, under whatever convention the source declared."""

    MANDATORY = "mandatory"
    ADVISORY = "advisory"
    OPTIONAL = "optional"
    NONE = "none"


class ModalityTrigger(BaseModel):
    """One modal word found in a piece of text, and what the policy makes of it."""

    model_config = ConfigDict(frozen=True)

    word: str = Field(min_length=1, description="The matched word, lowercased.")
    modality: Modality
    position: int = Field(ge=0, description="Character offset of the match in the text.")


class ModalityReading(BaseModel):
    """Everything the policy found in one piece of text.

    `modality` is the verdict — the first trigger in reading order. `triggers` is the whole
    finding, so a clause that says two things can be seen to.
    """

    model_config = ConfigDict(frozen=True)

    modality: Modality
    triggers: list[ModalityTrigger] = Field(default_factory=list)

    @property
    def trigger_word(self) -> str | None:
        """The word the verdict rests on, or None when no modal verb was present."""
        return self.triggers[0].word if self.triggers else None

    @property
    def is_mixed(self) -> bool:
        """Whether the text carries modal words of more than one kind.

        Not an error and not resolved here. It is the signal that a clause is doing two
        jobs, which is exactly the kind of thing an atomicity split exists to separate.
        """
        return len({trigger.modality for trigger in self.triggers}) > 1


class ModalityPolicy(BaseModel):
    """Which words mean mandatory, which advisory, which optional — for one source.

    Named, because a report that says "modality per the default policy" and one that says
    "modality per ITB-2.1" are making different claims, and the difference has to survive
    into the record.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description='e.g. "default" or "kessler_point ITB-2.1".')
    mandatory: frozenset[str] = Field(default_factory=frozenset)
    advisory: frozenset[str] = Field(default_factory=frozenset)
    optional: frozenset[str] = Field(default_factory=frozenset)
    source: str = Field(
        default="",
        description="Where the convention was stated, when the source stated one — a clause "
        "reference, a page. Empty means the policy is an assumption, which is worth being "
        "able to tell apart from a quotation.",
    )

    @model_validator(mode="after")
    def _words_mean_one_thing(self) -> "ModalityPolicy":
        """Reject a policy where a word carries two meanings, or carries none."""
        buckets = {
            Modality.MANDATORY: self.mandatory,
            Modality.ADVISORY: self.advisory,
            Modality.OPTIONAL: self.optional,
        }
        seen: dict[str, Modality] = {}
        clashes: list[str] = []
        for modality, words in buckets.items():
            for word in words:
                if not word.strip():
                    raise ValueError(f"policy {self.name!r} carries a blank word in {modality}.")
                if word.lower() != word:
                    raise ValueError(
                        f"policy {self.name!r} carries {word!r}, which is not lowercase. "
                        f"Matching lowercases the text, so a capitalised entry would never fire."
                    )
                if word in seen:
                    clashes.append(f"{word!r} is both {seen[word]} and {modality}")
                seen[word] = modality
        if clashes:
            raise ValueError(f"policy {self.name!r}: " + "; ".join(sorted(clashes)))
        if not seen:
            raise ValueError(
                f"policy {self.name!r} maps no words at all, so every clause would read as "
                f"{Modality.NONE}. An empty policy is almost always a configuration mistake."
            )
        return self

    @property
    def words(self) -> dict[str, Modality]:
        """Every word the policy knows, mapped to what it means."""
        return {
            **{word: Modality.MANDATORY for word in self.mandatory},
            **{word: Modality.ADVISORY for word in self.advisory},
            **{word: Modality.OPTIONAL for word in self.optional},
        }

    def classify(self, text: str) -> ModalityReading:
        """Read the modality of a piece of text under this policy.

        Whole-word, case-insensitive, in reading order. The first match is the verdict and
        every match is kept.
        """
        known = self.words
        if not known:  # pragma: no cover - the validator refuses an empty policy
            return ModalityReading(modality=Modality.NONE)

        pattern = re.compile(
            r"\b("
            + "|".join(re.escape(word) for word in sorted(known, key=len, reverse=True))
            + r")\b",
            re.IGNORECASE,
        )
        triggers = [
            ModalityTrigger(
                word=match.group(0).lower(),
                modality=known[match.group(0).lower()],
                position=match.start(),
            )
            for match in pattern.finditer(text)
        ]
        return ModalityReading(
            modality=triggers[0].modality if triggers else Modality.NONE,
            triggers=triggers,
        )


# The ordinary English reading, and nothing more authoritative than that. A source that
# states its own convention overrides this; see `ModalityPolicy.source`.
DEFAULT_POLICY = ModalityPolicy(
    name="default",
    mandatory=frozenset({"shall", "must", "will"}),
    advisory=frozenset({"should"}),
    optional=frozenset({"may", "can"}),
)
