"""Unit tests for req_core, entirely offline against a fake completion function.

No network, no model, no key. The model is reached through `StructuredCompletion`, so a
stub that returns whatever the test wants exercises the whole pass — which is the point of
taking a callable rather than holding a client.

What is pinned here is the division of labour the package rests on: modality comes from a
configured policy and not from a model, split children keep their lineage and their ids are
built by Python, references are cleaned by Python, and every anchor quotes text that is
really on the page it cites. The last one has both halves — it passes on a correct set, and
it fails, naming the requirement and the page, on a doctored one.
"""

import re

import pytest
from pydantic import ValidationError

from req_core.anchors import AnchorVerificationError, flatten, verify
from req_core.clauses import ClauseStyle, clauses_on_page, segment
from req_core.contracts import CHILD_SEPARATOR, Requirement, child_id
from req_core.corpus import (
    CorpusDocument,
    CorpusError,
    CorpusPage,
    DocumentLike,
    PageLike,
    SourceCorpus,
    corpus_from,
)
from req_core.extraction import (
    ClauseReading,
    PageReading,
    build_prompt,
    extract_requirements,
    merge_reading,
)
from req_core.policy import DEFAULT_POLICY, Modality, ModalityPolicy
from spine.contracts import EvidenceRef

# --- fixtures, built by hand -----------------------------------------------------------

PAGE_TEXT = """\
4. SYSTEM B — VIDEO SURVEILLANCE
TS-B.1 The Contractor shall supply and install the cameras listed at TS-B.13, and shall
commission each one in accordance with ISO 9001.
TS-B.2 Cameras should be mounted at a height of not less than 4 m.
TS-B.3 This clause is a heading with no obligation.
SENSITIVE SECURITY INFORMATION Page 4 of 9"""


def a_page(
    text: str = PAGE_TEXT, *, document_id: str = "04_Spec", page_number: int = 4
) -> CorpusPage:
    return CorpusPage(document_id=document_id, page_number=page_number, text=text)


def a_corpus(*pages: CorpusPage, withhold: frozenset[str] = frozenset()) -> SourceCorpus:
    pages = pages or (a_page(),)
    by_document: dict[str, list[CorpusPage]] = {}
    for page in pages:
        by_document.setdefault(page.document_id, []).append(page)
    return SourceCorpus(
        name="fixture",
        documents=[
            CorpusDocument(document_id=document_id, title=f"Title of {document_id}", pages=group)
            for document_id, group in by_document.items()
        ],
        withheld=withhold,
    )


class StubCompletion:
    """A `StructuredCompletion` that answers from a fixed script and records its prompts."""

    def __init__(self, readings: dict[str, ClauseReading] | None = None) -> None:
        """Answer with these readings, defaulting any clause not named to an empty one."""
        self.readings = readings or {}
        self.prompts: list[str] = []
        self.purposes: list[str] = []
        self.extra: list[ClauseReading] = []

    def __call__(self, prompt: str, schema: type, *, purpose: str) -> PageReading:
        self.prompts.append(prompt)
        self.purposes.append(purpose)
        identifiers = re.findall(r"^\[([^\]]+)\]", prompt, re.MULTILINE)
        return PageReading(
            clauses=[
                self.readings.get(identifier, ClauseReading(identifier=identifier))
                for identifier in identifiers
            ]
            + self.extra
        )


def an_anchor(**overrides: object) -> EvidenceRef:
    payload: dict[str, object] = {
        "source_id": "04_Spec",
        "document": "Spec",
        "page": 4,
        "clause": "TS-B.2",
        "quote": "TS-B.2 Cameras should be mounted at a height of not less than 4 m.",
    }
    payload.update(overrides)
    return EvidenceRef.model_validate(payload)


# --- the way in: a protocol, then a model ------------------------------------------------


def test_a_foreign_document_type_satisfies_the_protocol_without_importing_anything() -> None:
    # The whole point of the doorway: a caller's own record type walks in unchanged.
    class TheirPage:
        document_id = "X"
        page_number = 1
        text = "TS-1.1 The Contractor shall do the thing."

    class TheirDocument:
        document_id = "X"
        pages = (TheirPage(),)

    assert isinstance(TheirPage(), PageLike)
    assert isinstance(TheirDocument(), DocumentLike)

    corpus = corpus_from([TheirDocument()], name="theirs")

    assert corpus.document_ids == {"X"}
    assert corpus.page_count == 1


def test_a_corpus_refuses_to_hold_a_document_it_says_it_withheld() -> None:
    # Not a convention and not a comment: the mistake is not representable.
    with pytest.raises(CorpusError) as caught:
        corpus_from(
            [_document("04_Spec"), _document("09_Matrix")],
            name="t",
            withhold={"09_Matrix"},
        )

    assert "09_Matrix" in str(caught.value)
    assert "withheld" in str(caught.value)


def test_withholding_is_recorded_not_merely_omitted() -> None:
    corpus = corpus_from(
        [_document("04_Spec"), _document("09_Matrix")],
        name="t",
        include={"04_Spec"},
        withhold={"09_Matrix"},
    )

    assert corpus.document_ids == {"04_Spec"}
    assert corpus.withheld == {"09_Matrix"}


def test_asking_for_a_document_the_source_does_not_have_is_an_error() -> None:
    # A smaller corpus than intended would read as a worse result, not a missing input.
    with pytest.raises(CorpusError) as caught:
        corpus_from([_document("04_Spec")], name="t", include={"04_Spec", "05_Missing"})

    assert "05_Missing" in str(caught.value)


def test_two_documents_with_one_id_are_refused() -> None:
    with pytest.raises(CorpusError):
        corpus_from([_document("04_Spec"), _document("04_Spec")], name="t")


def test_a_page_filed_under_the_wrong_document_is_refused() -> None:
    with pytest.raises(ValidationError):
        CorpusDocument(
            document_id="04_Spec",
            pages=[CorpusPage(document_id="05_Other", page_number=1, text="x")],
        )


def _document(document_id: str) -> CorpusDocument:
    return CorpusDocument(
        document_id=document_id,
        pages=[CorpusPage(document_id=document_id, page_number=1, text=PAGE_TEXT)],
    )


# --- the modality policy, applied from configuration --------------------------------------


def test_the_default_policy_reads_the_ordinary_english_convention() -> None:
    assert DEFAULT_POLICY.classify("The Contractor shall do it.").modality is Modality.MANDATORY
    assert DEFAULT_POLICY.classify("The Contractor will do it.").modality is Modality.MANDATORY
    assert DEFAULT_POLICY.classify("The Contractor should do it.").modality is Modality.ADVISORY
    assert DEFAULT_POLICY.classify("The Contractor may do it.").modality is Modality.OPTIONAL
    assert DEFAULT_POLICY.classify("This section is informative.").modality is Modality.NONE


def test_a_different_convention_gives_a_different_answer_to_the_same_sentence() -> None:
    # This is why the mapping is configuration. A source that states its own convention and
    # is read under the default one gets a modality for every clause — the wrong one.
    house = ModalityPolicy(
        name="house style",
        source="ITB-2.1",
        mandatory=frozenset({"shall"}),
        advisory=frozenset({"may"}),
        optional=frozenset({"will", "should"}),
    )
    sentence = "The Contractor will provide a spares list."

    assert DEFAULT_POLICY.classify(sentence).modality is Modality.MANDATORY
    assert house.classify(sentence).modality is Modality.OPTIONAL


def test_the_policy_is_applied_to_the_text_not_asked_of_a_model() -> None:
    # A stub that says nothing about modality; the classification still happens.
    stub = StubCompletion()
    result = extract_requirements(a_corpus(), stub, policy=DEFAULT_POLICY)
    by_id = {requirement.requirement_id: requirement for requirement in result.requirements}

    assert by_id["TS-B.1"].modality is Modality.MANDATORY
    assert by_id["TS-B.2"].modality is Modality.ADVISORY
    assert by_id["TS-B.3"].modality is Modality.NONE
    assert result.policy_name == "default"


def test_every_modal_word_is_kept_and_the_first_one_decides() -> None:
    reading = DEFAULT_POLICY.classify("The Contractor shall supply it and may propose another.")

    assert reading.modality is Modality.MANDATORY
    assert reading.trigger_word == "shall"
    assert [trigger.word for trigger in reading.triggers] == ["shall", "may"]
    assert reading.is_mixed


def test_the_trigger_word_travels_onto_the_record() -> None:
    result = extract_requirements(a_corpus(), StubCompletion(), policy=DEFAULT_POLICY)
    by_id = {requirement.requirement_id: requirement for requirement in result.requirements}

    assert by_id["TS-B.2"].modality_trigger == "should"
    assert by_id["TS-B.3"].modality_trigger is None


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"mandatory": frozenset({"shall"}), "optional": frozenset({"shall"})}, "both"),
        ({"mandatory": frozenset({"Shall"})}, "lowercase"),
        ({"mandatory": frozenset({" "})}, "blank"),
        ({}, "maps no words"),
    ],
)
def test_an_unusable_policy_is_refused_at_construction(
    kwargs: dict[str, frozenset[str]], fragment: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        ModalityPolicy(name="broken", **kwargs)

    assert fragment in str(caught.value)


# --- clause segmentation, in Python ---------------------------------------------------------


def test_clauses_are_located_by_the_configured_style() -> None:
    found = clauses_on_page(a_page())

    assert [clause.identifier for clause in found] == ["TS-B.1", "TS-B.2", "TS-B.3"]
    assert found[0].text.startswith("TS-B.1 The Contractor shall supply")
    assert found[0].line_count == 2  # the clause wraps, and both lines are kept


def test_page_furniture_is_not_part_of_a_clause() -> None:
    # A quote that trailed off into the footer would still verify, and would still be wrong.
    found = clauses_on_page(a_page())

    assert "SENSITIVE SECURITY INFORMATION" not in found[-1].text
    assert "Page 4 of 9" not in found[-1].text


def test_a_narrower_style_refuses_what_the_default_accepts() -> None:
    # The style is configuration for the same reason the policy is: "4.7.1" is a clause in
    # one document and a section heading in another.
    page = a_page("4.1 General\nTS-B.1 The Contractor shall do it.")
    prefixed = ClauseStyle(
        name="prefixed", identifier_pattern=r"^([A-Z]{1,4}-[A-Z]?\.?\d+(?:\.\d+)*)(?=[ \t])"
    )

    assert [c.identifier for c in clauses_on_page(page)] == ["4.1", "TS-B.1"]
    assert [c.identifier for c in clauses_on_page(page, style=prefixed)] == ["TS-B.1"]


def test_a_clause_running_off_the_bottom_of_its_page_is_flagged() -> None:
    page = a_page("TS-B.1 The Contractor shall supply cameras of a type described in the")

    assert clauses_on_page(page)[0].truncated


def test_a_clause_ending_in_punctuation_is_not_flagged() -> None:
    page = a_page("TS-B.1 The Contractor shall supply cameras.")

    assert not clauses_on_page(page)[0].truncated


@pytest.mark.parametrize("pattern", ["([A-Z]+", "[A-Z]+", "([A-Z]+)-(\\d+)"])
def test_a_style_whose_pattern_is_unusable_is_refused(pattern: str) -> None:
    with pytest.raises(ValidationError):
        ClauseStyle(name="bad", identifier_pattern=pattern)


def test_segment_walks_every_document_and_page_in_order() -> None:
    corpus = a_corpus(
        a_page("TS-B.1 One shall.", document_id="04_Spec", page_number=1),
        a_page("TS-B.2 Two shall.", document_id="04_Spec", page_number=2),
        a_page("IF-1.1 Three shall.", document_id="05_Other", page_number=1),
    )

    assert [clause.identifier for clause in segment(corpus)] == ["TS-B.1", "TS-B.2", "IF-1.1"]


# --- what the model does: split with lineage ------------------------------------------------


def test_a_compound_clause_splits_into_children_that_keep_their_parent() -> None:
    stub = StubCompletion(
        {
            "TS-B.1": ClauseReading(
                identifier="TS-B.1",
                atomic_parts=[
                    "The Contractor shall supply the cameras listed at TS-B.13.",
                    "The Contractor shall commission each camera in accordance with ISO 9001.",
                ],
            )
        }
    )

    result = extract_requirements(a_corpus(), stub)
    family = [r for r in result.requirements if r.requirement_id.startswith("TS-B.1")]
    parent = next(r for r in family if r.requirement_id == "TS-B.1")
    children = [r for r in family if r.parent_id == "TS-B.1"]

    # The parent survives the split. Nothing is thrown away, so a count can be taken over
    # clauses or over obligations without either being a guess.
    assert not parent.is_atomic
    assert parent.parent_id is None
    assert [child.requirement_id for child in children] == ["TS-B.1/1", "TS-B.1/2"]
    assert all(child.is_atomic and child.is_child for child in children)
    assert result.split_count == 1


def test_a_child_id_is_built_by_python_not_supplied_by_the_model() -> None:
    # The model is never asked for an id. One it invented could collide, drift between runs,
    # or renumber a clause the document numbered itself.
    assert child_id("TS-B.1", 1) == f"TS-B.1{CHILD_SEPARATOR}1"
    with pytest.raises(ValueError, match="1-based"):
        child_id("TS-B.1", 0)


def test_a_child_carries_its_own_modality_read_from_its_own_words() -> None:
    stub = StubCompletion(
        {
            "TS-B.1": ClauseReading(
                identifier="TS-B.1",
                atomic_parts=[
                    "The Contractor shall supply the cameras.",
                    "The Contractor may propose an alternative mount.",
                ],
            )
        }
    )

    result = extract_requirements(a_corpus(), stub)
    children = {r.requirement_id: r for r in result.requirements if r.is_child}

    assert children["TS-B.1/1"].modality is Modality.MANDATORY
    assert children["TS-B.1/2"].modality is Modality.OPTIONAL


def test_an_unsplit_clause_keeps_the_documents_wording_not_the_models_echo() -> None:
    # A paraphrase accepted here would be a silent edit to the source.
    stub = StubCompletion(
        {"TS-B.2": ClauseReading(identifier="TS-B.2", atomic_parts=["Mount cameras high up."])}
    )

    result = extract_requirements(a_corpus(), stub)
    record = next(r for r in result.requirements if r.requirement_id == "TS-B.2")

    assert record.text.startswith("TS-B.2 Cameras should be mounted")
    assert "Mount cameras high up." != record.text
    assert record.is_atomic


def test_a_record_whose_lineage_contradicts_itself_is_refused() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        Requirement(
            requirement_id="A",
            text="t",
            modality=Modality.NONE,
            source=an_anchor(),
            parent_id="A",
            is_atomic=True,
        )
    with pytest.raises(ValidationError, match="did not finish"):
        Requirement(
            requirement_id="A/1",
            text="t",
            modality=Modality.NONE,
            source=an_anchor(),
            parent_id="A",
            is_atomic=False,
        )


# --- what the model does: references ---------------------------------------------------------


def test_references_are_taken_from_the_model_and_cleaned_by_python() -> None:
    stub = StubCompletion(
        {
            "TS-B.1": ClauseReading(
                identifier="TS-B.1",
                references=["ISO 9001", " TS-B.13 ", "ISO 9001", "TS-B.1", ""],
            )
        }
    )

    result = extract_requirements(a_corpus(), stub)
    record = next(r for r in result.requirements if r.requirement_id == "TS-B.1")

    # Sorted, de-duplicated, stripped, and the self-citation dropped. Order and duplicates
    # were the model's; the set is what matters, so Python decides both.
    assert record.references == ["ISO 9001", "TS-B.13"]


def test_a_self_citation_would_be_refused_if_it_ever_reached_the_record() -> None:
    with pytest.raises(ValidationError, match="cyclic"):
        Requirement(
            requirement_id="TS-B.1",
            text="t",
            modality=Modality.NONE,
            source=an_anchor(),
            references=["TS-B.1"],
            is_atomic=True,
        )


def test_children_inherit_the_clauses_references() -> None:
    stub = StubCompletion(
        {
            "TS-B.1": ClauseReading(
                identifier="TS-B.1",
                atomic_parts=["Supply them.", "Commission them."],
                references=["ISO 9001"],
            )
        }
    )

    result = extract_requirements(a_corpus(), stub)

    assert all(
        record.references == ["ISO 9001"]
        for record in result.requirements
        if record.requirement_id.startswith("TS-B.1")
    )


def test_a_clause_the_model_said_nothing_about_still_becomes_a_requirement() -> None:
    silent = StubCompletion()
    silent.readings = {}
    result = extract_requirements(a_corpus(), lambda *a, **k: PageReading(clauses=[]))

    assert len(result.requirements) == 3
    assert all(record.is_atomic and not record.references for record in result.requirements)


def test_a_reading_for_a_clause_not_on_the_page_is_discarded_and_counted() -> None:
    # A model naming a clause the page does not contain is the failure this count exists for.
    stub = StubCompletion()
    stub.extra = [ClauseReading(identifier="TS-Z.99", references=["invented"])]

    result = extract_requirements(a_corpus(), stub)

    assert result.unmatched_readings == 1
    assert "TS-Z.99" not in {record.requirement_id for record in result.requirements}


# --- the prompt and the call ------------------------------------------------------------------


def test_the_prompt_carries_the_spans_the_segmenter_selected() -> None:
    clauses = clauses_on_page(a_page())
    prompt = build_prompt(clauses)

    assert "[TS-B.1]" in prompt and "[TS-B.2]" in prompt
    assert "Page 4 of 9" not in prompt
    for clause in clauses:
        assert clause.text in prompt


def test_a_page_with_no_clauses_costs_no_call() -> None:
    stub = StubCompletion()
    corpus = a_corpus(
        a_page("A heading and some prose, numbered by nobody.", page_number=1),
        a_page("TS-B.9 The Contractor shall do it.", page_number=2),
    )

    extract_requirements(corpus, stub)

    assert len(stub.prompts) == 1


def test_the_purpose_travels_to_the_completion_function() -> None:
    # It is what the cost ledger groups spend by, so it has to arrive.
    stub = StubCompletion()

    extract_requirements(a_corpus(), stub, purpose="test.purpose")

    assert stub.purposes == ["test.purpose"]


def test_the_result_records_what_was_read_and_what_was_withheld() -> None:
    corpus = a_corpus(withhold=frozenset({"09_Matrix"}))

    result = extract_requirements(corpus, StubCompletion())

    assert result.documents_seen == {"04_Spec"}
    assert result.withheld == {"09_Matrix"}
    assert result.clauses_read == 3


def test_merge_reading_is_the_whole_decision_and_needs_no_model() -> None:
    clause = clauses_on_page(a_page())[1]
    corpus = a_corpus()

    records = merge_reading(clause, None, corpus=corpus, policy=DEFAULT_POLICY)

    assert [record.requirement_id for record in records] == ["TS-B.2"]
    assert records[0].modality is Modality.ADVISORY


# --- the hard assertion --------------------------------------------------------------------


def test_every_extracted_requirement_resolves_to_its_quoted_page_text() -> None:
    stub = StubCompletion(
        {
            "TS-B.1": ClauseReading(
                identifier="TS-B.1", atomic_parts=["Supply them.", "Commission them."]
            )
        }
    )
    corpus = a_corpus()

    result = extract_requirements(corpus, stub)
    report = verify(result.requirements, corpus)

    assert report.ok, report.describe()
    assert report.checked == len(result.requirements) == 5
    assert report.resolved == 5


def test_a_split_child_anchors_to_its_parents_span_not_to_its_own_words() -> None:
    # The child's sentence appears nowhere in the document. The anchor points at where it
    # came from, which is what keeps the assertion true of every record rather than of most.
    stub = StubCompletion(
        {"TS-B.1": ClauseReading(identifier="TS-B.1", atomic_parts=["Supply them.", "Do it."])}
    )
    corpus = a_corpus()

    result = extract_requirements(corpus, stub)
    child = next(record for record in result.requirements if record.requirement_id == "TS-B.1/1")

    assert child.text == "Supply them."
    assert child.text not in corpus.page("04_Spec", 4).text
    assert child.source.quote.startswith("TS-B.1 The Contractor shall supply")
    assert verify([child], corpus).ok


def test_a_quote_that_is_not_on_its_page_fails_and_names_the_requirement_and_the_page() -> None:
    # The positive control. Without it, "100% resolve" could mean the check does nothing.
    corpus = a_corpus()
    doctored = Requirement(
        requirement_id="TS-B.2",
        text="whatever",
        modality=Modality.ADVISORY,
        source=an_anchor(quote="a sentence that is not anywhere in this document"),
        is_atomic=True,
    )

    report = verify([doctored], corpus)

    assert not report.ok
    assert report.resolved == 0
    message = report.describe()
    assert "TS-B.2" in message
    assert "04_Spec p4" in message
    assert "does not appear on that page" in message
    with pytest.raises(AnchorVerificationError, match="TS-B.2"):
        report.raise_for_failures()


def test_a_rewrapped_quote_still_resolves() -> None:
    # A PDF breaks a clause wherever the column ends. A quote that is right must not fail
    # for being re-wrapped, which is the whole reason the comparison normalises whitespace.
    corpus = a_corpus()
    rewrapped = Requirement(
        requirement_id="TS-B.1",
        text="t",
        modality=Modality.MANDATORY,
        source=an_anchor(
            clause="TS-B.1",
            quote="TS-B.1 The Contractor shall supply and install the cameras listed at "
            "TS-B.13, and     shall\n\n  commission each one",
        ),
        is_atomic=True,
    )

    assert verify([rewrapped], corpus).ok


@pytest.mark.parametrize(
    ("anchor_kwargs", "reason"),
    [
        ({"source_id": "99_Nowhere"}, "does not contain"),
        ({"page": 99}, "does not have"),
        # Valid as an EvidenceRef — a clause is a locator — but not checkable, so it is a
        # failure here rather than a record that quietly skips verification.
        ({"page": None}, "carries no page number"),
        ({"quote": "   "}, "quotes nothing"),
    ],
)
def test_every_way_an_anchor_can_fail_is_reported_rather_than_raised(
    anchor_kwargs: dict[str, object], reason: str
) -> None:
    broken = Requirement(
        requirement_id="TS-B.2",
        text="t",
        modality=Modality.NONE,
        source=an_anchor(**anchor_kwargs),
        is_atomic=True,
    )

    report = verify([broken], a_corpus())

    assert not report.ok
    assert reason in report.describe()


def test_every_failure_is_reported_not_just_the_first() -> None:
    # One run, one list. Fixing them one at a time would take as many runs as there are.
    broken = [
        Requirement(
            requirement_id=f"TS-B.{n}",
            text="t",
            modality=Modality.NONE,
            source=an_anchor(quote=f"not on the page {n}"),
            is_atomic=True,
        )
        for n in range(1, 4)
    ]

    report = verify(broken, a_corpus())

    assert len(report.failures) == 3
    assert report.checked == 3 and report.resolved == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  a   b \n c ", "a b c"), ("a\tb", "a b"), ("", ""), ("already flat", "already flat")],
)
def test_flatten_collapses_whitespace_and_nothing_else(raw: str, expected: str) -> None:
    # Case and punctuation are left alone: a quote is the document's own words, and
    # loosening further would start excusing anchors that are not quotes at all.
    assert flatten(raw) == expected


def test_case_is_not_normalised_so_a_paraphrase_cannot_pass_as_a_quote() -> None:
    corpus = a_corpus()
    shouted = Requirement(
        requirement_id="TS-B.2",
        text="t",
        modality=Modality.NONE,
        source=an_anchor(quote="CAMERAS SHOULD BE MOUNTED AT A HEIGHT OF NOT LESS THAN 4 M."),
        is_atomic=True,
    )

    assert not verify([shouted], corpus).ok


def test_an_empty_run_verifies_vacuously_and_says_so() -> None:
    report = verify([], a_corpus())

    assert report.ok
    assert report.checked == 0
    assert "0 anchor(s) checked" in report.describe()
    report.raise_for_failures()  # returns quietly; the failing case is covered above


def test_trailing_blank_lines_are_not_part_of_a_clause() -> None:
    # A quote ending in whitespace still verifies, so this is not about correctness — it is
    # about a quote being the clause and nothing else.
    page = a_page("TS-B.1 The Contractor shall do it.\n\n   \n")

    assert clauses_on_page(page)[0].text == "TS-B.1 The Contractor shall do it."


def test_a_document_with_no_title_is_cited_by_its_id() -> None:
    # An EvidenceRef whose document is blank points nowhere a reader can follow.
    corpus = SourceCorpus(
        name="t",
        documents=[
            CorpusDocument(
                document_id="04_Spec",
                pages=[CorpusPage(document_id="04_Spec", page_number=1, text="TS-1.1 A shall.")],
            )
        ],
    )

    result = extract_requirements(corpus, StubCompletion())

    assert result.requirements[0].source.document == "04_Spec"
    assert corpus.title_of("99_Unknown") == "99_Unknown"


def test_cited_references_gathers_every_reference_across_the_set() -> None:
    stub = StubCompletion(
        {
            "TS-B.1": ClauseReading(identifier="TS-B.1", references=["ISO 9001"]),
            "TS-B.2": ClauseReading(identifier="TS-B.2", references=["ISO 9001", "SEC-201"]),
        }
    )

    result = extract_requirements(a_corpus(), stub)

    assert result.cited_references == {"ISO 9001", "SEC-201"}
