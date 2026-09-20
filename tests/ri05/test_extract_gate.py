"""The RI-05 extraction wiring and the gate, offline against the committed tender.

No network, no model, no key: the pass runs with a stub completion function, which is enough
because everything the gate compares is deterministic. The clause set comes from segmentation
and the matrix set from a workbook — the model contributes splits and citations, neither of
which is in the comparison. That is worth knowing about the gate rather than glossing: it
measures whether the reading layer found the right requirements, and a model that invented a
clause would show up in `unmatched_readings`, not here.

What is pinned: Document 9 cannot be reached by extraction, in two independent ways; the
tender's own ITB-2.1 convention gives different answers from the default one; and every
anchor over the real 301 clauses resolves to the page it cites.
"""

import re
from pathlib import Path

import pytest

from req_core.anchors import verify
from req_core.contracts import ExtractionResult
from req_core.corpus import CorpusError
from req_core.extraction import ClauseReading, PageReading, extract_requirements
from req_core.policy import DEFAULT_POLICY, Modality
from ri05_tender.extract.config import ITB_2_1_POLICY, KESSLER_POINT_CLAUSE_STYLE
from ri05_tender.extract.gate import (
    GATE_DOCUMENT_IDS,
    MATRIX_DOCUMENT_ID,
    GateError,
    compare,
    gate_corpus,
    matrix_clause_ids,
    render_markdown,
    verify_scope,
)
from ri05_tender.scope import ScopeError
from ri05_tender.tender.loader import load_tender
from ri05_tender.tender.models import TenderPackage

KESSLER_POINT = Path("data/tenders/kessler_point")

# Measured from the committed package, not chosen. 301 is also what the package's own index
# claims for Documents 4, 5 and 6, which is the first evidence the segmenter is right.
EXPECTED_CLAUSES = 301
EXPECTED_MATRIX_ROWS = 299


def no_op_completion(prompt: str, schema: type, *, purpose: str) -> PageReading:
    """Echo every clause back unsplit and citing nothing.

    Deliberately inert. The gate does not depend on what a model says, and this is how that
    claim is made checkable rather than asserted.
    """
    return PageReading(
        clauses=[
            ClauseReading(identifier=identifier)
            for identifier in re.findall(r"^\[([^\]]+)\]", prompt, re.MULTILINE)
        ]
    )


@pytest.fixture(scope="module")
def package() -> TenderPackage:
    """The committed tender, loaded once: reading 48 real PDF pages is the slow part."""
    return load_tender(KESSLER_POINT)


@pytest.fixture(scope="module")
def extracted(package: TenderPackage) -> ExtractionResult:
    """One inert extraction pass over Documents 4, 5 and 6."""
    return extract_requirements(
        gate_corpus(package),
        no_op_completion,
        policy=ITB_2_1_POLICY,
        style=KESSLER_POINT_CLAUSE_STYLE,
    )


# --- the matrix cannot be reached -----------------------------------------------------------


def test_the_gate_corpus_holds_only_the_specifications(package: TenderPackage) -> None:
    corpus = gate_corpus(package)

    assert corpus.document_ids == set(GATE_DOCUMENT_IDS)
    assert MATRIX_DOCUMENT_ID not in corpus.document_ids
    assert corpus.withheld == {MATRIX_DOCUMENT_ID}


def test_including_the_matrix_in_the_corpus_is_refused_by_the_type(package: TenderPackage) -> None:
    # The first of the two enforcements, and the load-bearing one: the mistake is not
    # representable, so it cannot be made by forgetting. `ScopeError` and not `GateError`
    # because building a corpus is `scope`'s job now — the gate is one of two callers, and
    # the refusal belongs to whichever scope was asked for.
    with pytest.raises(ScopeError) as caught:
        gate_corpus(package, include=[*GATE_DOCUMENT_IDS, MATRIX_DOCUMENT_ID])

    assert MATRIX_DOCUMENT_ID in str(caught.value)


def test_the_corpus_refusal_is_the_packages_own_not_this_modules(package: TenderPackage) -> None:
    # `ScopeError` wraps it for a caller's convenience; the rule lives in req_core, where any
    # other project gets it too.
    with pytest.raises(CorpusError):
        from req_core.corpus import corpus_from

        corpus_from(
            package.documents,
            name="x",
            include=[*GATE_DOCUMENT_IDS, MATRIX_DOCUMENT_ID],
            withhold=[MATRIX_DOCUMENT_ID],
        )


def test_nothing_extracted_cites_a_document_outside_the_corpus(
    package: TenderPackage, extracted: ExtractionResult
) -> None:
    # The second enforcement, on the way out: every anchor names a document the corpus held.
    verify_scope(extracted, gate_corpus(package))

    cited = {requirement.source.source_id for requirement in extracted.requirements}
    assert cited == set(GATE_DOCUMENT_IDS)
    assert extracted.withheld == {MATRIX_DOCUMENT_ID}


def test_verify_scope_catches_an_anchor_into_a_document_the_corpus_never_held(
    package: TenderPackage, extracted: ExtractionResult
) -> None:
    # The way out, checked separately from the way in: a record can claim to have seen only
    # the right documents and still carry an anchor that does not.
    corpus = gate_corpus(package)
    forged = extracted.requirements[0].model_copy(
        update={
            "source": extracted.requirements[0].source.model_copy(
                update={"source_id": MATRIX_DOCUMENT_ID}
            )
        }
    )
    sneaky = ExtractionResult(
        corpus_name="kessler_point",
        documents_seen=corpus.document_ids,
        policy_name="x",
        requirements=[forged],
        clauses_read=1,
    )

    with pytest.raises(GateError, match="cite document"):
        verify_scope(sneaky, corpus)


def test_verify_scope_catches_a_result_that_strayed(package: TenderPackage) -> None:
    corpus = gate_corpus(package)
    strayed = ExtractionResult(
        corpus_name="kessler_point",
        documents_seen=frozenset({*GATE_DOCUMENT_IDS, MATRIX_DOCUMENT_ID}),
        policy_name="x",
        clauses_read=0,
    )

    with pytest.raises(GateError, match=MATRIX_DOCUMENT_ID):
        verify_scope(strayed, corpus)


# --- segmentation over the real documents -----------------------------------------------------


def test_the_segmenter_finds_the_number_of_clauses_the_package_claims(
    extracted: ExtractionResult,
) -> None:
    # 00_INDEX.md states 301 numbered requirements in Documents 4, 5 and 6. Reaching that
    # from the text is the first evidence the clause style is right for this tender.
    assert extracted.clauses_read == EXPECTED_CLAUSES
    assert len(extracted.clause_ids) == EXPECTED_CLAUSES


def test_the_default_clause_style_would_have_been_wrong_here(package: TenderPackage) -> None:
    # It also matches bare dotted numbers, which in these documents are section headings
    # ("5.1 General") — and several collide across documents. This is why the style is
    # configuration rather than a constant in req_core.
    loose = extract_requirements(gate_corpus(package), no_op_completion)

    assert loose.clauses_read > EXPECTED_CLAUSES
    assert len(loose.clause_ids) < loose.clauses_read  # ids collided across documents


def test_truncated_clauses_are_counted_rather_than_silently_shortened(
    extracted: ExtractionResult,
) -> None:
    # Over-reports on purpose: the flag fires on any clause whose last line lacks terminal
    # punctuation, which catches a real page-wrap and also a trailing table row. Over-
    # reporting a shortened quote is the safe direction.
    assert 0 < extracted.clauses_truncated <= 10


def test_an_inert_pass_invents_nothing(extracted: ExtractionResult) -> None:
    assert extracted.unmatched_readings == 0
    assert extracted.split_count == 0
    assert extracted.atomic_count == len(extracted.requirements) == EXPECTED_CLAUSES


# --- the tender's own modality convention ------------------------------------------------------


def test_itb_2_1_is_quoted_not_assumed() -> None:
    assert ITB_2_1_POLICY.source.startswith("ITB-2.1")
    assert ITB_2_1_POLICY.mandatory == {"shall", "must"}
    assert ITB_2_1_POLICY.optional == {"will", "should"}
    assert ITB_2_1_POLICY.advisory == {"may"}


def test_the_tenders_convention_disagrees_with_the_default_on_real_clauses(
    extracted: ExtractionResult,
) -> None:
    """The case the configuration exists for, measured on the document rather than argued.

    Under the default policy `will` is mandatory and `may` is optional. ITB-2.1 says `will`
    is optional and `may` is a scored differentiator. Read the same clauses both ways and the
    disagreement is not hypothetical.
    """
    disagreed = [
        requirement
        for requirement in extracted.requirements
        if DEFAULT_POLICY.classify(requirement.text).modality is not requirement.modality
    ]

    assert disagreed, (
        "no clause in Documents 4/5/6 is read differently under ITB-2.1 than under the "
        "default convention. Either the policy or the documents have changed; a config "
        "object that never changes an answer is not earning its place."
    )
    # Every disagreement is a clause whose first modal word is one of the three ITB-2.1
    # re-defines. Nothing else can differ, because the two policies agree on "shall".
    assert {requirement.modality_trigger for requirement in disagreed} <= {
        "will",
        "should",
        "may",
    }


def test_shall_is_mandatory_under_both_conventions(extracted: ExtractionResult) -> None:
    shall = [r for r in extracted.requirements if r.modality_trigger == "shall"]

    assert shall
    assert all(requirement.modality is Modality.MANDATORY for requirement in shall)


# --- the hard assertion, over the real documents -------------------------------------------------


def test_every_requirement_extracted_from_the_real_tender_resolves_to_its_page(
    package: TenderPackage, extracted: ExtractionResult
) -> None:
    """The assertion the package rests on, on 301 real clauses across 16 real PDF pages."""
    report = verify(extracted.requirements, gate_corpus(package))

    assert report.ok, report.describe()
    assert report.checked == report.resolved == EXPECTED_CLAUSES


# --- the matrix side, and the comparison ----------------------------------------------------------


def test_the_matrix_rows_are_read_deterministically(package: TenderPackage) -> None:
    rows = matrix_clause_ids(package)

    assert len(rows) == EXPECTED_MATRIX_ROWS
    assert "TS-B.1" in rows
    # The sheet's own column guide prints a literal "TS-x.x" example, and the bidder block
    # prints labels. Neither is a requirement and neither may be counted as one.
    assert "TS-x.x" not in rows
    assert all(re.fullmatch(r"[A-Z]{1,4}-[A-Z]?\.?\d+(?:\.\d+)*", row) for row in rows)


def test_a_package_with_no_matrix_says_so_rather_than_comparing_against_nothing(
    package: TenderPackage,
) -> None:
    with pytest.raises(GateError, match="nothing to"):
        matrix_clause_ids(package, document_id="99_Not_A_Document")


def test_the_comparison_reports_both_directions(
    package: TenderPackage, extracted: ExtractionResult
) -> None:
    result = compare(extracted, package)

    assert result.extracted_count == EXPECTED_CLAUSES
    assert result.matrix_row_count == EXPECTED_MATRIX_ROWS
    # Both lists are reported whatever is in them. The gate never emits one without the
    # other, because each names a different defect and quoting one alone chooses a story.
    assert isinstance(result.absent_from_matrix, list)
    assert isinstance(result.absent_from_documents, list)
    assert result.agreed == result.extracted_count - len(result.absent_from_matrix)


def test_the_comparison_is_a_set_difference_and_nothing_more(
    package: TenderPackage, extracted: ExtractionResult
) -> None:
    # Re-derived here by hand from the two sides, so the module cannot quietly filter one.
    result = compare(extracted, package)
    rows = frozenset(matrix_clause_ids(package))

    assert set(result.absent_from_matrix) == set(extracted.clause_ids) - rows
    assert set(result.absent_from_documents) == rows - set(extracted.clause_ids)


def test_split_children_are_not_compared_against_the_matrix(package: TenderPackage) -> None:
    # The matrix indexes what the document printed; a child is this package's subdivision of
    # it. Comparing children would report every split as a missing row.
    def splitting(prompt: str, schema: type, *, purpose: str) -> PageReading:
        return PageReading(
            clauses=[
                ClauseReading(identifier=identifier, atomic_parts=["one part", "another part"])
                for identifier in re.findall(r"^\[([^\]]+)\]", prompt, re.MULTILINE)
            ]
        )

    split = extract_requirements(
        gate_corpus(package),
        splitting,
        policy=ITB_2_1_POLICY,
        style=KESSLER_POINT_CLAUSE_STYLE,
    )
    result = compare(split, package)

    assert len(split.requirements) == EXPECTED_CLAUSES * 3
    assert result.extracted_count == EXPECTED_CLAUSES
    assert not any("/" in name for name in result.absent_from_matrix)


def test_the_report_names_every_item_on_both_lists(
    package: TenderPackage, extracted: ExtractionResult
) -> None:
    result = compare(extracted, package)

    rendered = render_markdown(result, extracted)

    assert MATRIX_DOCUMENT_ID in rendered  # what was withheld is stated on the report
    assert ITB_2_1_POLICY.name in rendered
    for name in [*result.absent_from_matrix, *result.absent_from_documents]:
        assert f"`{name}`" in rendered
    assert "does not say which side is wrong" in rendered


def test_the_report_says_none_rather_than_printing_an_empty_list() -> None:
    empty = ExtractionResult(corpus_name="x", policy_name="p", clauses_read=0)
    from ri05_tender.extract.gate import MatrixComparison

    rendered = render_markdown(
        MatrixComparison(
            tender_name="x",
            scope=list(GATE_DOCUMENT_IDS),
            extracted_count=0,
            matrix_row_count=0,
            matrix_distinct_count=0,
        ),
        empty,
    )

    assert "_none_" in rendered
    assert "Withheld **nothing**" in rendered


# --- what the gate actually found, pinned --------------------------------------------------

# Measured on the committed package and checked by hand against the pages, one at a time.
# Pinned because each is a statement about this tender that a future change should have to
# re-justify rather than quietly revise.
ABSENT_FROM_MATRIX = ["TS-1.5", "TS-B.25", "TS-E.4"]
ABSENT_FROM_DOCUMENTS = ["TS-B.72"]


def test_the_gate_finds_three_requirements_the_matrix_omits(
    package: TenderPackage, extracted: ExtractionResult
) -> None:
    """Three real clauses the bidder is never asked to respond to.

    Each was opened and read before this list was written, so the reading is not "extraction
    invented three clauses":

    - **TS-1.5** (Document 4 p2) — a declaration of manufacturing origin for every proposed
      item, down to the image sensor and video encoder. This is the clause that makes
      Section 889 enforceable, and the matrix does not ask for it.
    - **TS-B.25** (Document 4 p4) — not less than eighty pixels per metre at the far edge of
      each detection zone. The hardest requirement in the package to demonstrate.
    - **TS-E.4** (Document 4 p8) — a crash-rated gate cycle of not more than ten seconds.

    A bidder completing the matrix in good faith answers 298 rows and has still not been
    asked about any of these three.
    """
    comparison = compare(extracted, package)

    assert comparison.absent_from_matrix == ABSENT_FROM_MATRIX, (
        f"the gate now reports {comparison.absent_from_matrix} as present in Documents 4/5/6 "
        f"and absent from the matrix, where {ABSENT_FROM_MATRIX} was measured and read by "
        f"hand. Open the new ones on the page before changing this list: the difference is "
        f"either a real defect in the tender or a bug in segmentation, and they look the same "
        f"from here."
    )
    # Each one really is in a specification, which is what makes it the tender's problem.
    for clause_id in ABSENT_FROM_MATRIX:
        assert clause_id in extracted.clause_ids


def test_the_gate_finds_one_matrix_row_pointing_at_no_clause(
    package: TenderPackage, extracted: ExtractionResult
) -> None:
    """A row the bidder must answer that points at a requirement in no specification.

    **TS-B.72** — "automatic recognition of container numbers at the pre-gate, with an
    accuracy of not less than 98%" — appears in the Compliance Matrix and nowhere else in the
    thirteen documents. A bidder must enter YES or NO against a requirement whose text exists
    only on the form they are filling in.
    """
    comparison = compare(extracted, package)

    assert comparison.absent_from_documents == ABSENT_FROM_DOCUMENTS, (
        f"the gate now reports {comparison.absent_from_documents} as indexed by the matrix and "
        f"absent from Documents 4/5/6, where {ABSENT_FROM_DOCUMENTS} was measured. A new entry "
        f"means either the matrix changed or extraction started missing clauses."
    )
    assert "TS-B.72" not in extracted.clause_ids


def test_the_two_sides_account_for_each_other_exactly(
    package: TenderPackage, extracted: ExtractionResult
) -> None:
    # 301 clauses = 298 both sides agree on + 3 the matrix omits.
    # 299 matrix rows = 298 agreed + 1 phantom.
    # Arithmetic, not a claim: if either side gained an item nobody accounted for, this fails.
    comparison = compare(extracted, package)

    assert comparison.agreed == 298
    assert comparison.extracted_count == comparison.agreed + len(comparison.absent_from_matrix)
    assert comparison.matrix_distinct_count == comparison.agreed + len(
        comparison.absent_from_documents
    )
    assert not comparison.duplicated_matrix_rows
    assert not comparison.is_clean  # there is something to read, and it is not extraction's
