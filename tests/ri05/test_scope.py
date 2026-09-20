"""Two scopes, two questions, and the chain from a planted defect to a matchable one.

The gate's scope and detection's scope were one default for a while. `findings_run` called
`extraction_corpus(package)` with no argument and inherited three documents that had been
chosen to make a comparison against the Compliance Matrix meaningful — a good reason, and
not a reason about detection. Every recall figure that run produced was bounded by a
decision nobody had made for it.

So both are pinned here, by name and by what they measure, along with the ceiling chain they
feed. The counts come from the repo's own loader and segmenter over the committed PDFs; they
are offline, exact, and they are what a widening or a narrowing has to move on purpose.

Deliberately does not: make a model call, assert a recall figure, or check that a detector
finds anything. What a detector finds is a measurement; what it is *shown* is configuration,
and this is about the configuration.
"""

from pathlib import Path

import pytest

from req_core.clauses import clauses_on_page, segment
from req_core.corpus import SourceCorpus
from ri05_tender.eval.findings_run import DETECTOR_TO_FINDING_TYPE, EMITTED_OUTPUTS
from ri05_tender.eval.loader import GoldSet, load_gold
from ri05_tender.eval.metrics import ceiling_chain
from ri05_tender.extract.config import KESSLER_POINT_CLAUSE_STYLE
from ri05_tender.scope import (
    DETECTION_DOCUMENT_IDS,
    GATE_DOCUMENT_IDS,
    MATRIX_DOCUMENT_ID,
    NO_CLAUSE_DOCUMENT_IDS,
    ScopeError,
    detection_corpus,
    document_numbers,
    gate_corpus,
)
from ri05_tender.tender.loader import load_tender
from ri05_tender.tender.models import TenderPackage

KESSLER_POINT = Path("data/tenders/kessler_point")

# Clauses the segmenter finds in each scope, and the pages carrying them. One model call is
# made per clause-bearing page per pass, so the page counts are the call counts: 14 + 14 for
# the gate's three documents, 44 + 44 for detection's ten.
GATE_CLAUSES = 301
GATE_CLAUSE_PAGES = 14
DETECTION_CLAUSES = 691
DETECTION_CLAUSE_PAGES = 44

# Per document, because a total hides a document that stopped segmenting. These are what the
# committed PDFs produce today.
CLAUSES_BY_DOCUMENT = {
    "01_Instructions_to_Bidders": 66,
    "02_Scope_of_Work": 75,
    "03_General_Conditions_of_Contract": 96,
    "04_Technical_Specification": 201,
    "05_Integration_and_Interface_Specification": 43,
    "06_Cybersecurity_and_Information_Security": 57,
    "10_Annex_A_Site_and_Environmental_Conditions": 39,
    "11_Annex_B_Schedule_of_Standards": 10,
    "12_Annex_C_SSI_Handling": 30,
    "13_Drawing_Register": 74,
}

# The chain, link by link, over the detection scope. Each number is what survives the
# condition above it, and the gap it leaves is a different kind of problem: a scope
# decision, then the answer key's own references, then unbuilt detectors, then unbuilt
# outputs. The last link needs a run and is not pinned here.
DETECTION_CHAIN = [
    ("scored items", 186),
    ("in a document the detection corpus reads", 150),
    ("carrying a reference the segmenter produces as a clause id", 149),
    ("addressable by a registered detector family", 32),
    ("not blocked on an output no detector yet emits", 10),
]

# The same chain over the gate's three documents: what the run was measuring before the
# widening. The ceiling was 6 of 186, and no recall figure printed beside it said so.
GATE_CEILING = 6


@pytest.fixture(scope="module")
def package() -> TenderPackage:
    return load_tender(KESSLER_POINT)


@pytest.fixture(scope="module")
def gold(package: TenderPackage) -> GoldSet:
    return load_gold(package)


def _clause_pages(corpus: SourceCorpus) -> int:
    """Pages carrying at least one clause — one model call each, per pass."""
    return sum(
        1
        for document in corpus.documents
        for page in document.pages
        if clauses_on_page(page, style=KESSLER_POINT_CLAUSE_STYLE)
    )


def _clauses(corpus: SourceCorpus) -> list[str]:
    return [clause.identifier for clause in segment(corpus, style=KESSLER_POINT_CLAUSE_STYLE)]


def test_the_two_scopes_are_different_and_named(package: TenderPackage) -> None:
    """The bug was one scope doing two jobs, so the test is that there are two."""
    assert set(GATE_DOCUMENT_IDS) < set(DETECTION_DOCUMENT_IDS)
    assert gate_corpus(package).document_ids == set(GATE_DOCUMENT_IDS)
    assert detection_corpus(package).document_ids == set(DETECTION_DOCUMENT_IDS)


def test_the_matrix_is_withheld_from_both(package: TenderPackage) -> None:
    """Not left out — withheld, which `SourceCorpus` enforces rather than remembers."""
    for corpus in (gate_corpus(package), detection_corpus(package)):
        assert MATRIX_DOCUMENT_ID not in corpus.document_ids
        assert corpus.withheld == {MATRIX_DOCUMENT_ID}


def test_including_the_withheld_matrix_is_refused(package: TenderPackage) -> None:
    with pytest.raises(ScopeError, match=MATRIX_DOCUMENT_ID):
        detection_corpus(package, include=[*DETECTION_DOCUMENT_IDS, MATRIX_DOCUMENT_ID])


def test_the_documents_left_out_of_both_scopes_carry_no_clauses(package: TenderPackage) -> None:
    """The reason they are excluded, checked rather than asserted in a comment.

    Sending a worksheet that segments to nothing would pay for a call that returns nothing.
    If either of these ever starts carrying clauses, this fails and the exclusion has to be
    argued again rather than inherited.
    """
    for document_id in NO_CLAUSE_DOCUMENT_IDS:
        document = package.document(document_id)
        assert document is not None
        found = sum(
            len(clauses_on_page(page, style=KESSLER_POINT_CLAUSE_STYLE)) for page in document.pages
        )
        assert found == 0, f"{document_id} now segments {found} clause(s)"


def test_what_each_scope_costs_in_pages_and_clauses(package: TenderPackage) -> None:
    """Pages are calls. A change here is a change to the bill for every labelled run."""
    gate, detection = gate_corpus(package), detection_corpus(package)

    assert len(_clauses(gate)) == GATE_CLAUSES
    assert _clause_pages(gate) == GATE_CLAUSE_PAGES
    assert len(_clauses(detection)) == DETECTION_CLAUSES
    assert _clause_pages(detection) == DETECTION_CLAUSE_PAGES


def test_the_clause_count_of_every_document_in_the_detection_scope(
    package: TenderPackage,
) -> None:
    """Per document, so one that stops segmenting cannot hide inside the total."""
    found = {
        document.document_id: sum(
            len(clauses_on_page(page, style=KESSLER_POINT_CLAUSE_STYLE)) for page in document.pages
        )
        for document in detection_corpus(package).documents
    }
    assert found == CLAUSES_BY_DOCUMENT


def test_document_numbers_are_how_the_gold_set_names_a_document() -> None:
    assert document_numbers(DETECTION_DOCUMENT_IDS) == frozenset({1, 2, 3, 4, 5, 6, 10, 11, 12, 13})
    assert document_numbers(GATE_DOCUMENT_IDS) == frozenset({4, 5, 6})
    with pytest.raises(ScopeError, match="document number"):
        document_numbers(["Annex_A"])


def test_the_ceiling_chain_over_the_detection_scope(package: TenderPackage, gold: GoldSet) -> None:
    """Every condition between a planted defect and a finding that could match it.

    Pinned link by link rather than as one ceiling, because the whole point of the chain is
    that the four causes are different. A ceiling that moved would otherwise say only that
    something changed.
    """
    corpus = detection_corpus(package)
    chain = ceiling_chain(
        gold.scored,
        corpus_documents=document_numbers(corpus.document_ids),
        clause_identifiers=_clauses(corpus),
        emitted_finding_types=frozenset(DETECTOR_TO_FINDING_TYPE.values()),
        emitted_outputs=EMITTED_OUTPUTS,
    )

    assert [(link.name, link.count) for link in chain.links] == DETECTION_CHAIN
    assert chain.scored_total == 186
    assert chain.ceiling == 10
    # Without a run there is nothing to say about the threshold, and the chain says nothing
    # rather than implying it cost nothing.
    assert not any(link.measured_after_the_run for link in chain.links)


def test_the_widening_is_what_moved_the_ceiling(package: TenderPackage, gold: GoldSet) -> None:
    """The same chain over the gate's three documents, which is what the run used to read.

    6 of 186 — and the run printed a recall figure over 186 with no ceiling beside it. The
    widening does not make a detector better; it stops the measurement being about a corpus
    nobody chose for it.
    """
    corpus = gate_corpus(package)
    chain = ceiling_chain(
        gold.scored,
        corpus_documents=document_numbers(corpus.document_ids),
        clause_identifiers=_clauses(corpus),
        emitted_finding_types=frozenset(DETECTOR_TO_FINDING_TYPE.values()),
        emitted_outputs=EMITTED_OUTPUTS,
    )
    assert chain.ceiling == GATE_CEILING


def test_the_abstention_link_appears_only_when_a_run_supplied_one(gold: GoldSet) -> None:
    """A link that cannot be computed from the configuration is marked as such."""
    chain = ceiling_chain(
        gold.scored,
        corpus_documents={4},
        clause_identifiers=["TS-B.25"],
        emitted_finding_types=frozenset(DETECTOR_TO_FINDING_TYPE.values()),
        emitted_outputs=EMITTED_OUTPUTS,
        withheld_gold_ids=[],
    )
    last = chain.links[-1]
    assert last.name == "not withheld by the abstention threshold"
    assert last.measured_after_the_run
