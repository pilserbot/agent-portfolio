"""How many gold items can be reached at all, pinned against the committed documents.

A gold item anchors when one of its `refs` equals a clause identifier the segmenter finds
in the tender. An item that anchors nowhere cannot be matched by any finding, however good
the detectors get: it is a row in the answer key that scores zero by construction, and it
does so silently, because a reference that names no clause looks exactly like a reference
to a clause nobody found.

That is what happened to 23 items. The gold-set builder's identifier regex could not match
the `XX-Y.NN` shape, so `'TS-B.36 + TS-B.38'` was stored as one reference equal to no
clause id at all. `refs` are now re-derived from `ref_raw` with the pattern `req_core`
already uses, and the counts below are what that re-derivation produced.

The numbers are pinned so the next ref change has to move them on purpose. A count that
falls is an answer key that quietly stopped asking for something.

Deliberately does not: check that an item's refs are the *right* clauses, or that a clause
it anchors to contains the planted defect. Both are readings a person makes against the
tender text; this only asks whether the reference names something that exists.
"""

import re
from pathlib import Path

import pytest

from req_core.clauses import segment
from req_core.corpus import SourceCorpus, corpus_from
from ri05_tender.eval.loader import GoldSet, load_gold
from ri05_tender.eval.models import GoldItem
from ri05_tender.extract.gate import extraction_corpus
from ri05_tender.tender.loader import load_tender
from ri05_tender.tender.models import TenderPackage

KESSLER_POINT = Path("data/tenders/kessler_point")

# The clause identifier shape `req_core.clauses` defines, without its line-start anchor: the
# same pattern the re-derivation read `ref_raw` with, so this guard and that rewrite cannot
# disagree about what counts as a reference.
CLAUSE_REFERENCE = re.compile(r"\b([A-Z]{1,4}-[A-Z]?\.?\d+(?:\.\d+)*)")

# Measured against the committed PDFs with the repo's own loader and segmenter, after the
# re-derivation. Before it: 49 and 138. The gap between the two figures is the corpus, not
# the answer key — extraction reads Documents 4, 5 and 6, and the other ten documents hold
# clauses the gold set legitimately references.
ANCHORING_IN_EXTRACTION_CORPUS = 66
ANCHORING_IN_EVERY_DOCUMENT = 152

# The nine demo-set items all anchor somewhere in the package. Five of them sit in the three
# documents extraction reads; the other four are in Documents 2 and 3, so no detector can
# reach them until the corpus widens, which is a scope decision and not this test's.
DEMO_ITEMS = 9
DEMO_ANCHORING_IN_EXTRACTION_CORPUS = 5

# Items whose `refs` the re-derivation changed. Each keeps its previous value in
# `refs_prior`, so the claim the answer key used to make is still on the record.
ITEMS_WITH_REDERIVED_REFS = 23

# Items whose references hold no clause-shaped token at all, by the document they sit in.
# Document 7's are the bill of quantities' own row codes (`A.07`, `C.08, D.07`) and 8's and
# 9's are prose — a spreadsheet tab, "Summary rows 24-27", "Guideline 5". `ref_raw` gave the
# re-derivation nothing to take, so it left them alone: deciding which clause a tab name
# means is a reading of the tender, not a regex, and not this change's to make.
#
# Only the one in Document 5 costs anything today, because Documents 7, 8 and 9 are outside
# the extraction corpus and unreachable for that reason anyway.
ITEMS_WITHOUT_A_CLAUSE_REFERENCE = {5: 1, 7: 27, 8: 4, 9: 2}


@pytest.fixture(scope="module")
def package() -> TenderPackage:
    return load_tender(KESSLER_POINT)


@pytest.fixture(scope="module")
def gold(package: TenderPackage) -> GoldSet:
    return load_gold(package)


def clause_identifiers(corpus: SourceCorpus) -> set[str]:
    """Every clause identifier the segmenter finds, upper-cased for comparison."""
    return {clause.identifier.upper() for clause in segment(corpus)}


def anchoring(items: list[GoldItem], identifiers: set[str]) -> list[GoldItem]:
    """The items carrying at least one reference that names a clause in `identifiers`."""
    return [item for item in items if any(ref.upper() in identifiers for ref in item.refs)]


@pytest.fixture(scope="module")
def extraction_identifiers(package: TenderPackage) -> set[str]:
    return clause_identifiers(extraction_corpus(package))


@pytest.fixture(scope="module")
def every_identifier(package: TenderPackage) -> set[str]:
    return clause_identifiers(
        corpus_from(
            package.documents,
            name=package.name,
            include=[document.document_id for document in package.documents],
        )
    )


def test_the_scored_items_anchoring_in_the_extraction_corpus(
    gold: GoldSet, extraction_identifiers: set[str]
) -> None:
    """The ceiling on what any detector over Documents 4, 5 and 6 could ever match."""
    reached = anchoring(gold.scored, extraction_identifiers)
    assert len(reached) == ANCHORING_IN_EXTRACTION_CORPUS, (
        f"{len(reached)} of {len(gold.scored)} scored items anchor in the extraction corpus, "
        f"not {ANCHORING_IN_EXTRACTION_CORPUS}. An item that anchors nowhere cannot be "
        f"matched by any finding, so a fall here is recall lost before a detector runs."
    )


def test_the_scored_items_anchoring_anywhere_in_the_package(
    gold: GoldSet, every_identifier: set[str]
) -> None:
    """The same count against all 13 documents, which the corpus decision does not bound."""
    reached = anchoring(gold.scored, every_identifier)
    assert len(reached) == ANCHORING_IN_EVERY_DOCUMENT, (
        f"{len(reached)} of {len(gold.scored)} scored items name a clause that exists "
        f"somewhere in the package, not {ANCHORING_IN_EVERY_DOCUMENT}."
    )


def test_every_demo_item_names_a_clause_that_exists(
    gold: GoldSet, every_identifier: set[str], extraction_identifiers: set[str]
) -> None:
    """The demo set is what gets shown, so an unanchorable item there is the worst kind."""
    demo = [item for item in gold.scored if item.demo_set]
    assert len(demo) == DEMO_ITEMS
    assert len(anchoring(demo, every_identifier)) == DEMO_ITEMS
    assert len(anchoring(demo, extraction_identifiers)) == DEMO_ANCHORING_IN_EXTRACTION_CORPUS


def test_the_re_derivation_kept_what_it_replaced(gold: GoldSet) -> None:
    """`refs_prior` is the receipt: a changed reference says what it used to claim."""
    changed = [item for item in gold.all_items if item.refs_prior is not None]
    assert len(changed) == ITEMS_WITH_REDERIVED_REFS
    for item in changed:
        assert item.refs_prior != item.refs, item.id
        assert item.refs, item.id


def test_every_clause_shaped_reference_stands_alone(gold: GoldSet) -> None:
    """The failure shape itself: one reference holding two clause ids and a connector.

    `['TS-B.36 + TS-B.38']` matched nothing because it was never a clause id. The guard is
    that a reference containing a clause identifier must BE that identifier and nothing
    else, which is what the re-derivation produces and what a hand edit would break.
    """
    unsplit = [
        (item.id, ref)
        for item in gold.all_items
        for ref in item.refs
        if (found := CLAUSE_REFERENCE.findall(ref)) and found != [ref.strip()]
    ]
    assert not unsplit, (
        f"references holding a clause id plus something else: {unsplit}. A compound token "
        f"equals no clause identifier, so the item silently anchors nowhere."
    )


def test_the_references_that_name_no_clause_are_the_known_ones(gold: GoldSet) -> None:
    """What the re-derivation could not reach, pinned where it can be read.

    These carry a spreadsheet row code or prose rather than a clause identifier, so there
    was no token to take. Pinned rather than fixed, and pinned by document because that is
    what says whether an unanchorable item costs any reachable recall.
    """
    by_document: dict[int, int] = {}
    for item in gold.all_items:
        if not any(CLAUSE_REFERENCE.fullmatch(ref.strip()) for ref in item.refs):
            by_document[item.document] = by_document.get(item.document, 0) + 1

    assert by_document == ITEMS_WITHOUT_A_CLAUSE_REFERENCE, (
        f"items whose references name no clause, by document: {by_document}. One added here "
        f"is an item that cannot anchor; one removed is a reading somebody made, and the "
        f"reading belongs in the commit that makes it."
    )
