# req_core

Domain-agnostic requirement extraction: unstructured source text in, typed requirement
structures out. Imported as `req_core`.

`req_core` decides *what a requirement is*, not what any particular industry does with
one. Domain vocabulary, scoring and response generation live in the applications. Nothing
about tenders, ports or any one client appears here: the package is written to be picked up
by the next project unchanged.

```python
from req_core.corpus import corpus_from
from req_core.extraction import extract_requirements
from req_core.anchors import verify

corpus = corpus_from(my_documents, name="acme-rfp", include={"spec"}, withhold={"answers"})
result = extract_requirements(corpus, complete, policy=house_style)
verify(result.requirements, corpus).raise_for_failures()
```

## What the model does, and only this

| | Who decides |
|---|---|
| Split a compound clause into atomic obligations | **the model** |
| Identify the clauses, standards and drawings a clause cites | **the model** |
| Where each clause starts and stops | Python (`clauses`) |
| What its modality is | Python (`policy`), from a configured word list |
| The source anchor, and the span it quotes | Python — **a model is never asked for a quote** |
| A split child's identifier | Python (`contracts.child_id`) |
| Every count in the result | Python, from the records |

Two model answers are taken as suggestions and checked. A reading naming a clause that is
not on the page is discarded and counted in `unmatched_readings`. And when a clause is *not*
split, the record keeps the document's wording rather than the model's echo of it — a
paraphrase accepted there would be a silent edit to the source.

## Modality is configuration, not a constant

Everyone knows `shall` is mandatory. Almost everyone assumes `will` is too. The tender this
package was first used on states, in a table on page 1, that `will` is **optional** and
`should` is treated the same way. A hardcoded mapping would have inverted the strongest
distinction in the document and reported it with a straight face — every clause would still
have come out with a modality, just the wrong one.

So `ModalityPolicy` is an object a caller supplies, carrying the words, a name, and where
the convention was stated. `DEFAULT_POLICY` is the ordinary English reading and is a
default, not a truth. `ClauseStyle` is configuration for the same reason: a bare `4.7.1` is
a clause in one document and a section heading in another.

## The way in is a protocol; everything after it is a model

`PageLike` and `DocumentLike` describe "a document with pages" structurally — anything
carrying `document_id` / `page_number` / `text` satisfies them, with no base class to inherit
and no import on the caller's side. `corpus_from` turns any of those into a `SourceCorpus`,
which is what every function here takes. The protocol is the doorway; the Pydantic model is
what crosses every module boundary after it.

## Withholding is a type, not a rule to remember

A corpus names the documents it deliberately does **not** carry, and refuses to be built if
one of them is present. "Do not let extraction read the answer sheet" stops being something
a caller has to keep in mind and becomes a state that cannot be represented. `documents_seen`
and `withheld` then travel on the result, so a coverage figure can be checked after the fact
against what the pass was actually allowed to read.

## The one assertion

> **100% of extracted requirements resolve to a source anchor whose page text actually
> contains the quoted span.**

Not a target — `anchors.verify`, run in the tests on every pass, offline and live. It
normalises whitespace before comparing, because a PDF wraps a clause wherever the column
ends and a correct quote must not fail for being re-wrapped. It reports **every** failure,
each naming the requirement id, the document and the page, rather than raising on the first.

It is also made structurally hard to fail: a quote is always text `clauses` selected, never
text a model produced. A split child's own sentence appears nowhere in the document, so its
anchor quotes the parent clause it came from — which is where a reader would go to check it.

A requirement whose anchor names a page that does not contain the quote is worse than no
requirement at all. It survives review, because checking it means opening the document, and
nobody opens the document for the one that looks fine.
