# RI-05 — Gold Set Labelling and Scoring Contract

**Step 4.4 · first pass · 13 September 2026 — amended at rev 3, 14 September 2026**

The gold set is the answer key. This document says what a label means, how an emitted finding is matched to a label, and how the KPIs are computed from those matches. It is written before any measurement is taken, so the numbers cannot be defined after the fact to suit the result.

> **Amended at rev 3.** The IMPLICIT class was wrong and has been split into UNSTATED (32) and DISPLACED (40) — the test is whether **an obligation is written anywhere**, not whether a requirement sentence exists. The claim in §1 and §4 that a "shall" search recovers none of the 72 was asserted here, never measured, and is false — a "shall" grep recovered 20 of them. Every superseded passage below is struck through rather than deleted, so this document reads as what it is: a contract written before the measurement, corrected by it. The full account is in **`CORRECTION_implicit_class.md`** beside this file.

---

## 1. What was generated

| | |
|---|---|
| Gold items | **187** (186 scored — see `REVIEW_ROUND_1.md`) |
| Class labels across them | **199** (many items carry two) |
| ~~Implicit — no "shall" statement anywhere~~ **Withdrawn at rev 3** | ~~**72 (38.5%)**~~ |
| → **UNSTATED** — **no obligation is written anywhere** in the package; it has to be constructed | **32 (17.1%)** |
| → → as a priced line item implying scope (27 Bill of Quantities, 2 Pricing Schedule) | 29 |
| → → as a declarative scope or applicability statement whose duty is left to the reader | 3 |
| → **DISPLACED** — written out as a "shall" clause, in a drawing note, annex or federal-provisions clause | **40 (21.4%)** |
| Cold-start detectable — documents alone | **176** |
| Needing client data for the full finding | **11** |
| Flagged for close review | **37** (21 reviewed at round 1) |

> **Correction to the ledger's headline.** The ledger reported "199 planted defects". 199 is the number of *class labels*; the number of distinct defects is **187**. The implicit count was reported as 68 (34%) from summing per-document subtotals that omitted secondary tags; counted from the rows it is **72 (38.5%)**. Both figures are corrected in the ledger and the package index. The implicit share went up, not down — but it is now counted rather than asserted.
>
> *Superseded at rev 3:* the 72 was counted correctly and labelled wrongly. It is 32 UNSTATED + 40 DISPLACED, and there is no combined figure.

Files:

| File | What it is |
|---|---|
| `gold_set.jsonl` | Machine form, **rev 3** — the IMPLICIT split applied. One JSON object per line. This is what the harness loads. Items with `scored: false` are excluded. |
| `gold_set_rev2.jsonl` | Rev 2, retained byte-for-byte: the pre-split file, with IMPLICIT still on all 72. |
| `gold_set_rev1.jsonl` | The pre-review first pass, retained for provenance. |
| `CORRECTION_implicit_class.md` | **What the claim was, that it was false, and what the measurement showed.** Read it before quoting any recovery figure. |
| `REVIEW_ROUND_1.md` | What the reviewer changed and why. |
| `gold_set.yaml` | The same records, readable and editable. |
| `GOLD_SET_REVIEW_rev2.xlsx` | Review workbook — now carries each item's review status and a fresh verdict column for round 2. |
| `LABELLING_GUIDE.md` | This document. |

---

## 1a. What UNSTATED means — sharpened at rev 3

**UNSTATED: no obligation is written anywhere in the package.** Not "no requirement sentence exists" — that phrasing was too narrow, and it made the boundary turn on whether a sentence happened to be present rather than on whether a duty was ever written down. A page can carry a perfectly well-formed sentence and still leave the obligation unwritten.

It covers two shapes:

| Shape | Items | What the package gives you | What is missing |
|---|---:|---|---|
| **A priced line item implying scope** | **29** | A Bill of Quantities line, or a priced system or cost category in the Pricing Schedule — a noun phrase, a unit, a quantity. *"A.07 Anti-climb collar to detector post · No · 820"* | Everything a duty is made of. Nobody wrote that 820 collars are to be supplied and fitted; the scheduling of a priced quantity is all there is. |
| **A declarative scope or applicability statement whose duty is left to the reader** | **3** | A grammatical sentence that states a fact about the contract. *"IF-5.4 The AIS receiver is within the scope of this Contract."* | The duty itself. Being in scope is a fact; what must be supplied, integrated, tested and handed over because of it is never written. |

The three of the second shape are **D5-08 (IF-5.4)**, **D11-01 (B-1.3)** and **D12-01 (C-1.2)**. They were flagged for a second reader precisely because a sentence *does* exist for them; this is that reader's answer, and it is that they belong with the other 29.

**Both shapes require the obligation to be constructed, which is why they score the same way.** A quantity and a scope declaration are different kinds of clue, but neither can be found by reading a duty off the page, because no duty is on the page. What separates UNSTATED from DISPLACED is not how hard the text is to find — DISPLACED text can be buried three annexes deep — but whether, once found, it *states the obligation*. DISPLACED text does. UNSTATED text does not exist to be found.

---

## 2. The record

```yaml
id: D4-29
document: 4
refs: [TS-B.13, TS-B.25]
classes: [COMPUTED, ENG-CONFLICT]
finding_type: computed_compliance
severity: critical
tier: cold_start
requires: [product catalogue attributes]
evidence_basis: arithmetic
statement: "The specified lens cannot meet the specified pixel density…"
expected_action: "…clarification question; both resolutions priced"
expects_clarification_question: true
expects_price_impact: true
demo_set: true
review_priority: high
```

**`id`, `document`, `refs`, `classes`, `statement`, `expected_action`** are transcribed from the ledger. Everything else is **inferred by rule** and is what the review is for.

### The inferred fields

**`severity`** — `no_bid` › `critical` › `major` › `minor`. Assigned from the primary class, then raised to `critical` for the demo set and to `no_bid` for the three General Conditions triggers. This is the only label that moves the published KPI, because recall is weighted by it. **Round 1 fixed the meaning of the top of the ladder:** `critical` is what you cannot ask your way out of; `major` is what a clarification question can resolve.

**`tier`** — `cold_start` means findable from the thirteen documents with no client data loaded; `configured` means it cannot be found at all without it. An item is `cold_start` if **any** of its classes is; `requires` then names what the *complete* finding still needs. D10-02 is the worked example: that a −40 °C camera is specified at a site whose minimum is −5 °C is a cross-document conflict findable cold; that removing the heater saves money needs the price book.

**`evidence_basis`** — where the proof lives: `single_clause`, `cross_clause`, `cross_document`, `arithmetic`, each optionally `+domain_knowledge` where the text alone is insufficient and the model must know something about the world. This field is what separates "retrieval found it" from "reasoning found it", and it is the axis the portfolio argument rests on.

**`expects_*`** — what the pipeline must actually **emit** for the item to count as found. A finding that identifies the defect but produces no clarification question does not satisfy an item whose `expects_clarification_question` is true. Detection without the right output is not a result.

---

## 3. Matching rule

An emitted finding `F` matches gold item `G` when **both** hold:

1. **Anchor** — `F.refs ∩ G.refs ≠ ∅` after normalisation (case, hyphen, whitespace).
2. **Type** — `F.finding_type` is in `G`'s allowed set: the mapping of every class on `G`, not only the primary.

Outcomes:

| Outcome | Definition | Counts as |
|---|---|---|
| **Match** | Anchor and type both hold | Found |
| **Partial** | Anchor holds, type does not | Reported separately; **not** counted as found |
| **Output miss** | Match holds but a required `expects_*` output is absent | **Not** counted as found; reported as its own class |
| **Duplicate** | Anchors and types onto an item another finding already holds | **Not** counted as found; counts against `precision_strict`, never adjudicated |
| **Unmatched finding** | `F` matches no gold item | → adjudication queue |
| **Miss** | `G` matched by nothing | Missed |

Assignment is one-to-one and greedy over a stated total order — completeness, then anchor overlap, then confidence, then finding id. The score is *defined* as the greedy result rather than as the best assignment obtainable, so anyone can re-derive it by hand.

Matching is deterministic Python. No model decides whether a finding counts.

### A recovered statement must be the finding's own words — added at rev 3

Every UNSTATED and DISPLACED item carries `expects_recovered_statement: true`. It is satisfied only when the finding carries a non-empty `recovered_statement` **and** that text, whitespace-collapsed and casefolded, is **not a verbatim span of any page of the tender**. A finding that quotes the source line scores `OUTPUT_MISS`.

This exists because the version of this document written at rev 1 asserted that a keyword rule could not recover an obligation, and the assertion was false. Quoting is what a regex can do, and the matcher used to credit it: `finding_type` was asserted by the finding rather than earned. It is now checkable, and checked.

### Unmatched findings are not automatically false positives

The ledger is a record of what was **planted**, not a census of every defect in the package. A thirteen-document synthetic tender contains accidental ambiguities nobody wrote on purpose. An unmatched finding therefore goes to an **adjudication queue** for a human verdict — `true_new` (a real defect, promoted into the gold set with a `U-` id) or `false_positive`.

Precision is reported two ways, always both:

```
precision_strict   = matches / (matches + unmatched)          # every unmatched finding penalised
precision_adjudicated = (matches + true_new) / all findings   # after human verdict
```

Publishing only the second would be self-serving. Publishing only the first understates a system that finds real things. The gap between them is itself informative, so both are on the dashboard.

---

## 4. The KPIs this gold set supports

```
recall_overall          = matched / 186        # scored items only
recall_weighted         = Σ w(severity) · matched / Σ w(severity)      w = {no_bid 8, critical 4, major 2, minor 1}
recall_by_class         = per the 19 classes
recall_cold_start       = matched / 176        # the number that generalises to a new client
recall_configured       = matched / 11
unstated_recovery       = matched UNSTATED / 32       # replaces implicit_recovery
displaced_recovery      = matched DISPLACED / 40      # never added to the line above
unstated_uplift         = unstated_recovery  − keyword_baseline_unstated
displaced_uplift        = displaced_recovery − keyword_baseline_displaced
no_bid_detection        = matched no_bid / 3   # must be 3/3 or the run fails the gate
time_to_first_finding   = wall-clock seconds
cost_per_run            = from the router's cost ledger, not from Langfuse
```

> ~~**`keyword_baseline` is 0 by construction.** A search for "shall" across all thirteen documents returns none of the 72 implicit items, because none of them is written as a "shall" statement.~~
>
> **Withdrawn at rev 3. This was asserted here and never measured, and it is false.** The grep recovered **20 of the 72** — 27.8% — because 40 of them are plain "shall" clauses displaced into annexes, drawing notes and federal-provisions text. The claim, and the single class it rested on, are replaced by the two classes above.

**The floor is now measured, per class, and both figures are 0.0.** `ri05_tender.eval.baseline` runs the "shall" rule through the same loader and the same matcher as a real run and writes `evals/results/<tender>_keyword_baseline.json`; the `test` CI job recomputes it on every push and fails if a figure moves. Each zero is established rather than assumed, and for a different reason:

- **UNSTATED 0/32 — structural.** No keyword finding anchors a single UNSTATED item, not even as a `PARTIAL`. There is no sentence to grep.
- **DISPLACED 0/40 — earned by the recovery rule.** The rule still reaches **37 of the 40** on anchor and type. Relax `expects_recovered_statement` and exactly the 20 that were once published as implicit recovery come back. It fails because it can only quote.

Neither figure may be added to the other, and neither uplift may be published until the pipeline exists and is measured against these floors.

**The no-bid gate is a hard gate.** A run that misses any of D3-01, D3-02, D3-03 fails, whatever the other numbers say. A system that reads a contract and does not notice unlimited liability has no business quoting its recall.

---

## 5. Leakage guard

The gold set and the ledger live outside the pipeline's import graph, and a test enforces it:

- No module under `apps/ri05_tender/pipeline/**` may import from or open any path under `gold/` or any file named `DEFECT_LEDGER*`.
- The eval harness reads the gold set **after** the pipeline run has returned, never before.
- The test fails the build, not the run.

Without this the recall figure means nothing, and there is no way to prove it after the fact.

---

## 6. The production path — a real tender, no ledger

The gold set exists only for measurement. On a real tender the pipeline receives documents and nothing else, and its intake stage does four things before any requirement is extracted:

1. **Reconcile the package against its own document list.** Find the clause that enumerates the tender documents (the equivalent of ITB-4.1) and compare it to what was actually received. Every listed document not present is a gap with a name.

2. **Reconcile the submission requirements against the same list.** Find the clause enumerating what the bidder must return (the equivalent of ITB-9.3 Packages A/B/C) and build the submission checklist from it. Where that clause is absent, that absence is itself recorded — the checklist is then reconstructed from scattered obligations and marked `derived`, not `stated`.

3. **Resolve every cross-reference.** Any document, drawing, standard or annex cited in the received text and not present becomes a **dangling reference**, carrying with it the list of requirements that depend on it. The demo already contains this case: the drawings are not in the package, yet TS-B.25 — the hardest requirement in the tender — is demonstrated against detection-zone extents that exist only on SEC-201 and SEC-202.

4. **Draft the bidder question for each gap.** Not a generic "please provide missing documents", but a specific, deadline-aware question naming the document and the requirements blocked by it:

   > *TS-B.25 requires ≥ 80 px/m at the far edge of the detection zone. Drawing note GN-09 defines those extents on SEC-201 and SEC-202, which were not included in the package. Please issue SEC-201 and SEC-202, or state the design range against which pixel density is to be demonstrated.*

Each question carries the requirement IDs it unblocks, the question deadline it must be filed by, and — where the tender has an ITB-6.5-equivalent deeming clause — the note that an unasked question becomes a priced obligation.

**Coverage confidence** is reported on the analysis as a whole: how many requirements are fully evidenced, how many rest on a document that was never received, and which conclusions are therefore provisional. The presale engineer sees what is solid and what is waiting on an answer, rather than a clean-looking report built on a hole.

Where a list exists, it is used. Where it does not, its absence is a finding. Nothing is silently assumed complete.

---

## 7. What the review needs from you

The 37 high-priority rows, in this order:

1. **The severity ladder** — 3 `no_bid`, 21 `critical`. Any grade you disagree with changes a published number.
2. **The 11 `configured` items** — is `requires` right about what data each actually needs? These set the honest boundary of the cold-start claim.
3. **The 12 `+domain_knowledge` items** — do these genuinely need world knowledge, or is the text sufficient? This is the axis the whole argument stands on, so it should be conservative.
4. **The 9 demo-set items** — the ones that get shown. Every label on these is load-bearing.

The remaining 150 are mostly UNSTATED items from the Bill of Quantities and DISPLACED ones from the drawing register and the annexes, all graded `minor` and all cold-start. Spot-check fifteen and accept the rest.

**Added at rev 3 — the split, and the boundary question it raised.** The 32/40 division was made by one rule: an item is DISPLACED if the clause its refs point to is itself written as a "shall" statement, and UNSTATED otherwise. The rule was applied independently to the loader's extracted text and agreed with the hand-made list on all 72, zero disagreements.

What was referred to a second reader was whether *the rule* is the right one — three UNSTATED items (IF-5.4, B-1.3, C-1.2) resolve to clauses that state a fact rather than impose a duty, so "no sentence exists" and "no duty is written" part company on them. **Answered: the boundary is the obligation, not the sentence.** See §1a. Nothing in the labelling moves; the class now says what it always meant.

---

*Next: step 4.5 — wire the eval harness to `gold_set.jsonl`, add the leakage guard test, and record the keyword baseline in CI.*
