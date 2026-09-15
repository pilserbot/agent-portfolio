# ri05_tender

The RI05 tender response engine: ingests a tender, extracts its requirements through
`req_core`, and assembles a response scored by `spine`.

## `ri05_tender.tender` — the loading layer

Every tender enters the system through one function, given the tender **folder**:

```python
from pathlib import Path
from ri05_tender.tender.loader import load_tender

package = load_tender(Path("data/tenders/kessler_point"))
```

```bash
python -m ri05_tender.tender.loader data/tenders/<folder>
```

The CLI prints document id, type, page count and sha256 as a table — the first thing to
run against a folder nobody has loaded before.

**No tender's name appears anywhere in the code.** The layout is the contract:

    <tender folder>/
        documents/   the tender as the client issued it — PDF, XLSX, DOCX
        gold/        an answer key. Optional, and absent in production.

A folder with no `gold/` loads identically; only `has_gold` and `gold_path` differ. That is
the production path and it is tested as such, not assumed. Nothing under `gold/` is opened —
enforced by a test that watches every file the process opens during a load.

### What the extraction promises

- **PDF** — one page per printed page, so an `EvidenceRef` can cite a page that exists.
  Read with pdfplumber; the module docstring records why, and what pypdf did instead.
- **XLSX** — one page per worksheet, `data_only=True` so a cell yields its value and not
  its formula. The sheet name is line 1 and row *r* is line *1 + r*, blank rows included,
  so "BOQ row 214" still resolves to row 214.
- **DOCX** — one page, because Word has no fixed pagination and any page number invented
  here would be a different number on the next machine. Tables are read alongside the
  paragraphs, since `python-docx` omits table text from `paragraphs` entirely.

An unreadable or unsupported file in `documents/` stops the load rather than being skipped:
a document silently missing is a document nobody bid on.

### What it does not do

No model call, no network, no interpretation. It turns files into text with the page
numbers kept, hashes each file so a change is visible, and stops there.

## `ri05_tender.eval` — the scoring harness

**The pipeline this measures does not exist yet, and that is deliberate.** The measurement
is defined first, so the pipeline is built against a target rather than the target being
drawn around whatever the pipeline happened to do. Everything is exercised against fixtures.

- **`models`** — `GoldItem` mirrors one line of a tender's `gold/gold_set.jsonl` field for
  field, `extra="forbid"` so a key the model does not know fails the load instead of
  vanishing. `Finding` is the shape the pipeline will emit; its evidence is
  `spine.contracts.EvidenceRef`, so a citation made here resolves like any other.
- **`loader`** — reads the answer key beside a loaded `TenderPackage`, and raises
  `NoGoldSetError` for a tender that has none: scoring an unlabelled tender is a caller
  mistake, not an empty run. Items marked `scored: false` load, are excluded from every
  denominator, and are counted in the report header.
- **`matcher`** — a finding matches when its normalised refs intersect the item's **and**
  its type is allowed by **all** of the item's classes. Six outcomes; only MATCH counts as
  found. Assignment is one-to-one, ranked by **completeness before overlap** so the
  instrument's own arbitration is never scored as the system's failure, and **greedy by
  definition** rather than by approximation — a published number needs a definition, not an
  optimum. It also enforces that **a recovered obligation is stated in the finding's own
  words**: a `recovered_statement` that is a verbatim span of any page of the tender scores
  OUTPUT_MISS, because quoting is retrieval. **No model call anywhere.**
- **`metrics`** — recall overall, by class, by tier, by severity; severity-weighted
  (no_bid 8, critical 4, major 2, minor 1); UNSTATED and DISPLACED recovery, **separately
  and never summed**; both precisions; the no-bid gate. Every ratio comes from
  `spine.eval.metrics` through a thin adapter, never a fork.
- **`adjudication`** — an unmatched finding is queued for a human, not counted as wrong.
  The key records what was *planted*, not every defect in the package.
- **`report`** — the score card as markdown, header stating what was scored and what was
  excluded, and the gate as its own PASS/FAIL line.

- **`baseline`** — the keyword floor. Every line containing "shall", scored through the
  same loader and the same matcher as a real run, so the comparison is like for like.
  `python -m ri05_tender.eval.baseline data/tenders/<folder> [--check]`; the record lives at
  `evals/results/<tender>_keyword_baseline.json` and CI fails if it drifts.

Three things the design refuses to allow:

- **One precision without the other.** `Precisions` has no default for either field, so
  half of it cannot be constructed. Strict counts every unmatched finding against the
  system; adjudicated counts only what a human called wrong. Quoting one is a choice about
  how flattering the number is.
- **Passing the no-bid gate on average.** It is True only when *every* scored `no_bid` item
  was fully matched. A bid going out on a tender the system should have refused is not
  offset by finding ninety other things.
- **A combined recovery figure.** `ScoreCard` has no field to put one in. No obligation is
  written anywhere for an UNSTATED item, while a DISPLACED one is written out somewhere nobody
  looks; reading finds the second and cannot find the first. One ratio over both is an average
  of two capabilities that moves when the mix changes rather than when the system does.

### Two guards, both offline, both in the `test` CI job

- **Gold leakage.** `tests/ri05/test_tender_loader.py` watches every file the process opens
  during a load — the runtime half. `tests/ri05/test_no_gold_leakage.py` parses every module
  outside `eval` and fails on any reference to the answer key at all, reachable or not — the
  static half, which catches what no test executes. The allowance is **one string literal in
  one file**, written out in the test. Growing it means the guard is being worked around.
- **The keyword floor.** A recovery figure means nothing without one, and it is measured in
  CI rather than asserted in prose.

### The floor caught the answer key

The gold set said none of its 72 IMPLICIT items was written as a "shall" statement anywhere
in the package. The keyword baseline recovered **20 of them**. The claim had been asserted in
a labelling guide written before the harness existed, and never measured.

**One class was doing two jobs**, and at rev 3 it became two:

| | Items | What it is | What finding it needs |
|---|---:|---|---|
| **UNSTATED** | 32 | **No obligation is written anywhere.** Two shapes: a priced line item implying scope (29), or a declarative scope statement whose duty is left to the reader (3) | Construction — both shapes score the same way because both require the obligation to be built |
| **DISPLACED** | 40 | Written out as a plain "shall" clause, in a drawing note, an annex or a federal-provisions clause | Reading the whole package |

The split rule — *DISPLACED if the clause its refs point to is itself a "shall" statement* —
was applied independently to the loader's extracted text and agreed with the hand-made list
on all 72 items, zero disagreements.

Splitting the class was not enough on its own, because the matcher credited a finding for
citing the clause an obligation hides in: `finding_type` was asserted rather than earned.
`expects_recovered_statement` now fixes that, and **both floors re-derive to 0.0** — each for
a different reason, each pinned by its own test:

- **UNSTATED 0/32 — structural.** No keyword finding anchors a single UNSTATED item, not even
  as a PARTIAL. There is no sentence to grep.
- **DISPLACED 0/40 — earned.** The rule still reaches **37 of the 40** on anchor and type.
  Relax `expects_recovered_statement` and exactly the 20 come back. It fails because it can
  only quote.

A zero no test can tell apart from a broken harness is not evidence of anything, which is why
those two tests exist. The full account is in
`data/tenders/kessler_point/gold/CORRECTION_implicit_class.md`. **No uplift figure on either
class is publishable until the pipeline exists and is measured against these floors.**
