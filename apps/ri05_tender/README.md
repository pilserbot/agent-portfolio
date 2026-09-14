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
  optimum. **No model call anywhere.**
- **`metrics`** — recall overall, by class, by tier, by severity; severity-weighted
  (no_bid 8, critical 4, major 2, minor 1); implicit recovery; both precisions; the no-bid
  gate. Every ratio comes from `spine.eval.metrics` through a thin adapter, never a fork.
- **`adjudication`** — an unmatched finding is queued for a human, not counted as wrong.
  The key records what was *planted*, not every defect in the package.
- **`report`** — the score card as markdown, header stating what was scored and what was
  excluded, and the gate as its own PASS/FAIL line.

Two things the design refuses to allow:

- **One precision without the other.** `Precisions` has no default for either field, so
  half of it cannot be constructed. Strict counts every unmatched finding against the
  system; adjudicated counts only what a human called wrong. Quoting one is a choice about
  how flattering the number is.
- **Passing the no-bid gate on average.** It is True only when *every* scored `no_bid` item
  was fully matched. A bid going out on a tender the system should have refused is not
  offset by finding ninety other things.
