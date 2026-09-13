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
