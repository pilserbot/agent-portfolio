# RI-05 — Kessler Point Tender Package

**Solicitation AHPA-2026-IFB-0417 · Rev 1 · 13 September 2026 — gold set amended at rev 3, 14 September 2026**

Synthetic tender package for RI-05, the Tender Response Engine. Invented authority, terminal, vendors and products; quantities and regulatory content derived from published US terminal data and verified primary sources.

> **The implicit count below was withdrawn at rev 3.** The claim that 72 defects have no "shall" statement anywhere was asserted, never measured, and is false: a "shall" grep recovered 20 of them. The class is now UNSTATED (32) and DISPLACED (40). See `gold/CORRECTION_implicit_class.md`.

---

## THE PACKAGE — 13 DOCUMENTS

Each document is provided in **PDF** (the issued form, with page numbers for source anchoring), **DOCX** (editable), and **Markdown** (the source). The three workbooks are native **XLSX**.

| # | Document | Issued format | Pages | Also as |
|---|---|---|---|---|
| 1 | Instructions to Bidders | `pdf/01_Instructions_to_Bidders.pdf` | 5 | docx · md |
| 2 | Scope of Work | `pdf/02_Scope_of_Work.pdf` | 7 | docx · md |
| 3 | General Conditions of Contract | `pdf/03_General_Conditions_of_Contract.pdf` | 6 | docx · md |
| 4 | Technical Specification | `pdf/04_Technical_Specification.pdf` | 9 | docx · md |
| 5 | Integration and Interface Specification | `pdf/05_Integration_and_Interface_Specification.pdf` | 3 | docx · md |
| 6 | Cybersecurity and Information Security | `pdf/06_Cybersecurity_and_Information_Security.pdf` | 4 | docx · md |
| 7 | **Bill of Quantities** | `07_Bill_of_Quantities.xlsx` | 10 tabs | — |
| 8 | **Pricing Schedule** | `08_Pricing_Schedule.xlsx` | 12 tabs | — |
| 9 | **Compliance Matrix** | `09_Compliance_Matrix.xlsx` | 298 rows | — |
| 10 | Annex A — Site and Environmental Conditions | `pdf/10_Annex_A_Site_and_Environmental_Conditions.pdf` | 3 | docx · md |
| 11 | Annex B — Schedule of Standards | `pdf/11_Annex_B_Schedule_of_Standards.pdf` | 3 | docx · md |
| 12 | Annex C — SSI Handling | `pdf/12_Annex_C_SSI_Handling.pdf` | 3 | docx · md |
| 13 | Drawing Register | `pdf/13_Drawing_Register.pdf` | 5 | docx · md |

**48 PDF pages.** Every page carries a running header with the solicitation number and document title, and a footer with the SSI marking and `Page n of m` — which is what makes `source_anchor` able to record a real page reference.

---

## NOT PART OF THE PACKAGE

| File | Purpose |
|---|---|
| `DEFECT_LEDGER.md` | **The gold-set answer key.** 187 planted defects carrying 199 class labels across 19 classes, each with its reference, class, description and the expected RI-05 action. Never ships with the tender. |
| `gold/CORRECTION_implicit_class.md` | **Why the IMPLICIT class was split at rev 3**, what the false claim was, and what the measurement showed. Read before quoting any recovery figure. |
| `00_INDEX.md` | This file. |

---

## WHAT THE PACKAGE CONTAINS

| Measure | Value |
|---|---|
| Documents | 13 |
| PDF pages | 48 |
| Numbered requirements in Documents 4, 5 and 6 | 301 |
| Rows in the Compliance Matrix | 298 + 1 phantom |
| Bill of Quantities line items | 103 across 9 systems |
| **Planted defects** | **187** (199 class labels) |
| ~~Of which are implicit — no "shall" statement anywhere~~ | ~~**72 (38.5%)**~~ |
| → Of which are **UNSTATED** — no requirement sentence exists anywhere in the package | **32 (17.1%)** |
| → Of which are **DISPLACED** — written out as a "shall" clause, in a drawing note, annex or federal-provisions clause | **40 (21.4%)** |

---

## THE DEMO SET

The seven findings the demonstration is built around.

| | Defect | Where | Why it lands |
|---|---|---|---|
| 1 | **NCIC query required at the pre-gate** | IF-4.2, IF-4.3 | A private port authority cannot lawfully query NCIC. Unbuildable at any price — the system must reason about the world, not the sentence. |
| 2 | **The specified lens cannot meet the specified pixel density** | TS-B.13 vs TS-B.25 | 12 mm at 55 m gives 74.8 px/m against a requirement of 80. Computed from a lens spec, a sensor spec, a performance clause and a drawing. |
| 3 | **A −40 °C camera on 15.4 W PoE, at a site whose minimum is −5 °C** | TS-B.36 + TS-B.38 vs A-2.1 | Impossible *and* unnecessary. The clarification question saves money rather than raising a problem. |
| 4 | **The Authority's own matrix omits three mandatory requirements** | Document 9 | Including the pixel-density constraint and the clause that enforces Section 889. In a visibly complete 298-row form, nobody suspects anything is missing. |
| 5 | **The taut-wire fence is redundant** | TS-A.1 vs TS-B.27 | $1.06 M saved on a solution with a *lower* nuisance alarm rate. The alternative-proposal capability in one example. |
| 6 | **The milestone table is arithmetically impossible** | SOW-9.4, SOW-9.7 vs M9/M11 | Day 510 + 90 fault-free days = 600, against a Final Acceptance date of 540. |
| 7 | **Three no-bid triggers in the General Conditions alone** | GCC-5.2, GCC-11.3, GCC-11.4 | Uncapped liquidated damages, unlimited liability, consequential losses including regulatory fines. All detectable in minutes, before any technical reading. |

---

## THE CENTRAL DILEMMA

```
Authority's published budget       $15.0 M     (ITB-1.4)
Fully compliant bid                $15.9 M     6% over
Alternative bid                    $14.8 M     fits, and better on NAR
Evaluation                         40% technical / 60% commercial   (ITB-8.1)
```

Under 40/60 the alternative wins by 2.1 points. Under 60/40 the compliant bid wins by 0.3. **The weighting decides, and the crossover is computable** — which is what the margin workbench exists to show.

---

## REGULATORY BASIS

Verified against primary sources in September 2026.

| Regime | Applies | Note |
|---|---|---|
| **MTSA — 33 CFR 101, 103, 105** | Yes | The controlling law. Performance-based: no prescriptive fence height, lighting level or camera spacing exists. |
| **33 CFR 101 Subpart F — Cybersecurity** | Yes | Final rule 17 Jan 2025, effective 16 Jul 2025. Cybersecurity Plan due to USCG by **16 Jul 2027**. |
| **TWIC — 49 CFR 1572** | Yes, visually | The electronic **Reader Rule does not reach container terminals** — Risk Group A is bulk dangerous cargo. |
| **49 CFR 1520 — SSI** | Yes | Facility Security Plans, assessments and security drawings are SSI, not CUI. |
| **2 CFR 200 Subpart D** | Yes | The operative procurement regime. A port authority spending FEMA grant money is **not** a federal agency. |
| **Section 889 (FAR 52.204-24/-25/-26)** | Yes | Huawei, ZTE, Hytera, Hikvision, Dahua — including OEM supply to another brand. |
| **FICAM / FIPS 201 PIV** | **No** | Applies to federal agency facilities. Cited in the tender anyway — a planted, and very common, conflation. |
| **UFC / CDSE / DoD criteria** | **No** | DoD has no authority over a commercial or landlord port. Scheduled at Annex B as "good engineering practice" — planted. |
| **CMMC** | **No** | Does not reach a non-DoD port authority contract. Deliberately excluded at CS-1.6. |

---

## NEXT

Step 4.4 complete — first-pass gold set at `ri05_gold/` (187 items, 37 flagged for review).
Step 4.5 — wire the eval harness to `gold_set.jsonl`, add the leakage guard, record the keyword baseline in CI. **Done, and it found this package's own labelling wrong** — see the note at the top.
Step 4.9 — the gold set is at rev 3. No recovery or uplift figure is publishable until the pipeline exists and is measured against the two keyword floors, both currently 0.0.

---

*RI-05 Phase 4 · Tender package Rev 1 · 13 September 2026*
