# Correction — the IMPLICIT class was wrong, and has been split

**Rev 3 · 14 September 2026 · supersedes the IMPLICIT labelling in rev 1 and rev 2**

---

## The claim

`LABELLING_GUIDE.md` and `00_INDEX.md` both stated it, and the KPI set was built on it:

> ~~**`keyword_baseline` is 0 by construction.** A search for "shall" across all thirteen
> documents returns none of the 72 implicit items, because none of them is written as a
> "shall" statement.~~

> ~~Of which are implicit — no "shall" statement anywhere: **72 (38.5%)**~~

**The claim was false.** It was never measured. It was asserted in a document written
before the harness existed, and every implicit-recovery and implicit-uplift figure the
project planned to publish rested on it.

## The measurement

Step 4.8 added `ri05_tender.eval.baseline` — the "shall" rule, run through the same loader
and scored by the same matcher as a real run, recorded in CI rather than asserted. On the
committed tender it produced **861 keyword hits** and recovered **20 of the 72 IMPLICIT
items — 27.8%**, not 0.

> D1-15, D1-18, D4-25, D4-26, D4-39, D5-09, D6-07, D6-08, D10-07, D11-03, D12-02, D12-03,
> D12-04, D13-01, D13-02, D13-03, D13-04, D13-05, D13-07, D13-09

D4-25 is the clearest case. Its gold statement is that no equipment may be *"scheduled for
end of production or end of support within five (5) years of Final Acceptance"*. TS-1.1
says, in the tender:

> TS-1.1 All equipment shall be new, unused, of current manufacture, and **shall not be
> scheduled for end of production or end of support within five (5) years** of Final
> Acceptance.

The obligation is written out, in a "shall" clause, in the Technical Specification. It is
not implicit by any reading.

## The diagnosis

One class was doing two jobs. Both are real problems a presale engineer has, and they are
not the same problem:

| | What it is | Why it is missed | What finding it needs |
|---|---|---|---|
| **UNSTATED** | No requirement sentence for it exists anywhere in the package. It lives in a Bill of Quantities line item, a Pricing Schedule row, or a scope word buried in prose. | There is nothing to read. | Inference — the obligation has to be derived from a quantity, a row, or an omission. |
| **DISPLACED** | Written out plainly as a "shall" clause — but in a drawing note, an annex, or a federal-provisions clause. | A requirements review never goes there. | Reading — of the whole package rather than of Documents 4, 5 and 6. |

Averaging them produced a figure that moved when the mix changed rather than when the
system did, and hid the fact that a one-line regex clears the whole of one half.

## The split

**72 items. 32 UNSTATED, 40 DISPLACED.** Applied in place to `gold_set.jsonl`; the
pre-split file is retained byte-for-byte as `gold_set_rev2.jsonl`.

The rule applied: **an item is DISPLACED if the clause its refs point to is itself written
as a "shall" statement in the tender text, and UNSTATED otherwise.** The rule was applied
independently to the extracted text by the tender loader, and the result agreed with the
hand-made list on all 72 items, with **zero disagreements**.

The 32 UNSTATED:

> D5-08, D7-02 … D7-28 (all 27), D8-03, D8-04, D11-01, D12-01

Twenty-seven of them are Bill of Quantities line items; two are Pricing Schedule structures
(`System J tab`, `Summary rows 24–27`) whose `refs` are not clause references at all, so the
rule reaches UNSTATED for them by there being no clause to inspect. The other three —
IF-5.4, B-1.3, C-1.2 — are clauses that state a fact rather than impose a duty
("The AIS receiver is within the scope of this Contract"), and the obligation that follows
from the fact is nowhere written.

The other **40 are DISPLACED**, which is exactly the count expected. Every one of the 20
items the keyword rule recovered falls in this half.

On each of the 72 items:

- `classes` — `IMPLICIT` replaced in place by `UNSTATED` or `DISPLACED`
- `finding_type` — `implicit_requirement` replaced by `unstated_requirement` or
  `displaced_requirement`
- `expects_recovered_statement: true` — new, on all 72
- `reclassified_rev3` — `"IMPLICIT -> UNSTATED"` or `"IMPLICIT -> DISPLACED"`

`CLASS_TO_FINDING_TYPE` in `ri05_tender.eval.matcher` no longer contains `IMPLICIT`. The
class cannot be used again by accident: an unmapped class stops the run.

## The second fix — the claim is now enforced, not asserted

Splitting the class alone would have left the deeper problem in place. The matcher credited
a finding for citing the clause an obligation hides in, because `finding_type` was asserted
by the finding rather than earned. A rule that understood nothing was scored as having
recovered something.

`expects_recovered_statement` is now enforced in `matcher`. The requirement is satisfied
only when the finding carries a non-empty `recovered_statement` **and** that text —
whitespace-collapsed and casefolded — is **not a verbatim span of any page of the tender**.
A finding that merely quotes the source line scores `OUTPUT_MISS`.

This is what makes *"a keyword baseline cannot recover an obligation"* true by enforcement
rather than by assertion. The previous claim was asserted and was false.

## The re-derived baseline

| | Items | Recovered | Recall |
|---|---:|---:|---:|
| UNSTATED | 32 | 0 | **0.0** |
| DISPLACED | 40 | 0 | **0.0** |

There is no combined figure, and no field to put one in.

Both zeroes are established rather than assumed, and for different reasons — each pinned by
its own test in `tests/ri05/test_scoring_baseline.py`:

- **DISPLACED is 0.0 because of the recovery rule.** The keyword findings still reach **37
  of the 40** DISPLACED items on anchor and type. Switch `expects_recovered_statement` off
  and exactly the same **20** are credited as before the split. The rule is doing the work;
  the rename is not.
- **UNSTATED is 0.0 structurally.** No keyword finding anchors a single UNSTATED item — not
  even as a `PARTIAL`. A rule that searches sentences cannot produce a candidate for an
  obligation that has no sentence.

A zero no test can tell apart from a broken harness is not evidence of anything, which is
why both of those tests exist.

## What is not publishable

**No uplift figure, on either class, until the pipeline exists and is measured against these
floors.** `evals/projects/ri05.yaml` replaces `implicit_recovery` and `implicit_uplift` with
`unstated_recovery`, `displaced_recovery`, `unstated_uplift` and `displaced_uplift`, and
every target in that file is UNSET. `spine.kpi.compute_kpis` refuses to measure a metric
with no target, so the instrument cannot produce a number against an invented standard.

Any prior statement of an implicit-recovery or implicit-uplift figure for RI-05 is
withdrawn. It divided by 72, and 72 was two denominators.

---

*Raised by the step 4.8 keyword baseline. Applied at rev 3, 14 September 2026.*
