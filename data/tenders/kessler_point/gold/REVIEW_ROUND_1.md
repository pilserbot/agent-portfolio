# Gold Set — Review Round 1

**Reviewer: AE · 13 September 2026 · gold set rev 1 → rev 2**

21 of 187 items carried a verdict. Everything else is recorded as `accepted_by_default` — machine-graded and not yet seen by a human. That distinction is kept in the record so the KPI report can state how much of the answer key was human-confirmed rather than implying all of it was.

---

## Verdicts

| Status | Items |
|---|---|
| `accepted` | 14 |
| `amended` | 6 |
| `dropped_disputed` | 1 |
| `accepted_by_default` | 166 |

**Scored items: 186.** One item is excluded from scoring pending a final decision.

---

## Severity amendments

| Item | Was | Now | The defect |
|---|---|---|---|
| D1-03 | critical | **major** | Compliant bid 6% over the published budget |
| D1-05 | critical | **major** | Mandatory site visit, TWIC required, no escorted access |
| D2-02 | critical | **major** | SAT day 510 + 90 fault-free days against Final Acceptance day 540 |
| D4-01 | critical | **minor** | Pixel density must be computed rather than read from a datasheet |
| D4-02 | critical | **major** | −40 °C operation within 15.4 W of 802.3af |
| D4-07 | critical | **major** | Crash-rated 24 ft sliding gate cycling in ≤ 10 s |

Reviewer note on D4-02 and D4-07: *"Explain and ask for waiver or change in Q&A session."*

Effect on the distribution of scored items:

| Severity | Rev 1 | Rev 2 |
|---|---|---|
| `no_bid` | 3 | 3 |
| `critical` | 21 | **14** |
| `major` | 62 | **67** |
| `minor` | 101 | **102** |

---

## The severity principle this establishes

The six amendments are consistent with one rule, which the machine grader did not have:

> **`critical` is what you cannot ask your way out of. `major` is what a clarification question can resolve.**

An engineering impossibility that the Authority will almost certainly waive once it is pointed out (a −40 °C camera at a Gulf Coast terminal, a 10-second crash gate) costs a question, not the bid. An impossibility that threatens the architecture whatever the answer (D4-08 and D4-09, the 20-second transaction with zero time budget; D5-04, no TOS modification against a synchronous API) stays `critical`. So does anything that eliminates the bidder outright (D1-04 bonding capacity) or cannot be delivered lawfully at any price (D5-01, NCIC).

**Two items sit awkwardly against that rule and are left as accepted:** D4-32 (30 fps with analytics plus a heater inside 15.4 W) is the same power-budget family as D4-02, now `major`, but was accepted as `critical`; and D2-01 (PDR at day 60 against 20–40 week lead times) is arguably a Q&A matter like D2-02, now `major`. One line settles both.

The rule has **not** been propagated to the 166 unreviewed items. Re-grading them by a rule inferred from 21 examples would put a machine's reading of the reviewer's judgement into the answer key and label it human review. They keep their machine grades and their `accepted_by_default` status until someone looks at them.

---

## D1-14 — dropped, and contested

**Reviewer verdict: Drop.** *"What is important is the brand and specifications. If our product is OEM labeled with our company name and part number, we are official brand."*

**Applied:** excluded from scoring. **Recorded as contested**, because the legal position runs the other way.

FAR 52.204-25, implementing NDAA §889, prohibits video surveillance equipment **produced by** a covered entity — Huawei, ZTE, Hytera, Hikvision, Dahua, or any subsidiary or affiliate — and reaches it where it is a substantial or essential component of any system. The test is the manufacturer of origin, established from the FCC ID grantee, not the nameplate. Relabelling is the mechanism the clause exists to catch, not an exemption from it: LTS, EZVIZ and Annke are Hikvision OEM lines; Lorex, Amcrest, IC Realtime and Q-See are Dahua's. A rebranded covered camera in a federally funded installation is a False Claims Act exposure for the contractor that certified compliance, whatever brand is silkscreened on the housing.

The tender's own wording at ITB-12.2 — *"as a component or original equipment manufacturer supply to another brand"* — was written to state exactly this.

This one is worth reversing. Say the word and it returns to the scored set as `critical`, `configured`, tier note *catalogue with covered-entity / OEM origin*.

---

## Unreviewed high-priority items

16 of the 37 flagged rows did not get a verdict, including all three `no_bid` triggers (D3-01, D3-02, D3-03) and the regulatory misassignments (D3-20, D6-01 — the Cybersecurity Plan wrongly assigned to the Contractor; D3-21 — FAR clauses incorporated as though the FAR applied directly). They keep their machine grades. The `no_bid` gate in the harness depends on those three grades being right, so they are the first thing to confirm when there is a spare ten minutes.

---

*Rev 2 is the scored answer key from here. Rev 1 is retained at `gold_set_rev1.jsonl`.*
