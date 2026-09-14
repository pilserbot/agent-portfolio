# evals/adjudication

One JSONL file per scoring run, written by `ri05_tender.eval.adjudication` as
`<tender_name>_<timestamp>.jsonl`. Each line is a finding the run could not match to
anything in the tender's answer key, with `"verdict": null`.

**An unmatched finding is not a false positive.** The gold set records what was *planted*
in a tender, not every defect the package contains, so a finding that matches nothing may
be a real problem nobody thought to label. Counting all of them as wrong would punish the
pipeline for reading more carefully than the person who wrote the key.

So a human rules on each one, by editing the `verdict` field in place:

- `"true_new"` — a real defect the key had not planted. It is then known by its finding id
  under a `U-` prefix (`U-` for unplanted), so it can never be mistaken for a gold item
  that was in the key all along, and it lifts `precision_adjudicated`.
- `"false_positive"` — not a defect.

Anything left `null` is pending, and the report says how many. Pending is not a quiet pass:
`precision_strict` counts every unmatched finding against the system regardless, and both
precisions are always reported together, so an unreviewed queue cannot flatter a run.

Nothing infers a verdict. A confirmed new finding is recorded here and nowhere else —
promoting one into the answer key is a separate, deliberate act.
