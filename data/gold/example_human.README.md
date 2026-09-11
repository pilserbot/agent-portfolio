# data/gold/example_human.jsonl — SMOKE-TEST FIXTURE, NOT REAL HUMAN LABELS

**These 30 labels were written to make `python -m spine.eval.calibrate` runnable end to end.
No human judged them. They are not a gold set, and a calibration measured against them
says nothing about whether any judge is fit for use.**

Every record carries `"provenance": "synthetic-fixture"`, and the calibration report stamps
a prominent warning whenever it sees a non-`human` provenance. That marker is in the data
rather than only in this file, so the warning survives the file being copied, renamed or
appended to.

To produce a real calibration set:

1. Take outputs from an actual run — `SPINE_MODE=record` cassettes are a good source.
2. Have a person who knows the domain label each one against the rubric, without seeing the
   judge's score.
3. Write them with `"provenance": "human"`, and keep the labeller and the date in
   `human_note`.
4. Aim for at least 50 items and both classes well represented; kappa on a handful of items
   is noise with a number attached.

Do not extend this file to do that. Start a new one, so the synthetic items can never be
mixed in with real ones.
