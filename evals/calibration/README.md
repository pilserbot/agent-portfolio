# evals/calibration

Reports written by `python -m spine.eval.calibrate`, one per rubric per day.

A report answers one question: **is this judge fit to be used?** It leads with the verdict,
and a judge whose Cohen's kappa falls below 0.6 is reported as `NOT FIT FOR USE`. The tool
exists to be able to say the instrument is bad — a judge that has never been calibrated is a
guess with a number attached, and one calibrated and found wanting must not be used anyway.

Kappa, not raw agreement, decides fitness. On a skewed set a judge that answers the same way
every time can look 90% accurate while carrying no information; kappa corrects for that and
scores it zero.

Reports are committed, so the history of an instrument is visible: when a rubric's version
changed, and whether its agreement with human judgement moved.

```bash
python -m spine.eval.calibrate --rubric example --labels data/gold/example_human.jsonl
```

The command exits non-zero when the judge is not fit for use, so a pipeline that runs it
fails rather than quietly filing the bad news.

Note: a report measured against labels whose provenance is not `human` is stamped as a
**smoke test, not a calibration**. `data/gold/example_human.jsonl` is such a fixture.
