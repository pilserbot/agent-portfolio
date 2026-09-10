# demo/cassettes

Recorded model calls, one JSONL file per run at `<project>/<run_id>.jsonl`.

These are **committed on purpose**. They are what lets a reviewer with a fresh clone, or a
demo deployed with no credentials, run the real code paths end to end:

```bash
SPINE_MODE=replay ...
```

Record new ones with `SPINE_MODE=record` against a live provider, then check them with
`python -m spine.replay verify <cassette>` before committing.

A cassette is a verbatim record of what a model returned, including the prompt that was
sent. Nothing redacts it — record only what may be committed to this repository.

Empty for now: no project records a demo run yet.
