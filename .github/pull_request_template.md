## What changes

<!-- One or two sentences. Link the issue if there is one: Fixes #123 -->

## How it was checked

<!--
Tests, or the manual path you walked. For anything touching ingest, autolabel,
export or the review UI, the useful answer is the end-to-end one:
upload -> autolabel -> review -> export, and what you saw.
-->

---

<!--
No checklist beyond this. Two things worth remembering:

- CI runs `pytest tests -q` from server/, `pytest labelers -q -m "not slow"`
  from the repo root, and `npm run build` from web/.
- After changing server/, labelers/ or sdk/ in dev mode:
  docker compose restart worker (the worker caches imports).

By submitting, you agree your contribution is licensed under AGPL-3.0.
-->
