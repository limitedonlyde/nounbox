# Contributing to AutoLabelUi

## Dev setup

```bash
cp .env.example .env
docker compose up -d --build   # dev: Vite HMR + uvicorn --reload
```

`docker-compose.yml` holds the release configuration (static bundle behind
nginx); `docker-compose.override.yml` is applied automatically and turns it
into the dev setup. To check the release build locally, exclude the override:
`docker compose -f docker-compose.yml up -d --build`.

Source is mounted into the containers, so code changes need no rebuild — but
only `api` (uvicorn `--reload`) and `web` (Vite HMR) pick them up live. The
worker caches imports: after changing `server/`, `labelers/` or `sdk/` code,
run `docker compose restart worker`.

- UI: http://localhost:5173 · API docs: http://localhost:8000/docs
- Worker logs: `docker compose logs -f worker`

## Tests

The test suite is still being established (pytest for `server/` and
`labelers/`, vitest for `web/` is planned). What exists today:

- Python tests live in a `tests/` directory next to the package they cover
  (example: `labelers/vlm/tests/`) and run with `pytest`. Follow the same
  layout for new tests.
- Frontend type-checks with `npm run build` in `web/`.
- For anything touching ingest/autolabel/export, describe your manual e2e
  check in the PR (upload → autolabel → review → export).

## Adding a labeler plugin

1. Create `labelers/<name>/` with a `pyproject.toml` depending on
   `autolabelui-sdk` and a package implementing the `Labeler` protocol —
   attributes `name`, `version`, `capabilities` and a method
   `predict(image: bytes, config: dict) -> list[Annotation]`
   (see `sdk/README.md` for a minimal example).
2. Register it via the entry point group:

   ```toml
   [project.entry-points."autolabelui.labelers"]
   my_ocr = "my_package:MyLabeler"
   ```

3. Add an editable install of your plugin to `server/Dockerfile` (the
   `pip install -e` line), rebuild: `docker compose up -d --build worker`.
4. The engine appears in the UI engine dropdown automatically. Keep `predict`
   pure (no platform imports): bytes in, annotations out. Confidence drives
   bulk-accept and the review queue — return honest values, or a synthetic
   low value (e.g. 0.5) to force human review.

Plugins that need a GPU should not run in the worker — expose them behind an
OpenAI-compatible API or the HTTP-labeler convention (`labelers/http`), and
add a Modal recipe under `deploy/modal/` if you want a one-command deploy.

## Style

- Python: type hints everywhere, dataclasses/Pydantic at boundaries; no
  comments that restate the code.
- Keep the core lightweight: heavy dependencies belong in plugins or
  `deploy/` recipes, never in `server/`.
