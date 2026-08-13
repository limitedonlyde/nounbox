# Contributing to Nounbox

## Dev setup

```bash
cp .env.example .env
docker compose up -d --build   # dev: Vite HMR + uvicorn --reload
```

(Users who only want to *run* Nounbox do not build anything — they pull
published images with `docker-compose.ghcr.yml`. See the README quickstart.)

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
   `nounbox-sdk` and a package implementing the `Labeler` protocol —
   attributes `name`, `version`, `capabilities` and a method
   `predict(image: bytes, config: dict) -> list[Annotation]`
   (see `sdk/README.md` for a minimal example).
2. Register it via the entry point group:

   ```toml
   [project.entry-points."nounbox.labelers"]
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

## Releasing images

`.github/workflows/publish.yml` builds and pushes to GHCR:

| Image | Contents | Built from |
|---|---|---|
| `nounbox-server` | `api` and `worker` — same image, different commands | `server/Dockerfile`, context = repo root |
| `nounbox-web` | static bundle behind nginx | `web/Dockerfile`, target `prod` |

Both are multi-arch (`linux/amd64`, `linux/arm64`). Tags:

- push to `main` → `latest` and `sha-<short>`
- tag `v0.2.0` → `0.2.0`, `0.2` and `sha-<short>` (`latest` stays on `main`)

Cutting a release is one command:

```bash
git tag -a v0.2.0 -m v0.2.0 && git push origin v0.2.0
```

Publishing from a fork? The account name lives in three places — the README
quickstart URL, `docker-compose.ghcr.yml`, and nowhere else (the workflow reads
`github.repository_owner` on its own):

```bash
grep -rn limitedonlyde README.md docker-compose.ghcr.yml
```

GHCR rejects capitals in image names, so lowercase yours even if the account has
them.

Things worth knowing before the first publish:

- **The packages are created private.** After the first green run, open the
  repository's Packages, and set both to public — otherwise every `docker
  compose -f docker-compose.ghcr.yml up -d` fails with `denied`. The workflow
  labels images with `org.opencontainers.image.source`, so GitHub links them to
  this repository on its own.
- **arm64 is emulated with QEMU and dominates the wall clock.** For a quick
  check, run the workflow from the Actions tab (`workflow_dispatch`) with
  `platforms = linux/amd64`. If the repository is public, moving arm64 onto
  native `ubuntu-24.04-arm` runners is the real fix — see the comment in the
  workflow.
- **Layer cache is the GitHub Actions cache**, scoped per image, and the
  repository quota is 10 GB. Evicted cache means a slow cold build, nothing
  worse.
- A new labeler plugin only reaches users once a new image is published: the
  plugins are installed into the image at build time.

## Style

- Python: type hints everywhere, dataclasses/Pydantic at boundaries; no
  comments that restate the code.
- Keep the core lightweight: heavy dependencies belong in plugins or
  `deploy/` recipes, never in `server/`.
