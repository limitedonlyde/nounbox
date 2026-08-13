# Security Policy

## Supported versions

Nounbox is pre-1.0 and moves fast. Fixes land on `main` and go out in the
next image build; there are no backports to older tags.

| Version | Supported |
| --- | --- |
| `main` / latest published image | Yes |
| Any earlier tag | No — upgrade first |

## Threat model — read this before reporting

**Nounbox is a single-user tool for localhost or a network you trust.** It
has no accounts, no login, no roles, and no per-project permissions. That is a
deliberate design decision, documented in the [roadmap](ROADMAP.md), not an
oversight.

What follows from it:

- **Anyone who can reach the port owns the instance.** They can read every
  project, upload images, edit or delete annotations, and export the dataset.
  The shipped compose files therefore bind published ports to `127.0.0.1`
  (`8080` for the UI, `9000` for MinIO, plus `8000`/`5173`/`9001` in the dev
  override). Do not change those to `0.0.0.0` and put the result on a public
  address.
- **Exposing it to the internet requires your own front door.** A reverse
  proxy terminating TLS and enforcing authentication in front of the web
  container is the supported way. Without it, every point above applies to the
  whole internet.
- **`APP_ACCESS_TOKEN` is not a login.** It protects exactly three endpoints —
  the ones that spend money or take a secret: storing the Modal token
  (`PUT /settings`), removing it (`DELETE /settings/modal`), and deploying the
  GPU app (`POST /settings/gpu/deploy`). Everything else stays open. When the
  variable is unset, those three are open too; the server warns about it in the
  log and in `GET /settings`.
- **The platform spends your money in your Modal account.** It takes a Modal
  API token, deploys a GPU recipe into your account, and calls it. The token
  secret is encrypted at rest (Fernet) with a key file stored `0600` in the
  `appdata` volume shared by `api` and `worker`, and is never returned by any
  endpoint — only the non-secret `token_id` is. Whoever can read both the
  database and that volume can recover the token.
- **Per-run engine config is stored in plaintext.** The engine config JSON you
  type in the UI lives in the database as-is. Put API keys in `.env`
  (`VLM_API_KEY`), not in that field.
- **Images are served straight from MinIO by presigned URL.** The browser
  fetches them directly, bypassing the API, and anyone holding such a link can
  fetch that image until it expires.
- **Uploads are parsed by native libraries.** Photos, HEIC and PDFs are decoded
  in the worker by Pillow, pillow-heif and pypdfium2. Feeding the platform
  files from untrusted sources means trusting those decoders.
- **Labeler plugins are code.** Installing one — the `vlm` and `http` labelers
  also send your images to whatever URL you configure — is a trust decision you
  make, not a boundary the platform enforces.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub Security Advisories: go to the repository's
**Security** tab and press **Report a vulnerability**
([direct link](https://github.com/limitedonlyde/nounbox/security/advisories/new)).
That opens a private thread visible only to you and the maintainer.

Helpful to include:

- what an attacker gains, and what access they need to start;
- how you run the platform — prebuilt GHCR images or a local build — and the
  image tag or commit;
- steps to reproduce, ideally the smallest possible ones;
- any log output, with secrets removed.

What to expect: this is a small project maintained by one person in their own
time. Acknowledgement usually within 7 days, a fix as fast as the severity
warrants. There is no bug bounty. You will be credited in the advisory unless
you would rather not be, and I will ask you to hold public details until a
fixed image is published.

### In scope

Things that break the model described above, for example:

- code execution or file writes outside the intended paths from a crafted
  image, PDF or imported dataset;
- path traversal or zip-slip in dataset export or import;
- the Modal token secret, the encryption key, or another stored secret leaking
  through an API response, an export, or the logs;
- bypassing `APP_ACCESS_TOKEN` on the three endpoints it protects;
- SQL injection, or stored XSS via a class name, file name or any other value
  rendered in the review UI;
- a web page you merely visit being able to drive your local instance
  (CSRF, DNS rebinding);
- container escape or privilege escalation from the worker.

### Not vulnerabilities

These are known properties of a single-user, localhost-first tool. Reports
about them will be closed with a link to this section:

- **No authentication and no multi-user support.** Known limitation, listed
  under "Not planned" in the [roadmap](ROADMAP.md). "Unauthenticated API allows
  reading all projects" describes the design.
- **Anything that follows from publishing the ports yourself** without a proxy
  that adds TLS and authentication.
- **The development credentials in `.env.example`.** They are documented as
  development-only, in that file and in the README.
- **The settings endpoints being open when `APP_ACCESS_TOKEN` is unset.** This
  is the documented default and the server warns about it.
- **Presigned image URLs working for whoever holds them** until they expire.
  That is how S3 presigning works, and the UI relies on it.
- **Missing HSTS or other TLS-related headers** on the bundled nginx: it serves
  plain HTTP on localhost by design, and TLS belongs to your reverse proxy.
- **No rate limiting or lockout.** There is nothing to brute-force.
- **A dependency advisory with no demonstrated path into this project.** Open a
  normal issue or a pull request bumping the version instead.
- **Automated scanner output without a working exploit path.**

## Hardening checklist

If more than one machine can reach the instance:

1. Put a reverse proxy with TLS and authentication in front of the web
   container; keep every container port bound to `127.0.0.1`.
2. Set `APP_ACCESS_TOKEN` — `openssl rand -hex 32`.
3. Replace every credential from `.env.example`: `POSTGRES_PASSWORD`,
   `MINIO_ROOT_PASSWORD`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`.
4. Set `S3_PUBLIC_ENDPOINT_URL` to an address the browser can reach — image
   previews are presigned links opened directly against MinIO, so plan the
   network for that.
5. Narrow `CORS_ORIGINS` to the origin you actually serve the UI from.
6. Treat the `appdata` volume as a key store: it holds the settings encryption
   key. Back it up separately and restrict access to it.
7. Use a dedicated Modal workspace with a spending limit, and revoke the token
   in Modal when you are done with GPU labeling.
