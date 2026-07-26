# Changelog

All notable changes to the InfoLang Python SDK are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.3.0] - 2026-07-25

**BREAKING**: the SDK now targets the InfoLang gateway `/v2` API
(https://api.infolang.ai/docs). The legacy `/v1` endpoints this SDK
called in 0.2.x (`/v1/banks`, `/v1/context-pack`, `/v1/repos/{ns}/ingest`, the
bare-batch `/v1/execute`) no longer exist; `/v1` as a whole is deprecated.

### Added
- `whoami()` (`GET /v2/whoami`) and automatic workspace resolution: a
  credential with exactly one workspace grant needs no configuration;
  multi-workspace credentials pass `workspace=` (or `INFOLANG_WORKSPACE`).
  Workspace ids are opaque strings — never parsed client-side.
- Recall retrieval extras: `golden`, `format`, `snippet_chars`, `adaptive`,
  `margin`.
  **Reserved:** these options are accepted today and activate in a
  future release — no SDK change needed.
- `list()` (gateway MemoryPage: `ns`/`q`/`limit`, `next_cursor`, search
  scores), `namespaces()` with per-namespace memory/chunk counts.
- `encode()`, `similarity()`, `stats()`, `execute()` (simple `{operations}`
  batch).
- `ingest()`: zip archives via `POST /v2/workspaces/{ws}/ingest`.
- `ingest_files()` (sync + async): multipart upload of individual text files
  (repeated `files` parts) on the same ingest endpoint — no zip step.
- `health.ready()` (`GET /readyz`) — the authoritative readiness signal.
- Remember response fields surfaced: `stored`, `deduplicated`,
  `deduped_against`, `total_memories`; `X-InfoLang-Overage` surfaced as
  `metering.overage`; `{error:{code,message}}` envelopes surfaced (plus an
  `error_code()` helper).

### Changed
- `forget()` sends the namespace (`DELETE /v2/.../memories/{id}?namespace=`).
- `remember_batch()` is client-side sugar over `execute` with one real
  `remember` sub-op per item (the `remember_batch` pseudo-op is gone).
- String `tags` are split into the list the API expects.
- All credentials now default to the managed cloud base URL; dev keys and
  mTLS no longer redirect to the removed direct endpoint.
- `lane_not_supported` is now **403** (was 401); both map to
  `AuthenticationError`, so no caller change is needed.
- Multi-workspace API keys now get their full workspace list from
  `GET /v2/whoami`, so the multi-grant resolution error can name the
  candidate ids.

### Removed
- `list_banks()`, `context_pack()`, `ingest_repo()` and the context resource —
  their endpoints no longer exist. `list_recent()` and `DIRECT_BASE_URL`
  remain as deprecated shims.

## [0.2.0] - 2026-07-13

### Changed
- Pinned OpenAPI contract to **v0.2.0**.
- `forget(memory_id)` now calls `DELETE /v1/memories/{id}` (was `POST /v1/forget`).
- `list_recent(..., n=)` now calls `GET /v1/memories?limit=` (was `GET /v1/recent`).

### Added
- `recall_hybrid(query, *, namespace, top_k, tag_filter, candidate_pool, use_hybrid)`
  — fetches up to `candidate_pool` IL-ranked candidates then applies client-side
  tag-inclusion ordering and truncates to `top_k`.
- Fixed `recall` / `investigate` parsing: the `/v1/recall` route returns
  `{"hits": [...]}` with a `similarity` score; these are mapped onto
  `RecallResult.chunks`.
- `remember_batch(items, *, namespace, source)` — store many memories via
  `POST /v1/execute` `remember_batch`.
- `reset_namespace(namespace)` — bulk clear a namespace (list + forget loop).
- `list_banks` maps `total_memories` onto `Bank.count`.
- PyPI publish workflow (trusted publisher / OIDC).

## [0.1.0] - Unreleased

### Added
- Initial release: `InfoLang` (sync) and `AsyncInfoLang` (async) clients.
- Auth providers: API key, dev key (`key:namespace`), OAuth session file, mTLS.
- Memory API: `recall`, `investigate`, `remember`, `memorize`, `forget`,
  `list_banks`, `list_recent`.
- Context API: `context_pack`, `ingest_repo`, `execute`.
- Typed error hierarchy, automatic retries with jitter, and metering metadata.
