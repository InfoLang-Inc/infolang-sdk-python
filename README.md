# InfoLang Python SDK

Official Python client for [InfoLang](https://infolang.ai) semantic memory.
Wraps the gateway `/v2` REST API (https://api.infolang.ai/docs) with one-line
construction, automatic workspace resolution, typed errors, automatic retries,
and sync/async clients. Package: `infolang` (PyPI).

## Install

```bash
pip install infolang
```

## Quickstart

```python
from infolang import InfoLang

il = InfoLang.from_api_key("il_live_...")          # managed cloud
result = il.investigate("How does auth middleware work?")
for chunk in result.chunks:
    print(chunk.score, chunk.text)
```

Three ways to call, depending on your runtime:

```python
# 1. One-shot
chunks = InfoLang.from_api_key("il_live_...").investigate("query").chunks

# 2. Durable client (connection pooling)
with InfoLang.from_session_file() as il:        # OAuth via ~/.config/infolang/session.json
    il.memorize("a fact worth keeping", source="docs/auth.md")

# 3. Async
import asyncio
from infolang import AsyncInfoLang

async def main():
    async with AsyncInfoLang.from_api_key("il_live_...") as il:
        result = await il.recall("auth middleware", top_k=5)
        print(len(result.chunks))

asyncio.run(main())
```

## Authentication

| Mode | Constructor | Target |
|------|-------------|--------|
| Managed cloud (API key) | `InfoLang.from_api_key("il_live_...")` | `api.infolang.ai` |
| Managed cloud (OAuth) | `InfoLang.from_session_file()` | `api.infolang.ai` |
| Dev key (deprecated) | `InfoLang.from_dev_key("key:namespace")` | `api.infolang.ai` (pass `base_url=` for self-hosted) |
| mTLS (deprecated) | `InfoLang.from_mtls("client.pem", "client-key.pem")` | `api.infolang.ai` (pass `base_url=` for self-hosted) |

All modes default to the managed cloud gateway; dev keys and mTLS certs no
longer redirect to the removed direct endpoint.

Credentials are also read from the environment: `INFOLANG_API_KEY`,
`INFOLANG_DEV_KEY`, `INFOLANG_BASE_URL`, `INFOLANG_NAMESPACE`.

### Workspaces

Every call operates in a workspace (`/v2/workspaces/{id}/...`). A credential
with exactly one workspace grant needs no configuration — the SDK resolves it
once via `whoami()` and caches it. Multi-workspace credentials must pass
`workspace=` (or set `INFOLANG_WORKSPACE`). Workspace ids are opaque strings.

## Core API

| Method | Purpose |
|--------|---------|
| `recall(query, *, namespace, top_k, verbose, golden, format, snippet_chars, adaptive, margin)` | Semantic recall (golden/format extras are reserved) |
| `recall_hybrid(query, *, namespace, top_k, tag_filter, candidate_pool)` | Recall over a candidate pool with tag-inclusion ordering |
| `investigate(query, *, namespace_hint, top_k=5)` | Agent-style recall |
| `remember(text, *, source, tags, namespace)` | Store a memory |
| `remember_batch(items, *, namespace, source)` | Store many memories in one call |
| `memorize(content, *, source, tags, namespace)` | Alias of `remember` |
| `forget(memory_id, *, namespace)` | Delete a memory |
| `reset_namespace(namespace)` | Bulk clear a namespace (list + forget) |
| `list(*, namespace, query, limit)` | List or search memories (paged) |
| `namespaces()` | Namespaces with memory/chunk counts |
| `whoami()` | Identity echo and workspace grants |
| `encode(text)` / `similarity(text1, text2)` | Embedding utilities |
| `execute(operations)` / `stats()` | Batch ops / server stats (OpResult) |
| `ingest(archive_bytes, *, namespace, tag_prefix)` | Ingest a zip of text content |
| `health.check()` / `health.ready()` | Liveness / readiness |

`list_recent()` remains as a deprecated shim over `list()`. `list_banks()`,
`context_pack()`, and `ingest_repo()` were removed in 0.3.0 — their endpoints
no longer exist.

### Bulk ingest + hybrid recall

Useful when running evals or benchmarks against a scratch namespace:

```python
il = InfoLang.from_dev_key("devsecret:default")   # deprecated; pass base_url= for self-hosted

il.reset_namespace("eval_run")                      # clean slate
il.remember_batch(
    [
        {"text": "Alice: I moved to Berlin in March 2024.",
         "tags": ["alice", "march", "2024", "session_1"]},
        {"text": "Bob: My flight is on the 12th.",
         "tags": ["bob", "session_2"]},
    ],
    namespace="eval_run",
)

hits = il.recall_hybrid(
    "When did Alice move to Berlin?",
    namespace="eval_run",
    top_k=20,
    tag_filter=["march", "2024"],   # restrict to chunks carrying these tags
    candidate_pool=500,
)
for chunk in hits.chunks:
    print(chunk.score, chunk.tags, chunk.text)
```

`recall_hybrid` fetches up to `candidate_pool` IL-ranked candidates, then (when
`tag_filter` is given) re-orders so chunks sharing a tag come first and
truncates to `top_k`. No server-side hybrid op required.

## Errors

All failures raise a subclass of `InfoLangError`: `AuthenticationError`,
`RateLimitError` (with `retry_after`), `NotFoundError`, `ValidationError`,
`ServerError`, plus `InfoLangConnectionError` for transport failures. Every
API error carries `status`, `body`, and `request_id`.

## Resilience

`recall`/`remember` and friends retry `429` and `5xx` with exponential backoff
plus full jitter (configurable via `max_retries`), and honor `Retry-After`.
Timeouts default to connect 5s / read 30s.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```

The REST API is documented at https://api.infolang.ai/docs.

## License

Apache-2.0
