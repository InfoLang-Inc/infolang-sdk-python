# infolang-sdk-python — agent instructions

Official **Python SDK** for InfoLang semantic memory. Wraps the InfoLang gateway
REST API. Package name: `infolang`.

## Architecture

- `src/infolang/client.py` — `InfoLang` (sync) + `AsyncInfoLang` (async) facades.
- `src/infolang/_transport.py` — httpx transport: retries, timeouts, error mapping.
- `src/infolang/auth/` — credential providers (API key, dev key, session file, mTLS).
- `src/infolang/resources/` — `memory`, `context`, `health`; request shaping in `_ops.py`.
- `src/infolang/errors.py` / `types.py` — typed errors and Pydantic models.

## Contract

The authoritative REST contract is documented at
https://api.infolang.ai/docs. Verify request/response shapes against it,
never against assumptions.

## Rules

- Sync and async must stay in lockstep — share builders in `resources/_ops.py`,
  never duplicate request shaping.
- New endpoints: add a builder + parser in `_ops.py`, then the sync and async
  resource methods, then the top-level alias on both clients.
- Keep `fetch`-equivalent ergonomics: one-line construction, typed errors.

## Commands

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```
