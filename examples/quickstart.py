"""InfoLang Python SDK — quickstart.

Run against the managed cloud:

    export INFOLANG_API_KEY=il_live_...
    python examples/quickstart.py

Or against a self-hosted base URL (dev key key:namespace):

    export INFOLANG_DEV_KEY=devsecret:default
    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from infolang import AsyncInfoLang, InfoLang


def sync_example() -> None:
    # Credentials resolved from INFOLANG_API_KEY / INFOLANG_DEV_KEY.
    with InfoLang() as il:
        result = il.investigate("How does auth middleware work?")
        if result.weak:
            print("(weak match — consider narrowing the query)")
        for chunk in result.chunks:
            print(f"[{chunk.score:.2f}] {chunk.text[:120]}")

        il.memorize("Auth middleware validates bearer tokens via Supabase.", source="docs/auth.md")


async def async_example() -> None:
    async with AsyncInfoLang() as il:
        result = await il.recall("context-pack token savings", top_k=3)
        print(f"recalled {len(result.chunks)} chunks from {result.namespace}")


if __name__ == "__main__":
    sync_example()
    asyncio.run(async_example())
