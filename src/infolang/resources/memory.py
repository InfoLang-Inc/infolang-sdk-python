"""Memory resource: recall, remember, forget, list, and agent-friendly aliases."""

from __future__ import annotations

import builtins
import warnings
from collections.abc import Awaitable, Callable
from typing import Any

from .._transport import AsyncTransport, Transport
from ..types import (
    MemoryItem,
    MemoryPage,
    NamespaceInfo,
    RecallResult,
    RememberResult,
)
from . import _ops


class Memory:
    """Synchronous memory operations."""

    def __init__(
        self,
        transport: Transport,
        workspace: Callable[[], str],
        default_namespace: str | None,
    ) -> None:
        self._t = transport
        self._workspace = workspace
        self._ns = default_namespace

    def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        verbose: bool | None = None,
        golden: bool | None = None,
        format: str | None = None,
        snippet_chars: int | None = None,
        adaptive: bool | None = None,
        margin: float | None = None,
    ) -> RecallResult:
        """Semantic recall over the workspace.

        ``golden`` is the token-optimal preset (adaptive top-k + snippet
        rendering); ``format`` is one of ``full | lossless | exact | snippet |
        meta``; ``snippet_chars`` sizes the snippet window; ``adaptive`` trims
        hits past a similarity cliff with ``margin`` as the cutoff.

        Reserved: accepted today, activates in a future release.
        """

        method, path, body = _ops.build_recall(
            self._workspace(),
            query,
            namespace=namespace or self._ns,
            top_k=top_k,
            verbose=verbose,
            golden=golden,
            format=format,
            snippet_chars=snippet_chars,
            adaptive=adaptive,
            margin=margin,
        )
        data, metering = self._t.request(method, path, json=body)
        return _ops.parse_recall(data, metering)

    def investigate(
        self,
        query: str,
        *,
        namespace_hint: str | None = None,
        top_k: int | None = 5,
    ) -> RecallResult:
        """Agent-style recall with a sensible default ``top_k`` of 5."""

        return self.recall(query, namespace=namespace_hint, top_k=top_k)

    def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        source: str | None = None,
        tags: str | list[str] | None = None,
    ) -> RememberResult:
        method, path, body = _ops.build_remember(
            self._workspace(),
            text,
            namespace=namespace or self._ns,
            source=source,
            tags=tags,
        )
        data, _ = self._t.request(method, path, json=body)
        return _ops.parse_remember(data)

    def memorize(
        self,
        content: str,
        *,
        source: str | None = None,
        namespace: str | None = None,
        tags: str | list[str] | None = None,
    ) -> RememberResult:
        """Alias for :meth:`remember` matching the ``auto_memorize`` tool."""

        return self.remember(content, namespace=namespace, source=source, tags=tags)

    def forget(self, memory_id: str, *, namespace: str | None = None) -> None:
        method, path, body = _ops.build_forget(
            self._workspace(), memory_id, namespace=namespace or self._ns
        )
        self._t.request(method, path, json=body)

    def list(
        self,
        *,
        namespace: str | None = None,
        query: str | None = None,
        limit: int | None = None,
    ) -> MemoryPage:
        """List (or, with ``query``, semantically search) memories."""

        method, path, _ = _ops.build_list(
            self._workspace(),
            namespace=namespace or self._ns,
            query=query,
            limit=limit,
        )
        data, _ = self._t.request(method, path)
        return _ops.parse_list(data)

    def namespaces(self) -> builtins.list[NamespaceInfo]:
        """Every namespace in the workspace with its memory and chunk counts."""

        method, path, _ = _ops.build_namespaces(self._workspace())
        data, _ = self._t.request(method, path)
        return _ops.parse_namespaces(data)

    def list_recent(
        self, *, namespace: str | None = None, n: int | None = None
    ) -> builtins.list[MemoryItem]:
        """Deprecated: use :meth:`list`; kept for 0.2.x callers."""

        warnings.warn(
            "list_recent() is deprecated; use list()", DeprecationWarning, stacklevel=2
        )
        return self.list(namespace=namespace, limit=n).memories

    def recall_hybrid(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = 10,
        tag_filter: builtins.list[str] | None = None,
        candidate_pool: int | None = 500,
    ) -> RecallResult:
        """Recall with tag-inclusion ordering over a candidate pool (client-side).

        Fetches up to ``candidate_pool`` ranked candidates, then (when
        ``tag_filter`` is given) re-orders so chunks sharing a tag come first,
        and truncates to ``top_k``.
        """

        ns = namespace or self._ns
        pool = max(candidate_pool or 0, top_k or 0) or top_k
        result = self.recall(query, namespace=ns, top_k=pool)
        result.chunks = _ops.filter_hits_by_tags(result.chunks, tag_filter, top_k)
        return result

    def remember_batch(
        self,
        items: builtins.list[Any],
        *,
        namespace: str | None = None,
        source: str | None = None,
    ) -> builtins.list[RememberResult]:
        """Store many memories in one round trip (execute with remember sub-ops).

        ``items`` may be a list of strings or a list of dicts with ``text`` and
        optional ``tags``/``source`` keys.
        """

        if not items:
            return []
        records = _ops.normalize_remember_items(items, default_source=source)
        method, path, body = _ops.build_remember_batch(
            self._workspace(), records, namespace=namespace or self._ns, source=source
        )
        data, _ = self._t.request(method, path, json=body)
        return _ops.parse_execute_remember_batch(data)

    def reset_namespace(self, namespace: str | None = None, *, batch: int = 500) -> int:
        """Bulk clear a namespace (list + forget loop).

        Returns the number of memories deleted. O(N) over HTTP — prefer
        per-unit namespaces.
        """

        ns = namespace or self._ns
        deleted = 0
        while True:
            page = self.list(namespace=ns, limit=batch)
            ids = [m.id for m in page.memories if m.id]
            if not ids:
                break
            for mid in ids:
                self.forget(mid, namespace=ns)
                deleted += 1
            if len(ids) < batch:
                break
        return deleted


class AsyncMemory:
    """Asynchronous memory operations."""

    def __init__(
        self,
        transport: AsyncTransport,
        workspace: Callable[[], Awaitable[str]],
        default_namespace: str | None,
    ) -> None:
        self._t = transport
        self._workspace = workspace
        self._ns = default_namespace

    async def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        verbose: bool | None = None,
        golden: bool | None = None,
        format: str | None = None,
        snippet_chars: int | None = None,
        adaptive: bool | None = None,
        margin: float | None = None,
    ) -> RecallResult:
        """Semantic recall over the workspace.

        Reserved: ``golden``/``format``/``snippet_chars``/``adaptive``/``margin``
        are accepted today and activate in a future release.
        """

        method, path, body = _ops.build_recall(
            await self._workspace(),
            query,
            namespace=namespace or self._ns,
            top_k=top_k,
            verbose=verbose,
            golden=golden,
            format=format,
            snippet_chars=snippet_chars,
            adaptive=adaptive,
            margin=margin,
        )
        data, metering = await self._t.request(method, path, json=body)
        return _ops.parse_recall(data, metering)

    async def investigate(
        self,
        query: str,
        *,
        namespace_hint: str | None = None,
        top_k: int | None = 5,
    ) -> RecallResult:
        return await self.recall(query, namespace=namespace_hint, top_k=top_k)

    async def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        source: str | None = None,
        tags: str | list[str] | None = None,
    ) -> RememberResult:
        method, path, body = _ops.build_remember(
            await self._workspace(),
            text,
            namespace=namespace or self._ns,
            source=source,
            tags=tags,
        )
        data, _ = await self._t.request(method, path, json=body)
        return _ops.parse_remember(data)

    async def memorize(
        self,
        content: str,
        *,
        source: str | None = None,
        namespace: str | None = None,
        tags: str | list[str] | None = None,
    ) -> RememberResult:
        return await self.remember(content, namespace=namespace, source=source, tags=tags)

    async def forget(self, memory_id: str, *, namespace: str | None = None) -> None:
        method, path, body = _ops.build_forget(
            await self._workspace(), memory_id, namespace=namespace or self._ns
        )
        await self._t.request(method, path, json=body)

    async def list(
        self,
        *,
        namespace: str | None = None,
        query: str | None = None,
        limit: int | None = None,
    ) -> MemoryPage:
        method, path, _ = _ops.build_list(
            await self._workspace(),
            namespace=namespace or self._ns,
            query=query,
            limit=limit,
        )
        data, _ = await self._t.request(method, path)
        return _ops.parse_list(data)

    async def namespaces(self) -> builtins.list[NamespaceInfo]:
        method, path, _ = _ops.build_namespaces(await self._workspace())
        data, _ = await self._t.request(method, path)
        return _ops.parse_namespaces(data)

    async def list_recent(
        self, *, namespace: str | None = None, n: int | None = None
    ) -> builtins.list[MemoryItem]:
        """Deprecated: use :meth:`list`; kept for 0.2.x callers."""

        warnings.warn(
            "list_recent() is deprecated; use list()", DeprecationWarning, stacklevel=2
        )
        page = await self.list(namespace=namespace, limit=n)
        return page.memories

    async def recall_hybrid(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = 10,
        tag_filter: builtins.list[str] | None = None,
        candidate_pool: int | None = 500,
    ) -> RecallResult:
        ns = namespace or self._ns
        pool = max(candidate_pool or 0, top_k or 0) or top_k
        result = await self.recall(query, namespace=ns, top_k=pool)
        result.chunks = _ops.filter_hits_by_tags(result.chunks, tag_filter, top_k)
        return result

    async def remember_batch(
        self,
        items: builtins.list[Any],
        *,
        namespace: str | None = None,
        source: str | None = None,
    ) -> builtins.list[RememberResult]:
        if not items:
            return []
        records = _ops.normalize_remember_items(items, default_source=source)
        method, path, body = _ops.build_remember_batch(
            await self._workspace(),
            records,
            namespace=namespace or self._ns,
            source=source,
        )
        data, _ = await self._t.request(method, path, json=body)
        return _ops.parse_execute_remember_batch(data)

    async def reset_namespace(
        self, namespace: str | None = None, *, batch: int = 500
    ) -> int:
        ns = namespace or self._ns
        deleted = 0
        while True:
            page = await self.list(namespace=ns, limit=batch)
            ids = [m.id for m in page.memories if m.id]
            if not ids:
                break
            for mid in ids:
                await self.forget(mid, namespace=ns)
                deleted += 1
            if len(ids) < batch:
                break
        return deleted
