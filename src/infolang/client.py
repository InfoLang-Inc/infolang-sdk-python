"""The InfoLang client: one-line construction over the gateway /v2 API.

Two clients are provided. :class:`InfoLang` is synchronous; :class:`AsyncInfoLang`
mirrors it with ``async``/``await``. Both expose grouped resources
(``client.memory``, ``client.health``) plus the common operations as top-level
aliases so the first call is a one-liner.

Workspace resolution is automatic: a credential with exactly one workspace
grant needs no configuration; multi-workspace credentials pass ``workspace=``
(or set ``INFOLANG_WORKSPACE``).
"""

from __future__ import annotations

import asyncio
import builtins
import os
from typing import Any

import httpx

from ._transport import AsyncTransport, Transport
from .auth import ApiKeyAuth, AuthProvider, DevKeyAuth, MtlsAuth, SessionFileAuth
from .errors import InfoLangConfigError
from .resources import AsyncHealth, AsyncMemory, Health, Memory, _ops
from .types import (
    IngestJob,
    MemoryItem,
    MemoryPage,
    NamespaceInfo,
    OpResult,
    RecallResult,
    RememberResult,
    Whoami,
)

CLOUD_BASE_URL = "https://api.infolang.ai"
#: Deprecated: the direct endpoint is gone; kept only so 0.2.x imports work.
DIRECT_BASE_URL = "http://127.0.0.1:8766"

_NO_GRANTS_MESSAGE = (
    "This credential has no visible workspace grants; mint a key from the "
    "Console or pass workspace= explicitly."
)


def _resolve_auth(
    auth: AuthProvider | None,
    api_key: str | None,
    dev_key: str | None,
) -> AuthProvider:
    if auth is not None:
        return auth
    if api_key:
        return ApiKeyAuth(api_key)
    if dev_key:
        return DevKeyAuth(dev_key)
    env_key = os.environ.get("INFOLANG_API_KEY")
    if env_key:
        return ApiKeyAuth(env_key)
    env_dev = os.environ.get("INFOLANG_DEV_KEY")
    if env_dev:
        return DevKeyAuth(env_dev)
    raise InfoLangConfigError(
        "No credentials. Pass api_key=, dev_key=, or auth=, "
        "set INFOLANG_API_KEY, or use InfoLang.from_session_file()."
    )


def _resolve_base_url(base_url: str | None) -> str:
    if base_url:
        return base_url
    env_url = os.environ.get("INFOLANG_BASE_URL")
    if env_url:
        return env_url
    # 0.3.0: everything defaults to the managed cloud gateway. Dev keys and
    # mTLS certs no longer redirect to DIRECT_BASE_URL (the direct endpoint is
    # gone); the auth classes remain functional against a custom base_url.
    return CLOUD_BASE_URL


def _default_namespace(namespace: str | None, auth: AuthProvider) -> str | None:
    if namespace:
        return namespace
    if isinstance(auth, DevKeyAuth):
        return auth.namespace
    return os.environ.get("INFOLANG_NAMESPACE")


def _default_workspace(workspace: str | None) -> str | None:
    if workspace:
        return workspace
    return os.environ.get("INFOLANG_WORKSPACE") or os.environ.get(
        "INFOLANG_WORKSPACE_ID"
    )


def _pick_workspace(who: Whoami) -> str:
    """Resolve a single workspace grant, or raise a config error."""

    ids = [w.workspace_id for w in who.workspaces if w.workspace_id]
    if len(ids) == 1:
        return ids[0]
    if not ids:
        raise InfoLangConfigError(_NO_GRANTS_MESSAGE)
    raise InfoLangConfigError(
        f"This credential can reach {len(ids)} workspaces ({', '.join(ids)}); "
        "pass workspace= (or set INFOLANG_WORKSPACE) to pick one."
    )


class InfoLang:
    """Synchronous InfoLang client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        dev_key: str | None = None,
        auth: AuthProvider | None = None,
        base_url: str | None = None,
        namespace: str | None = None,
        workspace: str | None = None,
        timeout: httpx.Timeout | None = None,
        max_retries: int = 2,
    ) -> None:
        resolved_auth = _resolve_auth(auth, api_key, dev_key)
        self._base_url = _resolve_base_url(base_url)
        self.namespace = _default_namespace(namespace, resolved_auth)
        self.workspace = _default_workspace(workspace)
        self._workspace_cached: str | None = None
        self._transport = Transport(
            base_url=self._base_url,
            auth=resolved_auth,
            timeout=timeout,
            max_retries=max_retries,
            workspace_id=self.workspace,
        )
        self.memory = Memory(self._transport, self.workspace_id, self.namespace)
        self.health = Health(self._transport)

    # --- constructors ---------------------------------------------------

    @classmethod
    def from_api_key(cls, api_key: str, **kwargs: Any) -> InfoLang:
        return cls(api_key=api_key, **kwargs)

    @classmethod
    def from_dev_key(cls, dev_key: str, **kwargs: Any) -> InfoLang:
        return cls(dev_key=dev_key, **kwargs)

    @classmethod
    def from_session_file(cls, path: str | None = None, **kwargs: Any) -> InfoLang:
        return cls(auth=SessionFileAuth(path), **kwargs)

    @classmethod
    def from_mtls(cls, cert: str, key: str, **kwargs: Any) -> InfoLang:
        return cls(auth=MtlsAuth(cert, key), **kwargs)

    # --- identity & workspace resolution --------------------------------

    def whoami(self) -> Whoami:
        """Identity echo for the current credential (``GET /v2/whoami``)."""

        method, path, _ = _ops.build_whoami()
        data, _ = self._transport.request(method, path)
        return _ops.parse_whoami(data)

    def workspace_id(self) -> str:
        """The workspace id every call operates in.

        Explicit configuration wins; otherwise resolved once via ``whoami``
        (single-grant credentials only — multi-workspace credentials must pass
        ``workspace=``). A failed resolution is not cached, so a retry re-asks.
        """

        if self.workspace:
            return self.workspace
        if self._workspace_cached is None:
            self._workspace_cached = _pick_workspace(self.whoami())
        return self._workspace_cached

    # --- top-level aliases ----------------------------------------------

    def recall(self, query: str, **kwargs: Any) -> RecallResult:
        return self.memory.recall(query, **kwargs)

    def recall_hybrid(self, query: str, **kwargs: Any) -> RecallResult:
        return self.memory.recall_hybrid(query, **kwargs)

    def investigate(self, query: str, **kwargs: Any) -> RecallResult:
        return self.memory.investigate(query, **kwargs)

    def remember(self, text: str, **kwargs: Any) -> RememberResult:
        return self.memory.remember(text, **kwargs)

    def remember_batch(self, items: list[Any], **kwargs: Any) -> list[RememberResult]:
        return self.memory.remember_batch(items, **kwargs)

    def memorize(self, content: str, **kwargs: Any) -> RememberResult:
        return self.memory.memorize(content, **kwargs)

    def forget(self, memory_id: str, **kwargs: Any) -> None:
        self.memory.forget(memory_id, **kwargs)

    def reset_namespace(self, namespace: str | None = None, **kwargs: Any) -> int:
        return self.memory.reset_namespace(namespace, **kwargs)

    def list(self, **kwargs: Any) -> MemoryPage:
        return self.memory.list(**kwargs)

    def namespaces(self) -> builtins.list[NamespaceInfo]:
        return self.memory.namespaces()

    def list_recent(self, **kwargs: Any) -> builtins.list[MemoryItem]:
        """Deprecated: use :meth:`list`."""

        return self.memory.list_recent(**kwargs)

    def execute(self, operations: builtins.list[dict[str, Any]]) -> OpResult:
        """Batch primitives in one round trip; one OpResult per sub-op, in order."""

        method, path, body = _ops.build_execute(self.workspace_id(), operations)
        data, _ = self._transport.request(method, path, json=body)
        return _ops.parse_op_result(data)

    def encode(self, text: str) -> dict[str, Any]:
        """Encode text to an embedding (response is the server payload, verbatim)."""

        method, path, body = _ops.build_encode(self.workspace_id(), text)
        data, _ = self._transport.request(method, path, json=body)
        return data if isinstance(data, dict) else {}

    def similarity(self, text1: str, text2: str) -> dict[str, Any]:
        """Similarity between two texts (response is the server payload, verbatim)."""

        method, path, body = _ops.build_similarity(self.workspace_id(), text1, text2)
        data, _ = self._transport.request(method, path, json=body)
        return data if isinstance(data, dict) else {}

    def stats(self) -> OpResult:
        """Server stats for the workspace (native OpResult envelope)."""

        method, path, _ = _ops.build_stats(self.workspace_id())
        data, _ = self._transport.request(method, path)
        return _ops.parse_op_result(data)

    def ingest(
        self,
        archive: bytes,
        *,
        namespace: str | None = None,
        tag_prefix: str | None = None,
        content_type: str = "application/zip",
    ) -> IngestJob:
        """Ingest a zip of text content (raw bytes body, 202 response)."""

        method, path = _ops.build_ingest(
            self.workspace_id(),
            namespace=namespace or self.namespace,
            tag_prefix=tag_prefix,
        )
        data, _ = self._transport.request(
            method, path, content=archive, content_type=content_type
        )
        return _ops.parse_ingest(data)

    def ingest_files(
        self,
        files: builtins.list[tuple[str, bytes | str]],
        *,
        namespace: str | None = None,
        tag_prefix: str | None = None,
    ) -> IngestJob:
        """Ingest individual text files (multipart ``files`` parts, no zip step)."""

        method, path = _ops.build_ingest(
            self.workspace_id(),
            namespace=namespace or self.namespace,
            tag_prefix=tag_prefix,
        )
        data, _ = self._transport.request(
            method, path, files=_ops.multipart_files(files)
        )
        return _ops.parse_ingest(data)

    # --- lifecycle ------------------------------------------------------

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> InfoLang:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncInfoLang:
    """Asynchronous InfoLang client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        dev_key: str | None = None,
        auth: AuthProvider | None = None,
        base_url: str | None = None,
        namespace: str | None = None,
        workspace: str | None = None,
        timeout: httpx.Timeout | None = None,
        max_retries: int = 2,
    ) -> None:
        resolved_auth = _resolve_auth(auth, api_key, dev_key)
        self._base_url = _resolve_base_url(base_url)
        self.namespace = _default_namespace(namespace, resolved_auth)
        self.workspace = _default_workspace(workspace)
        self._workspace_cached: str | None = None
        self._workspace_lock = asyncio.Lock()
        self._transport = AsyncTransport(
            base_url=self._base_url,
            auth=resolved_auth,
            timeout=timeout,
            max_retries=max_retries,
            workspace_id=self.workspace,
        )
        self.memory = AsyncMemory(self._transport, self.workspace_id, self.namespace)
        self.health = AsyncHealth(self._transport)

    @classmethod
    def from_api_key(cls, api_key: str, **kwargs: Any) -> AsyncInfoLang:
        return cls(api_key=api_key, **kwargs)

    @classmethod
    def from_dev_key(cls, dev_key: str, **kwargs: Any) -> AsyncInfoLang:
        return cls(dev_key=dev_key, **kwargs)

    @classmethod
    def from_session_file(cls, path: str | None = None, **kwargs: Any) -> AsyncInfoLang:
        return cls(auth=SessionFileAuth(path), **kwargs)

    @classmethod
    def from_mtls(cls, cert: str, key: str, **kwargs: Any) -> AsyncInfoLang:
        return cls(auth=MtlsAuth(cert, key), **kwargs)

    # --- identity & workspace resolution --------------------------------

    async def whoami(self) -> Whoami:
        """Identity echo for the current credential (``GET /v2/whoami``)."""

        method, path, _ = _ops.build_whoami()
        data, _ = await self._transport.request(method, path)
        return _ops.parse_whoami(data)

    async def workspace_id(self) -> str:
        """The workspace id every call operates in (see :meth:`InfoLang.workspace_id`)."""

        if self.workspace:
            return self.workspace
        if self._workspace_cached is not None:
            return self._workspace_cached
        async with self._workspace_lock:
            # Re-check inside the lock so concurrent first calls resolve once.
            if self._workspace_cached is None:
                # A raise here leaves the cache empty — failures are not cached.
                self._workspace_cached = _pick_workspace(await self.whoami())
            return self._workspace_cached

    # --- top-level aliases ----------------------------------------------

    async def recall(self, query: str, **kwargs: Any) -> RecallResult:
        return await self.memory.recall(query, **kwargs)

    async def recall_hybrid(self, query: str, **kwargs: Any) -> RecallResult:
        return await self.memory.recall_hybrid(query, **kwargs)

    async def investigate(self, query: str, **kwargs: Any) -> RecallResult:
        return await self.memory.investigate(query, **kwargs)

    async def remember(self, text: str, **kwargs: Any) -> RememberResult:
        return await self.memory.remember(text, **kwargs)

    async def remember_batch(
        self, items: list[Any], **kwargs: Any
    ) -> list[RememberResult]:
        return await self.memory.remember_batch(items, **kwargs)

    async def memorize(self, content: str, **kwargs: Any) -> RememberResult:
        return await self.memory.memorize(content, **kwargs)

    async def forget(self, memory_id: str, **kwargs: Any) -> None:
        await self.memory.forget(memory_id, **kwargs)

    async def reset_namespace(
        self, namespace: str | None = None, **kwargs: Any
    ) -> int:
        return await self.memory.reset_namespace(namespace, **kwargs)

    async def list(self, **kwargs: Any) -> MemoryPage:
        return await self.memory.list(**kwargs)

    async def namespaces(self) -> builtins.list[NamespaceInfo]:
        return await self.memory.namespaces()

    async def list_recent(self, **kwargs: Any) -> builtins.list[MemoryItem]:
        """Deprecated: use :meth:`list`."""

        return await self.memory.list_recent(**kwargs)

    async def execute(self, operations: builtins.list[dict[str, Any]]) -> OpResult:
        """Batch primitives in one round trip; one OpResult per sub-op, in order."""

        method, path, body = _ops.build_execute(await self.workspace_id(), operations)
        data, _ = await self._transport.request(method, path, json=body)
        return _ops.parse_op_result(data)

    async def encode(self, text: str) -> dict[str, Any]:
        """Encode text to an embedding (response is the server payload, verbatim)."""

        method, path, body = _ops.build_encode(await self.workspace_id(), text)
        data, _ = await self._transport.request(method, path, json=body)
        return data if isinstance(data, dict) else {}

    async def similarity(self, text1: str, text2: str) -> dict[str, Any]:
        """Similarity between two texts (response is the server payload, verbatim)."""

        method, path, body = _ops.build_similarity(
            await self.workspace_id(), text1, text2
        )
        data, _ = await self._transport.request(method, path, json=body)
        return data if isinstance(data, dict) else {}

    async def stats(self) -> OpResult:
        """Server stats for the workspace (native OpResult envelope)."""

        method, path, _ = _ops.build_stats(await self.workspace_id())
        data, _ = await self._transport.request(method, path)
        return _ops.parse_op_result(data)

    async def ingest(
        self,
        archive: bytes,
        *,
        namespace: str | None = None,
        tag_prefix: str | None = None,
        content_type: str = "application/zip",
    ) -> IngestJob:
        """Ingest a zip of text content (raw bytes body, 202 response)."""

        method, path = _ops.build_ingest(
            await self.workspace_id(),
            namespace=namespace or self.namespace,
            tag_prefix=tag_prefix,
        )
        data, _ = await self._transport.request(
            method, path, content=archive, content_type=content_type
        )
        return _ops.parse_ingest(data)

    async def ingest_files(
        self,
        files: builtins.list[tuple[str, bytes | str]],
        *,
        namespace: str | None = None,
        tag_prefix: str | None = None,
    ) -> IngestJob:
        """Ingest individual text files (multipart ``files`` parts, no zip step)."""

        method, path = _ops.build_ingest(
            await self.workspace_id(),
            namespace=namespace or self.namespace,
            tag_prefix=tag_prefix,
        )
        data, _ = await self._transport.request(
            method, path, files=_ops.multipart_files(files)
        )
        return _ops.parse_ingest(data)

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> AsyncInfoLang:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
