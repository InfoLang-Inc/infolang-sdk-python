"""HTTP transport for the InfoLang SDK.

Wraps ``httpx`` with the resilience defaults expected of a modern SDK: a shared
connection pool, explicit timeout budgets, and targeted retries (429 + 5xx)
using exponential backoff with full jitter. Both a sync and an async transport
are provided; they share request building, error mapping and retry policy.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx

from ._version import __version__
from .auth import AuthProvider
from .errors import InfoLangConnectionError, error_from_response
from .types import MeteringMeta

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
_DEFAULT_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)


class _RetryPolicy:
    def __init__(self, max_retries: int, backoff_base: float, backoff_cap: float) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap

    def delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return retry_after
        # Full jitter: random between 0 and the capped exponential window.
        window = min(self.backoff_cap, self.backoff_base * (2**attempt))
        return random.uniform(0, window)


def _build_headers(
    auth: AuthProvider,
    extra: dict[str, str] | None,
    workspace_id: str | None = None,
    *,
    content_type: str | None = None,
) -> dict[str, str]:
    headers = {
        "User-Agent": f"infolang-python/{__version__}",
        "Accept": "application/json",
    }
    headers.update(auth.headers())
    if workspace_id:
        headers["X-InfoLang-Workspace-Id"] = workspace_id
    if content_type:
        # Raw (pre-encoded) bodies carry an explicit content type; JSON bodies
        # get theirs from httpx's json= encoding.
        headers["Content-Type"] = content_type
    if extra:
        headers.update(extra)
    return headers


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_metering(headers: httpx.Headers) -> MeteringMeta:
    """Extract managed-cloud usage metadata from response headers."""

    def _int(name: str) -> int | None:
        value = headers.get(name)
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    def _float(name: str) -> float | None:
        value = headers.get(name)
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    return MeteringMeta(
        tokens_saved=_int("x-infolang-tokens-saved"),
        chunks_used=_int("x-infolang-chunks-used"),
        repo_coverage=_float("x-infolang-repo-coverage"),
        overage=_int("x-infolang-overage"),
        request_id=headers.get("x-request-id"),
    )


def _decode(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


class Transport:
    """Synchronous HTTP transport."""

    def __init__(
        self,
        *,
        base_url: str,
        auth: AuthProvider,
        timeout: httpx.Timeout | None = None,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        workspace_id: str | None = None,
    ) -> None:
        self._auth = auth
        self._workspace_id = workspace_id
        self._policy = _RetryPolicy(max_retries, backoff_base, backoff_cap)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout or _DEFAULT_TIMEOUT,
            limits=_DEFAULT_LIMITS,
            **auth.transport_options(),
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        content: bytes | None = None,
        content_type: str | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[Any, MeteringMeta]:
        if content is not None and content_type is None:
            content_type = "application/octet-stream"
        # Multipart: httpx owns the boundary — never set Content-Type for files.
        last_exc: Exception | None = None
        for attempt in range(self._policy.max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    path,
                    json=json if content is None and files is None else None,
                    content=content,
                    files=files,
                    headers=_build_headers(
                        self._auth,
                        headers,
                        self._workspace_id,
                        content_type=None if files is not None else content_type,
                    ),
                )
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt >= self._policy.max_retries:
                    raise InfoLangConnectionError(str(exc)) from exc
                time.sleep(self._policy.delay(attempt, None))
                continue

            if response.status_code in _RETRY_STATUSES and attempt < self._policy.max_retries:
                time.sleep(self._policy.delay(attempt, _retry_after(response)))
                continue
            return _finish(response)

        raise InfoLangConnectionError(str(last_exc))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Transport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncTransport:
    """Asynchronous HTTP transport."""

    def __init__(
        self,
        *,
        base_url: str,
        auth: AuthProvider,
        timeout: httpx.Timeout | None = None,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        workspace_id: str | None = None,
    ) -> None:
        self._auth = auth
        self._workspace_id = workspace_id
        self._policy = _RetryPolicy(max_retries, backoff_base, backoff_cap)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout or _DEFAULT_TIMEOUT,
            limits=_DEFAULT_LIMITS,
            **auth.transport_options(),
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        content: bytes | None = None,
        content_type: str | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[Any, MeteringMeta]:
        if content is not None and content_type is None:
            content_type = "application/octet-stream"
        # Multipart: httpx owns the boundary — never set Content-Type for files.
        last_exc: Exception | None = None
        for attempt in range(self._policy.max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    json=json if content is None and files is None else None,
                    content=content,
                    files=files,
                    headers=_build_headers(
                        self._auth,
                        headers,
                        self._workspace_id,
                        content_type=None if files is not None else content_type,
                    ),
                )
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt >= self._policy.max_retries:
                    raise InfoLangConnectionError(str(exc)) from exc
                await asyncio.sleep(self._policy.delay(attempt, None))
                continue

            if response.status_code in _RETRY_STATUSES and attempt < self._policy.max_retries:
                await asyncio.sleep(self._policy.delay(attempt, _retry_after(response)))
                continue
            return _finish(response)

        raise InfoLangConnectionError(str(last_exc))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncTransport:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _finish(response: httpx.Response) -> tuple[Any, MeteringMeta]:
    metering = parse_metering(response.headers)
    if response.is_success:
        return _decode(response), metering
    raise error_from_response(
        response.status_code,
        _decode(response),
        metering.request_id,
        _retry_after(response),
    )
