"""Health resource: liveness/readiness checks against the gateway."""

from __future__ import annotations

from typing import Any

from .._transport import AsyncTransport, Transport
from . import _ops


class Health:
    """Synchronous health checks."""

    def __init__(self, transport: Transport) -> None:
        self._t = transport

    def check(self) -> dict[str, Any]:
        """Liveness (``GET /healthz``)."""

        method, path, _ = _ops.build_health()
        data, _ = self._t.request(method, path)
        return data if isinstance(data, dict) else {"status": data}

    def ready(self) -> dict[str, Any]:
        """Readiness (``GET /readyz``) — the authoritative "is the engine up" signal."""

        method, path, _ = _ops.build_ready()
        data, _ = self._t.request(method, path)
        return data if isinstance(data, dict) else {"status": data}


class AsyncHealth:
    """Asynchronous health checks."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._t = transport

    async def check(self) -> dict[str, Any]:
        """Liveness (``GET /healthz``)."""

        method, path, _ = _ops.build_health()
        data, _ = await self._t.request(method, path)
        return data if isinstance(data, dict) else {"status": data}

    async def ready(self) -> dict[str, Any]:
        """Readiness (``GET /readyz``) — the authoritative "is the engine up" signal."""

        method, path, _ = _ops.build_ready()
        data, _ = await self._t.request(method, path)
        return data if isinstance(data, dict) else {"status": data}
