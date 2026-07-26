from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..errors import InfoLangConfigError

DEFAULT_SESSION_PATH = Path.home() / ".config" / "infolang" / "session.json"


@runtime_checkable
class AuthProvider(Protocol):
    """Supplies per-request auth headers and optional transport options.

    ``headers()`` is called before every request so providers can refresh
    short-lived tokens. ``transport_options()`` returns httpx client kwargs
    (used by mTLS to attach a client certificate).
    """

    def headers(self) -> dict[str, str]: ...

    def transport_options(self) -> dict[str, Any]: ...


class _BaseAuth:
    def transport_options(self) -> dict[str, Any]:
        return {}


class ApiKeyAuth(_BaseAuth):
    """Bearer authentication with an InfoLang API key (``il_live_...``)."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise InfoLangConfigError("api_key must not be empty")
        self._api_key = api_key

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}


class DevKeyAuth(_BaseAuth):
    """Self-hosted dev key in ``key:namespace`` form (matches INFOLANG_API_KEYS)."""

    def __init__(self, dev_key: str) -> None:
        if ":" not in dev_key:
            raise InfoLangConfigError("dev key must be in 'key:namespace' form")
        self._dev_key = dev_key

    @property
    def namespace(self) -> str:
        return self._dev_key.split(":", 1)[1]

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._dev_key.split(':', 1)[0]}"}


class SessionFileAuth(_BaseAuth):
    """OAuth bearer token read from the cursor-setup session file.

    The file (default ``~/.config/infolang/session.json``) is written by
    ``npx @infolang/cursor-setup``. When it contains a ``refresh_token`` and a
    ``token_url``, an expired access token is refreshed transparently.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = Path(path) if path else DEFAULT_SESSION_PATH
        self._cache: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            raise InfoLangConfigError(
                f"session file not found at {self._path}; run 'npx @infolang/cursor-setup'"
            )
        try:
            data: dict[str, Any] = json.loads(self._path.read_text())
        except json.JSONDecodeError as exc:
            raise InfoLangConfigError(f"session file at {self._path} is not valid JSON") from exc
        return data

    def headers(self) -> dict[str, str]:
        session = self._cache or self._load()
        if self._is_expired(session):
            session = self._refresh(session)
        self._cache = session
        token = session.get("access_token")
        if not token:
            raise InfoLangConfigError("session file is missing 'access_token'")
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _is_expired(session: dict[str, Any]) -> bool:
        expires_at = session.get("expires_at")
        if not isinstance(expires_at, (int, float)):
            return False
        # Refresh 30s early to avoid races at the boundary.
        return time.time() >= float(expires_at) - 30

    def _refresh(self, session: dict[str, Any]) -> dict[str, Any]:
        refresh_token = session.get("refresh_token")
        token_url = session.get("token_url")
        if not refresh_token or not token_url:
            # Can't refresh; hand back what we have and let the API reject it
            # with a clear AuthenticationError.
            return session

        import httpx

        resp = httpx.post(
            token_url,
            json={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        session = {**session, **data}
        if "expires_in" in data:
            session["expires_at"] = time.time() + float(data["expires_in"])
        with contextlib.suppress(OSError):
            self._path.write_text(json.dumps(session))
        return session


class MtlsAuth(_BaseAuth):
    """Enterprise mutual-TLS using a client certificate issued by the InfoLang PKI."""

    def __init__(self, cert: str | os.PathLike[str], key: str | os.PathLike[str]) -> None:
        self._cert = str(cert)
        self._key = str(key)
        if not Path(self._cert).exists():
            raise InfoLangConfigError(f"client cert not found at {self._cert}")
        if not Path(self._key).exists():
            raise InfoLangConfigError(f"client key not found at {self._key}")

    def headers(self) -> dict[str, str]:
        return {}

    def transport_options(self) -> dict[str, Any]:
        return {"cert": (self._cert, self._key)}


class OriginAuth(_BaseAuth):
    """Worker-to-origin shared secret. For internal integrators only."""

    def __init__(self, workspace: str, shared_secret: str) -> None:
        self._workspace = workspace
        self._secret = shared_secret

    def headers(self) -> dict[str, str]:
        return {
            "X-InfoLang-Workspace": self._workspace,
            "X-InfoLang-Origin-Secret": self._secret,
        }
