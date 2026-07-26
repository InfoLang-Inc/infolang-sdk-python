"""Typed error hierarchy for the InfoLang SDK.

Callers should catch these instead of inspecting raw HTTP responses. Every
error carries the originating ``request_id`` (from the ``x-request-id`` response
header) when the server provides one, so failures are traceable end to end.
"""

from __future__ import annotations

from typing import Any


class InfoLangError(Exception):
    """Base class for every error raised by the SDK."""


class InfoLangConfigError(InfoLangError):
    """Raised for client misconfiguration (missing credentials, bad base URL)."""


class InfoLangConnectionError(InfoLangError):
    """Raised when the server could not be reached or timed out."""


class InfoLangAPIError(InfoLangError):
    """Raised when the server returns a non-2xx response.

    Subclasses map well-known HTTP statuses to actionable types. The raw
    ``status``, parsed ``body`` and ``request_id`` are always available.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int,
        body: Any = None,
        request_id: str | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self.request_id = request_id
        suffix = f" (request_id={request_id})" if request_id else ""
        super().__init__(f"{message}{suffix}")


class AuthenticationError(InfoLangAPIError):
    """401/403 — the credential was missing, invalid, or lacked permission."""


class NotFoundError(InfoLangAPIError):
    """404 — the namespace, bank, or memory id does not exist."""


class ValidationError(InfoLangAPIError):
    """400/422 — the request payload was rejected by the server."""


class RateLimitError(InfoLangAPIError):
    """429 — quota exceeded. ``retry_after`` is seconds when the server sets it."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        body: Any = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, status=status, body=body, request_id=request_id)


class ServerError(InfoLangAPIError):
    """5xx — the server failed to process the request."""


def error_from_response(
    status: int,
    body: Any,
    request_id: str | None,
    retry_after: float | None,
) -> InfoLangAPIError:
    """Map an HTTP status to the most specific error type."""

    message = _message_from_body(body) or f"InfoLang request failed with status {status}"
    if status in (401, 403):
        return AuthenticationError(message, status=status, body=body, request_id=request_id)
    if status == 404:
        return NotFoundError(message, status=status, body=body, request_id=request_id)
    if status in (400, 422):
        return ValidationError(message, status=status, body=body, request_id=request_id)
    if status == 429:
        return RateLimitError(
            message,
            status=status,
            body=body,
            request_id=request_id,
            retry_after=retry_after,
        )
    if status >= 500:
        return ServerError(message, status=status, body=body, request_id=request_id)
    return InfoLangAPIError(message, status=status, body=body, request_id=request_id)


def _message_from_body(body: Any) -> str | None:
    if isinstance(body, dict):
        # Gateway /v2 envelope: {"error": {"code", "message"}}.
        nested = body.get("error")
        if isinstance(nested, dict):
            message = nested.get("message")
            code = nested.get("code")
            message = message if isinstance(message, str) and message else None
            code = code if isinstance(code, str) and code else None
            if message:
                return f"{code}: {message}" if code else message
            if code:
                return code
        for key in ("error", "message", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(body, str) and body:
        return body
    return None


def error_code(body: Any) -> str | None:
    """The gateway's machine-readable error code (``{error:{code}}``), when present."""

    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict):
            code = nested.get("code")
            if isinstance(code, str):
                return code
    return None
