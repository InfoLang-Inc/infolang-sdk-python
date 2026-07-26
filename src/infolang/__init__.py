"""InfoLang — official Python SDK for InfoLang semantic memory.

Targets the gateway /v2 API (https://api.infolang.ai/docs).

Quickstart::

    from infolang import InfoLang

    il = InfoLang.from_api_key("il_live_...")
    il.remember("The auth middleware validates bearer tokens.")
    result = il.recall("How does auth work?", golden=True)
    for chunk in result.chunks:
        print(chunk.score, chunk.text)
"""

from __future__ import annotations

from ._version import __version__
from .auth import (
    ApiKeyAuth,
    AuthProvider,
    DevKeyAuth,
    MtlsAuth,
    OriginAuth,
    SessionFileAuth,
)
from .client import (
    CLOUD_BASE_URL,
    DIRECT_BASE_URL,
    AsyncInfoLang,
    InfoLang,
)
from .errors import (
    AuthenticationError,
    InfoLangAPIError,
    InfoLangConfigError,
    InfoLangConnectionError,
    InfoLangError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    error_code,
)
from .types import (
    Chunk,
    IngestJob,
    MemoryItem,
    MemoryPage,
    MeteringMeta,
    NamespaceInfo,
    OpError,
    OpResult,
    RecallResult,
    RememberResult,
    Whoami,
    WhoamiWorkspace,
)

__all__ = [
    "__version__",
    "InfoLang",
    "AsyncInfoLang",
    "CLOUD_BASE_URL",
    "DIRECT_BASE_URL",
    # auth
    "AuthProvider",
    "ApiKeyAuth",
    "DevKeyAuth",
    "SessionFileAuth",
    "MtlsAuth",
    "OriginAuth",
    # errors
    "InfoLangError",
    "InfoLangConfigError",
    "InfoLangConnectionError",
    "InfoLangAPIError",
    "AuthenticationError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
    "error_code",
    # types
    "Chunk",
    "RecallResult",
    "RememberResult",
    "MemoryItem",
    "MemoryPage",
    "NamespaceInfo",
    "Whoami",
    "WhoamiWorkspace",
    "IngestJob",
    "OpResult",
    "OpError",
    "MeteringMeta",
]
