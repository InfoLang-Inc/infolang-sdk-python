"""Request and response models mirroring the InfoLang gateway ``/v2`` contract.

API routes return the server shapes verbatim; the SDK normalizes
recall ``hits`` into :class:`Chunk` s and keeps every id opaque — never parse
or validate id prefixes client-side.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A single recalled memory (normalized from the ``hits[]`` wire shape)."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    #: Memory id (wire: ``id``). Opaque — do not parse.
    id: str = ""
    #: Cosine similarity (wire: ``similarity``).
    score: float | None = None
    #: Memory text. Empty when ``format="meta"`` was requested.
    text: str = ""
    #: Comma-joined tags; present only when the memory has tags.
    tags: str | None = None
    #: Origin label; present only on ``verbose=True`` recalls.
    source: str | None = None
    #: Epoch seconds; present only on ``verbose=True`` recalls.
    timestamp: float | None = None


class MeteringMeta(BaseModel):
    """Usage metadata parsed from gateway response headers."""

    tokens_saved: int | None = None
    chunks_used: int | None = None
    repo_coverage: float | None = None
    #: Overage units incurred by this call, when the plan is over its inclusion.
    overage: int | None = None
    request_id: str | None = None


class RecallResult(BaseModel):
    """Result of a ``recall`` / ``investigate`` call."""

    model_config = ConfigDict(extra="allow")

    chunks: list[Chunk] = Field(default_factory=list)
    namespace: str | None = None
    #: Server-observed latency in milliseconds, when reported.
    latency_ms: float | None = None
    metering: MeteringMeta | None = None

    @property
    def weak(self) -> bool:
        """True when the top match scores below the 0.85 confidence floor."""

        top = self.chunks[0].score if self.chunks else None
        return top is not None and top < 0.85


class RememberResult(BaseModel):
    """Result of a ``remember`` / ``memorize`` call (server shape)."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    memory_id: str | None = Field(default=None, alias="id")
    namespace: str | None = None
    #: False when the write was absorbed by a near-duplicate (never billed).
    stored: bool | None = None
    #: Present and true only on a dedup absorb.
    deduplicated: bool | None = None
    #: The id of the existing memory that absorbed this write.
    deduped_against: str | None = None
    total_memories: int | None = None


class MemoryItem(BaseModel):
    """One memory row from ``list`` (gateway MemoryPage item)."""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    text: str = ""
    namespace: str = ""
    source: str | None = None
    tags: list[str] | None = None
    timestamp: str | float | None = None
    #: Relevance score; present only on search (``query``) results.
    score: float | None = None


class MemoryPage(BaseModel):
    """A page of memories; ``next_cursor`` is absent on the last page."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    memories: list[MemoryItem] = Field(default_factory=list)
    next_cursor: str | None = Field(default=None, alias="nextCursor")


class NamespaceInfo(BaseModel):
    """A namespace with its logical memory count and chunk row count."""

    model_config = ConfigDict(extra="allow")

    namespace: str = ""
    memories: int | None = None
    chunks: int | None = None


class WhoamiWorkspace(BaseModel):
    """A workspace grant visible to the current credential."""

    model_config = ConfigDict(extra="allow")

    #: Opaque workspace id.
    workspace_id: str = ""
    role: str = ""
    #: Scopes that actually work on this workspace, when scoping is active.
    scopes: list[str] | None = None


class Whoami(BaseModel):
    """Identity echo for the current credential (``GET /v2/whoami``).

    Kept structurally open: the gateway adds fields (scope sources, credential
    descriptors) without notice, and ids are opaque.
    """

    model_config = ConfigDict(extra="allow")

    lane: str = ""
    principal: str | None = None
    workspaces: list[WhoamiWorkspace] = Field(default_factory=list)
    scopes: list[str] | None = None


class IngestJob(BaseModel):
    """Result of an ingest call (``POST /v2/workspaces/{ws}/ingest``)."""

    model_config = ConfigDict(extra="allow")

    status: str | None = None


class OpError(BaseModel):
    """The error half of a native OpResult envelope."""

    model_config = ConfigDict(extra="allow")

    code: str = ""
    message: str = ""
    capability: str | None = None


class OpResult(BaseModel):
    """The native OpResult envelope (``execute`` and ``stats``)."""

    model_config = ConfigDict(extra="allow")

    ok: bool = False
    payload: dict[str, Any] | None = None
    error: OpError | None = None
