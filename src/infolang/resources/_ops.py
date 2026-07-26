"""Pure request builders and response parsers for the gateway ``/v2`` surface.

Keeping these free of any I/O lets the sync and async resource classes share
identical request shaping and parsing, so the two transports can never drift.
Each builder returns ``(method, path, json_body)``; each parser turns a decoded
response (plus metering metadata where relevant) into a typed model. Every
workspace-scoped builder takes the resolved workspace id; ids are opaque
strings throughout — never parsed or validated client-side.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

from ..types import (
    Chunk,
    IngestJob,
    MemoryPage,
    MeteringMeta,
    NamespaceInfo,
    OpResult,
    RecallResult,
    RememberResult,
    Whoami,
)


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None so the gateway applies its own defaults."""

    return {key: value for key, value in payload.items() if value is not None}


def _ws(workspace_id: str) -> str:
    return f"/v2/workspaces/{quote(workspace_id, safe='')}"


def _split_tags(tags: str | list[str] | None) -> list[str] | None:
    """Accept a comma-joined string or a list; always emit the list the API expects."""

    if tags is None:
        return None
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return tags


# --- recall -------------------------------------------------------------------


def build_recall(
    workspace_id: str,
    query: str,
    *,
    namespace: str | None,
    top_k: int | None,
    verbose: bool | None,
    golden: bool | None = None,
    format: str | None = None,
    snippet_chars: int | None = None,
    adaptive: bool | None = None,
    margin: float | None = None,
) -> tuple[str, str, dict[str, Any]]:
    body = _compact(
        {
            "query": query,
            "namespace": namespace,
            "top_k": top_k,
            "verbose": verbose,
            "golden": golden,
            "format": format,
            "snippet_chars": snippet_chars,
            "adaptive": adaptive,
            "margin": margin,
        }
    )
    return "POST", f"{_ws(workspace_id)}/recall", body


def _hit_to_chunk(hit: dict[str, Any]) -> Chunk:
    """Normalize a recall ``hit`` (``similarity`` -> ``score``)."""

    return Chunk(
        id=str(hit.get("id", "") or ""),
        score=hit.get("similarity") if isinstance(hit.get("similarity"), (int, float)) else None,
        text=str(hit.get("text", "") or ""),
        tags=hit.get("tags") if isinstance(hit.get("tags"), str) else None,
        source=hit.get("source") if isinstance(hit.get("source"), str) else None,
        timestamp=(
            hit.get("timestamp") if isinstance(hit.get("timestamp"), (int, float)) else None
        ),
    )


def parse_recall(data: Any, metering: MeteringMeta) -> RecallResult:
    record = data if isinstance(data, dict) else {}
    hits = record.get("hits")
    if isinstance(hits, list):
        chunks = [_hit_to_chunk(h) for h in hits if isinstance(h, dict)]
    else:
        chunks = []
    namespace = record.get("namespace")
    latency = record.get("latency_ms")
    return RecallResult(
        chunks=chunks,
        namespace=namespace if isinstance(namespace, str) else None,
        latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
        metering=metering,
    )


# --- remember -----------------------------------------------------------------


def build_remember(
    workspace_id: str,
    text: str,
    *,
    namespace: str | None,
    source: str | None,
    tags: str | list[str] | None,
) -> tuple[str, str, dict[str, Any]]:
    body = _compact(
        {
            "text": text,
            "namespace": namespace,
            "source": source,
            "tags": _split_tags(tags),
        }
    )
    return "POST", f"{_ws(workspace_id)}/remember", body


def parse_remember(data: Any) -> RememberResult:
    payload = data if isinstance(data, dict) else {}
    return RememberResult.model_validate(payload)


# --- forget -------------------------------------------------------------------


def build_forget(
    workspace_id: str, memory_id: str, *, namespace: str | None
) -> tuple[str, str, None]:
    query = urlencode({"namespace": namespace}) if namespace else ""
    path = f"{_ws(workspace_id)}/memories/{quote(memory_id, safe='')}"
    return "DELETE", path + (f"?{query}" if query else ""), None


# --- list / namespaces --------------------------------------------------------


def build_list(
    workspace_id: str,
    *,
    namespace: str | None,
    query: str | None,
    limit: int | None,
) -> tuple[str, str, None]:
    params = _compact({"ns": namespace, "q": query, "limit": limit})
    encoded = urlencode(params)
    path = f"{_ws(workspace_id)}/memories" + (f"?{encoded}" if encoded else "")
    return "GET", path, None


def parse_list(data: Any) -> MemoryPage:
    payload = data if isinstance(data, dict) else {}
    return MemoryPage.model_validate(payload)


def build_namespaces(workspace_id: str) -> tuple[str, str, None]:
    return "GET", f"{_ws(workspace_id)}/namespaces", None


def parse_namespaces(data: Any) -> list[NamespaceInfo]:
    record = data if isinstance(data, dict) else {}
    items = record.get("namespaces")
    if not isinstance(items, list):
        return []
    out: list[NamespaceInfo] = []
    for item in items:
        if isinstance(item, str):
            out.append(NamespaceInfo(namespace=item))
        elif isinstance(item, dict):
            out.append(NamespaceInfo.model_validate(item))
    return out


# --- encode / similarity ------------------------------------------------------


def build_encode(workspace_id: str, text: str) -> tuple[str, str, dict[str, Any]]:
    return "POST", f"{_ws(workspace_id)}/encode", {"text": text}


def build_similarity(
    workspace_id: str, text1: str, text2: str
) -> tuple[str, str, dict[str, Any]]:
    return "POST", f"{_ws(workspace_id)}/similarity", {"text1": text1, "text2": text2}


# --- execute / stats ----------------------------------------------------------


def build_execute(
    workspace_id: str, operations: list[dict[str, Any]]
) -> tuple[str, str, dict[str, Any]]:
    """``POST /v2/workspaces/{ws}/execute`` takes the simple ``{operations}`` batch.

    The gateway accepts this shape directly — the SDK must NOT wrap it
    in any further envelope.
    """

    return "POST", f"{_ws(workspace_id)}/execute", {"operations": operations}


def parse_op_result(data: Any) -> OpResult:
    payload = data if isinstance(data, dict) else {}
    return OpResult.model_validate(payload)


def build_stats(workspace_id: str) -> tuple[str, str, None]:
    return "GET", f"{_ws(workspace_id)}/stats", None


# --- whoami -------------------------------------------------------------------


def build_whoami() -> tuple[str, str, None]:
    return "GET", "/v2/whoami", None


def parse_whoami(data: Any) -> Whoami:
    payload = data if isinstance(data, dict) else {}
    return Whoami.model_validate(payload)


# --- ingest -------------------------------------------------------------------


def build_ingest(
    workspace_id: str,
    *,
    namespace: str | None,
    tag_prefix: str | None,
) -> tuple[str, str]:
    """Path for a raw-zip ingest; the caller supplies the bytes body."""

    params = _compact({"ns": namespace, "tag_prefix": tag_prefix})
    encoded = urlencode(params)
    path = f"{_ws(workspace_id)}/ingest" + (f"?{encoded}" if encoded else "")
    return "POST", path


def parse_ingest(data: Any) -> IngestJob:
    payload = data if isinstance(data, dict) else {}
    return IngestJob.model_validate(payload)


# --- health -------------------------------------------------------------------


def build_health() -> tuple[str, str, None]:
    return "GET", "/healthz", None


def build_ready() -> tuple[str, str, None]:
    return "GET", "/readyz", None


# --- client-side helpers ------------------------------------------------------


def filter_hits_by_tags(
    chunks: list[Chunk], tag_filter: list[str] | None, top_k: int | None
) -> list[Chunk]:
    """Tag-inclusion ordering over already-ranked chunks (stable, then truncate).

    Chunks that share at least one tag with ``tag_filter`` are moved ahead of the
    rest while preserving the original (similarity) order within each group, then
    the list is truncated to ``top_k``. With no ``tag_filter`` this is just a
    truncation. Done client-side because the gateway's recall does not accept a
    tag-inclusion filter.
    """

    if tag_filter:
        wanted = {t.strip().lower() for t in tag_filter if t and t.strip()}

        def _matches(chunk: Chunk) -> bool:
            if not chunk.tags:
                return False
            have = {t.strip().lower() for t in chunk.tags.split(",") if t.strip()}
            return bool(have & wanted)

        matched = [c for c in chunks if _matches(c)]
        rest = [c for c in chunks if not _matches(c)]
        chunks = matched + rest
    return chunks[:top_k] if top_k is not None else chunks


def build_remember_batch(
    workspace_id: str,
    items: list[dict[str, Any]],
    *,
    namespace: str | None,
    source: str | None,
) -> tuple[str, str, dict[str, Any]]:
    """``remember_batch`` is client-side sugar over execute: one ``remember``
    sub-op per item (there is no ``remember_batch`` primitive). Sub-op results
    come back as one OpResult each, in order."""

    operations = [
        {
            "op": "remember",
            "args": _compact(
                {
                    "text": item.get("text"),
                    "source": item.get("source", source),
                    "tags": _split_tags(item.get("tags")),
                    "namespace": namespace,
                }
            ),
        }
        for item in items
    ]
    return build_execute(workspace_id, operations)


def parse_execute_remember_batch(data: Any) -> list[RememberResult]:
    envelope = parse_op_result(data)
    results = (envelope.payload or {}).get("results")
    if not isinstance(results, list):
        return []
    out: list[RememberResult] = []
    for result in results:
        payload = result.get("payload") if isinstance(result, dict) else None
        out.append(parse_remember(payload if isinstance(payload, dict) else {}))
    return out


def normalize_remember_items(
    items: list[Any], *, default_source: str | None
) -> list[dict[str, Any]]:
    """Accept ``list[str]`` or ``list[dict]`` (with ``text``) and return remember dicts."""

    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            out.append({"text": item, "source": default_source})
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            out.append(item)
        else:
            raise TypeError(
                "remember_batch items must be str or a dict with a 'text' key, "
                f"got {type(item).__name__}"
            )
    return out


def multipart_files(
    files: list[tuple[str, bytes | str]],
) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Repeated ``files`` parts for multipart ingest (gateway dispatches on
    content type; httpx owns the multipart boundary)."""

    parts: list[tuple[str, tuple[str, bytes, str]]] = []
    for name, content in files:
        payload = content.encode("utf-8") if isinstance(content, str) else content
        parts.append(("files", (name, payload, "text/plain")))
    return parts
