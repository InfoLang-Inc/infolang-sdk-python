from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from infolang import AsyncInfoLang, InfoLangConfigError
from infolang.resources import _ops
from infolang.types import Chunk, MeteringMeta
from tests.conftest import BASE_URL, WS, WS_PREFIX


def _body(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls.last.request.content)


def _client(**kwargs: Any) -> AsyncInfoLang:
    return AsyncInfoLang.from_api_key(
        "il_live_test", base_url=BASE_URL, workspace=WS, **kwargs
    )


# --- async workspace resolution ----------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_explicit_workspace_never_calls_whoami() -> None:
    whoami = respx.get(f"{BASE_URL}/v2/whoami")
    respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(200, json={"hits": [], "count": 0})
    )
    async with _client() as il:
        await il.recall("q")
    assert not whoami.called


@pytest.mark.asyncio
@respx.mock
async def test_async_single_grant_resolves_once_and_caches() -> None:
    whoami = respx.get(f"{BASE_URL}/v2/whoami").mock(
        return_value=httpx.Response(
            200,
            json={
                "lane": "key",
                "principal": None,
                "workspaces": [{"workspace_id": "ws_solo", "role": "Member"}],
            },
        )
    )
    recall = respx.post(f"{BASE_URL}/v2/workspaces/ws_solo/recall").mock(
        return_value=httpx.Response(200, json={"hits": [], "count": 0})
    )
    async with AsyncInfoLang.from_api_key("il_live_test", base_url=BASE_URL) as il:
        await il.recall("first")
        await il.recall("second")
    assert whoami.call_count == 1
    assert recall.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_async_multi_grant_errors_and_failure_not_cached() -> None:
    whoami = respx.get(f"{BASE_URL}/v2/whoami").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "lane": "key",
                    "principal": None,
                    "workspaces": [
                        {"workspace_id": "ws_a", "role": "Member"},
                        {"workspace_id": "ws_b", "role": "Member"},
                    ],
                },
            ),
            httpx.Response(
                200,
                json={
                    "lane": "key",
                    "principal": None,
                    "workspaces": [{"workspace_id": "ws_a", "role": "Member"}],
                },
            ),
        ]
    )
    respx.post(f"{BASE_URL}/v2/workspaces/ws_a/recall").mock(
        return_value=httpx.Response(200, json={"hits": [], "count": 0})
    )
    async with AsyncInfoLang.from_api_key("il_live_test", base_url=BASE_URL) as il:
        with pytest.raises(InfoLangConfigError, match="ws_a, ws_b"):
            await il.recall("q")
        # Second attempt re-asks whoami (single grant now) and succeeds.
        await il.recall("q")
    assert whoami.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_async_zero_grant_errors() -> None:
    respx.get(f"{BASE_URL}/v2/whoami").mock(
        return_value=httpx.Response(
            200, json={"lane": "key", "principal": None, "workspaces": []}
        )
    )
    async with AsyncInfoLang.from_api_key("il_live_test", base_url=BASE_URL) as il:
        with pytest.raises(InfoLangConfigError, match="no visible workspace grants"):
            await il.recall("q")


@pytest.mark.asyncio
@respx.mock
async def test_async_whoami_parses() -> None:
    respx.get(f"{BASE_URL}/v2/whoami").mock(
        return_value=httpx.Response(
            200,
            json={
                "lane": "oauth",
                "principal": "user_1",
                "workspaces": [{"workspace_id": WS, "role": "Owner"}],
            },
        )
    )
    async with _client() as il:
        who = await il.whoami()
    assert who.lane == "oauth"
    assert who.principal == "user_1"
    assert who.workspaces[0].workspace_id == WS


# --- async per-op coverage ----------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_recall_golden_extras_and_parse() -> None:
    route = respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            200,
            headers={"x-infolang-overage": "1", "x-request-id": "req_a"},
            json={
                "hits": [{"id": "1", "similarity": 0.9, "text": "x", "tags": "m"}],
                "count": 1,
                "namespace": "default",
                "latency_ms": 3,
            },
        )
    )
    async with _client() as il:
        result = await il.recall("q", golden=True, format="meta", top_k=2)
    assert _body(route) == {"query": "q", "top_k": 2, "golden": True, "format": "meta"}
    assert result.chunks[0].score == 0.9
    assert result.latency_ms == 3.0
    assert result.metering is not None
    assert result.metering.overage == 1


@pytest.mark.asyncio
@respx.mock
async def test_async_investigate_defaults_top_k() -> None:
    route = respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(200, json={"hits": [], "count": 0})
    )
    async with _client() as il:
        await il.investigate("q")
    assert _body(route)["top_k"] == 5


@pytest.mark.asyncio
@respx.mock
async def test_async_remember_and_memorize_split_tags() -> None:
    route = respx.post(f"{WS_PREFIX}/remember").mock(
        return_value=httpx.Response(
            200, json={"id": "m1", "stored": True, "total_memories": 1}
        )
    )
    async with _client() as il:
        result = await il.remember("fact", tags="a, b")
        assert _body(route)["tags"] == ["a", "b"]
        memorized = await il.memorize("fact 2", tags=["c"])
    assert result.memory_id == "m1"
    assert result.stored is True
    assert memorized.memory_id == "m1"
    assert _body(route)["tags"] == ["c"]


@pytest.mark.asyncio
@respx.mock
async def test_async_forget_sends_namespace_param() -> None:
    route = respx.delete(f"{WS_PREFIX}/memories/mem_1?namespace=notes").mock(
        return_value=httpx.Response(200, json={"removed": True})
    )
    async with _client() as il:
        await il.forget("mem_1", namespace="notes")
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_async_list_namespaces_list_recent() -> None:
    respx.get(f"{WS_PREFIX}/memories?ns=notes&limit=2").mock(
        return_value=httpx.Response(
            200,
            json={
                "memories": [{"id": "m1", "text": "t", "namespace": "notes"}],
                "nextCursor": "cur",
            },
        )
    )
    respx.get(f"{WS_PREFIX}/namespaces").mock(
        return_value=httpx.Response(
            200, json={"namespaces": [{"namespace": "notes", "memories": 1, "chunks": 2}]}
        )
    )
    respx.get(f"{WS_PREFIX}/memories?limit=5").mock(
        return_value=httpx.Response(
            200, json={"memories": [{"id": "m2", "text": "t", "namespace": "default"}]}
        )
    )
    async with _client() as il:
        page = await il.list(namespace="notes", limit=2)
        assert page.memories[0].id == "m1"
        assert page.next_cursor == "cur"
        namespaces = await il.namespaces()
        assert namespaces[0].chunks == 2
        with pytest.warns(DeprecationWarning):
            rows = await il.list_recent(n=5)
        assert rows[0].id == "m2"


@pytest.mark.asyncio
@respx.mock
async def test_async_execute_stats_encode_similarity() -> None:
    execute = respx.post(f"{WS_PREFIX}/execute").mock(
        return_value=httpx.Response(200, json={"ok": True, "payload": {"results": []}})
    )
    respx.get(f"{WS_PREFIX}/stats").mock(
        return_value=httpx.Response(200, json={"ok": True, "payload": {"memories": 3}})
    )
    respx.post(f"{WS_PREFIX}/encode").mock(
        return_value=httpx.Response(200, json={"dims": 2})
    )
    respx.post(f"{WS_PREFIX}/similarity").mock(
        return_value=httpx.Response(200, json={"similarity": 0.5})
    )
    async with _client() as il:
        result = await il.execute([{"op": "stats", "args": {}}])
        assert result.ok is True
        assert _body(execute) == {"operations": [{"op": "stats", "args": {}}]}
        stats = await il.stats()
        assert stats.payload == {"memories": 3}
        encoded = await il.encode("hello")
        assert encoded["dims"] == 2
        sim = await il.similarity("a", "b")
        assert sim["similarity"] == 0.5


@pytest.mark.asyncio
@respx.mock
async def test_async_ingest_raw_bytes() -> None:
    route = respx.post(f"{WS_PREFIX}/ingest?ns=docs").mock(
        return_value=httpx.Response(202, json={"status": "accepted"})
    )
    archive = b"PK\x03\x04"
    async with _client() as il:
        job = await il.ingest(archive, namespace="docs")
    sent = route.calls.last.request
    assert sent.content == archive
    assert sent.headers["content-type"] == "application/zip"
    assert job.status == "accepted"


@pytest.mark.asyncio
@respx.mock
async def test_async_health_check_and_ready() -> None:
    respx.get(f"{BASE_URL}/healthz").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    respx.get(f"{BASE_URL}/readyz").mock(
        return_value=httpx.Response(200, json={"ml_ready": True})
    )
    async with _client() as il:
        assert await il.health.check() == {"ok": True}
        assert await il.health.ready() == {"ml_ready": True}


@pytest.mark.asyncio
@respx.mock
async def test_async_hybrid_batch_reset() -> None:
    respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            200,
            json={"hits": [{"id": "h1", "text": "t", "similarity": 0.8, "tags": "m"}]},
        )
    )
    respx.post(f"{WS_PREFIX}/execute").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "payload": {"results": [{"ok": True, "payload": {"id": "b1"}}]},
            },
        )
    )
    pages = iter(
        [
            httpx.Response(
                200, json={"memories": [{"id": "z1", "text": "t", "namespace": "d"}]}
            ),
            httpx.Response(200, json={"memories": []}),
        ]
    )
    respx.get(url__regex=rf"{WS_PREFIX}/memories\?.*").mock(
        side_effect=lambda request: next(pages)
    )
    respx.delete(url__regex=rf"{WS_PREFIX}/memories/.*").mock(
        return_value=httpx.Response(200, json={})
    )

    async with _client() as il:
        hybrid = await il.recall_hybrid("q", tag_filter=["m"], top_k=5)
        assert hybrid.chunks[0].id == "h1"
        batch = await il.remember_batch(["one"], namespace="default")
        assert batch[0].memory_id == "b1"
        deleted = await il.reset_namespace("default", batch=500)
        assert deleted == 1


@pytest.mark.asyncio
@respx.mock
async def test_async_remember_batch_empty_short_circuits() -> None:
    route = respx.post(f"{WS_PREFIX}/execute")
    async with _client() as il:
        assert await il.remember_batch([]) == []
    assert not route.called


# --- pure builder / parser edge cases ----------------------------------------


def test_parsers_tolerate_garbage() -> None:
    metering = MeteringMeta(request_id="r1")
    assert _ops.parse_recall("nope", metering).chunks == []
    assert _ops.parse_recall({"hits": "nope"}, metering).chunks == []
    assert _ops.parse_remember("raw").memory_id is None
    assert _ops.parse_list("nope").memories == []
    assert _ops.parse_namespaces("nope") == []
    assert _ops.parse_namespaces({"namespaces": "nope"}) == []
    assert _ops.parse_namespaces({"namespaces": [42]}) == []
    assert _ops.parse_op_result("nope").ok is False
    assert _ops.parse_execute_remember_batch("nope") == []
    assert _ops.parse_execute_remember_batch({"ok": True, "payload": {}}) == []
    assert _ops.parse_whoami("nope").workspaces == []
    assert _ops.parse_ingest("nope").status is None


def test_builders_url_encode_opaque_ids() -> None:
    # Workspace and memory ids are opaque; URL-significant chars must be
    # percent-encoded into the path, never parsed or validated.
    method, path, body = _ops.build_forget("ws/1 a", "mem 2#x", namespace="team a&b")
    assert method == "DELETE"
    assert path.startswith("/v2/workspaces/ws%2F1%20a/memories/mem%202%23x?")
    from urllib.parse import parse_qs, urlparse

    assert parse_qs(urlparse(path).query)["namespace"] == ["team a&b"]
    assert body is None

    _, list_path, _ = _ops.build_list("ws", namespace="team a&b", query=None, limit=3)
    qs = parse_qs(urlparse(list_path).query)
    assert qs["ns"] == ["team a&b"]
    assert qs["limit"] == ["3"]

    _, plain, _ = _ops.build_list("ws", namespace="ns", query="q1", limit=None)
    assert plain == "/v2/workspaces/ws/memories?ns=ns&q=q1"


def test_build_recall_compacts_none_fields() -> None:
    method, path, body = _ops.build_recall(
        "ws", "q", namespace=None, top_k=None, verbose=None
    )
    assert method == "POST"
    assert path == "/v2/workspaces/ws/recall"
    assert body == {"query": "q"}


def test_filter_hits_by_tags_orders_and_truncates() -> None:
    hits = [
        Chunk(id="a", text="", tags="bob"),
        Chunk(id="b", text="", tags="alice,march"),
        Chunk(id="c", text=""),
    ]
    ordered = _ops.filter_hits_by_tags(hits, ["MARCH"], 2)
    assert [h.id for h in ordered] == ["b", "a"]
    assert [h.id for h in _ops.filter_hits_by_tags(hits, None, 1)] == ["a"]
    assert len(_ops.filter_hits_by_tags(hits, None, None)) == 3


def test_remember_batch_builder_emits_one_sub_op_per_item() -> None:
    _, path, body = _ops.build_remember_batch(
        "ws",
        [{"text": "x", "tags": "t1, t2"}, {"text": "y", "source": "own"}],
        namespace="ns",
        source="s",
    )
    assert path == "/v2/workspaces/ws/execute"
    assert body["operations"] == [
        {
            "op": "remember",
            "args": {"text": "x", "source": "s", "tags": ["t1", "t2"], "namespace": "ns"},
        },
        {"op": "remember", "args": {"text": "y", "source": "own", "namespace": "ns"}},
    ]


def test_normalize_remember_items() -> None:
    norm = _ops.normalize_remember_items(
        ["a", {"text": "b", "tags": ["t"]}], default_source="s"
    )
    assert norm[0] == {"text": "a", "source": "s"}
    assert norm[1]["tags"] == ["t"]
    with pytest.raises(TypeError):
        _ops.normalize_remember_items([123], default_source=None)
    with pytest.raises(TypeError):
        _ops.normalize_remember_items([{"no_text": True}], default_source=None)


def test_split_tags_variants() -> None:
    assert _ops._split_tags(None) is None
    assert _ops._split_tags("a, b, ,c") == ["a", "b", "c"]
    assert _ops._split_tags(["x"]) == ["x"]
