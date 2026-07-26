from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from infolang import InfoLang, NotFoundError, RateLimitError
from tests.conftest import BASE_URL, WS_PREFIX


def _body(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls.last.request.content)


# --- recall -------------------------------------------------------------------


@respx.mock
def test_recall_posts_v2_route_and_parses_hits(client_ns: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            200,
            headers={
                "x-infolang-tokens-saved": "1200",
                "x-infolang-chunks-used": "3",
                "x-infolang-overage": "2",
                "x-request-id": "req_123",
            },
            json={
                "hits": [
                    {
                        "id": "abc",
                        "similarity": 0.91,
                        "text": "auth middleware uses bearer tokens",
                        "source": "docs/auth.md",
                        "timestamp": 1721900000.5,
                    },
                    {"id": "def", "similarity": 0.42, "text": "unrelated", "tags": "auth,api"},
                ],
                "count": 2,
                "namespace": "default",
                "latency_ms": 12.5,
            },
        )
    )

    result = client_ns.recall("how does auth work?", top_k=2)

    assert route.called
    sent = route.calls.last.request
    assert sent.method == "POST"
    assert sent.headers["authorization"] == "Bearer il_live_test"
    assert _body(route) == {"query": "how does auth work?", "namespace": "default", "top_k": 2}
    assert len(result.chunks) == 2
    assert result.chunks[0].id == "abc"
    assert result.chunks[0].score == 0.91
    assert result.chunks[0].source == "docs/auth.md"
    assert result.chunks[0].timestamp == 1721900000.5
    assert result.chunks[1].tags == "auth,api"
    assert result.namespace == "default"
    assert result.latency_ms == 12.5
    assert result.weak is False
    assert result.metering is not None
    assert result.metering.tokens_saved == 1200
    assert result.metering.overage == 2
    assert result.metering.request_id == "req_123"


@respx.mock
def test_recall_passes_golden_extras_verbatim(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(200, json={"hits": [], "count": 0})
    )

    client.recall(
        "q",
        golden=True,
        format="snippet",
        snippet_chars=200,
        adaptive=True,
        margin=0.1,
    )

    assert _body(route) == {
        "query": "q",
        "golden": True,
        "format": "snippet",
        "snippet_chars": 200,
        "adaptive": True,
        "margin": 0.1,
    }


@respx.mock
def test_recall_weak_below_confidence_floor(client: InfoLang) -> None:
    respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            200, json={"hits": [{"id": "x", "similarity": 0.5, "text": "low"}], "count": 1}
        )
    )
    assert client.recall("q").weak is True


@respx.mock
def test_investigate_defaults_top_k(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(200, json={"hits": [], "count": 0})
    )
    result = client.investigate("weak query")
    assert _body(route)["top_k"] == 5
    assert result.weak is False


# --- remember -----------------------------------------------------------------


@respx.mock
def test_remember_splits_string_tags_and_parses_fields(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/remember").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "mem_1",
                "namespace": "default",
                "stored": True,
                "total_memories": 7,
                "properties": {},
            },
        )
    )

    result = client.remember("a fact", source="docs/x.md", tags="auth, api")

    assert _body(route) == {
        "text": "a fact",
        "source": "docs/x.md",
        "tags": ["auth", "api"],
    }
    assert result.memory_id == "mem_1"
    assert result.stored is True
    assert result.total_memories == 7


@respx.mock
def test_remember_accepts_list_tags(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/remember").mock(
        return_value=httpx.Response(200, json={"id": "mem_2", "stored": True})
    )
    client.remember("a fact", tags=["one", "two"])
    assert _body(route)["tags"] == ["one", "two"]


@respx.mock
def test_remember_surfaces_dedup_absorbs(client: InfoLang) -> None:
    respx.post(f"{WS_PREFIX}/remember").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "mem_dup",
                "stored": False,
                "deduplicated": True,
                "deduped_against": "mem_orig",
                "total_memories": 7,
            },
        )
    )
    result = client.remember("same text")
    assert result.stored is False
    assert result.deduplicated is True
    assert result.deduped_against == "mem_orig"


@respx.mock
def test_memorize_is_alias_for_remember(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/remember").mock(
        return_value=httpx.Response(200, json={"id": "mem_2", "stored": True})
    )
    result = client.memorize("a fact")
    assert route.called
    assert result.memory_id == "mem_2"


# --- forget -------------------------------------------------------------------


@respx.mock
def test_forget_deletes_with_namespace_param(client: InfoLang) -> None:
    route = respx.delete(f"{WS_PREFIX}/memories/mem_9?namespace=notes").mock(
        return_value=httpx.Response(200, json={"removed": True})
    )
    client.forget("mem_9", namespace="notes")
    assert route.called
    assert route.calls.last.request.method == "DELETE"


@respx.mock
def test_forget_omits_namespace_when_none(client: InfoLang) -> None:
    route = respx.delete(f"{WS_PREFIX}/memories/mem_9").mock(
        return_value=httpx.Response(200, json={"removed": True})
    )
    client.forget("mem_9")
    assert route.calls.last.request.url.query == b""


@respx.mock
def test_forget_uses_client_default_namespace(client_ns: InfoLang) -> None:
    route = respx.delete(f"{WS_PREFIX}/memories/mem_9?namespace=default").mock(
        return_value=httpx.Response(200, json={"removed": True})
    )
    client_ns.forget("mem_9")
    assert route.called


@respx.mock
def test_forget_404_raises_typed_error(client: InfoLang) -> None:
    respx.delete(url__regex=rf"{WS_PREFIX}/memories/.*").mock(
        return_value=httpx.Response(404, json={"error": "no such memory"})
    )
    with pytest.raises(NotFoundError) as exc:
        client.forget("missing")
    assert exc.value.status == 404


# --- list / namespaces --------------------------------------------------------


@respx.mock
def test_list_sends_ns_limit_params_and_parses_page(client: InfoLang) -> None:
    route = respx.get(f"{WS_PREFIX}/memories?ns=notes&limit=10").mock(
        return_value=httpx.Response(
            200,
            json={
                "memories": [{"id": "m1", "text": "t", "namespace": "notes"}],
                "nextCursor": "cur_2",
            },
        )
    )
    page = client.list(namespace="notes", limit=10)
    assert route.called
    assert page.memories[0].id == "m1"
    assert page.next_cursor == "cur_2"


@respx.mock
def test_list_searches_with_q_and_surfaces_scores(client: InfoLang) -> None:
    route = respx.get(f"{WS_PREFIX}/memories?q=auth").mock(
        return_value=httpx.Response(
            200,
            json={"memories": [{"id": "m1", "text": "t", "namespace": "default", "score": 0.9}]},
        )
    )
    page = client.list(query="auth")
    assert route.called
    assert page.memories[0].score == 0.9
    assert page.next_cursor is None


@respx.mock
def test_namespaces_parses_counts(client: InfoLang) -> None:
    route = respx.get(f"{WS_PREFIX}/namespaces").mock(
        return_value=httpx.Response(
            200, json={"namespaces": [{"namespace": "default", "memories": 12, "chunks": 40}]}
        )
    )
    namespaces = client.namespaces()
    assert route.called
    assert namespaces[0].namespace == "default"
    assert namespaces[0].memories == 12
    assert namespaces[0].chunks == 40


@respx.mock
def test_namespaces_tolerates_bare_strings(client: InfoLang) -> None:
    respx.get(f"{WS_PREFIX}/namespaces").mock(
        return_value=httpx.Response(200, json={"namespaces": ["default", "notes"]})
    )
    namespaces = client.namespaces()
    assert [n.namespace for n in namespaces] == ["default", "notes"]
    assert namespaces[0].memories is None


@respx.mock
def test_list_recent_deprecated_rides_list(client: InfoLang) -> None:
    route = respx.get(f"{WS_PREFIX}/memories?limit=5").mock(
        return_value=httpx.Response(
            200, json={"memories": [{"id": "m1", "text": "t", "namespace": "default"}]}
        )
    )
    with pytest.warns(DeprecationWarning):
        rows = client.list_recent(n=5)
    assert route.called
    assert len(rows) == 1
    assert rows[0].id == "m1"


# --- recall_hybrid ------------------------------------------------------------


@respx.mock
def test_recall_hybrid_overfetches_and_orders_by_tag(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            200,
            json={
                "count": 3,
                "namespace": "eval",
                "hits": [
                    {"id": "m1", "text": "untagged top", "similarity": 0.95, "tags": "bob"},
                    {"id": "m2", "text": "alice march", "similarity": 0.80, "tags": "alice,march"},
                    {"id": "m3", "text": "other", "similarity": 0.10, "tags": "carol"},
                ],
            },
        )
    )

    result = client.recall_hybrid(
        "when did alice move?",
        namespace="eval",
        top_k=2,
        tag_filter=["march", "2024"],
        candidate_pool=500,
    )

    # Over-fetches the candidate pool, not top_k.
    assert _body(route)["top_k"] == 500
    # Tag-matching chunk is promoted ahead of the higher-similarity untagged one.
    assert [c.id for c in result.chunks] == ["m2", "m1"]


@respx.mock
def test_recall_hybrid_without_tag_filter_truncates(client: InfoLang) -> None:
    respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {"id": "a", "similarity": 0.9, "text": "x"},
                    {"id": "b", "similarity": 0.5, "text": "y"},
                ]
            },
        )
    )
    result = client.recall_hybrid("q", namespace="eval", top_k=1)
    assert [c.id for c in result.chunks] == ["a"]


# --- remember_batch -----------------------------------------------------------


@respx.mock
def test_remember_batch_sends_one_remember_sub_op_per_item(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/execute").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "payload": {
                    "results": [
                        {"ok": True, "payload": {"id": "m1", "stored": True}},
                        {"ok": True, "payload": {"id": "m2", "stored": True}},
                    ]
                },
            },
        )
    )

    results = client.remember_batch(
        ["one", {"text": "two", "tags": "a,b"}],
        namespace="notes",
        source="batch",
    )

    sent = _body(route)
    assert len(sent["operations"]) == 2
    assert sent["operations"][0] == {
        "op": "remember",
        "args": {"text": "one", "source": "batch", "namespace": "notes"},
    }
    assert sent["operations"][1]["args"]["tags"] == ["a", "b"]
    assert [r.memory_id for r in results] == ["m1", "m2"]


@respx.mock
def test_remember_batch_empty_short_circuits(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/execute")
    assert client.remember_batch([]) == []
    assert not route.called


def test_remember_batch_rejects_bad_items(client: InfoLang) -> None:
    with pytest.raises(TypeError):
        client.remember_batch([123])
    with pytest.raises(TypeError):
        client.remember_batch([{"tags": ["no-text"]}])


# --- reset_namespace ----------------------------------------------------------


@respx.mock
def test_reset_namespace_lists_then_forgets(client: InfoLang) -> None:
    pages = iter(
        [
            httpx.Response(
                200,
                json={
                    "memories": [
                        {"id": "a", "text": "x", "namespace": "tmp"},
                        {"id": "b", "text": "y", "namespace": "tmp"},
                    ]
                },
            ),
            httpx.Response(200, json={"memories": []}),
        ]
    )
    respx.get(url__regex=rf"{WS_PREFIX}/memories\?.*").mock(
        side_effect=lambda request: next(pages)
    )
    forget_route = respx.delete(url__regex=rf"{WS_PREFIX}/memories/.*").mock(
        return_value=httpx.Response(200, json={"removed": True})
    )

    deleted = client.reset_namespace("tmp", batch=500)

    assert deleted == 2
    assert forget_route.call_count == 2
    assert b"namespace=tmp" in forget_route.calls.last.request.url.query


# --- rate limiting ------------------------------------------------------------


@respx.mock
def test_rate_limit_carries_retry_after() -> None:
    respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            429, headers={"retry-after": "1"}, json={"error": "slow down"}
        )
    )
    il = InfoLang.from_api_key(
        "il_live_test", base_url=BASE_URL, workspace="ws_test", max_retries=0
    )
    with pytest.raises(RateLimitError) as exc:
        il.recall("q")
    assert exc.value.retry_after == 1.0
    il.close()
