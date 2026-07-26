from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from infolang import (
    CLOUD_BASE_URL,
    AuthenticationError,
    InfoLang,
    InfoLangConfigError,
    ServerError,
    ValidationError,
)
from tests.conftest import BASE_URL, WS, WS_PREFIX


def _body(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls.last.request.content)


# --- workspace resolution (whoami) -------------------------------------------


@respx.mock
def test_explicit_workspace_never_calls_whoami(client: InfoLang) -> None:
    whoami = respx.get(f"{BASE_URL}/v2/whoami")
    recall = respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(200, json={"hits": [], "count": 0})
    )
    client.recall("q")
    assert recall.called
    assert not whoami.called


@respx.mock
def test_env_workspace_wins_without_whoami(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFOLANG_WORKSPACE", "ws_env")
    whoami = respx.get(f"{BASE_URL}/v2/whoami")
    recall = respx.post(f"{BASE_URL}/v2/workspaces/ws_env/recall").mock(
        return_value=httpx.Response(200, json={"hits": [], "count": 0})
    )
    with InfoLang.from_api_key("il_live_test", base_url=BASE_URL) as il:
        il.recall("q")
    assert recall.called
    assert not whoami.called


@respx.mock
def test_single_grant_auto_resolves_and_caches() -> None:
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
    with InfoLang.from_api_key("il_live_test", base_url=BASE_URL) as il:
        il.recall("first")
        il.recall("second")
    assert whoami.call_count == 1
    assert recall.call_count == 2


@respx.mock
def test_multi_grant_raises_config_error_naming_ids() -> None:
    respx.get(f"{BASE_URL}/v2/whoami").mock(
        return_value=httpx.Response(
            200,
            json={
                "lane": "key",
                "principal": None,
                "workspaces": [
                    {"workspace_id": "ws_a", "role": "Member"},
                    {"workspace_id": "ws_b", "role": "Member"},
                ],
            },
        )
    )
    with (
        InfoLang.from_api_key("il_live_test", base_url=BASE_URL) as il,
        pytest.raises(InfoLangConfigError, match="ws_a, ws_b"),
    ):
        il.recall("q")


@respx.mock
def test_zero_grant_raises_config_error() -> None:
    respx.get(f"{BASE_URL}/v2/whoami").mock(
        return_value=httpx.Response(
            200, json={"lane": "key", "principal": None, "workspaces": []}
        )
    )
    with (
        InfoLang.from_api_key("il_live_test", base_url=BASE_URL) as il,
        pytest.raises(InfoLangConfigError, match="no visible workspace grants"),
    ):
        il.recall("q")


@respx.mock
def test_failed_resolution_is_not_cached() -> None:
    whoami = respx.get(f"{BASE_URL}/v2/whoami").mock(
        side_effect=[
            httpx.Response(200, json={"lane": "key", "principal": None, "workspaces": []}),
            httpx.Response(
                200,
                json={
                    "lane": "key",
                    "principal": None,
                    "workspaces": [{"workspace_id": "ws_later", "role": "Member"}],
                },
            ),
        ]
    )
    recall = respx.post(f"{BASE_URL}/v2/workspaces/ws_later/recall").mock(
        return_value=httpx.Response(200, json={"hits": [], "count": 0})
    )
    with InfoLang.from_api_key("il_live_test", base_url=BASE_URL) as il:
        with pytest.raises(InfoLangConfigError):
            il.recall("q")
        # Retry re-asks whoami instead of reusing the failed resolution.
        il.recall("q")
    assert whoami.call_count == 2
    assert recall.called


@respx.mock
def test_whoami_parses_lane_principal_workspaces(client: InfoLang) -> None:
    route = respx.get(f"{BASE_URL}/v2/whoami").mock(
        return_value=httpx.Response(
            200,
            json={
                "lane": "key",
                "principal": "svc_ci",
                "workspaces": [
                    {"workspace_id": WS, "role": "Owner", "scopes": ["memories:*"]}
                ],
                "scopes": ["memories:*"],
            },
        )
    )
    who = client.whoami()
    assert route.called
    assert who.lane == "key"
    assert who.principal == "svc_ci"
    assert who.workspaces[0].workspace_id == WS
    assert who.workspaces[0].scopes == ["memories:*"]


# --- execute / stats ----------------------------------------------------------


@respx.mock
def test_execute_sends_simple_batch_and_parses_op_result(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/execute").mock(
        return_value=httpx.Response(
            200,
            json={"ok": True, "payload": {"results": [{"ok": True, "payload": {"id": "m1"}}]}},
        )
    )
    result = client.execute([{"op": "stats", "args": {}}])
    # The gateway accepts the simple batch directly — the SDK must NOT wrap it.
    assert _body(route) == {"operations": [{"op": "stats", "args": {}}]}
    assert result.ok is True
    assert result.payload is not None


@respx.mock
def test_stats_gets_v2_route(client: InfoLang) -> None:
    route = respx.get(f"{WS_PREFIX}/stats").mock(
        return_value=httpx.Response(200, json={"ok": True, "payload": {"ml_ready": True}})
    )
    result = client.stats()
    assert route.called
    assert result.payload == {"ml_ready": True}


# --- encode / similarity ------------------------------------------------------


@respx.mock
def test_encode_posts_v2_route(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/encode").mock(
        return_value=httpx.Response(200, json={"vector": [0.1, 0.2], "dims": 2})
    )
    result = client.encode("hello")
    assert _body(route) == {"text": "hello"}
    assert result["dims"] == 2


@respx.mock
def test_similarity_posts_text1_text2(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/similarity").mock(
        return_value=httpx.Response(200, json={"similarity": 0.87})
    )
    result = client.similarity("a", "b")
    assert _body(route) == {"text1": "a", "text2": "b"}
    assert result["similarity"] == 0.87


# --- ingest -------------------------------------------------------------------


@respx.mock
def test_ingest_posts_raw_zip_bytes_with_params(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/ingest?ns=docs&tag_prefix=release-1").mock(
        return_value=httpx.Response(202, json={"status": "complete", "files_ingested": 3})
    )
    archive = b"PK\x03\x04zipbytes"

    job = client.ingest(archive, namespace="docs", tag_prefix="release-1")

    assert route.called
    sent = route.calls.last.request
    assert sent.content == archive
    assert sent.headers["content-type"] == "application/zip"
    assert job.status == "complete"


@respx.mock
def test_ingest_content_type_override_and_default_namespace(client_ns: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/ingest?ns=default").mock(
        return_value=httpx.Response(202, json={"status": "accepted"})
    )
    client_ns.ingest(b"\x00\x01", content_type="application/octet-stream")
    sent = route.calls.last.request
    assert sent.headers["content-type"] == "application/octet-stream"


# --- health -------------------------------------------------------------------


@respx.mock
def test_health_check_and_ready(client: InfoLang) -> None:
    check = respx.get(f"{BASE_URL}/healthz").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    ready = respx.get(f"{BASE_URL}/readyz").mock(
        return_value=httpx.Response(200, json={"ml_ready": True})
    )
    assert client.health.check() == {"ok": True}
    assert client.health.ready() == {"ml_ready": True}
    assert check.called
    assert ready.called


@respx.mock
def test_health_non_dict_body(client: InfoLang) -> None:
    respx.get(f"{BASE_URL}/healthz").mock(
        return_value=httpx.Response(200, content=b"alive")
    )
    assert client.health.check() == {"status": "alive"}


# --- error envelope mapping ---------------------------------------------------


@respx.mock
def test_v2_error_envelope_maps_code_message(client: InfoLang) -> None:
    respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            401, json={"error": {"code": "unauthorized", "message": "bad key"}}
        )
    )
    with pytest.raises(AuthenticationError, match="unauthorized: bad key"):
        client.recall("q")


@respx.mock
def test_422_maps_to_validation_error(client: InfoLang) -> None:
    respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            422, json={"error": {"code": "invalid_body", "message": "query required"}}
        )
    )
    with pytest.raises(ValidationError):
        client.recall("")


@respx.mock
def test_500_maps_to_server_error() -> None:
    respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    il = InfoLang.from_api_key(
        "il_live_test", base_url=BASE_URL, workspace=WS, max_retries=0
    )
    with pytest.raises(ServerError):
        il.recall("q")
    il.close()


# --- construction -------------------------------------------------------------


def test_missing_credentials_raises() -> None:
    with pytest.raises(InfoLangConfigError):
        InfoLang()


def test_cloud_default_for_api_and_dev_keys() -> None:
    cloud = InfoLang.from_api_key("il_live_x")
    assert cloud._base_url == CLOUD_BASE_URL  # noqa: SLF001
    cloud.close()

    # 0.3.0: dev keys no longer redirect to the direct base URL.
    dev = InfoLang.from_dev_key("k:ns")
    assert dev._base_url == CLOUD_BASE_URL  # noqa: SLF001
    assert dev.namespace == "ns"
    dev.close()


def test_env_base_url_and_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFOLANG_API_KEY", "il_live_env")
    monkeypatch.setenv("INFOLANG_BASE_URL", BASE_URL)
    monkeypatch.setenv("INFOLANG_NAMESPACE", "from-env")
    il = InfoLang()
    assert il._base_url == BASE_URL  # noqa: SLF001
    assert il.namespace == "from-env"
    il.close()


def test_from_mtls_patches_transport(tmp_path) -> None:
    cert = tmp_path / "c.pem"
    key = tmp_path / "k.pem"
    cert.write_text("c")
    key.write_text("k")
    with patch("infolang.client.Transport") as mock_transport:
        il = InfoLang.from_mtls(str(cert), str(key))
        mock_transport.assert_called_once()
        il.close()


@respx.mock
def test_sync_context_manager() -> None:
    respx.get(f"{BASE_URL}/healthz").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    with InfoLang.from_api_key("il_live_x", base_url=BASE_URL, workspace=WS) as il:
        assert il.health.check()["status"] == "ok"


def test_from_session_file(tmp_path) -> None:
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"access_token": "tok"}))
    il = InfoLang.from_session_file(str(session), base_url=BASE_URL, namespace="ns")
    assert il.namespace == "ns"
    il.close()


@respx.mock
def test_ingest_files_multipart_parts(client: InfoLang) -> None:
    route = respx.post(f"{WS_PREFIX}/ingest?ns=docs").mock(
        return_value=httpx.Response(202, json={"status": "complete", "files_ingested": 2})
    )

    job = client.ingest_files(
        [("notes.md", "# hello"), ("spec.txt", b"hi")], namespace="docs"
    )

    sent = route.calls.last.request
    content_type = sent.headers["content-type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    body = sent.content
    assert body.count(b'name="files"') == 2
    assert b'filename="notes.md"' in body
    assert b'filename="spec.txt"' in body
    assert b"# hello" in body
    assert job.status == "complete"


@respx.mock
def test_lane_not_supported_is_403_authentication_error(client: InfoLang) -> None:
    # A valid il_* credential on a session-only route is 403, not 401.
    respx.post(f"{WS_PREFIX}/recall").mock(
        return_value=httpx.Response(
            403, json={"error": {"code": "lane_not_supported", "message": "wrong lane"}}
        )
    )
    with pytest.raises(AuthenticationError, match="lane_not_supported: wrong lane"):
        client.recall("q")
