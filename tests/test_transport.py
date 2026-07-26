from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from infolang._transport import (
    AsyncTransport,
    Transport,
    _finish,
    _RetryPolicy,
    parse_metering,
)
from infolang.auth import ApiKeyAuth
from infolang.errors import InfoLangConnectionError, ServerError
from tests.conftest import BASE_URL


def test_retry_policy_honors_retry_after() -> None:
    policy = _RetryPolicy(2, 0.5, 8.0)
    assert policy.delay(0, 3.5) == 3.5


def test_retry_policy_jitter_window() -> None:
    policy = _RetryPolicy(2, 0.5, 8.0)
    with patch("infolang._transport.random.uniform", return_value=0.25):
        assert policy.delay(2, None) == 0.25


def test_parse_metering_valid_and_invalid() -> None:
    headers = httpx.Headers(
        {
            "x-infolang-tokens-saved": "10",
            "x-infolang-chunks-used": "bad",
            "x-infolang-repo-coverage": "0.5",
            "x-infolang-overage": "3",
            "x-request-id": "rid",
        }
    )
    meta = parse_metering(headers)
    assert meta.tokens_saved == 10
    assert meta.chunks_used is None
    assert meta.repo_coverage == 0.5
    assert meta.overage == 3
    assert meta.request_id == "rid"

    empty = parse_metering(httpx.Headers({"x-infolang-repo-coverage": "nope"}))
    assert empty.repo_coverage is None
    assert empty.overage is None


def test_finish_non_json_success() -> None:
    response = httpx.Response(200, content=b"ok", headers={"content-type": "text/plain"})
    body, meta = _finish(response)
    assert body == "ok"
    assert meta.request_id is None


@respx.mock
@patch("time.sleep")
def test_sync_retry_on_500(mock_sleep: object) -> None:
    route = respx.post(f"{BASE_URL}/v1/recall").mock(
        side_effect=[
            httpx.Response(500, json={"error": "fail"}),
            httpx.Response(200, json={"chunks": []}),
        ]
    )
    auth = ApiKeyAuth("il_live_x")
    with Transport(base_url=BASE_URL, auth=auth, max_retries=1) as transport:
        data, _ = transport.request("POST", "/v1/recall", json={"query": "q"})
    assert route.call_count == 2
    assert data == {"chunks": []}


@respx.mock
@patch("time.sleep")
def test_sync_connection_error_then_success(mock_sleep: object) -> None:
    route = respx.post(f"{BASE_URL}/v1/recall").mock(
        side_effect=[
            httpx.ConnectError("down"),
            httpx.Response(200, json={"chunks": []}),
        ]
    )
    auth = ApiKeyAuth("il_live_x")
    with Transport(base_url=BASE_URL, auth=auth, max_retries=1) as transport:
        data, _ = transport.request("POST", "/v1/recall", json={"query": "q"})
    assert route.call_count == 2
    assert data == {"chunks": []}


@respx.mock
def test_sync_server_error_after_retries() -> None:
    respx.post(f"{BASE_URL}/v1/recall").mock(return_value=httpx.Response(500, json={"error": "x"}))
    auth = ApiKeyAuth("il_live_x")
    with (
        Transport(base_url=BASE_URL, auth=auth, max_retries=0) as transport,
        pytest.raises(ServerError),
    ):
        transport.request("POST", "/v1/recall", json={})


@respx.mock
@patch("time.sleep")
def test_sync_connection_error_exhausted(mock_sleep: object) -> None:
    respx.post(f"{BASE_URL}/v1/recall").mock(side_effect=httpx.ConnectError("down"))
    auth = ApiKeyAuth("il_live_x")
    with (
        Transport(base_url=BASE_URL, auth=auth, max_retries=0) as transport,
        pytest.raises(InfoLangConnectionError),
    ):
        transport.request("POST", "/v1/recall", json={})


@respx.mock
@pytest.mark.asyncio
async def test_async_retry_on_500() -> None:
    async def fake_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    with patch("asyncio.sleep", new=fake_sleep):
        route = respx.post(f"{BASE_URL}/v1/recall").mock(
            side_effect=[
                httpx.Response(502),
                httpx.Response(200, json={"chunks": []}),
            ]
        )
        auth = ApiKeyAuth("il_live_x")
        async with AsyncTransport(base_url=BASE_URL, auth=auth, max_retries=1) as transport:
            data, _ = await transport.request("POST", "/v1/recall", json={})
        assert route.call_count == 2
        assert data == {"chunks": []}


@respx.mock
@pytest.mark.asyncio
async def test_async_connection_error_exhausted() -> None:
    async def fake_sleep(*_args: object, **_kwargs: object) -> None:
        return None

    with patch("asyncio.sleep", new=fake_sleep):
        respx.post(f"{BASE_URL}/v1/recall").mock(side_effect=httpx.ConnectError("down"))
        auth = ApiKeyAuth("il_live_x")
        async with AsyncTransport(base_url=BASE_URL, auth=auth, max_retries=0) as transport:
            with pytest.raises(InfoLangConnectionError):
                await transport.request("POST", "/v1/recall", json={})


@respx.mock
def test_sync_raw_body_skips_json_encoding() -> None:
    route = respx.post(f"{BASE_URL}/v2/workspaces/ws/ingest").mock(
        return_value=httpx.Response(202, json={"status": "accepted"})
    )
    auth = ApiKeyAuth("il_live_x")
    payload = b"PK\x03\x04rawzip"
    with Transport(base_url=BASE_URL, auth=auth) as transport:
        data, _ = transport.request(
            "POST",
            "/v2/workspaces/ws/ingest",
            content=payload,
            content_type="application/zip",
        )
    sent = route.calls.last.request
    assert sent.content == payload
    assert sent.headers["content-type"] == "application/zip"
    assert data == {"status": "accepted"}


@respx.mock
def test_sync_raw_body_defaults_to_octet_stream() -> None:
    route = respx.post(f"{BASE_URL}/v2/workspaces/ws/ingest").mock(
        return_value=httpx.Response(202, json={})
    )
    auth = ApiKeyAuth("il_live_x")
    with Transport(base_url=BASE_URL, auth=auth) as transport:
        transport.request("POST", "/v2/workspaces/ws/ingest", content=b"\x00")
    assert route.calls.last.request.headers["content-type"] == "application/octet-stream"


@respx.mock
@pytest.mark.asyncio
async def test_async_raw_body_skips_json_encoding() -> None:
    route = respx.post(f"{BASE_URL}/v2/workspaces/ws/ingest").mock(
        return_value=httpx.Response(202, json={"status": "accepted"})
    )
    auth = ApiKeyAuth("il_live_x")
    payload = b"PK\x03\x04rawzip"
    async with AsyncTransport(base_url=BASE_URL, auth=auth) as transport:
        data, _ = await transport.request(
            "POST",
            "/v2/workspaces/ws/ingest",
            content=payload,
            content_type="application/zip",
        )
    sent = route.calls.last.request
    assert sent.content == payload
    assert sent.headers["content-type"] == "application/zip"
    assert data == {"status": "accepted"}


@respx.mock
def test_extra_headers_merged() -> None:
    route = respx.get(f"{BASE_URL}/v1/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    auth = ApiKeyAuth("il_live_x")
    with Transport(base_url=BASE_URL, auth=auth) as transport:
        transport.request("GET", "/v1/health", headers={"X-Custom": "1"})
    assert route.calls.last.request.headers["x-custom"] == "1"
