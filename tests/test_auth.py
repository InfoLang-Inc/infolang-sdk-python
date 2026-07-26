from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
import respx

from infolang import (
    CLOUD_BASE_URL,
    DevKeyAuth,
    InfoLang,
    InfoLangConfigError,
    MtlsAuth,
    OriginAuth,
    SessionFileAuth,
)


def test_api_key_empty_raises() -> None:
    with pytest.raises(InfoLangConfigError, match="empty"):
        from infolang.auth import ApiKeyAuth

        ApiKeyAuth("")


def test_dev_key_headers() -> None:
    auth = DevKeyAuth("mykey:myns")
    assert auth.headers()["Authorization"] == "Bearer mykey"
    assert auth.namespace == "myns"


def test_origin_auth_headers() -> None:
    auth = OriginAuth("ws", "secret")
    headers = auth.headers()
    assert headers["X-InfoLang-Workspace"] == "ws"
    assert headers["X-InfoLang-Origin-Secret"] == "secret"


def test_session_missing_access_token(tmp_path: Path) -> None:
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"refresh_token": "r"}))
    auth = SessionFileAuth(session)
    with pytest.raises(InfoLangConfigError, match="access_token"):
        auth.headers()


def test_session_invalid_json(tmp_path: Path) -> None:
    session = tmp_path / "session.json"
    session.write_text("{not json")
    auth = SessionFileAuth(session)
    with pytest.raises(InfoLangConfigError, match="not valid JSON"):
        auth.headers()


@respx.mock
def test_session_refresh_on_expiry(tmp_path: Path) -> None:
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "access_token": "old",
                "refresh_token": "rt",
                "token_url": "https://auth.test/token",
                "expires_at": time.time() - 60,
            }
        )
    )
    respx.post("https://auth.test/token").mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "new", "expires_in": 3600},
        )
    )
    auth = SessionFileAuth(session)
    assert auth.headers()["Authorization"] == "Bearer new"
    saved = json.loads(session.read_text())
    assert saved["access_token"] == "new"


def test_session_expired_without_refresh(tmp_path: Path) -> None:
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps({"access_token": "stale", "expires_at": time.time() - 60})
    )
    auth = SessionFileAuth(session)
    assert auth.headers()["Authorization"] == "Bearer stale"


def test_mtls_transport_options(tmp_path: Path) -> None:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("cert")
    key.write_text("key")
    auth = MtlsAuth(cert, key)
    assert auth.transport_options() == {"cert": (str(cert), str(key))}
    assert auth.headers() == {}


def test_mtls_missing_cert(tmp_path: Path) -> None:
    with pytest.raises(InfoLangConfigError, match="cert not found"):
        MtlsAuth(tmp_path / "missing.pem", tmp_path / "key.pem")


def test_mtls_missing_key(tmp_path: Path) -> None:
    cert = tmp_path / "cert.pem"
    cert.write_text("cert")
    with pytest.raises(InfoLangConfigError, match="key not found"):
        MtlsAuth(cert, tmp_path / "missing-key.pem")


def test_api_key_defaults_to_cloud() -> None:
    il = InfoLang.from_api_key("il_live_x")
    assert il._base_url == CLOUD_BASE_URL  # noqa: SLF001
    il.close()


def test_dev_key_defaults_to_cloud() -> None:
    # 0.3.0: dev keys no longer redirect to the (removed) direct endpoint.
    il = InfoLang.from_dev_key("secret:acme")
    assert il._base_url == CLOUD_BASE_URL  # noqa: SLF001
    assert il.namespace == "acme"
    il.close()


def test_missing_credentials_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFOLANG_API_KEY", raising=False)
    monkeypatch.delenv("INFOLANG_DEV_KEY", raising=False)
    with pytest.raises(InfoLangConfigError):
        InfoLang()


def test_dev_key_requires_namespace_form() -> None:
    with pytest.raises(InfoLangConfigError):
        DevKeyAuth("no-colon")


def test_session_file_reads_token(tmp_path: Path) -> None:
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"access_token": "tok_abc"}))
    auth = SessionFileAuth(session)
    assert auth.headers()["Authorization"] == "Bearer tok_abc"


def test_session_file_missing_raises() -> None:
    auth = SessionFileAuth("/nonexistent/session.json")
    with pytest.raises(InfoLangConfigError):
        auth.headers()
