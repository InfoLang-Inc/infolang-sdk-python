from __future__ import annotations

from collections.abc import Iterator

import pytest

from infolang import InfoLang

BASE_URL = "https://api.test.infolang.ai"
WS = "ws_test"
WS_PREFIX = f"{BASE_URL}/v2/workspaces/{WS}"

_ENV_VARS = (
    "INFOLANG_API_KEY",
    "INFOLANG_DEV_KEY",
    "INFOLANG_BASE_URL",
    "INFOLANG_NAMESPACE",
    "INFOLANG_WORKSPACE",
    "INFOLANG_WORKSPACE_ID",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient InfoLang env vars from leaking into tests."""

    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client() -> Iterator[InfoLang]:
    """A client with an explicit workspace — no whoami round trip."""

    il = InfoLang.from_api_key("il_live_test", base_url=BASE_URL, workspace=WS)
    yield il
    il.close()


@pytest.fixture
def client_ns() -> Iterator[InfoLang]:
    """Same as ``client`` but with a default namespace configured."""

    il = InfoLang.from_api_key(
        "il_live_test", base_url=BASE_URL, workspace=WS, namespace="default"
    )
    yield il
    il.close()
