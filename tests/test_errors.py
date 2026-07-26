from __future__ import annotations

import pytest

from infolang.errors import (
    AuthenticationError,
    InfoLangAPIError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
    _message_from_body,
    error_code,
    error_from_response,
)


@pytest.mark.parametrize(
    ("status", "expected_cls"),
    [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (404, NotFoundError),
        (400, ValidationError),
        (422, ValidationError),
        (429, RateLimitError),
        (500, ServerError),
        (418, InfoLangAPIError),
    ],
)
def test_error_from_response(status: int, expected_cls: type[InfoLangAPIError]) -> None:
    err = error_from_response(status, {"error": "boom"}, "req_1", 2.0)
    assert isinstance(err, expected_cls)
    assert err.status == status
    assert err.request_id == "req_1"
    if isinstance(err, RateLimitError):
        assert err.retry_after == 2.0


def test_v2_envelope_message_is_code_colon_message() -> None:
    body = {"error": {"code": "lane_not_supported", "message": "wrong lane"}}
    assert _message_from_body(body) == "lane_not_supported: wrong lane"
    err = error_from_response(401, body, None, None)
    assert "lane_not_supported: wrong lane" in str(err)


def test_v2_envelope_partial_fields() -> None:
    assert _message_from_body({"error": {"message": "just message"}}) == "just message"
    assert _message_from_body({"error": {"code": "just_code"}}) == "just_code"
    # An empty envelope falls through to nothing.
    assert _message_from_body({"error": {}}) is None


def test_flat_message_keys_still_supported() -> None:
    assert _message_from_body({"error": "e"}) == "e"
    assert _message_from_body({"message": "m"}) == "m"
    assert _message_from_body({"detail": "d"}) == "d"
    assert _message_from_body("plain") == "plain"
    assert _message_from_body({}) is None
    assert _message_from_body(None) is None


def test_error_code_helper() -> None:
    assert error_code({"error": {"code": "quota_exceeded", "message": "m"}}) == "quota_exceeded"
    assert error_code({"error": "flat"}) is None
    assert error_code({"error": {"code": 42}}) is None
    assert error_code("nope") is None
    assert error_code(None) is None


def test_default_message_when_body_unusable() -> None:
    err = error_from_response(502, None, None, None)
    assert "status 502" in str(err)


def test_request_id_suffix() -> None:
    err = error_from_response(404, {"error": "gone"}, "req_9", None)
    assert "(request_id=req_9)" in str(err)
