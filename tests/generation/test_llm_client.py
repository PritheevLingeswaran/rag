"""GeminiClient transport tests via httpx.MockTransport: every failure
mode the API can produce maps to exactly one typed exception."""

import httpx
import pytest

from app.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMMalformedError,
    LLMQuotaError,
    LLMServerError,
    LLMTimeoutError,
)
from app.generation.llm_client import GeminiClient, GroqClient


def make_client(handler) -> GeminiClient:
    return GeminiClient(
        api_key="test-key", transport=httpx.MockTransport(handler)
    )


def gemini_ok_body(text: str) -> dict:
    return {
        "candidates": [
            {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
    }


def test_happy_path_parses_text_and_usage():
    client = make_client(
        lambda req: httpx.Response(200, json=gemini_ok_body("Grounded answer [1]."))
    )
    resp = client.generate("prompt")
    assert resp.text == "Grounded answer [1]."
    assert resp.prompt_tokens == 100
    assert resp.output_tokens == 20


def test_429_raises_quota_error_with_retry_after():
    client = make_client(lambda req: httpx.Response(
        429, headers={"retry-after": "37"},
        json={"error": {"status": "RESOURCE_EXHAUSTED"}},
    ))
    with pytest.raises(LLMQuotaError) as exc_info:
        client.generate("prompt")
    assert exc_info.value.retry_after_s == 37.0


def test_429_without_retry_after_still_typed():
    client = make_client(lambda req: httpx.Response(429, json={}))
    with pytest.raises(LLMQuotaError) as exc_info:
        client.generate("prompt")
    assert exc_info.value.retry_after_s is None


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures(status):
    client = make_client(lambda req: httpx.Response(status, json={}))
    with pytest.raises(LLMAuthError):
        client.generate("prompt")


def test_500_raises_server_error():
    client = make_client(lambda req: httpx.Response(500, text="oops"))
    with pytest.raises(LLMServerError):
        client.generate("prompt")


def test_timeout_raises_timeout_error():
    def handler(req):
        raise httpx.ReadTimeout("too slow")
    client = make_client(handler)
    with pytest.raises(LLMTimeoutError):
        client.generate("prompt")


def test_network_error_raises_server_error():
    def handler(req):
        raise httpx.ConnectError("refused")
    client = make_client(handler)
    with pytest.raises(LLMServerError):
        client.generate("prompt")


@pytest.mark.parametrize("status", [400, 404])
def test_config_rejections_raise_config_error_not_malformed(status):
    """A retired/unknown model id 404s; that is OUR config, not a
    provider response-shape problem (regression: prod 404 surfaced as
    degraded_llm_malformed and sent diagnosis the wrong way)."""
    client = make_client(lambda req: httpx.Response(
        status, json={"error": {"message": "model not found"}}
    ))
    with pytest.raises(LLMConfigError, match=str(status)):
        client.generate("prompt")


def test_unparseable_json_raises_malformed():
    client = make_client(lambda req: httpx.Response(200, text="<html>not json"))
    with pytest.raises(LLMMalformedError, match="unparseable"):
        client.generate("prompt")


def test_empty_candidates_raises_malformed():
    client = make_client(lambda req: httpx.Response(200, json={"candidates": []}))
    with pytest.raises(LLMMalformedError):
        client.generate("prompt")


def test_safety_blocked_empty_text_raises_malformed():
    body = {"candidates": [
        {"content": {"parts": [{"text": ""}]}, "finishReason": "SAFETY"}
    ]}
    client = make_client(lambda req: httpx.Response(200, json=body))
    with pytest.raises(LLMMalformedError, match="SAFETY"):
        client.generate("prompt")


def test_empty_api_key_rejected_at_construction():
    with pytest.raises(LLMAuthError):
        GeminiClient(api_key="")


# ---- GroqClient: same taxonomy, OpenAI-compatible shape ----

def make_groq(handler) -> GroqClient:
    return GroqClient(api_key="test-key",
                      transport=httpx.MockTransport(handler))


def test_groq_happy_path_parses_text_and_usage():
    body = {
        "choices": [{"message": {"role": "assistant",
                                 "content": "Grounded answer [1]."}}],
        "usage": {"prompt_tokens": 90, "completion_tokens": 15},
    }
    resp = make_groq(lambda req: httpx.Response(200, json=body)).generate("p")
    assert resp.text == "Grounded answer [1]."
    assert resp.model == "llama-3.1-8b-instant"
    assert resp.output_tokens == 15


def test_groq_sends_bearer_and_temperature_zero():
    captured = {}

    def handler(req):
        captured["auth"] = req.headers.get("authorization")
        import json
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "x"}}], "usage": {},
        })

    make_groq(handler).generate("p")
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["messages"] == [{"role": "user", "content": "p"}]


@pytest.mark.parametrize("status,exc", [
    (429, LLMQuotaError), (401, LLMAuthError), (403, LLMAuthError),
    (500, LLMServerError), (404, LLMConfigError), (400, LLMConfigError),
])
def test_groq_error_statuses_map_to_same_taxonomy(status, exc):
    client = make_groq(lambda req: httpx.Response(status, json={}))
    with pytest.raises(exc):
        client.generate("p")


def test_groq_empty_answer_raises_malformed():
    body = {"choices": [{"message": {"content": "  "}}]}
    with pytest.raises(LLMMalformedError):
        make_groq(lambda req: httpx.Response(200, json=body)).generate("p")


def test_request_carries_key_header_and_temperature_zero():
    captured = {}

    def handler(req):
        captured["key"] = req.headers.get("x-goog-api-key")
        import json
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=gemini_ok_body("x"))

    make_client(handler).generate("prompt")
    assert captured["key"] == "test-key"
    assert captured["body"]["generationConfig"]["temperature"] == 0.0
