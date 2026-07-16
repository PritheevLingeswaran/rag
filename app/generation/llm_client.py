"""LLM clients. GeminiClient talks to the Gemini REST API via httpx.

Every failure maps to exactly one typed exception from app.errors (the
table lives in app/generation/service.py, which decides what the caller
receives). This module never swallows or retries anything -- retry policy
belongs to the service layer, transport truth belongs here.

REST rather than the google-genai SDK: one fewer heavyweight pinned
dependency, and the error surface (status codes, retry-after) is exactly
what Stage 2.5 documented instead of being wrapped in SDK exceptions.

temperature=0 for reproducibility-in-intent; the API does not guarantee
bit-identical sampling, which is one reason live-LLM eval runs are tagged
and kept separate from the deterministic retrieval eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from app.errors import (
    LLMAuthError,
    LLMConfigError,
    LLMMalformedError,
    LLMQuotaError,
    LLMServerError,
    LLMTimeoutError,
)
from app.logging_config import get_logger

logger = get_logger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    output_tokens: int | None


class LLMClient(Protocol):
    def generate(self, prompt: str) -> LLMResponse: ...


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite",
                 timeout_s: float = 20.0, max_output_tokens: int = 1024,
                 base_url: str = GEMINI_BASE_URL,
                 transport: httpx.BaseTransport | None = None) -> None:
        if not api_key:
            raise LLMAuthError("GEMINI_API_KEY is empty")
        self._model = model
        self._timeout_s = timeout_s
        self._max_output_tokens = max_output_tokens
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout_s, transport=transport,
            headers={"x-goog-api-key": api_key},
        )

    def generate(self, prompt: str,
                 max_output_tokens: int | None = None) -> LLMResponse:
        # Per-request cap can only LOWER the configured ceiling, never
        # raise it (cost guardrail).
        cap = self._max_output_tokens
        if max_output_tokens is not None:
            cap = min(cap, max_output_tokens)
        try:
            resp = self._client.post(
                f"/models/{self._model}:generateContent",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": cap,
                    },
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"gemini request exceeded {self._timeout_s}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"network error calling gemini: {exc}") from exc

        if resp.status_code == 429:
            retry_after = None
            header = resp.headers.get("retry-after")
            if header is not None:
                try:
                    retry_after = float(header)
                except ValueError:
                    logger.warning("unparseable_retry_after", value=header)
            raise LLMQuotaError(
                "gemini quota exhausted (429 RESOURCE_EXHAUSTED)",
                retry_after_s=retry_after,
            )
        if resp.status_code in (401, 403):
            raise LLMAuthError(f"gemini auth failed: HTTP {resp.status_code}")
        if resp.status_code >= 500:
            raise LLMServerError(f"gemini server error: HTTP {resp.status_code}")
        if resp.status_code != 200:
            # Remaining 4xx (404 unknown/retired model, 400 bad request):
            # OUR configuration is wrong, not Gemini's response shape.
            raise LLMConfigError(
                f"gemini rejected the request as configured (HTTP "
                f"{resp.status_code}, model={self._model!r}): {resp.text[:200]}"
            )

        try:
            body = resp.json()
        except ValueError as exc:
            raise LLMMalformedError(
                f"gemini returned unparseable JSON: {resp.text[:200]}"
            ) from exc

        try:
            candidates = body["candidates"]
            if not candidates:
                raise KeyError("empty candidates")
            parts = candidates[0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMMalformedError(
                f"gemini response missing expected fields: {exc} "
                f"(finishReason={body.get('candidates', [{}])[0].get('finishReason') if body.get('candidates') else None})"
            ) from exc
        if not text.strip():
            raise LLMMalformedError(
                "gemini returned an empty answer "
                f"(finishReason={candidates[0].get('finishReason')})"
            )

        usage = body.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            model=self._model,
            prompt_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
        )


class GroqClient:
    """Secondary provider (Stage 2.5 planned fallback). OpenAI-compatible
    chat-completions REST; maps every failure to the SAME typed taxonomy
    as GeminiClient so GenerationService's handling is provider-blind."""

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant",
                 timeout_s: float = 20.0, max_output_tokens: int = 1024,
                 base_url: str = GROQ_BASE_URL,
                 transport: httpx.BaseTransport | None = None) -> None:
        if not api_key:
            raise LLMAuthError("GROQ_API_KEY is empty")
        self._model = model
        self._timeout_s = timeout_s
        self._max_output_tokens = max_output_tokens
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout_s, transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def generate(self, prompt: str,
                 max_output_tokens: int | None = None) -> LLMResponse:
        cap = self._max_output_tokens
        if max_output_tokens is not None:
            cap = min(cap, max_output_tokens)
        try:
            resp = self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": cap,
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"groq request exceeded {self._timeout_s}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"network error calling groq: {exc}") from exc

        if resp.status_code == 429:
            retry_after = None
            header = resp.headers.get("retry-after")
            if header is not None:
                try:
                    retry_after = float(header)
                except ValueError:
                    logger.warning("unparseable_retry_after", value=header)
            raise LLMQuotaError("groq quota exhausted (429)",
                                retry_after_s=retry_after)
        if resp.status_code in (401, 403):
            raise LLMAuthError(f"groq auth failed: HTTP {resp.status_code}")
        if resp.status_code >= 500:
            raise LLMServerError(f"groq server error: HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise LLMConfigError(
                f"groq rejected the request as configured (HTTP "
                f"{resp.status_code}, model={self._model!r}): {resp.text[:200]}"
            )

        try:
            body = resp.json()
            text = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMMalformedError(
                f"groq response missing expected fields: {exc}: {resp.text[:200]}"
            ) from exc
        if not text or not text.strip():
            raise LLMMalformedError("groq returned an empty answer")

        usage = body.get("usage", {})
        return LLMResponse(
            text=text,
            model=self._model,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
