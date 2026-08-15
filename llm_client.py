"""Unified LLM client supporting Groq, Ollama, and OpenAI-compatible endpoints.

Provides a single `generate()` interface so evaluation suites don't care
which backend is in use.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""

    text: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict)


class LLMClient:
    """Unified client for querying LLMs across providers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.provider = provider or LLM_PROVIDER
        self._api_key = api_key
        self._base_url = base_url

        if self.provider == "groq":
            self.model = model or GROQ_MODEL
            self._api_key = self._api_key or GROQ_API_KEY
            self._base_url = "https://api.groq.com/openai/v1"
        elif self.provider == "ollama":
            self.model = model or OLLAMA_MODEL
            self._base_url = self._base_url or OLLAMA_BASE_URL
        elif self.provider == "openai_compatible":
            self.model = model or OPENAI_MODEL
            self._api_key = self._api_key or OPENAI_API_KEY
            self._base_url = self._base_url or OPENAI_BASE_URL
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a prompt and return a standardized response."""
        if self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt, temperature, max_tokens)
        else:
            return self._call_openai_compatible(
                prompt, system_prompt, temperature, max_tokens
            )

    @staticmethod
    def _retry_wait_seconds(resp: "requests.Response", fallback: float) -> float:
        """Determine how long to wait before retrying a 429 response.

        Prefers the Retry-After header, then falls back to parsing the
        "retry in Xs" hint some providers (e.g. Gemini) embed in the error
        body, then a plain exponential backoff.
        """
        header_val = resp.headers.get("retry-after")
        if header_val:
            try:
                return float(header_val)
            except ValueError:
                pass

        match = re.search(r"retry in ([\d.]+)s", resp.text, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 1.0

        return fallback

    def _call_openai_compatible(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Groq or any OpenAI-compatible chat completions endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        max_retries = 5
        backoff = 2.0
        start = time.perf_counter()
        for attempt in range(max_retries + 1):
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429 and attempt < max_retries:
                wait = self._retry_wait_seconds(resp, backoff)
                logger.warning(
                    "Rate limited (429), retrying in %.1fs (attempt %d/%d)",
                    wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
                backoff *= 2
                continue
            break
        latency = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=self.model,
            provider=self.provider,
            latency_ms=round(latency, 1),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )

    def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Ollama's /api/chat endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        start = time.perf_counter()
        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        latency = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=self.model,
            provider="ollama",
            latency_ms=round(latency, 1),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            raw=data,
        )

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"
