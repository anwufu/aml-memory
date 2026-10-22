"""Minimal async client for OpenAI-compatible chat endpoints."""
from __future__ import annotations

import asyncio
import secrets

import httpx

RETRYABLE = {408, 425, 429, 500, 502, 503, 504, 524}


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        concurrency: int = 8,
        timeout: float = 180.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with self._semaphore:
            for attempt in range(6):
                try:
                    response = await self._client.post(url, json=payload, headers=headers)
                    if response.status_code in RETRYABLE:
                        if attempt == 5:
                            raise LLMError(f"chat endpoint kept returning {response.status_code}")
                        await asyncio.sleep(min(30.0, 2.0**attempt) + secrets.randbelow(1000) / 1000)
                        continue
                    if response.status_code >= 400:
                        raise LLMError(
                            f"chat endpoint returned {response.status_code}: "
                            f"{response.text[:300]}"
                        )
                    data = response.json()
                    return str(data["choices"][0]["message"].get("content") or "").strip()
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == 5:
                        raise LLMError(f"chat request failed: {exc}") from exc
                    await asyncio.sleep(min(30.0, 2.0**attempt) + secrets.randbelow(1000) / 1000)
        raise LLMError("chat request exhausted retries")
