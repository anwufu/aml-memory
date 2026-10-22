"""Alibaba Cloud Bailian text-embedding-v4 client."""
from __future__ import annotations

import asyncio
import secrets

import httpx
import numpy as np

from .config import Settings

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 524}


class EmbeddingError(RuntimeError):
    """The embedding provider could not return usable vectors."""


class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(max(1, settings.embedding_concurrency))

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.embedding_timeout_s),
            limits=httpx.Limits(max_connections=self.settings.embedding_concurrency * 2),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = max(1, self.settings.embedding_batch_size)
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        vectors = [vector for batch in results for vector in batch]
        if len(vectors) != len(texts):
            raise EmbeddingError("embedding provider returned a different number of vectors")
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self._client is None:
            raise EmbeddingError("embedding client is not started")

        payload = {
            "model": self.settings.embedding_model,
            "input": [text[: self.settings.max_embed_chars] for text in batch],
            "dimensions": self.settings.embedding_dims,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        url = self.settings.embedding_base_url.rstrip("/") + "/embeddings"

        async with self._semaphore:
            for attempt in range(6):
                try:
                    response = await self._client.post(url, json=payload, headers=headers)
                    if response.status_code in RETRYABLE_STATUS:
                        if attempt == 5:
                            raise EmbeddingError(
                                f"embedding provider kept returning {response.status_code}"
                            )
                        await asyncio.sleep(_backoff(attempt))
                        continue
                    if response.status_code >= 400:
                        body = response.text[:300]
                        raise EmbeddingError(
                            f"embedding provider returned {response.status_code}: {body}"
                        )
                    data = response.json().get("data")
                    if not isinstance(data, list) or len(data) != len(batch):
                        raise EmbeddingError("embedding response has an unexpected shape")
                    data.sort(key=lambda item: item.get("index", 0))
                    vectors: list[list[float]] = []
                    for item in data:
                        vector = np.asarray(item["embedding"], dtype=np.float32)
                        if vector.size != self.settings.embedding_dims:
                            raise EmbeddingError(
                                f"expected {self.settings.embedding_dims} dimensions, "
                                f"got {vector.size}"
                            )
                        norm = float(np.linalg.norm(vector))
                        if norm > 0:
                            vector = vector / norm
                        vectors.append(vector.tolist())
                    return vectors
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt == 5:
                        raise EmbeddingError(f"embedding request failed: {exc}") from exc
                    await asyncio.sleep(_backoff(attempt))

        raise EmbeddingError("embedding request exhausted retries")


def _backoff(attempt: int) -> float:
    # Jitter only spreads retries; it is not used as key material.
    return min(30.0, 2.0**attempt) + secrets.randbelow(1000) / 1000.0
