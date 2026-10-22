"""Test fixtures: real PostgreSQL, deterministic fake embedder."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["MEMORY_API_KEY"] = "test-key"

from aml_memory.api import create_app  # noqa: E402
from aml_memory.config import Settings  # noqa: E402
from aml_memory.db import Database  # noqa: E402
from aml_memory.service import MemoryService  # noqa: E402


class FakeEmbedder:
    """Deterministic unit vectors; semantics are irrelevant for contract tests."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            generator = np.random.default_rng(seed)
            vector = generator.normal(size=self.dimensions).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            vectors.append((vector / norm).tolist())
        return vectors

    async def close(self) -> None:
        return None


@pytest_asyncio.fixture
async def service():
    settings = Settings()
    database = Database(settings)
    await database.connect()
    await database.init_schema()
    async with database.pool.acquire() as conn:
        await conn.execute("TRUNCATE chunks, add_requests, sessions RESTART IDENTITY")
    memory_service = MemoryService(
        database,
        FakeEmbedder(settings.embedding_dims),
        settings,
    )
    try:
        yield memory_service
    finally:
        await database.close()


@pytest_asyncio.fixture
async def client(service):
    app = create_app()
    app.state.service = service
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
