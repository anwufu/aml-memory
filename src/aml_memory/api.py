"""FastAPI surface for the AML Add/Search contract."""
from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .db import Database
from .embedding import EmbeddingClient, EmbeddingError
from .models import AddRequest, SearchRequest
from .service import ContractConflict, MemoryService

logger = logging.getLogger("aml_memory")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    database = Database(settings)
    await database.connect()
    await database.init_schema()
    embedder = EmbeddingClient(settings)
    await embedder.start()
    app.state.service = MemoryService(database, embedder, settings)
    logger.info("service ready: %s %s", settings.service_name, settings.service_version)
    try:
        yield
    finally:
        await embedder.close()
        await database.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        lifespan=lifespan,
    )

    async def require_auth(request: Request) -> None:
        expected = request.app.state.service.settings.memory_api_key
        token: str | None = None
        authorization = request.headers.get("authorization")
        if authorization:
            parts = authorization.split(None, 1)
            if len(parts) == 2 and parts[0].lower() in {"bearer", "token"}:
                token = parts[1].strip()
        if not token:
            token = request.headers.get("x-api-key")
        if not token or not secrets.compare_digest(token, expected):
            raise HTTPException(
                status_code=401,
                detail={"reason": "invalid or missing credentials"},
            )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/add")
    @app.post("/v1/memories/add")
    async def add(
        request: Request,
        payload: AddRequest,
        _: None = Depends(require_auth),
    ) -> dict:
        return await request.app.state.service.add(payload)

    @app.post("/search")
    @app.post("/v1/memories/search")
    async def search(
        request: Request,
        payload: SearchRequest,
        _: None = Depends(require_auth),
    ) -> dict:
        return await request.app.state.service.search(payload)

    @app.exception_handler(ContractConflict)
    async def contract_conflict(_: Request, exc: ContractConflict) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": {"reason": str(exc)}},
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_failure(_: Request, exc: EmbeddingError) -> JSONResponse:
        logger.error("embedding provider failure: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": {"reason": "embedding provider unavailable"}},
        )

    return app


app = create_app()
