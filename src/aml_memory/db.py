"""Async PostgreSQL access.

All DDL below is static, repository-controlled text. No request data is ever
interpolated into SQL; every query elsewhere in this service uses asyncpg
placeholders ($1, $2, ...).
"""
from __future__ import annotations

import asyncpg

from .config import Settings


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(
            dsn=self.settings.database_url,
            min_size=self.settings.db_pool_min,
            max_size=self.settings.db_pool_max,
            command_timeout=120,
        )

    async def init_schema(self) -> None:
        if self.pool is None:
            raise RuntimeError("database is not connected")
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS add_requests (
                    user_id        TEXT        NOT NULL,
                    request_id     TEXT        NOT NULL,
                    payload_sha256 TEXT        NOT NULL,
                    response       JSONB       NOT NULL DEFAULT '{}'::jsonb,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (user_id, request_id)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id    TEXT   NOT NULL,
                    session_id TEXT   NOT NULL,
                    next_seq   BIGINT NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, session_id)
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          BIGSERIAL    PRIMARY KEY,
                    user_id     TEXT         NOT NULL,
                    session_id  TEXT         NOT NULL,
                    request_id  TEXT         NOT NULL,
                    start_seq   BIGINT       NOT NULL,
                    end_seq     BIGINT       NOT NULL,
                    content     TEXT         NOT NULL,
                    embedding   BYTEA        NOT NULL,
                    first_ts_ms BIGINT,
                    last_ts_ms  BIGINT,
                    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS chunks_user_idx
                    ON chunks (user_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS chunks_user_session_seq_idx
                    ON chunks (user_id, session_id, start_seq, end_seq)
            """)
            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS chunks_request_window_idx
                    ON chunks (user_id, request_id, start_seq, end_seq)
            """)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
