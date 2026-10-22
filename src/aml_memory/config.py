"""Application configuration."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = "postgresql://aml:aml@127.0.0.1:5432/aml_memory"

    memory_api_key: str = "change-me"

    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dims: int = 1024
    embedding_batch_size: int = 10
    embedding_concurrency: int = 8
    embedding_timeout_s: float = 60.0
    max_embed_chars: int = 8000

    window_messages: int = 6
    rrf_k: int = 60
    neighbor_seeds: int = 20
    neighbor_radius: int = 1
    neighbor_decay: float = 0.35
    top_k_max: int = 100
    max_item_chars: int = 4000
    max_total_chars: int = 350000
    window_date_header: bool = False

    db_pool_min: int = 1
    db_pool_max: int = 20

    service_name: str = "aml-memory"
    service_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
