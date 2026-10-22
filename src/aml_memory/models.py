"""Request/response models and content helpers."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

ROLES = {"user", "assistant"}


def content_to_text(content: str | list[dict[str, Any]]) -> str:
    """Flatten the contract's content field to text.

    Textual runs send strings. Content-part arrays are accepted for forward
    compatibility; image parts are ignored because this service is text-only.
    """
    if isinstance(content, str):
        return content
    pieces: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                pieces.append(text)
    return "\n".join(pieces)


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[dict[str, Any]]
    timestamp: int | None = Field(default=None, ge=0)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in ROLES:
            raise ValueError("role must be 'user' or 'assistant'")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        text = content_to_text(value)
        if not text.strip():
            raise ValueError("content must be non-empty")
        return value


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

    def canonical_payload(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str | list[dict[str, Any]]
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    options: list[str] | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if not content_to_text(value).strip():
            raise ValueError("query must be non-empty")
        return value
