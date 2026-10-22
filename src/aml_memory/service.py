"""Add/Search business logic."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import Settings
from .db import Database
from .models import AddRequest, Message, SearchRequest, content_to_text

WINDOW_MAX_CHARS = 2000
SPLIT_OVERLAP = 100
DENSE_CANDIDATES = 200
SPARSE_CANDIDATES = 200


class ContractConflict(Exception):
    """The same (user_id, request_id) was reused with a different payload."""


@dataclass
class Entry:
    seq: int
    role: str
    timestamp: int | None
    text: str


@dataclass
class Window:
    start_local: int
    end_local: int
    content: str
    first_ts: int | None
    last_ts: int | None


def format_timestamp(ms: int | None) -> str:
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_text(text: str, max_chars: int = WINDOW_MAX_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            window = text[start:end]
            cut = max(
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind("。"),
            )
            if cut > max_chars * 0.5:
                end = start + cut + 1
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= length:
            break
        start = max(start + 1, end - SPLIT_OVERLAP)
    return parts


def expand_entries(messages: list[Message], max_chars: int = WINDOW_MAX_CHARS) -> list[Entry]:
    entries: list[Entry] = []
    seq = 0
    for message in messages:
        text = content_to_text(message.content)
        for part in split_text(text, max_chars):
            entries.append(
                Entry(seq=seq, role=message.role, timestamp=message.timestamp, text=part)
            )
            seq += 1
    return entries


def build_windows(entries: list[Entry], window_size: int) -> list[Window]:
    window_size = max(1, window_size)
    windows: list[Window] = []
    for start in range(0, len(entries), window_size):
        group = entries[start : start + window_size]
        lines: list[str] = []
        for entry in group:
            stamp = format_timestamp(entry.timestamp)
            speaker = "User" if entry.role == "user" else "Assistant"
            prefix = f"[{stamp}] " if stamp else ""
            lines.append(f"{prefix}{speaker}: {entry.text}")
        timestamps = [entry.timestamp for entry in group if entry.timestamp is not None]
        windows.append(
            Window(
                start_local=group[0].seq,
                end_local=group[-1].seq,
                content="\n".join(lines),
                first_ts=min(timestamps) if timestamps else None,
                last_ts=max(timestamps) if timestamps else None,
            )
        )
    return windows


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]]) -> list[float]:
    count = len(docs_tokens)
    if count == 0:
        return []
    average_length = sum(len(doc) for doc in docs_tokens) / count or 1.0
    document_frequency: Counter[str] = Counter()
    for doc in docs_tokens:
        document_frequency.update(set(doc))

    k1, b = 1.5, 0.75
    scores: list[float] = []
    for doc in docs_tokens:
        frequencies = Counter(doc)
        length = len(doc)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if frequency == 0:
                continue
            idf = math.log(1.0 + (count - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            score += idf * (frequency * (k1 + 1)) / (
                frequency + k1 * (1 - b + b * length / average_length)
            )
        scores.append(score)
    return scores


def rrf_fuse(dense: np.ndarray, sparse: np.ndarray, k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    dense_order = np.argsort(-dense)[:DENSE_CANDIDATES].tolist()
    for rank, index in enumerate(dense_order):
        fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank + 1)

    sparse_order = [int(i) for i in np.argsort(-sparse) if sparse[i] > 0][:SPARSE_CANDIDATES]
    for rank, index in enumerate(sparse_order):
        fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank + 1)
    return fused


def trim_content(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return content[: max(1, limit - 1)].rstrip() + "…"


WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

RELATIVE_PATTERN = re.compile(
    r"\b(?:last|previous)\s+"
    r"(?:weekend|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def resolve_relative_dates(text: str, anchor_ms: int | None) -> str:
    """Append a resolved date to relative time expressions.

    This is a display-time annotation only: the stored/embedded content is
    unchanged, and the original phrase is preserved in front of the parenthesis.
    """
    if anchor_ms is None:
        return text
    anchor = datetime.fromtimestamp(anchor_ms / 1000, tz=timezone.utc)
    anchor_date = anchor.strftime("%Y-%m-%d")

    def replace(match: re.Match) -> str:
        phrase = match.group(0)
        lowered = phrase.lower()
        if "weekend" in lowered:
            return f"{phrase} (the weekend before {anchor_date})"
        if "week" in lowered:
            return f"{phrase} (the week before {anchor_date})"
        if "month" in lowered:
            previous_month = anchor.replace(day=1) - timedelta(days=1)
            return f"{phrase} ({previous_month.strftime('%Y-%m')})"
        if "year" in lowered:
            return f"{phrase} ({anchor.year - 1})"
        weekday = lowered.split()[-1]
        target = WEEKDAYS.get(weekday)
        if target is None:
            return phrase
        delta = (anchor.weekday() - target) % 7
        if delta == 0:
            delta = 7
        resolved = (anchor - timedelta(days=delta)).strftime("%Y-%m-%d")
        return f"{phrase} ({resolved})"

    text = RELATIVE_PATTERN.sub(replace, text)
    text = re.sub(
        r"\byesterday\b",
        f"yesterday ({(anchor - timedelta(days=1)).strftime('%Y-%m-%d')})",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\btoday\b", f"today ({anchor_date})", text, flags=re.IGNORECASE)
    return text


TEMPORAL_MARKERS = (
    "when",
    "what date",
    "which month",
    "which year",
    "what time",
    "what day",
    "what year",
    "which day",
)


def is_temporal_query(query: str) -> bool:
    lowered = query.lower()
    if "how long" in lowered or "duration" in lowered:
        return False
    return any(marker in lowered for marker in TEMPORAL_MARKERS)


class MemoryService:
    def __init__(self, db: Database, embedder, settings: Settings) -> None:
        self.db = db
        self.embedder = embedder
        self.settings = settings

    async def add(self, request: AddRequest) -> dict:
        response = {
            "success": True,
            "request_id": request.request_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }
        payload_hash = hashlib.sha256(
            request.canonical_payload().encode("utf-8")
        ).hexdigest()

        entries = expand_entries(request.messages)
        if not entries:
            raise ValueError("no usable content in messages")
        windows = build_windows(entries, self.settings.window_messages)

        if self.db.pool is None:
            raise RuntimeError("database is not connected")

        async with self.db.pool.acquire() as conn:
            async with conn.transaction():
                inserted = await conn.fetchval(
                    """
                    INSERT INTO add_requests (user_id, request_id, payload_sha256, response)
                    VALUES ($1, $2, $3, '{}'::jsonb)
                    ON CONFLICT (user_id, request_id) DO NOTHING
                    RETURNING 1
                    """,
                    request.user_id,
                    request.request_id,
                    payload_hash,
                )
                if inserted is None:
                    row = await conn.fetchrow(
                        """
                        SELECT payload_sha256, response
                        FROM add_requests
                        WHERE user_id = $1 AND request_id = $2
                        """,
                        request.user_id,
                        request.request_id,
                    )
                    if row is None:
                        raise RuntimeError("idempotency row disappeared")
                    if row["payload_sha256"] != payload_hash:
                        raise ContractConflict(
                            "request_id was reused with a different payload"
                        )
                    return json.loads(row["response"])

                await conn.execute(
                    """
                    INSERT INTO sessions (user_id, session_id, next_seq)
                    VALUES ($1, $2, 0)
                    ON CONFLICT DO NOTHING
                    """,
                    request.user_id,
                    request.session_id,
                )
                base_seq = await conn.fetchval(
                    """
                    UPDATE sessions
                    SET next_seq = next_seq + $3
                    WHERE user_id = $1 AND session_id = $2
                    RETURNING next_seq - $3
                    """,
                    request.user_id,
                    request.session_id,
                    len(entries),
                )

                vectors = await self.embedder.embed([window.content for window in windows])
                for window, vector in zip(windows, vectors):
                    await conn.execute(
                        """
                        INSERT INTO chunks (
                            user_id, session_id, request_id,
                            start_seq, end_seq, content, embedding,
                            first_ts_ms, last_ts_ms
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        request.user_id,
                        request.session_id,
                        request.request_id,
                        base_seq + window.start_local,
                        base_seq + window.end_local,
                        window.content,
                        np.asarray(vector, dtype=np.float32).tobytes(),
                        window.first_ts,
                        window.last_ts,
                    )

                await conn.execute(
                    """
                    UPDATE add_requests
                    SET response = $3::jsonb
                    WHERE user_id = $1 AND request_id = $2
                    """,
                    request.user_id,
                    request.request_id,
                    json.dumps(response, ensure_ascii=False),
                )
        return response

    async def search(self, request: SearchRequest) -> dict:
        top_k = min(request.top_k, self.settings.top_k_max)
        query_text = content_to_text(request.query)
        apply_dates = self.settings.resolve_relative_dates and is_temporal_query(query_text)
        if self.db.pool is None:
            raise RuntimeError("database is not connected")

        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, start_seq, end_seq, content, embedding, first_ts_ms
                FROM chunks
                WHERE user_id = $1
                """,
                request.user_id,
            )
        if not rows:
            return {"data": []}

        query_vector = np.asarray(
            (await self.embedder.embed([query_text]))[0], dtype=np.float32
        )
        matrix = np.stack(
            [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
        )
        dense_scores = matrix @ query_vector

        docs_tokens = [tokenize(row["content"]) for row in rows]
        sparse_scores = np.asarray(
            bm25_scores(tokenize(query_text), docs_tokens), dtype=np.float32
        )
        fused = rrf_fuse(dense_scores, sparse_scores, self.settings.rrf_k)

        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        selected: dict[int, float] = dict(ranked[:top_k])

        by_session: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            by_session.setdefault(row["session_id"], []).append(index)

        radius = max(0, self.settings.neighbor_radius) * max(
            1, self.settings.window_messages
        )
        for index, score in ranked[: self.settings.neighbor_seeds]:
            seed = rows[index]
            for other in by_session.get(seed["session_id"], []):
                if other == index or other in selected:
                    continue
                row = rows[other]
                if row["end_seq"] < seed["start_seq"] - radius:
                    continue
                if row["start_seq"] > seed["end_seq"] + radius:
                    continue
                candidate = score * self.settings.neighbor_decay
                if candidate > selected.get(other, -1.0):
                    selected[other] = candidate

        final = sorted(selected.items(), key=lambda item: (-item[1], item[0]))[:top_k]

        data: list[dict] = []
        budget = self.settings.max_total_chars
        for index, score in final:
            row = rows[index]
            content = row["content"]
            if apply_dates:
                content = resolve_relative_dates(content, row["first_ts_ms"])
            if self.settings.window_date_header and row["first_ts_ms"] is not None:
                content = f"Conversation date: {format_timestamp(row['first_ts_ms'])}\n{content}"
            content = trim_content(content, self.settings.max_item_chars)
            if data and len(content) > budget:
                break
            budget -= len(content)
            item = {
                "id": str(row["id"]),
                "content": content,
                "score": float(score),
            }
            if row["first_ts_ms"] is not None:
                item["created_at"] = format_timestamp(row["first_ts_ms"])
            data.append(item)
            if budget <= 0:
                break
        return {"data": data}
