"""Normalize public datasets into one JSONL shape.

Output record:
{
  "dataset": "locomo",
  "sample_id": "conv-0",
  "sessions": [{"session_id": "...", "messages": [{"role","content","timestamp","msg_id"}]}],
  "questions": [{"qid","question","answer","options","evidence_msg_ids","evidence_session_ids","category"}]
}
"""
from __future__ import annotations

import csv
import json
import ast
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "normalized"

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_locomo_datetime(value: str) -> int | None:
    match = re.search(
        r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})",
        value or "",
        re.IGNORECASE,
    )
    if not match:
        return None
    hour, minute, meridiem, day, month_name, year = match.groups()
    hour = int(hour) % 12
    if meridiem.lower() == "pm":
        hour += 12
    month = MONTHS.get(month_name.lower())
    if month is None:
        return None
    dt = datetime(int(year), month, int(day), hour, int(minute), tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_iso_datetime(value: str) -> int | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            dt = datetime.fromisoformat(text) if fmt is None else datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return None


def locomo() -> Iterable[dict[str, Any]]:
    data = json.loads((RAW / "locomo" / "data" / "locomo10.json").read_text(encoding="utf-8"))
    for sample_index, sample in enumerate(data):
        sample_id = sample.get("sample_id") or f"conv-{sample_index}"
        conversation = sample["conversation"]
        speaker_a = conversation.get("speaker_a", "Speaker A")
        speaker_b = conversation.get("speaker_b", "Speaker B")
        sessions = []
        evidence_text: dict[str, str] = {}
        index = 1
        while f"session_{index}" in conversation:
            session_id = f"session_{index}"
            timestamp = parse_locomo_datetime(conversation.get(f"{session_id}_date_time", ""))
            messages = []
            for turn in conversation[session_id]:
                text = turn.get("text") or turn.get("blip_caption") or ""
                if not text.strip():
                    continue
                speaker = turn.get("speaker")
                role = "user" if speaker == speaker_a else "assistant"
                msg_id = turn.get("dia_id") or f"{session_id}:{len(messages)}"
                evidence_text[msg_id] = text
                messages.append(
                    {
                        "role": role,
                        "content": text,
                        "timestamp": timestamp,
                        "msg_id": msg_id,
                    }
                )
            if messages:
                sessions.append({"session_id": session_id, "messages": messages})
            index += 1

        questions = []
        for qa_index, qa in enumerate(sample.get("qa", [])):
            questions.append(
                {
                    "qid": f"{sample_id}:q{qa_index}",
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "options": None,
                    "evidence_msg_ids": qa.get("evidence") or [],
                    "evidence_session_ids": None,
                    "category": qa.get("category"),
                }
            )
        yield {
            "dataset": "locomo",
            "sample_id": sample_id,
            "sessions": sessions,
            "questions": questions,
            "evidence_text": evidence_text,
        }


def longmemeval(path: Path) -> Iterable[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for record in data:
        sample_id = str(record["question_id"])
        session_ids = record.get("haystack_session_ids") or []
        dates = record.get("haystack_dates") or []
        sessions = []
        for index, raw_session in enumerate(record.get("haystack_sessions") or []):
            session_id = session_ids[index] if index < len(session_ids) else f"session-{index}"
            timestamp = parse_iso_datetime(dates[index]) if index < len(dates) else None
            messages = []
            for turn_index, turn in enumerate(raw_session):
                content = turn.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                messages.append(
                    {
                        "role": (
                            turn.get("role", "user")
                            if turn.get("role") in {"user", "assistant"}
                            else "user"
                        ),
                        "content": content,
                        "timestamp": timestamp,
                        "msg_id": f"{session_id}:{turn_index}",
                    }
                )
            if messages:
                sessions.append({"session_id": session_id, "messages": messages})
        yield {
            "dataset": "longmemeval",
            "sample_id": sample_id,
            "sessions": sessions,
            "questions": [
                {
                    "qid": sample_id,
                    "question": record.get("question", ""),
                    "answer": record.get("answer", ""),
                    "options": None,
                    "evidence_msg_ids": None,
                    "evidence_session_ids": record.get("answer_session_ids") or [],
                    "category": record.get("question_type"),
                }
            ],
            "evidence_text": {},
        }


def parse_options(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return None


def personamem() -> Iterable[dict[str, Any]]:
    contexts: dict[str, list[dict]] = {}
    context_path = RAW / "personamem" / "data" / "shared_contexts_32k.jsonl"
    for line in context_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            for context_id, messages in item.items():
                contexts[context_id] = messages

    questions_path = RAW / "personamem" / "data" / "questions_32k.csv"
    by_persona: dict[str, list[dict]] = {}
    with questions_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            context_id = row["shared_context_id"]
            messages = contexts.get(context_id, [])
            by_persona.setdefault(context_id, []).append(
                {
                    "qid": row["question_id"],
                    "question": row["user_question_or_message"],
                    "answer": row["correct_answer"],
                    "options": parse_options(row["all_options"]),
                    "evidence_msg_ids": None,
                    "evidence_session_ids": None,
                    "category": row.get("question_type"),
                }
            )

    for context_id, questions in by_persona.items():
        messages = contexts.get(context_id, [])
        normalized = []
        for index, message in enumerate(messages):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                normalized.append(
                    {
                        "role": (
                            message.get("role", "user")
                            if message.get("role") in {"user", "assistant"}
                            else "user"
                        ),
                        "content": content,
                        "timestamp": None,
                        "msg_id": f"{context_id}:{index}",
                    }
                )
        if not normalized:
            continue
        yield {
            "dataset": "personamem",
            "sample_id": context_id,
            "sessions": [{"session_id": "context", "messages": normalized}],
            "questions": questions,
            "evidence_text": {},
        }


def scriptmem() -> Iterable[dict[str, Any]]:
    conversations = {}
    for line in (RAW / "scriptmem" / "data" / "public" / "conversations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        conversations[item["conversation_id"]] = item
    questions: dict[str, list[dict]] = {}
    for line in (RAW / "scriptmem" / "data" / "public" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        options = item.get("option")
        if isinstance(options, str):
            options = parse_options(options)
        questions.setdefault(item["conversation_id"], []).append(
            {
                "qid": item["qa_id"],
                "question": item["question"],
                "answer": item.get("answer_letters") or item.get("answer"),
                "options": options,
                "evidence_msg_ids": None,
                "evidence_session_ids": None,
                "category": item.get("qa_type"),
            }
        )
    for conversation_id, item in conversations.items():
        conversation = item.get("conversation", {})
        speakers = conversation.get("format_example", {}).get("speakers", ["Speaker A", "Speaker B"])
        sessions = []
        index = 1
        while f"session_{index}" in conversation:
            session = conversation[f"session_{index}"]
            timestamp = parse_iso_datetime(conversation.get(f"session_{index}_date_time", ""))
            messages = []
            for turn in session:
                text = turn.get("text") or ""
                if not text.strip():
                    continue
                role = "user" if turn.get("speaker") == speakers[0] else "assistant"
                messages.append(
                    {
                        "role": role,
                        "content": text,
                        "timestamp": timestamp,
                        "msg_id": turn.get("dia_id") or f"{conversation_id}:{index}:{len(messages)}",
                    }
                )
            if messages:
                sessions.append({"session_id": f"session_{index}", "messages": messages})
            index += 1
        if sessions:
            yield {
                "dataset": "scriptmem",
                "sample_id": conversation_id,
                "sessions": sessions,
                "questions": questions.get(conversation_id, []),
                "evidence_text": {},
            }


def write(name: str, records: Iterable[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"{name}: {count} samples -> {path}")


if __name__ == "__main__":
    write("locomo", locomo())
    write("longmemeval_oracle", longmemeval(RAW / "longmemeval" / "data" / "longmemeval_oracle.json"))
    write("personamem_32k", personamem())
    write("scriptmem", scriptmem())
