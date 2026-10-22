"""Offline evaluation harness.

Runs the real MemoryService against normalized public datasets and records
retrieval evidence metrics per question.

Usage:
    python eval/harness.py --datasets locomo longmemeval_oracle personamem_32k \
        --dims 1024 --out data/eval/run_1024.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml_memory.config import Settings  # noqa: E402
from aml_memory.db import Database  # noqa: E402
from aml_memory.embedding import EmbeddingClient  # noqa: E402
from aml_memory.models import AddRequest, Message, SearchRequest  # noqa: E402
from aml_memory.service import MemoryService  # noqa: E402
from answer_judge import (  # noqa: E402
    answer_multiple_choice,
    answer_question,
    judge_answer,
    judge_multiple_choice,
)
from llm import LLMClient  # noqa: E402

NORMALIZED = ROOT / "data" / "normalized"


def load_records(name: str) -> list[dict[str, Any]]:
    path = NORMALIZED / f"{name}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chunk_messages(messages: list[dict], max_messages: int = 20, max_words: int = 2000) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    words = 0
    for message in messages:
        message_words = len(str(message["content"]).split())
        if current and (len(current) >= max_messages or words + message_words > max_words):
            chunks.append(current)
            current = []
            words = 0
        current.append(message)
        words += message_words
    if current:
        chunks.append(current)
    return chunks


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def build_add_request(
    run_id: str,
    sample_id: str,
    session: dict,
    messages: list[dict],
    chunk_index: int,
) -> AddRequest:
    return AddRequest(
        request_id=f"eval:{run_id}:{sample_id}:{session['session_id']}:c{chunk_index}",
        user_id=f"eval:{run_id}:{sample_id}",
        session_id=f"eval:{run_id}:{sample_id}:{session['session_id']}",
        messages=[
            Message(
                role=(
                    message["role"]
                    if message.get("role") in {"user", "assistant"}
                    else "user"
                ),
                content=message["content"],
                timestamp=message.get("timestamp"),
            )
            for message in messages
        ],
    )


async def evaluate_sample(
    service: MemoryService,
    db: Database,
    run_id: str,
    dataset: str,
    sample: dict,
    max_questions: int | None,
    progress: list[int],
    done_qids: set[str] | None = None,
    qid_filter: set[str] | None = None,
    top_k: int = 100,
    answer_llm: LLMClient | None = None,
    judge_llm: LLMClient | None = None,
    max_context_chars: int = 80000,
) -> list[dict[str, Any]]:
    user_id = f"eval:{run_id}:{sample['sample_id']}"
    for session in sample["sessions"]:
        for chunk_index, chunk in enumerate(chunk_messages(session["messages"])):
            await service.add(
                build_add_request(run_id, sample["sample_id"], session, chunk, chunk_index)
            )

    evidence_text = sample.get("evidence_text") or {}
    chunk_sessions: dict[str, str] = {}
    if any(question.get("evidence_session_ids") for question in sample["questions"]):
        rows = await db.pool.fetch(
            "SELECT id, session_id FROM chunks WHERE user_id = $1", user_id
        )
        prefix = f"eval:{run_id}:{sample['sample_id']}:"
        chunk_sessions = {
            str(row["id"]): (
                row["session_id"][len(prefix):]
                if row["session_id"].startswith(prefix)
                else row["session_id"]
            )
            for row in rows
        }

    results: list[dict[str, Any]] = []
    questions = sample["questions"]
    if done_qids:
        questions = [question for question in questions if question["qid"] not in done_qids]
    if qid_filter is not None:
        questions = [question for question in questions if question["qid"] in qid_filter]
    if max_questions is not None:
        questions = questions[:max_questions]

    for question in questions:
        started = time.perf_counter()
        response = await service.search(
            SearchRequest(
                query=question["question"],
                user_id=user_id,
                top_k=top_k,
            )
        )
        latency_ms = (time.perf_counter() - started) * 1000
        data = response["data"]
        contents = [normalize_text(item["content"]) for item in data]

        evidence_ids = question.get("evidence_msg_ids") or []
        evidence_sessions = question.get("evidence_session_ids") or []
        covered = 0
        first_hit_rank = 0

        if evidence_ids:
            for rank, content in enumerate(contents, start=1):
                hit = False
                for msg_id in evidence_ids:
                    needle = normalize_text(evidence_text.get(msg_id, ""))
                    if needle and needle in content:
                        hit = True
                        break
                if hit and first_hit_rank == 0:
                    first_hit_rank = rank
            for msg_id in evidence_ids:
                needle = normalize_text(evidence_text.get(msg_id, ""))
                if needle and any(needle in content for content in contents):
                    covered += 1
            total = len(evidence_ids)
        elif evidence_sessions:
            returned_sessions = [
                chunk_sessions.get(str(item["id"])) for item in data
            ]
            for rank, session_id in enumerate(returned_sessions, start=1):
                if session_id in evidence_sessions and first_hit_rank == 0:
                    first_hit_rank = rank
            covered = len({session_id for session_id in returned_sessions if session_id in evidence_sessions})
            total = len(set(evidence_sessions))
        else:
            total = 0

        generated_answer = None
        judge_label = None
        judge_response = None
        if answer_llm is not None:
            memories = "\n\n".join(item["content"] for item in data)[:max_context_chars]
            options = question.get("options") or []
            if options:
                generated_answer = await answer_multiple_choice(
                    answer_llm, question["question"], options, memories
                )
            else:
                generated_answer = await answer_question(
                    answer_llm, question["question"], memories
                )
        if generated_answer is not None:
            options = question.get("options") or []
            if options:
                judge_label, judge_response = judge_multiple_choice(
                    question.get("answer", ""), generated_answer
                )
            elif judge_llm is not None:
                judge_label, judge_response = await judge_answer(
                    judge_llm,
                    question["question"],
                    str(question.get("answer", "")),
                    generated_answer,
                )

        results.append(
            {
                "dataset": dataset,
                "sample_id": sample["sample_id"],
                "qid": question["qid"],
                "question": question["question"],
                "category": question.get("category"),
                "n_results": len(data),
                "evidence_total": total,
                "evidence_covered": covered,
                "evidence_recall": (covered / total) if total else None,
                "full_recall": (1.0 if covered == total else 0.0) if total else None,
                "mrr": (1.0 / first_hit_rank) if first_hit_rank else (0.0 if total else None),
                "zero_recall": (1.0 if covered == 0 else 0.0) if total else None,
                "latency_ms": latency_ms,
                "generated_answer": generated_answer,
                "judge_label": judge_label,
                "judge_correct": (
                    1.0 if judge_label == "CORRECT" else 0.0
                ) if judge_label else None,
            }
        )
        progress[0] += 1
        if progress[0] % 100 == 0:
            print(f"  questions done: {progress[0]}")
    return results


async def run(
    datasets: list[str],
    dims: int,
    output: Path,
    run_id: str,
    max_questions: int | None,
    answer_model: str | None = None,
    judge_model: str | None = None,
    max_context_chars: int = 80000,
    llm_concurrency: int = 8,
    resume: bool = False,
    sample_concurrency: int = 4,
    top_k: int = 100,
    qid_filter: set[str] | None = None,
    reuse_db: bool = False,
    window_date_header: bool = False,
    resolve_relative_dates: bool = False,
) -> None:
    settings = Settings().model_copy(
        update={
            "embedding_dims": dims,
            "embedding_concurrency": 8,
            "window_date_header": window_date_header,
            "resolve_relative_dates": resolve_relative_dates,
        }
    )
    database = Database(settings)
    await database.connect()
    await database.init_schema()

    done_qids: set[str] = set()
    mode = "w"
    if resume and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done_qids.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    continue
        mode = "a"
        print(f"resuming with {len(done_qids)} completed questions")
    elif not reuse_db:
        async with database.pool.acquire() as conn:
            await conn.execute("TRUNCATE chunks, add_requests, sessions RESTART IDENTITY")

    embedder = EmbeddingClient(settings)
    await embedder.start()
    answer_llm = (
        LLMClient(
            settings.embedding_api_key,
            settings.embedding_base_url,
            answer_model,
            concurrency=llm_concurrency,
        )
        if answer_model
        else None
    )
    judge_llm = (
        LLMClient(
            settings.embedding_api_key,
            settings.embedding_base_url,
            judge_model,
            concurrency=llm_concurrency,
        )
        if judge_model
        else None
    )
    service = MemoryService(database, embedder, settings)
    progress = [0]
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output.open(mode, encoding="utf-8") as handle:
            for dataset in datasets:
                records = load_records(dataset)
                print(f"[{dataset}] samples={len(records)} dims={dims}")
                semaphore = asyncio.Semaphore(max(1, sample_concurrency))

                async def process_sample(sample: dict) -> None:
                    async with semaphore:
                        results = await evaluate_sample(
                            service,
                            database,
                            run_id,
                            dataset,
                            sample,
                            max_questions,
                            progress,
                            done_qids=done_qids,
                            qid_filter=qid_filter,
                            top_k=top_k,
                            answer_llm=answer_llm,
                            judge_llm=judge_llm,
                            max_context_chars=max_context_chars,
                        )
                        for result in results:
                            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()

                await asyncio.gather(*(process_sample(sample) for sample in records))
    finally:
        await embedder.close()
        if answer_llm is not None:
            await answer_llm.close()
        if judge_llm is not None:
            await judge_llm.close()
        await database.close()

    print(f"wrote {output}")


def summarize(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    print(f"=== {path.name} ===")
    for dataset, items in by_dataset.items():
        scored = [item for item in items if item["evidence_recall"] is not None]
        judged = [item for item in items if item.get("judge_correct") is not None]
        accuracy_text = ""
        if judged:
            accuracy = sum(item["judge_correct"] for item in judged) / len(judged)
            accuracy_text = f" accuracy={accuracy:.4f}"
        if not scored:
            print(f"{dataset}: {len(items)} questions (no evidence labels){accuracy_text}")
            continue
        recall = sum(item["evidence_recall"] for item in scored) / len(scored)
        full = sum(item["full_recall"] for item in scored) / len(scored)
        mrr = sum(item["mrr"] for item in scored) / len(scored)
        zero = sum(item["zero_recall"] for item in scored) / len(scored)
        latency = sum(item["latency_ms"] for item in items) / len(items)
        print(
            f"{dataset}: n={len(scored)} recall={recall:.4f} full={full:.4f} "
            f"mrr={mrr:.4f} zero={zero:.4f} latency={latency:.0f}ms{accuracy_text}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--max-context-chars", type=int, default=80000)
    parser.add_argument("--llm-concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-concurrency", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--qid-file", default=None)
    parser.add_argument("--reuse-db", action="store_true")
    parser.add_argument("--window-date-header", action="store_true")
    parser.add_argument("--resolve-relative-dates", action="store_true")
    args = parser.parse_args()
    run_id = args.run_id or f"d{args.dims}"
    qid_filter = None
    if args.qid_file:
        qid_filter = {
            line.strip()
            for line in Path(args.qid_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    asyncio.run(
        run(
            args.datasets,
            args.dims,
            Path(args.out),
            run_id,
            args.max_questions,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            max_context_chars=args.max_context_chars,
            llm_concurrency=args.llm_concurrency,
            resume=args.resume,
            sample_concurrency=args.sample_concurrency,
            top_k=args.top_k,
            qid_filter=qid_filter,
            reuse_db=args.reuse_db,
            window_date_header=args.window_date_header,
            resolve_relative_dates=args.resolve_relative_dates,
        )
    )
    summarize(Path(args.out))
