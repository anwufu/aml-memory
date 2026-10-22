"""Compare two judge models on the same (question, gold, generated) triples.

Usage:
    python eval/judge_agreement.py --results data/eval/judge_sample.jsonl \
        --a qwen3.8-max --b deepseek-v4.1-flash-1017
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from aml_memory.config import Settings  # noqa: E402
from answer_judge import judge_answer  # noqa: E402
from llm import LLMClient  # noqa: E402

NORMALIZED = ROOT / "data" / "normalized"


def load_gold() -> dict[str, str]:
    gold: dict[str, str] = {}
    for path in NORMALIZED.glob("locomo.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            for question in sample["questions"]:
                gold[question["qid"]] = str(question.get("answer", ""))
    return gold


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    settings = Settings()
    rows = [
        json.loads(line)
        for line in Path(args.results).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold = load_gold()

    client_a = LLMClient(
        settings.embedding_api_key,
        settings.embedding_base_url,
        args.a,
        concurrency=args.concurrency,
    )
    client_b = LLMClient(
        settings.embedding_api_key,
        settings.embedding_base_url,
        args.b,
        concurrency=args.concurrency,
    )

    try:
        labels_a = await asyncio.gather(
            *[
                judge_answer(
                    client_a,
                    row["question"],
                    gold.get(row["qid"], ""),
                    row["generated_answer"],
                )
                for row in rows
                if row.get("generated_answer")
            ]
        )
        labels_b = await asyncio.gather(
            *[
                judge_answer(
                    client_b,
                    row["question"],
                    gold.get(row["qid"], ""),
                    row["generated_answer"],
                )
                for row in rows
                if row.get("generated_answer")
            ]
        )
    finally:
        await client_a.close()
        await client_b.close()

    pairs = [(a[0], b[0]) for a, b in zip(labels_a, labels_b)]
    n = len(pairs)
    agreement = sum(1 for a, b in pairs if a == b) / n if n else 0.0
    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    expected = sum(
        (count_a[label] / n) * (count_b[label] / n)
        for label in {"CORRECT", "WRONG"}
    ) if n else 0.0
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else 1.0

    print(f"judge A={args.a}  judge B={args.b}  n={n}")
    print(f"agreement={agreement:.4f}  cohen_kappa={kappa:.4f}")
    print(f"A labels={dict(count_a)}  B labels={dict(count_b)}")
    if n:
        print(
            f"A accuracy={count_a.get('CORRECT', 0) / n:.4f}  "
            f"B accuracy={count_b.get('CORRECT', 0) / n:.4f}"
        )
    print("confusion (A rows, B cols):")
    for a_label in ("CORRECT", "WRONG"):
        row = [sum(1 for a, b in pairs if a == a_label and b == b_label) for b_label in ("CORRECT", "WRONG")]
        print(f"  {a_label:7s} CORRECT={row[0]:4d}  WRONG={row[1]:4d}")
    disagreements = [
        (row, a, b)
        for row, (a, b) in zip([r for r in rows if r.get("generated_answer")], pairs)
        if a != b
    ]
    for row, a, b in disagreements[:10]:
        print("\nDISAGREE", row["qid"])
        print("  Q:", row["question"][:160])
        print("  gold:", gold.get(row["qid"], "")[:120])
        print("  gen :", str(row["generated_answer"])[:160])
        print(f"  A={a}  B={b}")


if __name__ == "__main__":
    asyncio.run(main())
