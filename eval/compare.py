"""Paired comparison for two evaluation runs (e.g. 1024 vs 2048 dims).

Usage:
    python eval/compare.py --a data/eval/retrieval_1024.jsonl \
        --b data/eval/retrieval_2048.jsonl

For binary metrics (judge_correct, full_recall, zero_recall) it reports
McNemar's exact test on discordant pairs. For continuous metrics (MRR,
evidence_recall) it reports a paired bootstrap confidence interval.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["qid"]] = row
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Two-sided exact p = 2 * P(X <= k | n, 0.5), capped at 1.
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(diffs: np.ndarray, iterations: int = 4000) -> tuple[float, float]:
    if len(diffs) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(20261021)
    means = []
    for _ in range(iterations):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        means.append(float(sample.mean()))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def compare(a_path: Path, b_path: Path) -> None:
    a = load(a_path)
    b = load(b_path)
    shared = sorted(set(a) & set(b))
    print(f"a={a_path.name} n={len(a)}  b={b_path.name} n={len(b)}  paired={len(shared)}")

    by_dataset: dict[str, list[str]] = defaultdict(list)
    for qid in shared:
        by_dataset[a[qid]["dataset"]].append(qid)

    for dataset, qids in by_dataset.items():
        print(f"\n=== {dataset} (n={len(qids)}) ===")
        for metric in ["judge_correct", "full_recall", "zero_recall", "evidence_recall", "mrr", "latency_ms"]:
            pairs = [
                (a[qid].get(metric), b[qid].get(metric))
                for qid in qids
                if a[qid].get(metric) is not None and b[qid].get(metric) is not None
            ]
            if not pairs:
                continue
            av = np.array([p[0] for p in pairs], dtype=float)
            bv = np.array([p[1] for p in pairs], dtype=float)
            diff = bv - av
            mean_a = float(av.mean())
            mean_b = float(bv.mean())
            if metric in {"judge_correct", "full_recall", "zero_recall"}:
                b_wins = int(np.sum((bv == 1) & (av == 0)))
                a_wins = int(np.sum((av == 1) & (bv == 0)))
                p_value = mcnemar_exact(b_wins, a_wins)
                print(
                    f"{metric:16s} a={mean_a:.4f} b={mean_b:.4f} diff={mean_b-mean_a:+.4f} "
                    f"b_wins={b_wins} a_wins={a_wins} p={p_value:.4g}"
                )
            else:
                low, high = bootstrap_ci(diff)
                print(
                    f"{metric:16s} a={mean_a:.4f} b={mean_b:.4f} diff={mean_b-mean_a:+.4f} "
                    f"95%CI=[{low:+.4f},{high:+.4f}]"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    args = parser.parse_args()
    compare(Path(args.a), Path(args.b))
