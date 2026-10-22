"""Official LoCoMo answer/judge templates used as the offline proxy.

Templates are taken from the public AML evaluation pipeline
(data/locomo-refined/pipeline.py at the repository commit referenced in README).
"""
from __future__ import annotations

import json
import re

from llm import LLMClient

OPEN_ENDED_ANSWER_TEMPLATE = """You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Use only the provided memories. Prefer the memory that answers the question most directly.
2. Your memories are episodic raw observations. Reason about what they imply. Do not refuse just because the answer is not stated verbatim.
3. The question may contain typos. Match it to the most relevant memory even if the wording differs.
4. When multiple answers are possible, list all supported answers, not just the first.
5. For counts or time intervals, enumerate carefully before answering.
6. Preserve specific names, titles, places, and labels from the memories. Use "Rob" not "a colleague", "Sweden" not "home country".
7. Convert relative times like "yesterday", "last month", and "last year" into dates, months, or years when the memory timestamp makes it clear. Keep week-based expressions relative.
8. If memories conflict, prefer the most recent supported memory.
9. For list questions, include all required items and no extras.
10. Keep the final answer minimal. Do not add explanation, background, or extra dates unless needed for correctness.
</instructions>

<memories>
{memories}
</memories>

Question: {question}
Answer with the shortest correct phrase or sentence. No preamble, no fluff:"""


ACCURACY_PROMPT = """Your task is to label an answer as 'CORRECT' or 'WRONG' given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold's key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold's content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps.
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.

Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts, the generated answer must cover all of them.
- Extra non-contradictory items generally count as WRONG.
- Exception: if a gold element is elaborated into finer details, it is still CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include any one of them without contradiction.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Just return the label CORRECT or WRONG in a json format with the key as "label":
{{"label": "CORRECT" or "WRONG"}}"""


MC_ANSWER_TEMPLATE = """You are asked to answer a multiple-choice question based on your memories of a conversation.

<instructions>
1. Use only the provided memories.
2. The question may be single-choice, multi-select, or ordering. Obey the format stated in the question.
3. Do not add unsupported options. Do not omit supported options. Do not hedge.
4. Preserve option letters exactly.
5. If memories conflict, prefer the most recent supported memory.
6. Keep reasoning internal. The visible output must be only the option letter(s).
</instructions>

<memories>
{memories}
</memories>

Question: {question}

Options:
{options}

Return only the option letter(s), e.g. "B" or "A, C"."""


def _letters(value: object) -> list[str]:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return [letter.upper() for letter in re.findall(r"[A-Fa-f]", text)]


async def answer_multiple_choice(
    llm: LLMClient,
    question: str,
    options: list[str],
    memories: str,
    max_tokens: int = 512,
) -> str:
    prompt = MC_ANSWER_TEMPLATE.format(
        memories=memories or "(no memories)",
        question=question,
        options="\n".join(options),
    )
    return await llm.chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )


def judge_multiple_choice(gold: object, generated: str) -> tuple[str, str]:
    gold_letters = sorted(set(_letters(gold)))
    predicted = sorted(set(_letters(generated)))
    correct = bool(gold_letters) and gold_letters == predicted
    return ("CORRECT" if correct else "WRONG"), json.dumps(
        {"gold": gold_letters, "predicted": predicted}, ensure_ascii=False
    )


async def answer_question(
    llm: LLMClient,
    question: str,
    memories: str,
    max_tokens: int = 512,
) -> str:
    prompt = OPEN_ENDED_ANSWER_TEMPLATE.format(
        memories=memories or "(no memories)",
        question=question,
    )
    return await llm.chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )


async def judge_answer(
    llm: LLMClient,
    question: str,
    gold: str,
    generated: str,
    max_tokens: int = 512,
) -> tuple[str, str]:
    prompt = ACCURACY_PROMPT.format(
        question=question,
        gold_answer=gold,
        generated_answer=generated,
    )
    response = await llm.chat(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if not match:
        return "WRONG", response
    try:
        payload = json.loads(match.group(0))
        label = str(payload.get("label", "")).upper()
    except json.JSONDecodeError:
        return "WRONG", response
    return ("CORRECT" if label == "CORRECT" else "WRONG"), response
