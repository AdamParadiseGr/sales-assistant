#!/usr/bin/env python3
"""
Runs a suite of test dialogs through the agent and prints quality metrics.

Usage:
    python scripts/run_evals.py
    python scripts/run_evals.py --no-reset     # keep previous log entries
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.WARNING,   # suppress verbose agent output during evals
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LOG_PATH = ROOT / "logs" / "evaluation_log.jsonl"
EVAL_SESSION_PREFIX = "eval_"

# ---------------------------------------------------------------------------
# Test dialog suite
# ---------------------------------------------------------------------------

TEST_DIALOGS = [
    {
        "id": "eval_001",
        "description": "ИП открывает бизнес — полный цикл до заявки",
        "turns": [
            "Здравствуйте, хочу открыть счёт для бизнеса",
            "У меня ИП, розничный магазин одежды",
            "Оборот пока небольшой, тысяч 300 в месяц",
            "Хотел бы ещё принимать карты",
            "Хорошо, как подать заявку? Меня зовут Дмитрий, телефон 79161234567",
        ],
    },
    {
        "id": "eval_002",
        "description": "Вопрос об эквайринге — квалификация и расчёт",
        "turns": [
            "Сколько стоит эквайринг?",
            "У меня кофейня, оборот по картам примерно 400 тысяч в месяц",
            "Нужен терминал, есть ли аренда бесплатно?",
            "А что такое QR-оплата?",
        ],
    },
    {
        "id": "eval_003",
        "description": "Кредит на оборудование",
        "turns": [
            "Нам нужен кредит на покупку производственного оборудования",
            "ООО, работаем уже 3 года, оборот 2.5 миллиона в месяц",
            "Нужно около 8 миллионов рублей, хотели бы на 5 лет",
            "Какие документы нужны?",
        ],
    },
    {
        "id": "eval_004",
        "description": "Сравнение тарифов РКО",
        "turns": [
            "Чем отличается тариф Старт от тарифа Бизнес?",
            "У меня ООО, оборот около 1.2 миллиона в месяц",
        ],
    },
    {
        "id": "eval_005",
        "description": "Срок открытия счёта",
        "turns": [
            "Как быстро можно открыть расчётный счёт?",
            "Я ИП, все документы есть",
        ],
    },
    {
        "id": "eval_006",
        "description": "Интернет-магазин — полный пакет",
        "turns": [
            "У меня интернет-магазин, ищу банк для бизнеса",
            "ИП, продаём товары онлайн, оборот 1.5 млн в месяц",
            "Нужен счёт и приём оплаты на сайте",
            "Какова ставка интернет-эквайринга при нашем обороте?",
        ],
    },
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_dialog(agent, dialog: dict) -> list[dict]:
    results = []
    session_id = dialog["id"]
    for turn_text in dialog["turns"]:
        t0 = time.perf_counter()
        response = await agent.chat(session_id, turn_text)
        latency_ms = (time.perf_counter() - t0) * 1000
        results.append({
            "user": turn_text,
            "agent": response,
            "latency_ms": latency_ms,
        })
    return results


def wait_for_evals(session_ids: list[str], timeout: float = 30.0) -> list[dict]:
    """Poll evaluation_log.jsonl until all expected sessions appear."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        scores = _read_eval_scores(session_ids)
        if scores:
            return scores
        time.sleep(1)
    return _read_eval_scores(session_ids)


def _read_eval_scores(session_ids: list[str]) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    scores = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("session_id", "").startswith(EVAL_SESSION_PREFIX):
                scores.append(entry["scores"])
    return scores


def print_results(
    dialogs: list[dict],
    all_turns: list[dict],
    scores: list[dict],
    total_elapsed: float,
) -> None:
    total_turns = sum(len(d["turns"]) for d in dialogs)
    avg_latency = statistics.mean(t["latency_ms"] for d in all_turns for t in d) if all_turns else 0

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    if scores:
        relevance_vals = [s["relevance"] for s in scores]
        ground_vals = [s["groundedness"] for s in scores]
        sales_vals = [s["sales_effectiveness"] for s in scores]
        avg_all = [(s["relevance"] + s["groundedness"] + s["sales_effectiveness"]) / 3 for s in scores]

        high_conv = sum(1 for s in scores if s["sales_effectiveness"] >= 7)

        print(f"  Dialogs evaluated:       {len(dialogs)}")
        print(f"  Total turns:             {total_turns}")
        print(f"  Turns scored by judge:   {len(scores)}")
        print()
        print(f"  Avg Relevance:           {statistics.mean(relevance_vals):.2f} / 10")
        print(f"  Avg Groundedness:        {statistics.mean(ground_vals):.2f} / 10")
        print(f"  Avg Sales Effectiveness: {statistics.mean(sales_vals):.2f} / 10")
        print(f"  Avg Overall:             {statistics.mean(avg_all):.2f} / 10")
        print()
        print(f"  High-conversion turns:   {high_conv}/{len(scores)} ({100*high_conv/len(scores):.0f}%)")
        print()
        print(f"  Avg turn latency:        {avg_latency:.0f} ms")
        print(f"  Total wall time:         {total_elapsed:.1f} s")
    else:
        print("  No scores collected (check logs/evaluation_log.jsonl)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(reset_log: bool) -> None:
    from app.agent import SalesAgent

    if reset_log and LOG_PATH.exists():
        LOG_PATH.unlink()
        LOG_PATH.touch()
        print(f"Cleared {LOG_PATH}")

    print(f"Initialising agent...")
    agent = SalesAgent()

    all_results: list[list[dict]] = []
    t_start = time.perf_counter()

    for dialog in TEST_DIALOGS:
        print(f"\n[{dialog['id']}] {dialog['description']}")
        results = await run_dialog(agent, dialog)
        all_results.append(results)
        for i, r in enumerate(results, 1):
            print(f"  Turn {i}: {r['user'][:70]}")
            print(f"       → {r['agent'][:120]}  ({r['latency_ms']:.0f}ms)")

    total_elapsed = time.perf_counter() - t_start

    # Give the evaluator tasks time to finish
    print("\nWaiting for LLM judge to finish scoring...")
    await asyncio.sleep(15)

    session_ids = [d["id"] for d in TEST_DIALOGS]
    scores = wait_for_evals(session_ids, timeout=20)

    print_results(TEST_DIALOGS, all_results, scores, total_elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Keep existing entries in evaluation_log.jsonl",
    )
    args = parser.parse_args()
    asyncio.run(main(reset_log=not args.no_reset))
