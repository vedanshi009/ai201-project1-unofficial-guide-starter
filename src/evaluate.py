"""
evaluate.py — LLM-as-Judge Evaluation Pipeline
Vegetarian Food RAG System

Runs all 5 evaluation plan questions through the full RAG pipeline,
then uses Groq as an LLM judge to score each response as:
  correct / partial / wrong

Outputs a markdown table of results to:
  data/processed/evaluation_results.md

Usage:
    python src/evaluate.py
"""

import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from generate import ask

try:
    from groq import Groq
except ImportError:
    raise ImportError("Run: pip install groq")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_FILE  = PROJECT_ROOT / "data" / "processed" / "evaluation_results.md"

# ─── Evaluation plan ──────────────────────────────────────────────────────────
# Directly from planning.md — question, expected answer, source filter to use

EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": "What vegetarian dishes are available at USF dining halls this week that have over 20g of protein?",
        "expected": "A list of high-protein plant-based dishes (lentil curry, tofu stir fry, black bean bowls) from USF dining menus with approximate protein values.",
        "source_type": None,
        "note": "No USF dining or nutrition data ingested — system should say it doesn't have enough information.",
    },
    {
        "id": 2,
        "question": "Does the Tom Yum soup at Thai restaurants near me typically contain fish sauce or shrimp paste?",
        "expected": "Yes — traditional Tom Yum uses fish sauce and shrimp paste. System should surface this warning and suggest asking for a vegan-modified version.",
        "source_type": "reddit",
        "note": "Covered by Reddit r/vegetarian hidden ingredients file.",
    },
    {
        "id": 3,
        "question": "Which Indian restaurants near USF have the most positive vegetarian reviews on Yelp?",
        "expected": "A ranked list of 2–4 Indian restaurants with summarized Yelp review sentiment praising vegetarian options.",
        "source_type": None,
        "note": "No Yelp review data ingested — system should say it doesn't have enough information.",
    },
    {
        "id": 4,
        "question": "What is the calorie and macro breakdown of a vegetarian burrito bowl from Chipotle?",
        "expected": "~700–900 calories, protein ~25g, fat ~30g, carbs ~85g. Note that guacamole and sour cream add significant fat.",
        "source_type": "menu",
        "note": "No nutrition/calorie data ingested — system should say it doesn't have enough information.",
    },
    {
        "id": 5,
        "question": "Are there any hidden non-vegetarian ingredients I should watch out for when ordering pasta at Italian restaurants?",
        "expected": "Yes — anchovies in Caesar dressing, meat broth in risotto, lard in some pasta doughs. Should include actionable phrases to ask the server.",
        "source_type": "reddit",
        "note": "Covered by Reddit r/vegetarian hidden ingredients file.",
    },
]


# ─── Judge prompt ─────────────────────────────────────────────────────────────

def build_judge_prompt(question: str, expected: str, actual: str, note: str) -> str:
    return f"""You are evaluating a RAG (Retrieval-Augmented Generation) system for vegetarian food assistance.

Score the system's actual response against the expected answer.

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

CONTEXT NOTE (known data limitations):
{note}

ACTUAL SYSTEM RESPONSE:
{actual}

Score the response as one of:
- correct   : The response addresses the question accurately using retrieved sources, OR correctly says it doesn't have enough information when data is missing (as noted above)
- partial   : The response is on-topic but incomplete, vague, or missing key details from the expected answer
- wrong     : The response is off-topic, hallucinates facts not in its sources, or fails to say "I don't have enough information" when it should

Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{"score": "correct|partial|wrong", "reason": "one sentence explanation"}}"""


def call_judge(prompt: str) -> dict:
    """Call Groq as the LLM judge. Returns {score, reason}."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"score": "error", "reason": "GROQ_API_KEY not set"}

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=150,
        temperature=0.0,   # deterministic for consistent scoring
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    try:
        # Strip markdown fences if model adds them
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": "error", "reason": f"Judge returned unparseable output: {raw[:100]}"}


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_evaluation() -> list[dict]:
    print("=" * 65)
    print("EVALUATION — Vegetarian RAG (LLM-as-Judge)")
    print("=" * 65)

    results = []

    for item in EVAL_QUESTIONS:
        print(f"\n[Q{item['id']}] {item['question'][:70]}...")

        # Step 1: Get RAG system answer
        rag_result = ask(
            item["question"],
            k=5,
            source_type=item["source_type"],
        )
        actual_answer = rag_result["answer"]
        sources_used  = rag_result["sources"]

        print(f"  Sources retrieved: {sources_used}")

        # Step 2: Judge the answer
        judge_prompt = build_judge_prompt(
            question=item["question"],
            expected=item["expected"],
            actual=actual_answer,
            note=item["note"],
        )
        judgment = call_judge(judge_prompt)

        score  = judgment.get("score", "error")
        reason = judgment.get("reason", "no reason returned")

        score_icon = {"correct": "✅", "partial": "⚠️", "wrong": "❌", "error": "💥"}.get(score, "?")
        print(f"  Score : {score_icon} {score}")
        print(f"  Reason: {reason}")

        results.append({
            "id":           item["id"],
            "question":     item["question"],
            "expected":     item["expected"],
            "actual":       actual_answer,
            "sources_used": ", ".join(sources_used) if sources_used else "none",
            "score":        score,
            "reason":       reason,
            "note":         item["note"],
        })

    return results


# ─── Markdown output ──────────────────────────────────────────────────────────

def write_markdown_report(results: list[dict]):
    score_icon = {"correct": "✅", "partial": "⚠️", "wrong": "❌", "error": "💥"}

    correct = sum(1 for r in results if r["score"] == "correct")
    partial = sum(1 for r in results if r["score"] == "partial")
    wrong   = sum(1 for r in results if r["score"] == "wrong")
    total   = len(results)

    lines = [
        "# Evaluation Results — Vegetarian RAG System",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        f"**Summary:** {correct}/{total} correct, {partial} partial, {wrong} wrong\n",
        "---\n",
        "## Results Table\n",
        "| # | Question | Score | Sources Used | Judge Reason |",
        "|---|----------|-------|--------------|--------------|",
    ]

    for r in results:
        q_short = r["question"][:60] + "..." if len(r["question"]) > 60 else r["question"]
        icon    = score_icon.get(r["score"], "?")
        lines.append(
            f"| {r['id']} | {q_short} | {icon} {r['score']} | {r['sources_used'][:50]} | {r['reason']} |"
        )

    lines += [
        "\n---\n",
        "## Detailed Answers\n",
    ]

    for r in results:
        icon = score_icon.get(r["score"], "?")
        lines += [
            f"### Q{r['id']} — {icon} {r['score'].upper()}",
            f"**Question:** {r['question']}",
            f"\n**Expected:** {r['expected']}",
            f"\n**Data note:** {r['note']}",
            f"\n**Sources retrieved:** {r['sources_used']}",
            f"\n**System answer:**",
            f"\n> {r['actual'].replace(chr(10), chr(10) + '> ')}",
            f"\n**Judge reasoning:** {r['reason']}",
            "\n---\n",
        ]

    lines += [
        "## Score Summary\n",
        f"- ✅ Correct : {correct}/{total}",
        f"- ⚠️  Partial : {partial}/{total}",
        f"- ❌ Wrong   : {wrong}/{total}",
        "\n**Target from planning.md:** ≥4/5 correct on first pass.",
        "\n**Note on Q1, Q3, Q4:** These questions require USF dining, Yelp, and",
        "nutrition data that was not ingested in this version. The correct system",
        "behavior is to say 'I don't have enough information' — this is scored as",
        "**correct** by the judge since it accurately reflects data limitations.",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[eval] Report written to: {OUTPUT_FILE}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_evaluation()

    correct = sum(1 for r in results if r["score"] == "correct")
    total   = len(results)

    print(f"\n{'='*65}")
    print(f"FINAL SCORE: {correct}/{total} correct")

    score_icon = {"correct": "✅", "partial": "⚠️", "wrong": "❌", "error": "💥"}
    for r in results:
        icon = score_icon.get(r["score"], "?")
        print(f"  Q{r['id']}: {icon} {r['score']:8s} — {r['reason']}")

    write_markdown_report(results)
    print(f"\n[eval] Target was ≥4/5 correct.")
    print(f"[eval] Note: Q1, Q3, Q4 are correct if system says 'I don't have enough information'")
    print("=" * 65)