"""
generate.py — RAG Generation Pipeline
Vegetarian Food RAG System

Takes a user question, retrieves top-k chunks from ChromaDB,
and passes them to an LLM with a grounding prompt that forces
the model to answer from context only — not general training knowledge.

LLM backend: Groq (llama-3.1-8b-instant)
  - Free tier, no credit card needed
  - Much faster than OpenAI/Anthropic (~200ms responses)
  - Install: pip install groq
  - Get key: https://console.groq.com → API Keys
  - Set env var: GROQ_API_KEY=your_key

Usage:
    python src/generate.py
    or import ask() for use in the Gradio interface
"""

import os
from pathlib import Path
from query import retrieve

# Load .env file from project root automatically —
# this reads GROQ_API_KEY=... from your .env file into os.environ
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed, fall back to system env vars

try:
    from groq import Groq
    LLM_BACKEND = "groq"
except ImportError:
    LLM_BACKEND = None

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the grounded RAG prompt.

    Critical design: the instruction 'answer ONLY from the context below'
    prevents the LLM from hallucinating menu items or ingredient facts
    that aren't in your documents. Without this, Claude will confidently
    invent Chipotle calorie counts or Tom Yum ingredients from training data.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["restaurant"] or chunk["source_name"]
        context_blocks.append(
            f"[Source {i}: {source} ({chunk['source_type']})]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a vegetarian food assistant helping users — especially international students — find safe, vegetarian-friendly meals at restaurants.

Answer the user's question using ONLY the context provided below. Do not use your general training knowledge about restaurants, menus, or ingredients. If the context does not contain enough information to answer the question, say exactly: "I don't have enough information in my sources to answer this confidently."

If the context contains information about hidden non-vegetarian ingredients (fish sauce, meat broth, gelatin, anchovies, etc.), always surface that warning even if the user didn't ask about it directly.

Always end your answer by citing which sources you used, e.g. "Sources: Shalimar Indian Cuisine menu, r/vegetarian"

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return prompt


# ─── Generation function ──────────────────────────────────────────────────────

def ask(
    question: str,
    k: int = 5,
    source_type: str = None,
) -> dict:
    """
    Full RAG pipeline: retrieve → prompt → generate → return.

    Args:
        question    : user's natural language question
        k           : number of chunks to retrieve
        source_type : optional filter ("menu" or "reddit")

    Returns dict with:
        "answer"  : LLM-generated grounded answer
        "chunks"  : retrieved chunks used as context
        "prompt"  : the full prompt sent to the LLM (useful for debugging)
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question, k=k, source_type=source_type)

    if not chunks:
        return {
            "answer": "No relevant documents found in the knowledge base.",
            "chunks": [],
            "prompt": "",
        }

    # Step 2: Build grounded prompt
    prompt = build_prompt(question, chunks)

    # Step 3: Generate answer
    if LLM_BACKEND == "groq":
        answer = _call_groq(prompt)
    else:
        answer = (
            "[LLM not configured]\n\n"
            "Install groq: pip install groq\n"
            "Set env var: $env:GROQ_API_KEY='your_key'  (PowerShell)\n\n"
            "Retrieved context that would be sent to LLM:\n\n"
            + "\n\n".join(c["text"][:200] for c in chunks)
        )

    # Step 4: Programmatic source attribution
    # Appended after the LLM answer regardless of whether the model cited them.
    # This guarantees attribution is always present and traceable.
    unique_sources = []
    seen = set()
    for c in chunks:
        label = c["restaurant"] or c["source_name"]
        key   = (label, c["source_type"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(f"{label} ({c['source_type']})")

    attribution = "\n\n---\nSources consulted: " + ", ".join(unique_sources)

    # Only append if the model didn't already end with a sources line
    if "sources:" not in answer.lower()[-120:]:
        answer = answer + attribution

    return {
        "answer":  answer,
        "chunks":  chunks,
        "prompt":  prompt,
        # "sources" list for app.py and callers that want clean attribution
        "sources": unique_sources,
    }


def _call_groq(prompt: str) -> str:
    """
    Call Groq API with llama-3.1-8b-instant.
    Fast, free-tier friendly, accurate enough for RAG generation.
    Upgrade to llama-3.3-70b-versatile for stronger reasoning if needed.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "GROQ_API_KEY environment variable not set.\n"
            "Get a free key at: https://console.groq.com\n"
            "Then run: $env:GROQ_API_KEY='your_key_here'  (PowerShell)"
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=512,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ─── CLI test ─────────────────────────────────────────────────────────────────

def run_generation_test():
    """
    End-to-end grounding test on 4 queries:
      - 3 queries your data covers  → should cite specific sources
      - 1 query your data does NOT cover → must say "I don't have enough information"

    For each answer ask: could this have come from anywhere other than the
    retrieved chunks? If yes, the grounding instruction needs tightening.
    """
    test_cases = [
        # Covered by Reddit data
        ("Does Tom Yum soup contain fish sauce or shrimp paste?",         5, "reddit"),
        # Covered by menu data
        ("What vegetarian options does Chipotle have?",                   5, "menu"),
        # Covered by Reddit hidden ingredients data
        ("What hidden non-vegetarian ingredients should I watch for?",    5, "reddit"),
        # NOT covered — no USF dining, nutrition, or calorie data ingested
        # System must say it doesn't know, not hallucinate numbers
        ("What is the calorie count of a vegetarian burrito bowl?",       5, None),
    ]

    print("=" * 65)
    print("GENERATION TEST — Vegetarian RAG (Full Pipeline)")
    print("Grounding check: answers must be traceable to retrieved chunks")
    print("=" * 65)

    for question, k, source_type in test_cases:
        print(f"\n{'─'*65}")
        print(f"Q: {question}")
        print(f"   filter={source_type or 'none'}  k={k}")
        print()

        result = ask(question, k=k, source_type=source_type)

        print("RETRIEVED SOURCES:")
        for i, chunk in enumerate(result["chunks"], 1):
            src = chunk["restaurant"] or chunk["source_name"]
            print(f"  [{i}] score={chunk['score']:.4f}  {src}  ({chunk['source_type']})")
        print()
        print("ANSWER:")
        print(result["answer"])
        print()
        # Grounding self-check prompt for the developer
        print("✔ GROUNDING CHECK: Is every claim above traceable to a retrieved chunk?")
        print("  If the answer mentions facts not in the sources above → grounding failure.")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    run_generation_test()