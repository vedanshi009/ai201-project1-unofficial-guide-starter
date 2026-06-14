"""
inspect.py — Pipeline Inspection & Checkpoint
Vegetarian Food RAG System

Run this AFTER ingest.py and chunk_text.py to verify your output
meets the milestone checklist requirements:

  ✓ Print one cleaned document and read it
  ✓ Print 5 representative chunks with quality assessment
  ✓ Count total chunks and flag if out of range (50–2000)
  ✓ Check for leftover HTML, encoding artifacts, or bad content

Usage:
    python inspect.py
"""

import json
import re
import random
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR     = PROJECT_ROOT / "data"
DOCUMENTS_FILE = BASE_DIR / "processed" / "documents.jsonl"
CHUNKS_FILE    = BASE_DIR / "processed" / "chunks.jsonl"

# ─── Artifact detectors ───────────────────────────────────────────────────────

HTML_TAG_RE      = re.compile(r"<[a-zA-Z/][^>]{0,100}>")
HTML_ENTITY_RE   = re.compile(r"&[a-zA-Z]+;|&#\d+;")
NAV_BOILERPLATE  = re.compile(
    r"\b(cookie|privacy policy|terms of service|subscribe|newsletter|"
    r"sign up|log in|follow us|share this|read more|click here|"
    r"advertisement|sponsored)\b",
    re.IGNORECASE,
)

def check_artifacts(text: str) -> list[str]:
    """Return a list of warnings about leftover cleaning artifacts."""
    warnings = []
    if HTML_TAG_RE.search(text):
        warnings.append("⚠️  Contains HTML tags (e.g. <div>, <p>) — cleaning incomplete")
    if HTML_ENTITY_RE.search(text):
        warnings.append("⚠️  Contains HTML entities (e.g. &amp;, &nbsp;) — cleaning incomplete")
    nav_hits = NAV_BOILERPLATE.findall(text)
    if nav_hits:
        warnings.append(f"⚠️  Possible boilerplate: {set(nav_hits)}")
    if "\x00" in text or "\ufffd" in text:
        warnings.append("⚠️  Contains null bytes or replacement chars — encoding issue")
    return warnings


def assess_chunk_quality(text: str) -> str:
    """
    Return a one-line quality verdict for a chunk.
    Mirrors the milestone rubric: good / too small / too large / artifact.
    """
    # Strip the metadata header line before assessing content
    body = text.split("\n", 1)[1] if text.startswith("[") else text
    word_count = len(body.split())

    if word_count < 20:
        return "❌ TOO SMALL — fragment, no standalone meaning"
    if word_count > 400:
        return "❌ TOO LARGE — multiple unrelated topics, dilutes retrieval"
    if HTML_TAG_RE.search(body) or HTML_ENTITY_RE.search(body):
        return "❌ HTML ARTIFACT — cleaning didn't finish"
    if NAV_BOILERPLATE.search(body):
        return "⚠️  POSSIBLE BOILERPLATE — check for nav/cookie text"
    return "✅ GOOD — complete, retrievable thought"


# ─── Step 1: Print one cleaned document ──────────────────────────────────────

def inspect_one_document():
    print("\n" + "═" * 70)
    print("STEP 1 — One cleaned document (read this carefully)")
    print("═" * 70)

    if not DOCUMENTS_FILE.exists():
        print(f"[ERROR] {DOCUMENTS_FILE} not found. Run ingest.py first.")
        return

    docs = []
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    if not docs:
        print("[ERROR] No documents found in documents.jsonl")
        return

    print(f"Total documents loaded: {len(docs)}")
    print(f"  → Menu docs   : {sum(1 for d in docs if d['source_type'] == 'menu')}")
    print(f"  → Reddit docs : {sum(1 for d in docs if d['source_type'] == 'reddit')}")

    # Pick one menu doc and one reddit doc to show
    menu_docs   = [d for d in docs if d["source_type"] == "menu"]
    reddit_docs = [d for d in docs if d["source_type"] == "reddit"]

    for label, sample_list in [("MENU", menu_docs), ("REDDIT", reddit_docs)]:
        if not sample_list:
            print(f"\n[SKIP] No {label} documents found.")
            continue

        doc = sample_list[0]
        text = doc["text"]
        preview = text[:1500]  # show first 1500 chars

        print(f"\n{'─'*60}")
        print(f"[{label} DOCUMENT SAMPLE]")
        print(f"  doc_id     : {doc['doc_id']}")
        print(f"  file_path  : {doc['file_path']}")
        print(f"  restaurant : {doc.get('restaurant', 'N/A')}")
        print(f"  cuisine    : {doc.get('cuisine', 'N/A')}")
        print(f"  char count : {len(text)}")
        print(f"{'─'*60}")
        print(preview)
        if len(text) > 1500:
            print(f"\n  ... [{len(text) - 1500} more characters]")

        # Artifact check
        warnings = check_artifacts(text)
        if warnings:
            print("\n[CLEANING CHECK]")
            for w in warnings:
                print(f"  {w}")
        else:
            print("\n[CLEANING CHECK] ✅ No HTML, entities, or boilerplate detected")


# ─── Step 2: Print 5 representative chunks ───────────────────────────────────

def inspect_five_chunks():
    print("\n" + "═" * 70)
    print("STEP 2 — 5 representative chunks (quality check)")
    print("═" * 70)

    if not CHUNKS_FILE.exists():
        print(f"[ERROR] {CHUNKS_FILE} not found. Run chunk_text.py first.")
        return

    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        print("[ERROR] No chunks found in chunks.jsonl")
        return

    # Select representative samples:
    # - first menu chunk, middle menu chunk, last menu chunk
    # - first reddit chunk, one random chunk
    menu_chunks   = [c for c in chunks if c["source_type"] == "menu"]
    reddit_chunks = [c for c in chunks if c["source_type"] == "reddit"]

    samples = []
    if menu_chunks:
        samples.append(("Menu — first chunk",  menu_chunks[0]))
        if len(menu_chunks) > 2:
            samples.append(("Menu — middle chunk", menu_chunks[len(menu_chunks) // 2]))
        samples.append(("Menu — last chunk",   menu_chunks[-1]))
    if reddit_chunks:
        samples.append(("Reddit — first chunk", reddit_chunks[0]))
    # One fully random chunk for variety
    random.seed(42)
    samples.append(("Random chunk", random.choice(chunks)))

    # Deduplicate by chunk_id, keep order
    seen = set()
    unique_samples = []
    for label, chunk in samples:
        if chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            unique_samples.append((label, chunk))

    for i, (label, chunk) in enumerate(unique_samples[:5], 1):
        text = chunk["text"]
        verdict = assess_chunk_quality(text)
        token_count = chunk.get("token_count", "?")

        print(f"\n{'─'*60}")
        print(f"CHUNK {i} — {label}")
        print(f"  chunk_id   : {chunk['chunk_id']}")
        print(f"  restaurant : {chunk.get('restaurant', 'N/A')}")
        print(f"  tokens     : {token_count}")
        print(f"  quality    : {verdict}")
        print(f"{'─'*60}")
        # Show full chunk text (capped at 600 chars for readability)
        print(text[:600])
        if len(text) > 600:
            print(f"  ... [{len(text) - 600} more characters]")

        # Per-chunk artifact warnings
        warnings = check_artifacts(text)
        if warnings:
            for w in warnings:
                print(f"  {w}")


# ─── Step 3: Count total chunks and validate range ───────────────────────────

def count_and_validate_chunks():
    print("\n" + "═" * 70)
    print("STEP 3 — Chunk count & range validation")
    print("═" * 70)

    if not CHUNKS_FILE.exists():
        print(f"[ERROR] {CHUNKS_FILE} not found. Run chunk_text.py first.")
        return

    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    total = len(chunks)
    menu_count   = sum(1 for c in chunks if c["source_type"] == "menu")
    reddit_count = sum(1 for c in chunks if c["source_type"] == "reddit")

    token_counts = [c.get("token_count", 0) for c in chunks]
    avg_tokens = sum(token_counts) / total if total else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0

    # Count quality issues
    too_small = sum(1 for c in chunks if len(c["text"].split()) < 20)
    too_large = sum(1 for c in chunks if len(c["text"].split()) > 400)
    has_html  = sum(1 for c in chunks if HTML_TAG_RE.search(c["text"]))

    print(f"\n  Total chunks  : {total}")
    print(f"  Menu chunks   : {menu_count}")
    print(f"  Reddit chunks : {reddit_count}")
    print(f"\n  Token stats:")
    print(f"    min  : {min_tokens}")
    print(f"    avg  : {avg_tokens:.0f}")
    print(f"    max  : {max_tokens}")
    print(f"\n  Quality issues:")
    print(f"    Too small (<20 words) : {too_small}")
    print(f"    Too large (>400 words): {too_large}")
    print(f"    HTML artifacts        : {has_html}")

    # Range verdict
    print(f"\n  {'─'*40}")
    if total < 50:
        print(f"  ❌ UNDER RANGE ({total} chunks < 50)")
        print("     Chunks are too large — specific queries can't match precisely.")
        print("     Try reducing chunk_size (e.g. 250–300) in chunk_text.py")
        print("     and update planning.md to explain the change.")
    elif total > 2000:
        print(f"  ⚠️  OVER RANGE ({total} chunks > 2000)")
        print("     Chunks may be too small — embeddings carry too little meaning.")
        print("     Try increasing chunk_size (e.g. 450–500) in chunk_text.py")
        print("     and update planning.md to explain the change.")
    else:
        print(f"  ✅ IN RANGE ({total} chunks — within 50–2000 target)")

    if too_small > total * 0.1:
        print(f"  ⚠️  {too_small} tiny chunks ({too_small/total:.0%}) — check split logic for empty sections")
    if has_html > 0:
        print(f"  ❌ {has_html} chunks still contain HTML — re-run cleaning")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("PIPELINE INSPECTION — Vegetarian RAG")
    print("Milestone 3 Checkpoint")
    print("=" * 70)

    inspect_one_document()
    inspect_five_chunks()
    count_and_validate_chunks()

    print("\n" + "═" * 70)
    print("INSPECTION COMPLETE")
    print("  If all checks show ✅, you're ready for Milestone 4 (embedding).")
    print("  Fix any ❌ or ⚠️  issues above before proceeding.")
    print("═" * 70)