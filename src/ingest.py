"""
ingest.py — Document Ingestion Pipeline
Vegetarian Food RAG System

Loads raw menu text files and Reddit files from:
  data/raw/        → restaurant menu .txt files
  data/clean/      → Reddit .txt files (already cleaned)

Cleans and normalizes text, then writes structured documents to:
  data/processed/documents.jsonl

Each output record:
{
  "doc_id":      str,   # unique identifier
  "source_type": str,   # "menu" | "reddit"
  "source_name": str,   # e.g. "Allmenus" | "r/vegetarian"
  "restaurant":  str,   # restaurant name (menus only, else "")
  "cuisine":     str,   # inferred cuisine type where possible
  "text":        str,   # cleaned full document text
  "file_path":   str    # originating file path
}
"""

import os
import re
import json
import hashlib
from pathlib import Path

# ─── Directory layout ────────────────────────────────────────────────────────
# Resolve paths relative to THIS script file so the script works regardless
# of which directory you invoke it from (project root, src/, etc.)
SCRIPT_DIR   = Path(__file__).resolve().parent   # the src/ folder
PROJECT_ROOT = SCRIPT_DIR.parent                  # ai201-project1-... root
BASE_DIR     = PROJECT_ROOT / "data"

MENU_DIR     = BASE_DIR / "raw"        # .txt menu files (any subfolders ok)
REDDIT_DIR   = BASE_DIR / "clean"      # Reddit .txt files
OUTPUT_DIR   = BASE_DIR / "processed"
OUTPUT_FILE  = OUTPUT_DIR / "documents.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[ingest] Project root : {PROJECT_ROOT}")
print(f"[ingest] Menu dir     : {MENU_DIR}  (exists: {MENU_DIR.exists()})")
print(f"[ingest] Reddit dir   : {REDDIT_DIR}  (exists: {REDDIT_DIR.exists()})")


# ─── Encoding-safe file reader ────────────────────────────────────────────────

def _read_any_encoding(fp: Path) -> str | None:
    """
    Try multiple encodings in order. Windows saves files in various encodings:
    - utf-8-sig  : UTF-8 with BOM (VS Code default)
    - utf-16     : UTF-16 with BOM (Windows Notepad "Unicode")
    - utf-16-le  : UTF-16 little-endian without BOM
    - cp1252     : Windows ANSI Western European (very common on Windows)
    - latin-1    : ISO-8859-1 fallback, never raises but may have garbage chars
    Returns the decoded string, or None if file is truly empty after decoding.
    """
    # First: read raw bytes to detect BOM manually
    try:
        raw_bytes = fp.read_bytes()
    except Exception as e:
        print(f"  [WARN] Cannot read bytes from {fp.name}: {e}")
        return None

    if len(raw_bytes) == 0:
        print(f"  [WARN] {fp.name} is 0 bytes on disk — file is empty.")
        return None

    # BOM detection
    if raw_bytes[:3] == b'\xef\xbb\xbf':
        return raw_bytes[3:].decode("utf-8", errors="replace")
    if raw_bytes[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw_bytes.decode("utf-16", errors="replace")

    # No BOM — try encodings in order
    for enc in ["utf-8", "cp1252", "latin-1"]:
        try:
            text = raw_bytes.decode(enc)
            if text.strip():
                return text
        except (UnicodeDecodeError, UnicodeError):
            continue

    print(f"  [WARN] Could not decode {fp.name} — try re-saving as UTF-8 in VS Code.")
    return None

# ─── Cuisine keyword mapping (basic heuristic) ───────────────────────────────
CUISINE_KEYWORDS = {
    "indian":     ["curry", "masala", "paneer", "dal", "naan", "biryani", "tikka"],
    "mexican":    ["taco", "burrito", "quesadilla", "enchilada", "tamale", "salsa"],
    "thai":       ["pad thai", "tom yum", "thai", "basil", "lemongrass", "coconut milk"],
    "italian":    ["pasta", "pizza", "risotto", "marinara", "pesto", "lasagna"],
    "chinese":    ["fried rice", "dim sum", "wonton", "noodle", "szechuan", "kung pao"],
    "japanese":   ["sushi", "ramen", "miso", "tempura", "udon", "edamame"],
    "american":   ["burger", "fries", "sandwich", "wings", "bbq", "mac and cheese"],
    "mediterranean": ["hummus", "falafel", "pita", "tzatziki", "shawarma", "gyro"],
    "vietnamese": ["pho", "banh mi", "vietnamese", "spring roll", "vermicelli"],
}

def infer_cuisine(text: str) -> str:
    """Return best-matching cuisine label or 'unknown'."""
    text_lower = text.lower()
    scores = {cuisine: 0 for cuisine in CUISINE_KEYWORDS}
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cuisine] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


# ─── Menu cleaning ────────────────────────────────────────────────────────────

def extract_menu_header(text: str) -> dict:
    """
    Pull RESTAURANT and SOURCE from the header block.
    Expected format:
        RESTAURANT: Maya's
        SOURCE: Allmenus
    """
    restaurant = ""
    source_name = "Allmenus"  # default

    restaurant_match = re.search(r"^RESTAURANT:\s*(.+)", text, re.MULTILINE)
    source_match     = re.search(r"^SOURCE:\s*(.+)",     text, re.MULTILINE)

    if restaurant_match:
        restaurant = restaurant_match.group(1).strip()
    if source_match:
        source_name = source_match.group(1).strip()

    return {"restaurant": restaurant, "source_name": source_name}


def clean_menu_text(text: str) -> str:
    """
    Normalize a raw menu text file.
    - Collapse excessive blank lines
    - Strip trailing whitespace per line
    - Remove null/empty DESCRIPTION lines (noise reduction)
    - Keep structural tags (SECTION:, ITEM:, PRICE:, DESCRIPTION:)
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.rstrip()

        # Drop lines that are just "DESCRIPTION:" with nothing after them
        if re.match(r"^DESCRIPTION:\s*$", line):
            continue

        cleaned.append(line)

    # Collapse 3+ consecutive blank lines into 2
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ─── Reddit cleaning ─────────────────────────────────────────────────────────

def clean_reddit_text(text: str) -> str:
    """
    Light cleaning for Reddit files that are already pre-cleaned.
    - Strip markdown artifacts (**, __, ~~)
    - Collapse whitespace
    - Remove URLs (not useful for embedding)
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove bold/italic markdown
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text)
    text = re.sub(r"~~(.+?)~~", r"\1", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove Reddit vote/score artifacts like "[+123]" or "(upvotes: 45)"
    text = re.sub(r"\[\+?\d+\]", "", text)
    text = re.sub(r"\(upvotes?:\s*\d+\)", "", text, flags=re.IGNORECASE)

    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_reddit_source(file_path: Path) -> str:
    """
    Infer subreddit from filename or first line.
    E.g. 'r_vegetarian_posts.txt' → 'r/vegetarian'
    """
    name = file_path.stem.lower()
    if "vegetarian" in name:
        return "r/vegetarian"
    if "vegan" in name:
        return "r/vegan"

    # Fallback: check first 200 chars of file
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(200).lower()
        if "r/vegetarian" in head:
            return "r/vegetarian"
        if "r/vegan" in head:
            return "r/vegan"
    except Exception:
        pass

    return "reddit"


# ─── Document ID ─────────────────────────────────────────────────────────────

def make_doc_id(file_path: Path, source_type: str) -> str:
    """Stable short ID: source_type prefix + hash of path string."""
    h = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
    prefix = "menu" if source_type == "menu" else "reddit"
    return f"{prefix}_{h}"


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_menu_documents() -> list[dict]:
    """
    Walk MENU_DIR recursively, load every .txt file as a menu document.
    """
    docs = []
    pattern = MENU_DIR.rglob("*.txt")

    for fp in sorted(pattern):
        raw = _read_any_encoding(fp)
        if raw is None:
            continue

        print(f"  [READ] {fp.name}  ({len(raw.strip())} chars)")
        if len(raw.strip()) < 10:
            print(f"  [SKIP] Truly empty, skipping: {fp.name}")
            continue

        header   = extract_menu_header(raw)
        cleaned  = clean_menu_text(raw)
        cuisine  = infer_cuisine(cleaned)
        doc_id   = make_doc_id(fp, "menu")

        docs.append({
            "doc_id":      doc_id,
            "source_type": "menu",
            "source_name": header["source_name"],
            "restaurant":  header["restaurant"],
            "cuisine":     cuisine,
            "text":        cleaned,
            "file_path":   str(fp),
        })

    print(f"[ingest] Loaded {len(docs)} menu documents from {MENU_DIR}")
    return docs


def load_reddit_documents() -> list[dict]:
    """
    Walk REDDIT_DIR recursively, load every .txt file as a Reddit document.
    """
    docs = []
    pattern = REDDIT_DIR.rglob("*.txt")

    for fp in sorted(pattern):
        raw = _read_any_encoding(fp)
        if raw is None:
            continue

        print(f"  [READ] {fp.name}  ({len(raw.strip())} chars)")
        if len(raw.strip()) < 10:
            print(f"  [SKIP] Truly empty, skipping: {fp.name}")
            continue

        source_name = detect_reddit_source(fp)
        cleaned     = clean_reddit_text(raw)
        cuisine     = infer_cuisine(cleaned)
        doc_id      = make_doc_id(fp, "reddit")

        docs.append({
            "doc_id":      doc_id,
            "source_type": "reddit",
            "source_name": source_name,
            "restaurant":  "",          # Reddit posts aren't restaurant-specific
            "cuisine":     cuisine,
            "text":        cleaned,
            "file_path":   str(fp),
        })

    print(f"[ingest] Loaded {len(docs)} Reddit documents from {REDDIT_DIR}")
    return docs


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_ingestion() -> list[dict]:
    print("=" * 60)
    print("INGESTION PIPELINE — Vegetarian RAG")
    print("=" * 60)

    all_docs = []
    all_docs.extend(load_menu_documents())
    all_docs.extend(load_reddit_documents())

    # Write to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\n[ingest] Total documents: {len(all_docs)}")
    print(f"[ingest] Output written to: {OUTPUT_FILE}")
    print("=" * 60)
    return all_docs


if __name__ == "__main__":
    run_ingestion()