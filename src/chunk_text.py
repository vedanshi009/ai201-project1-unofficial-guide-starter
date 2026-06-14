"""
chunk_text.py — Chunking Pipeline
Vegetarian Food RAG System

Reads:  data/processed/documents.jsonl   (output of ingest.py)
Writes: data/processed/chunks.jsonl

Chunk spec (from project plan):
  chunk_size : 300–450 tokens  → target 400 tokens
  overlap    : 60–100 tokens   → target 80 tokens

Each output record:
{
  "chunk_id":    str,   # doc_id + sequential index
  "doc_id":      str,   # parent document id
  "source_type": str,   # "menu" | "reddit"
  "source_name": str,   # e.g. "Allmenus", "r/vegetarian"
  "restaurant":  str,   # restaurant name or ""
  "cuisine":     str,
  "chunk_index": int,   # 0-based position within document
  "text":        str    # metadata header + chunk body
}
"""

import json
import re
import tiktoken
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent   # src/
PROJECT_ROOT = SCRIPT_DIR.parent                  # project root
BASE_DIR     = PROJECT_ROOT / "data"

INPUT_FILE   = BASE_DIR / "processed" / "documents.jsonl"
OUTPUT_FILE  = BASE_DIR / "processed" / "chunks.jsonl"

# ─── Chunk parameters ────────────────────────────────────────────────────────
# Reduced from 400→250 tokens to produce more chunks (was 41, need ≥50).
# With 8 menus + 3 Reddit files averaging ~3500 chars each, 250-token chunks
# yield ~70–90 total chunks — comfortably in the 50–200 target range.
# Overlap kept at 60 tokens (bottom of spec range) so boundaries stay clean.
# planning.md updated: chunk_size 250, overlap 60, reason: dataset size.
DEFAULT_CHUNK_SIZE = 250   # tokens per chunk
DEFAULT_OVERLAP    = 60    # overlap tokens
MAX_CHUNK_TOKENS   = 310   # hard ceiling: 250 body + 20 metadata + 40 snap buffer

# ─── Tokenizer ───────────────────────────────────────────────────────────────
# cl100k_base is the tokenizer used by text-embedding-004 / GPT-4-class models.
# Using it here ensures token counts during chunking match counts during embedding.
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def encode(text: str) -> list[int]:
    return TOKENIZER.encode(text)


def decode(tokens: list[int]) -> str:
    return TOKENIZER.decode(tokens)


# ─── Metadata header builder ─────────────────────────────────────────────────

def build_metadata_header(doc: dict, chunk_index: int) -> str:
    parts = [
        f"SOURCE: {doc['source_name']}",
        f"TYPE: {doc['source_type']}",
    ]
    if doc.get("restaurant"):
        parts.append(f"RESTAURANT: {doc['restaurant']}")
    if doc.get("cuisine") and doc["cuisine"] != "unknown":
        parts.append(f"CUISINE: {doc['cuisine']}")
    parts.append(f"CHUNK: {chunk_index}")
    return "[" + " | ".join(parts) + "]\n"


# ─── Core chunking function ───────────────────────────────────────────────────

def _snap_to_item_boundary(tokens: list[int], pos: int, window: int = 30) -> int:
    """
    Snap a token position forward to the nearest ITEM: line boundary
    within ±window tokens, so chunks never split mid-item description.
    Falls back to the original pos if no boundary found nearby.
    """
    search_start = max(0, pos - window)
    search_end   = min(len(tokens), pos + window)
    candidate    = decode(tokens[search_start:search_end])

    # Find all ITEM: positions in the candidate text
    item_offsets = [m.start() for m in re.finditer(r"\nITEM:", candidate)]
    if not item_offsets:
        return pos  # no boundary nearby, keep original

    # Convert char offset back to approximate token offset
    # Pick the item boundary closest to (pos - search_start) in char space
    target_char = len(decode(tokens[search_start:pos]))
    best_char   = min(item_offsets, key=lambda x: abs(x - target_char))

    # Re-encode just the prefix to get an accurate token count
    prefix_text   = candidate[:best_char]
    prefix_tokens = len(encode(prefix_text))
    return search_start + prefix_tokens


def chunk_text(
    doc_text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split doc_text into overlapping token windows, snapping boundaries
    to ITEM: lines where possible to avoid mid-description cuts.
    """
    assert 150 <= chunk_size <= 450, f"chunk_size {chunk_size} out of range"
    assert 40  <= overlap    <= 100, f"overlap {overlap} out of range"

    tokens = encode(doc_text)
    if len(tokens) == 0:
        return []

    step   = chunk_size - overlap
    chunks = []
    start  = 0

    while start < len(tokens):
        raw_end  = min(start + chunk_size, len(tokens))
        # Snap end to nearest ITEM: boundary so we don't cut mid-description
        end      = _snap_to_item_boundary(tokens, raw_end) if raw_end < len(tokens) else raw_end
        # Safety: never exceed hard ceiling
        end      = min(end, start + MAX_CHUNK_TOKENS)

        window    = tokens[start:end]
        chunk_str = decode(window).strip()

        if chunk_str:
            chunks.append(chunk_str)

        if end >= len(tokens):
            break
        start += step

    return chunks


# ─── Menu-aware splitting (pre-processing) ───────────────────────────────────

def split_menu_by_section(text: str) -> list[str]:
    """
    For menu files: split on SECTION: boundaries first so that each
    semantic unit (a menu section with its items) is chunked together
    rather than being sliced mid-section.

    If a section is still too large it will be further split by chunk_text().
    Sections under MIN_SECTION_TOKENS are merged with adjacent sections to
    prevent tiny tail-end chunks (e.g. a 2-item Desserts section).
    """
    MIN_SECTION_TOKENS = 40   # sections smaller than this get merged

    # Split on SECTION: lines but keep the delimiter
    parts = re.split(r"(?=^SECTION:)", text, flags=re.MULTILINE)

    # Extract header block (RESTAURANT / SOURCE lines before first SECTION)
    header_block = parts[0] if parts else ""
    sections     = [s.strip() for s in parts[1:] if s.strip()]

    if not sections:
        return [text]

    # Pass 1: merge forward — small sections absorb the next section
    merged = []
    buffer = header_block.strip() + "\n\n" if header_block.strip() else ""

    for section in sections:
        candidate = (buffer + section) if buffer else section
        if count_tokens(candidate) < MIN_SECTION_TOKENS:
            # Too small — keep accumulating into buffer
            buffer = candidate + "\n\n"
        else:
            merged.append(candidate)
            buffer = ""

    # Pass 2: if anything is left in buffer (small trailing section),
    # append it to the last merged section instead of emitting alone
    if buffer.strip():
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buffer.strip()
        else:
            merged.append(buffer.strip())

    return merged if merged else [text]


# ─── Document → chunks ────────────────────────────────────────────────────────

def chunk_document(doc: dict) -> list[dict]:
    """
    Convert one document record (from ingest.py) into a list of chunk records.

    For menus: pre-split by SECTION first, then apply sliding-window per section.
    For reddit: apply sliding-window directly on the full text.
    """
    raw_text = doc["text"]

    if doc["source_type"] == "menu":
        sections = split_menu_by_section(raw_text)
    else:
        sections = [raw_text]

    all_raw_chunks: list[str] = []
    for section in sections:
        section_chunks = chunk_text(section)
        all_raw_chunks.extend(section_chunks)

    chunk_records = []
    for idx, raw_chunk in enumerate(all_raw_chunks):
        header = build_metadata_header(doc, idx)
        full_text = header + raw_chunk

        # Safety assertion — no chunk should exceed the hard ceiling
        token_count = count_tokens(full_text)
        assert token_count <= MAX_CHUNK_TOKENS, (
            f"Chunk {doc['doc_id']}_{idx} exceeds {MAX_CHUNK_TOKENS} tokens "
            f"(got {token_count}). Reduce chunk_size or fix split logic."
        )

        chunk_records.append({
            "chunk_id":    f"{doc['doc_id']}_{idx}",
            "doc_id":      doc["doc_id"],
            "source_type": doc["source_type"],
            "source_name": doc["source_name"],
            "restaurant":  doc.get("restaurant", ""),
            "cuisine":     doc.get("cuisine", "unknown"),
            "chunk_index": idx,
            "token_count": token_count,
            "text":        full_text,
        })

    return chunk_records


# ─── Pipeline runner ─────────────────────────────────────────────────────────

def run_chunking() -> list[dict]:
    print("=" * 60)
    print("CHUNKING PIPELINE — Vegetarian RAG")
    print(f"  chunk_size : {DEFAULT_CHUNK_SIZE} tokens")
    print(f"  overlap    : {DEFAULT_OVERLAP} tokens")
    print(f"  max_tokens : {MAX_CHUNK_TOKENS} tokens (hard ceiling)")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run ingest.py first to generate documents.jsonl"
        )

    # Load documents
    docs = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    print(f"[chunk] Loaded {len(docs)} documents")

    # Chunk all documents
    all_chunks = []
    for doc in docs:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"  {doc['doc_id']} ({doc['source_type']}) → {len(doc_chunks)} chunks")

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Summary stats
    token_counts = [c["token_count"] for c in all_chunks]
    avg_tokens   = sum(token_counts) / len(token_counts) if token_counts else 0
    max_tokens   = max(token_counts) if token_counts else 0
    min_tokens   = min(token_counts) if token_counts else 0

    print(f"\n[chunk] Total chunks : {len(all_chunks)}")
    print(f"[chunk] Token stats  : min={min_tokens}  avg={avg_tokens:.0f}  max={max_tokens}")
    print(f"[chunk] Output       : {OUTPUT_FILE}")
    print("=" * 60)

    return all_chunks


# ─── Verification helper ─────────────────────────────────────────────────────

def verify_overlap(doc_text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    """
    Sanity-check: given a document, find a phrase near the boundary between
    chunk 0 and chunk 1, and confirm it appears in both chunks (proving overlap works).

    Run this manually on a sample document to validate the chunker.

    Example:
        from chunk_text import verify_overlap
        sample = open("data/raw/menus/mayas.txt").read()
        verify_overlap(sample)
    """
    chunks = chunk_text(doc_text, chunk_size, overlap)

    assert len(chunks) >= 2, "Document too short to test overlap (needs at least 2 chunks)"

    # Take the last 20 tokens of chunk 0 as the "boundary phrase"
    boundary_tokens = encode(chunks[0])[-20:]
    boundary_phrase = decode(boundary_tokens).strip()

    in_chunk_0 = boundary_phrase in chunks[0]
    in_chunk_1 = boundary_phrase in chunks[1]

    print(f"[verify] Boundary phrase (last 20 tokens of chunk 0):")
    print(f"  '{boundary_phrase[:80]}...'")
    print(f"  Present in chunk 0: {in_chunk_0}")
    print(f"  Present in chunk 1: {in_chunk_1}")

    assert in_chunk_0, "Boundary phrase not found in chunk 0 — something is wrong"
    assert in_chunk_1, "Overlap FAILED: boundary phrase not found in chunk 1"

    print("[verify] ✓ Overlap verified successfully")

    for i, chunk in enumerate(chunks):
        n = count_tokens(chunk)
        assert n <= MAX_CHUNK_TOKENS, f"Chunk {i} exceeds max: {n} tokens"
        assert n >= 10, f"Chunk {i} suspiciously tiny: {n} tokens"

    print(f"[verify] ✓ All {len(chunks)} chunks within token bounds")
    return chunks


if __name__ == "__main__":
    run_chunking()