"""
embed_and_store.py — Embedding + Vector Store Pipeline
Vegetarian Food RAG System

Reads:  data/processed/chunks.jsonl       (output of chunk_text.py)
Writes: data/processed/chroma_db/         (ChromaDB persistent store)

Embedding model: all-MiniLM-L6-v2 (sentence-transformers)
  - Runs fully locally, no API key, no rate limits
  - 384-dimension vectors, strong semantic accuracy on short food text
  - Fast: ~70 chunks embed in <2 seconds on CPU

Vector store: ChromaDB (persistent on disk)
  - Collection name: "vegetarian_rag"
  - Each chunk stored with full metadata for attribution at query time
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR     = PROJECT_ROOT / "data" / "processed"

CHUNKS_FILE  = BASE_DIR / "chunks.jsonl"
CHROMA_DIR   = BASE_DIR / "chroma_db"

COLLECTION_NAME = "vegetarian_rag"

# ─── Embedding model ──────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: local, free, no API key, 384-dim vectors
# Fast enough that all 70 chunks embed in under 2 seconds on CPU
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    """Load all chunks from chunks.jsonl."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"chunks.jsonl not found at {CHUNKS_FILE}\n"
            "Run chunk_text.py first."
        )
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[embed] Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks


def embed_and_store(chunks: list[dict]) -> chromadb.Collection:
    """
    Embed all chunks with all-MiniLM-L6-v2 and upsert into ChromaDB.

    ChromaDB stores three things per chunk:
      - document : the raw chunk text (what the LLM will read)
      - embedding: the 384-dim vector (what similarity search uses)
      - metadata : source, restaurant, cuisine etc. (for attribution)
    """
    # Load embedding model (downloads ~90MB on first run, cached after)
    print(f"[embed] Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Extract texts to embed
    texts = [c["text"] for c in chunks]

    # Embed all chunks in one batch — faster than one-by-one
    print(f"[embed] Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print(f"[embed] Done. Embedding shape: {embeddings.shape}")

    # Set up ChromaDB persistent client
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if re-running (clean slate)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[embed] Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # collection didn't exist yet

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # cosine similarity is standard for sentence-transformer embeddings
        metadata={"hnsw:space": "cosine"},
    )

    # Build lists for ChromaDB upsert
    ids        = [c["chunk_id"]    for c in chunks]
    metadatas  = [
        {
            "doc_id":      c["doc_id"],
            "source_type": c["source_type"],
            "source_name": c["source_name"],
            "restaurant":  c.get("restaurant", ""),
            "cuisine":     c.get("cuisine", "unknown"),
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    # Upsert in batches of 50 (safe for any ChromaDB version)
    BATCH = 50
    for i in range(0, len(chunks), BATCH):
        batch_slice = slice(i, i + BATCH)
        collection.add(
            ids        = ids[batch_slice],
            documents  = texts[batch_slice],
            embeddings = embeddings[batch_slice].tolist(),
            metadatas  = metadatas[batch_slice],
        )
    print(f"[embed] Stored {collection.count()} chunks in ChromaDB")
    print(f"[embed] Database saved to: {CHROMA_DIR}")
    return collection


def run_embedding():
    print("=" * 60)
    print("EMBEDDING PIPELINE — Vegetarian RAG")
    print(f"  Model : {EMBEDDING_MODEL}")
    print(f"  Store : ChromaDB (persistent)")
    print("=" * 60)

    chunks     = load_chunks()
    collection = embed_and_store(chunks)

    print("\n[embed] ✅ Embedding complete.")
    print(f"[embed] Collection '{COLLECTION_NAME}' contains {collection.count()} vectors.")
    print("=" * 60)


if __name__ == "__main__":
    run_embedding()