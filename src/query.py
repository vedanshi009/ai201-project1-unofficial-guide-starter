"""
query.py — Retrieval Function
Vegetarian Food RAG System

Loads the ChromaDB collection built by embed_and_store.py and exposes:

  retrieve(query, k=5) -> list[dict]
    Returns the top-k most relevant chunks with text + source metadata.

  query_rag(user_question, k=5) -> str        [placeholder for Milestone 5]
    Calls retrieve() then formats results — LLM generation added in M5.

Architecture position:
  chunks.jsonl → [embed_and_store.py] → chroma_db/
                                             ↓
                               query string → retrieve() → top-k chunks
                                                               ↓
                                                     query_rag() → answer
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CHROMA_DIR   = PROJECT_ROOT / "data" / "processed" / "chroma_db"

COLLECTION_NAME = "vegetarian_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─── Lazy-loaded singletons ───────────────────────────────────────────────────
# Load model and DB once, reuse across calls (important for query loops)
_model      = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DIR}\n"
                "Run embed_and_store.py first."
            )
        client      = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ─── Core retrieval function ──────────────────────────────────────────────────

def retrieve(query: str, k: int = 5, source_type: str = None) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.

    Args:
        query       : natural language question from the user
        k           : number of chunks to return (default 5)
        source_type : optional filter — "menu" or "reddit" only
                      Use when you know the answer lives in one source type.
                      e.g. ingredient warnings → source_type="reddit"
                           dish availability  → source_type="menu"

    Returns:
        List of dicts with text, score, chunk_id, and source metadata.
    """
    model      = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(query).tolist()

    # Build optional where-filter for ChromaDB
    where = {"source_type": source_type} if source_type else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]
    ids       = results["ids"][0]

    chunks = []
    for doc, meta, dist, cid in zip(docs, metas, distances, ids):
        chunks.append({
            "text":        doc,
            "score":       round(1 - dist, 4),
            "chunk_id":    cid,
            "source_name": meta.get("source_name", ""),
            "source_type": meta.get("source_type", ""),
            "restaurant":  meta.get("restaurant", ""),
            "cuisine":     meta.get("cuisine", ""),
            "chunk_index": meta.get("chunk_index", 0),
        })

    return chunks


# ─── RAG query function (generation added in Milestone 5) ────────────────────

def query_rag(user_question: str, k: int = 5, source_type: str = None) -> str:
    """
    Retrieve top-k chunks and format them for inspection.
    LLM generation will be added in Milestone 5.
    """
    chunks = retrieve(user_question, k=k, source_type=source_type)

    if not chunks:
        return "No relevant chunks found."

    lines = [
        f"Query: {user_question}",
        f"Top {len(chunks)} retrieved chunks:\n",
    ]
    for i, chunk in enumerate(chunks, 1):
        source = chunk["restaurant"] or chunk["source_name"]
        score  = chunk["score"]
        # Flag weak matches — scores below 0.5 indicate poor relevance
        flag   = "  ⚠️  weak match" if score < 0.5 else ""
        lines.append(f"[{i}] score={score}  source={source}  ({chunk['source_type']}){flag}")
        # Show full chunk text so you can judge relevance properly
        lines.append(chunk["text"][:600])
        lines.append("")

    return "\n".join(lines)


# ─── Retrieval test ───────────────────────────────────────────────────────────

def run_retrieval_test():
    """
    Test retrieval against 3 evaluation plan queries that match the actual data.

    Notes on queries not tested here:
      Q1 (USF protein dishes) — no USF dining data in dataset yet, will always drift
      Q3 (Yelp reviews)       — no Yelp data in dataset, Indian menu chunks retrieved instead
    These are data gaps to note in planning.md, not retrieval bugs.
    """
    # Query, k, optional source_type filter
    test_cases = [
        # Q2 — ingredient warning, Reddit is the right source
        ("Does Tom Yum soup contain fish sauce or shrimp paste?",          5, "reddit"),
        # Q4 — Chipotle menu, menus are the right source
        ("What are the vegetarian options at Chipotle?",                   5, "menu"),
        # Q5 — hidden ingredients, Reddit is the right source
        ("What hidden non-vegetarian ingredients should I watch for?",     5, "reddit"),
        # Bonus: unfiltered Indian restaurant query to see drift
        ("Which Indian restaurants have good vegetarian options?",         5, None),
    ]

    print("=" * 60)
    print("RETRIEVAL TEST — Vegetarian RAG")
    print("Checkpoint: scores below 0.5 = weak match")
    print("=" * 60)

    collection = _get_collection()
    print(f"[query] Collection: '{COLLECTION_NAME}' — {collection.count()} chunks\n")

    for query, k, source_type in test_cases:
        filter_label = f" [filter: {source_type}]" if source_type else " [no filter]"
        print("─" * 60)
        print(query_rag(query, k=k, source_type=source_type) + filter_label)

    print("=" * 60)
    print("\nREMINDER — Data gaps to note in planning.md:")
    print("  Q1 (protein content): no USF dining / nutrition data ingested yet")
    print("  Q3 (Yelp reviews)   : no Yelp review data ingested yet")
    print("  These queries will drift until those sources are added.")


if __name__ == "__main__":
    run_retrieval_test()