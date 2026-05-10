"""Interactive RAG query CLI for the UPSC question collection.

Loads the persistent ChromaDB collection built by ingest.py, embeds each
user query with all-MiniLM-L6-v2, and prints the top-k matching rows.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "upsc_questions"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


def format_result(rank: int, distance: float, md: dict) -> str:
    question = md.get("Question", "")
    options = [
        f"  A) {md.get('Option A', '')}",
        f"  B) {md.get('Option B', '')}",
        f"  C) {md.get('Option C', '')}",
        f"  D) {md.get('Option D', '')}",
    ]
    similarity = 1.0 - distance  # cosine distance -> similarity
    header = (
        f"[{rank}] similarity={similarity:.3f}  "
        f"{md.get('Subject', '?')} / {md.get('Topic', '?')} "
        f"({md.get('Year', '?')}, {md.get('Paper', '?')})"
    )
    wrapped_q = textwrap.fill(question, width=100, subsequent_indent="    ")
    parts = [header, f"  Q: {wrapped_q}"]
    parts.extend(options)
    parts.append(f"  Answer: {md.get('Correct Answer', '?')}")
    return "\n".join(parts)


def main() -> None:
    if not DB_PATH.exists():
        sys.exit(
            f"ChromaDB not found at {DB_PATH}. Run `python ingest.py` first."
        )

    print(f"Loading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} items.")
    print(f"Type a query (or :k <n> to change top-k, :q to quit). Default top-k = {DEFAULT_TOP_K}.\n")

    top_k = DEFAULT_TOP_K
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query in {":q", ":quit", "exit"}:
            break
        if query.startswith(":k "):
            try:
                top_k = max(1, int(query.split()[1]))
                print(f"top-k set to {top_k}")
            except (ValueError, IndexError):
                print("usage: :k <n>")
            continue

        embedding = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=top_k,
        )

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        if not metadatas:
            print("(no results)\n")
            continue
        print()
        for i, (md, dist) in enumerate(zip(metadatas, distances), start=1):
            print(format_result(i, dist, md))
            print()


if __name__ == "__main__":
    main()
