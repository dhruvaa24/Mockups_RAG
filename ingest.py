"""Build a ChromaDB index over upscpyqs.csv.

Embeds: Question + Subject + Topic for each row using all-MiniLM-L6-v2.
Stores: persistent collection at ./chroma_db with full row metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

CSV_PATH = Path(__file__).parent / "upscpyqs.csv"
DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "upsc_questions"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def build_embed_text(row: pd.Series) -> str:
    parts = [str(row.get("Question", "")).strip()]
    subject = str(row.get("Subject", "")).strip()
    topic = str(row.get("Topic", "")).strip()
    if subject:
        parts.append(f"Subject: {subject}")
    if topic:
        parts.append(f"Topic: {topic}")
    return "\n".join(p for p in parts if p)


def row_to_metadata(row: pd.Series) -> dict:
    keys = [
        "Paper", "Question", "Option A", "Option B", "Option C", "Option D",
        "Correct Answer", "Explanation", "Subject", "Topic", "Year",
    ]
    md = {}
    for k in keys:
        v = row.get(k, "")
        if pd.isna(v):
            v = ""
        md[k] = str(v)
    return md


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"CSV not found at {CSV_PATH}")

    print(f"Reading {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH)
    df = df[df["Question"].notna() & (df["Question"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)
    print(f"  {len(df)} rows with non-empty questions")

    print(f"Loading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building embedding inputs...")
    texts = [build_embed_text(row) for _, row in df.iterrows()]
    ids = [f"q-{i}" for i in range(len(df))]
    metadatas = [row_to_metadata(row) for _, row in df.iterrows()]

    print(f"Encoding {len(texts)} rows...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    print(f"Writing to ChromaDB at {DB_PATH}...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    # Reset the collection so re-runs are idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"Done. Collection '{COLLECTION_NAME}' has {collection.count()} items.")


if __name__ == "__main__":
    main()
