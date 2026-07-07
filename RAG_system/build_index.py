"""
Step 2a - Build the retrieval index.

Chunks each paper's cleaned text, embeds each chunk with a passage embedder
(bge-small-en-v1.5), and stores the vectors in a FAISS index. A parallel
chunks.json records which paper each vector came from, so a retrieval hit can
be traced back to a citation.

Outputs:
    data/index/faiss.index   - FAISS IndexFlatIP over L2-normalized vectors
    data/index/chunks.json   - one entry per vector, aligned by position

Usage:
    python build_index.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


METADATA_PATH = Path("data/metadata/papers_metadata.csv")
TEXT_DIR = Path("data/processed/texts")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, fast, tuned for passage retrieval

# Chunking is word-based so nothing splits mid-word. ~180 words is roughly a
# paragraph or two: big enough to hold a complete idea, small enough to stay
# focused. 30-word overlap keeps ideas that straddle a chunk boundary
# retrievable from either side.
TARGET_WORDS = 180
OVERLAP_WORDS = 30
MIN_TAIL_WORDS = 40  # drop a tiny trailing fragment unless it's the only chunk


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def chunk_text(text: str):
    words = text.split()
    if not words:
        return []

    chunks = []
    step = TARGET_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        piece = words[start:start + TARGET_WORDS]
        if not piece:
            break
        if len(piece) < MIN_TAIL_WORDS and chunks:
            break
        chunks.append(" ".join(piece))
        if start + TARGET_WORDS >= len(words):
            break
    return chunks


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)
    df = df[df["extract_status"] == "extracted"].reset_index(drop=True)
    print(f"Papers with usable text: {len(df)}")

    chunk_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        text_path = row.get("text_path", "")
        if not isinstance(text_path, str) or not Path(text_path).exists():
            continue

        text = Path(text_path).read_text(encoding="utf-8")
        title = str(row.get("title", "")).strip()
        arxiv_base_id = row["arxiv_base_id"]

        for j, chunk in enumerate(chunk_text(text)):
            chunk_records.append({
                "arxiv_base_id": arxiv_base_id,
                "title": title,
                "chunk_index": j,
                "text": chunk,
            })

    if not chunk_records:
        print("No chunks produced - check that text_path values are valid.")
        return

    print(f"Total chunks: {len(chunk_records)}")

    # Prepend the title to each chunk so it carries its own topical context.
    # bge passages take no instruction prefix; the query side adds one instead.
    device = pick_device()
    print(f"Loading {MODEL_NAME} on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    passages = [f"{r['title']}\n\n{r['text']}" for r in chunk_records]
    embeddings = model.encode(
        passages,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via inner product
        convert_to_numpy=True,
    )
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact search - fine at this corpus size
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False)

    print("\nDone.")
    print(f"  vectors:   {index.ntotal}")
    print(f"  dimension: {dim}")
    print(f"  index ->   {INDEX_PATH}")
    print(f"  chunks ->  {CHUNKS_PATH}")


if __name__ == "__main__":
    main()