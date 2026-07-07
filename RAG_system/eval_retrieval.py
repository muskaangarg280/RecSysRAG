"""
Step 2b - Retrieval evaluation (no LLM involved).

Runs every question in the eval set through retrieval only and checks
whether the paper(s) that should answer it (gold_sources) actually appear in
the top-k. This is the gate before wiring in an LLM: if retrieval can't find
the right paper, no amount of prompting fixes the answer.

Scoring by gold_match:
    "any"  - hit if any gold paper is in the top-k
    "all"  - hit if every gold paper is in the top-k (used for comparisons)
    "none" - refusal questions; retrieval hit-rate isn't meaningful here, so
             we report the top similarity score instead, used to calibrate a
             routing/refusal threshold.

Usage:
    python eval_retrieval.py           # top-k = 5
    python eval_retrieval.py --k 10
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("data/index/faiss.index")
CHUNKS_PATH = Path("data/index/chunks.json")
EVAL_PATH = Path("data/eval_set.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge's asymmetric retrieval convention: queries get this instruction prefix,
# passages do not.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def retrieve(model, index, chunks, question, k):
    """Return (ranked unique paper IDs, paper -> rank map, top score, top hits)."""
    q_emb = model.encode(
        [QUERY_INSTRUCTION + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(q_emb, k)
    scores, ids = scores[0], ids[0]

    ranked_papers = []
    paper_rank = {}
    top_hits = []  # (title, arxiv_id, score) in retrieval order, chunk-level
    for score, vec_id in zip(scores, ids):
        if vec_id < 0:
            continue
        rec = chunks[vec_id]
        pid = rec["arxiv_base_id"]
        top_hits.append((rec["title"], pid, float(score)))
        if pid not in paper_rank:
            paper_rank[pid] = len(ranked_papers) + 1  # 1-indexed paper rank
            ranked_papers.append(pid)

    top_score = float(scores[0]) if len(scores) else 0.0
    return ranked_papers, paper_rank, top_score, top_hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="top-k chunks to retrieve")
    args = parser.parse_args()

    for p in (INDEX_PATH, CHUNKS_PATH, EVAL_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    eval_set = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    device = pick_device()
    print(f"Loading {MODEL_NAME} on {device} ...\n")
    model = SentenceTransformer(MODEL_NAME, device=device)

    graded = 0
    hits = 0
    reciprocal_ranks = []
    refusal_rows = []

    print(f"{'ID':<5}{'TYPE':<6}{'HIT':<5}{'GOLD FOUND':<12}{'1ST RANK':<9}TOP RETRIEVED")
    print("-" * 100)

    for q in eval_set:
        ranked, paper_rank, top_score, top_hits = retrieve(
            model, index, chunks, q["question"], args.k
        )
        gold = q.get("gold_sources", [])
        match = q.get("gold_match", "any")

        if match == "none":
            refusal_rows.append((q["id"], top_score))
            top_titles = " | ".join(f"{t[:32]}({s:.2f})" for t, _, s in top_hits[:2])
            print(f"{q['id']:<5}{q['type']:<6}{'-':<5}{'(refusal)':<12}{'-':<9}{top_titles}")
            continue

        found = [g for g in gold if g in paper_rank]
        hit = len(found) == len(gold) if match == "all" else len(found) > 0

        first_rank = min((paper_rank[g] for g in found), default=0)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        graded += 1
        hits += int(hit)

        top_titles = " | ".join(f"{t[:32]}({s:.2f})" for t, _, s in top_hits[:2])
        print(
            f"{q['id']:<5}{q['type']:<6}{('Y' if hit else 'N'):<5}"
            f"{f'{len(found)}/{len(gold)}':<12}{(first_rank or '-'):<9}{top_titles}"
        )

    print("-" * 100)
    print("\nSUMMARY (A/B/C questions only)")
    print(f"  top-k:            {args.k}")
    print(f"  questions graded: {graded}")
    if graded:
        print(f"  hit-rate:         {hits}/{graded} = {hits / graded:.2%}")
    if reciprocal_ranks:
        print(f"  mean recip. rank: {np.mean(reciprocal_ranks):.3f}")

    if refusal_rows:
        print("\nREFUSAL QUESTIONS (top similarity score - lower means the corpus")
        print("likely doesn't cover it; use these to calibrate a refusal threshold)")
        for qid, score in refusal_rows:
            print(f"  {qid}: top score {score:.3f}")


if __name__ == "__main__":
    main()