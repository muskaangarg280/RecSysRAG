"""
Step 3 - Connect the LLM.

Retrieves relevant chunks for a question, assembles a citation-tagged context,
sends it to a hosted LLM (Groq) with a strict grounding prompt, and returns an
answer plus the arXiv IDs it actually drew from.

Includes three safeguards beyond a plain retrieve-then-generate pipeline:
  1. A similarity-threshold check that skips the LLM call entirely for
     clearly off-topic questions.
  2. An automated citation-validity check: every [arXiv:ID] the model cites
     must be one of the papers actually in its context. This is a hard,
     checkable fact, not an LLM judgment call.
  3. A one-shot corrective retry if a citation fails that check, followed by
     a fail-safe refusal (rather than a second guess) if the retry still
     produces an unverified citation. A research assistant should decline
     rather than risk showing an unverified claim.

Setup (one time):
    pip install groq
    export GROQ_API_KEY="your-key-from-console.groq.com/keys"

Usage:
    python answer.py                          # run the full eval set
    python answer.py --ask "your question"    # ask a single question
    python answer.py --model llama-3.3-70b-versatile   # override the model
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import faiss
import torch
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------------
# Paths and index config
# --------------------------------------------------------------------------
INDEX_PATH = Path("data/index/faiss.index")
CHUNKS_PATH = Path("data/index/chunks.json")
EVAL_PATH = Path("data/eval_set.json")

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Groq's free tier tracks rate limits per model, each with its own quota
# pool. llama-3.1-8b-instant is the higher-throughput option recommended for
# prototyping; llama-3.3-70b-versatile gives higher-quality answers but a
# smaller daily token budget. Override at runtime with --model.
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# --------------------------------------------------------------------------
# Retrieval config
# --------------------------------------------------------------------------
# For comparison questions (e.g. "how do X and Y differ"), a single query
# embedding can rank one paper's chunks so highly that the other paper never
# appears in a narrow candidate pool at all. RAW_K is intentionally wide so
# both papers get a real chance to surface; MAX_PER_PAPER then caps how many
# chunks from any single paper make it into the final context, so one paper
# can't crowd out a genuine comparison.
RAW_K = 40
CONTEXT_CHUNKS = 8
MAX_PER_PAPER = 2

# Calibrated from a retrieval-only evaluation: clearly off-topic questions
# scored ~0.65 similarity, on-topic questions scored 0.71-0.91. Below this
# threshold we skip the LLM call entirely and return a refusal directly.
REFUSAL_SCORE_THRESHOLD = 0.68

# --------------------------------------------------------------------------
# Groq call config
# --------------------------------------------------------------------------
# Free tier: 12,000 tokens/minute. A short pause between calls plus
# retry-with-backoff on 429s keeps a full eval-set run from tripping it.
SECONDS_BETWEEN_CALLS = 3
MAX_RETRIES = 4

REFUSAL_TEXT = "This isn't covered in the provided papers."

SYSTEM_PROMPT = f"""You are a research assistant answering questions about \
recommender systems using ONLY the paper excerpts provided below.

Rules:
1. Answer only using information in the excerpts. Never use outside knowledge, \
even if you already know the answer from general training, and even for \
well-known models or papers you can name from memory.
2. Every factual claim must cite the paper it came from, using the exact format \
[arXiv:ID] where ID matches the tag shown before each excerpt. Only cite arXiv \
IDs that actually appear in the excerpts below - never cite a paper from memory. \
Do NOT copy any bracketed numbers that appear inside the excerpt text itself \
(like [4], [12, 26], [7,8]) - those are the source paper's own internal \
bibliography references and are meaningless outside that paper. The ONLY valid \
citation format is [arXiv:ID].
3. If a question has multiple parts and the excerpts only cover some of them, \
answer the covered parts and explicitly name which part isn't addressed by the \
excerpts - do not fill that gap with outside knowledge, even if you know the \
real answer.
4. If the excerpts don't contain enough information to answer at all, respond \
exactly: "{REFUSAL_TEXT}" Do not guess, and do not fabricate paper names, \
methods, or results that aren't in the excerpts below.
"""


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_resources():
    """Load the Groq client, the FAISS index, chunk metadata, and the embedder."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY before running (see file header).")
    llm = Groq(api_key=os.environ["GROQ_API_KEY"])

    for p in (INDEX_PATH, CHUNKS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}. Run build_index.py first.")

    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    device = pick_device()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    return index, chunks, embed_model, llm


def retrieve_diverse(embed_model, index, chunks, question, raw_k=RAW_K,
                      context_chunks=CONTEXT_CHUNKS, max_per_paper=MAX_PER_PAPER):
    """
    Retrieve a wide pool of raw chunks, then keep the top-scoring ones subject
    to a per-paper cap, so a single dominant paper can't crowd out a genuine
    multi-paper comparison question.
    """
    q_emb = embed_model.encode(
        [QUERY_INSTRUCTION + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(q_emb, raw_k)
    scores, ids = scores[0], ids[0]

    per_paper_count = {}
    selected = []
    top_score = float(scores[0]) if len(scores) else 0.0

    for score, vec_id in zip(scores, ids):
        if vec_id < 0 or len(selected) >= context_chunks:
            continue
        rec = chunks[vec_id]
        pid = rec["arxiv_base_id"]
        if per_paper_count.get(pid, 0) >= max_per_paper:
            continue
        per_paper_count[pid] = per_paper_count.get(pid, 0) + 1
        selected.append({**rec, "score": float(score)})

    return selected, top_score


def build_context(selected_chunks):
    """Assemble retrieved chunks into a citation-tagged context block."""
    blocks = []
    for c in selected_chunks:
        tag = f"[arXiv:{c['arxiv_base_id']}]"
        blocks.append(f"{tag} {c['title']}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def call_llm_with_retry(llm, prompt):
    """
    Call Groq with generation parameters chosen to guard against two
    observed failure modes:

      - Repetition loops, especially on results-table text with near-duplicate
        phrasing across rows. Guarded against with a token cap and a frequency
        penalty. presence_penalty is deliberately NOT used: it penalizes any
        repeated token regardless of whether the context supports repeating
        it, which can push the model away from the (sometimes sparse) grounded
        excerpts and toward inventing new, ungrounded content instead.

      - 429 rate limits, handled with backoff parsed from Groq's own
        "please try again in Xs" message where available.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=700,
                frequency_penalty=0.5,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            last_error = e
            wait = 20.0
            match = re.search(r"try again in ([\d.]+)s", str(e))
            if match:
                wait = float(match.group(1)) + 1.0
            print(f"  [rate limited - waiting {wait:.1f}s, attempt {attempt + 1}/{MAX_RETRIES}]")
            time.sleep(wait)

    raise last_error


def extract_cited_ids(answer_text):
    return sorted(set(re.findall(r"\[arXiv:([\w\./]+)\]", answer_text)))


def answer_question(embed_model, index, chunks, llm, question):
    """
    Full answer pipeline for one question: retrieve, generate, verify
    citations, and self-correct or fail safe if verification fails.
    """
    selected, top_score = retrieve_diverse(embed_model, index, chunks, question)
    retrieved_ids = sorted({c["arxiv_base_id"] for c in selected})

    if top_score < REFUSAL_SCORE_THRESHOLD:
        return {
            "answer": REFUSAL_TEXT,
            "retrieved_ids": retrieved_ids,
            "top_score": top_score,
            "skipped_llm": True,
            "corrected": False,
        }

    context = build_context(selected)
    prompt = f"Context excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"

    answer_text = call_llm_with_retry(llm, prompt)
    cited_ids = extract_cited_ids(answer_text)
    hallucinated = set(cited_ids) - set(retrieved_ids)

    # Automated citation-validity check: any cited ID that isn't in
    # retrieved_ids was never in the model's context, i.e. fabricated. If
    # that happens, give the model one chance to self-correct by telling it
    # exactly what it got wrong and which IDs are actually valid.
    if hallucinated:
        correction_prompt = (
            f"{prompt}\n\n---\n\nYour previous answer cited [arXiv:"
            f"{', arXiv:'.join(sorted(hallucinated))}], but that paper is NOT "
            f"among the excerpts above. Only these arXiv IDs are valid to cite: "
            f"{', '.join(retrieved_ids)}. Revise your answer using ONLY "
            f"information and citations from the excerpts above. If part of "
            f"the question isn't covered by these excerpts, say so explicitly "
            f"instead of citing anything else."
        )
        answer_text = call_llm_with_retry(llm, correction_prompt)
        cited_ids = extract_cited_ids(answer_text)
        still_hallucinated = set(cited_ids) - set(retrieved_ids)

        # Fail-safe rather than a second guess: if the correction retry still
        # produces an unverified citation, don't gamble on a further retry
        # fixing it - it may simply fabricate something different. Refuse
        # cleanly instead. For a research assistant, declining is safer than
        # risking an unverified claim reaching the user.
        if still_hallucinated:
            answer_text = (
                f"{REFUSAL_TEXT} (The retrieved papers "
                f"{', '.join(retrieved_ids)} address part of this question, but "
                f"a reliable, fully-grounded answer could not be generated - "
                f"please review those papers directly.)"
            )
            cited_ids = []

    return {
        "answer": answer_text,
        "retrieved_ids": retrieved_ids,
        "top_score": top_score,
        "skipped_llm": False,
        "corrected": bool(hallucinated),
    }


def run_eval(embed_model, index, chunks, llm):
    """Run every question in the eval set end-to-end and print per-question checks."""
    eval_set = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    for q in eval_set:
        print("=" * 100)
        print(f"{q['id']} ({q['type']}): {q['question']}")
        result = answer_question(embed_model, index, chunks, llm, q["question"])
        if not result["skipped_llm"]:
            time.sleep(SECONDS_BETWEEN_CALLS)

        cited_ids = extract_cited_ids(result["answer"])
        gold = set(q.get("gold_sources", []))
        match = q.get("gold_match", "any")

        print(f"\ntop score: {result['top_score']:.3f}  |  skipped LLM: {result['skipped_llm']}"
              f"  |  correction retried: {result['corrected']}")
        print(f"retrieved: {result['retrieved_ids']}")
        print(f"cited:     {cited_ids}")
        print(f"\nANSWER:\n{result['answer']}")

        # Automated hallucination check, re-verified here in case the
        # in-pipeline correction/fail-safe above still let something through.
        hallucinated = set(cited_ids) - set(result["retrieved_ids"])
        if hallucinated:
            print(f"\n[check] HALLUCINATED CITATION: {sorted(hallucinated)}")

        if match == "none":
            ok = result["skipped_llm"] or REFUSAL_TEXT.lower() in result["answer"].lower()
            print(f"[check] correct refusal: {'YES' if ok else 'NO - investigate'}")
        else:
            cited_gold_overlap = gold.intersection(cited_ids)
            print(f"\n[check] gold cited: {len(cited_gold_overlap)}/{len(gold) if match == 'all' else 1}"
                  f"  |  has_any_citation: {len(cited_ids) > 0}")


def main():
    global GROQ_MODEL_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, default=None,
                         help="ask a single question instead of running the eval set")
    parser.add_argument("--model", type=str, default=None,
                         help="override the Groq model name, e.g. llama-3.3-70b-versatile")
    args = parser.parse_args()

    if args.model:
        GROQ_MODEL_NAME = args.model
        print(f"Using model override: {GROQ_MODEL_NAME}\n")

    index, chunks, embed_model, llm = load_resources()

    if args.ask:
        result = answer_question(embed_model, index, chunks, llm, args.ask)
        print(f"top score: {result['top_score']:.3f}  |  skipped LLM: {result['skipped_llm']}")
        print(f"retrieved: {result['retrieved_ids']}\n")
        print(result["answer"])
    else:
        run_eval(embed_model, index, chunks, llm)


if __name__ == "__main__":
    main()