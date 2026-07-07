# RecSysRAG

A citation-grounded research assistant over the recommender-systems literature.
Ask a technical question about recommenders (collaborative filtering,
sequential models, learning-to-rank, cold-start, etc.) and get back an answer
built entirely from real arXiv papers, with every claim traceable to a source
- or an honest refusal when the corpus doesn't support an answer.

## Why this project

Most "chat with your PDFs" demos stop at retrieve-then-generate. This one
treats **citation faithfulness** as the actual engineering problem: an
automated check verifies every citation the model produces is one it was
actually given, catches it when the model invents one anyway, and fails safe
- refusing rather than showing an unverified claim - when a correction
attempt doesn't fix it. That behavior is demonstrated and logged, not just
claimed.

## Architecture

```
arXiv API  ─▶  PDF download  ─▶  column-aware text extraction
                                        │
                                        ▼
                              chunk + embed (bge-small-en-v1.5)
                                        │
                                        ▼
                                  FAISS index
                                        │
   question ──▶ retrieve (diversity-capped) ──▶ Groq LLM ──▶ citation check
                                        │                         │
                                        │              hallucination? ──▶ corrective retry
                                        │                         │
                                        │              still hallucinated? ──▶ fail-safe refusal
                                        ▼
                              answer + verified citations
```

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1a | `collect_papers.py` | Query the arXiv API for ~150 recommender-systems papers (broad keyword search + canonical named papers), dedupe, save metadata. |
| 1b | `download_pdfs.py` | Download each paper's PDF. |
| 1c | `extract_text.py` | Extract clean text with column-aware block ordering (avoids the sentence-scrambling that naive top-to-bottom sorting produces on two-column academic PDFs). |
| - | `add_anchor_papers.py` | Optional: fetch specific papers by exact arXiv ID, useful for guaranteeing a canonical paper is in the corpus. |
| 2a | `build_index.py` | Chunk each paper (~180 words, 30-word overlap), embed with `bge-small-en-v1.5`, build a FAISS index. |
| 2b | `eval_retrieval.py` | Retrieval-only evaluation against a hand-written eval set - the gate before adding an LLM. |
| 3 | `answer.py` | Full pipeline: retrieve, generate via Groq, verify citations, self-correct or fail safe. |

## The evaluation set

14 hand-written questions (`data/eval/eval_set.json`), covering four types:

- **Single-paper factual** - e.g. "What problem does BPR loss solve?"
- **Two-paper comparison** - e.g. "How do SASRec and BERT4Rec differ?"
- **Multi-part / compound** - e.g. comparing two models on two distinct axes
- **Refusal cases** - off-topic, fabricated-but-plausible, and out-of-corpus
  questions, where the correct behavior is declining to answer

Each question is tagged with `gold_sources` (the arXiv ID(s) that should
answer it) and `gold_match` (`any`, `all`, or `none`), so both retrieval and
end-to-end answer quality can be scored automatically.

## Retrieval results

At top-k = 5, retrieval hit-rate across the graded questions was **82%**
(9/11), with the two misses being the hardest comparison questions - where a
single query embedding for "compare X and Y" can rank one paper's chunks so
highly that the other paper doesn't surface in a narrow candidate pool. Fixed
in the answer pipeline with a wider raw retrieval pool (`RAW_K = 40`) and a
per-paper cap (`MAX_PER_PAPER = 2`) so a single paper can't crowd out a
genuine comparison.

Refusal calibration: off-topic questions scored ~0.65 similarity; genuinely
on-topic questions scored 0.71-0.91. Plausible-but-fabricated questions
("explain the HyperGraph-BPR model," which doesn't exist) scored *within* the
on-topic range (0.73-0.75) - meaning similarity alone can't catch that case.
That's handled by the prompt's explicit anti-fabrication instruction instead,
not by the score threshold.

## The citation safety mechanism

This is the part worth reading closely if you're evaluating the project:

1. **Retrieve** with a wide, per-paper-capped pool (see above).
2. **Generate** an answer with Groq, instructed to cite only from the
   provided excerpts and to explicitly flag any part of a question the
   excerpts don't cover.
3. **Verify**: every `[arXiv:ID]` the model cites is checked against the set
   of papers actually in its context. This is a hard, checkable fact - no
   LLM judgment involved.
4. **Correct**: if a citation fails verification, one corrective retry tells
   the model exactly what it got wrong and which IDs are valid.
5. **Fail safe**: if the retry still produces an unverified citation, the
   system declines rather than gambling on a second guess - it names the
   retrieved papers and suggests the user check them directly.

This was tested against a genuinely hard compound question ("compare SASRec
and BERT4Rec on training objective *and* inference/serving characteristics")
where the corpus doesn't discuss the second half of the question in those
terms. Both a 70B and an 8B model hallucinated a citation to a real but
unretrieved paper when asked directly. The verification step caught it every
time; the fail-safe produced a clean refusal instead of showing the
fabricated citation.

## Model notes

Answer generation uses Groq (`llama-3.1-8b-instant` by default; override with
`--model llama-3.3-70b-versatile`). Groq's free tier tracks rate limits per
model with separate quota pools, so `answer.py` includes retry-with-backoff
on rate limits and paces requests to stay under the token-per-minute budget
during a full eval run.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key-from-console.groq.com/keys"

python collect_papers.py
python download_pdfs.py
python extract_text.py
python add_anchor_papers.py   # optional
python download_pdfs.py       # re-run to fetch any newly-added anchors
python extract_text.py        # re-run to extract them

python build_index.py
python eval_retrieval.py

python answer.py                          # run the full eval set
python answer.py --ask "your question"    # ask a single question
```

## What's next

- Weights & Biases logging of per-query retrieval scores, latency, token
  usage, and citation-validity outcomes across the eval set.
- Flask API (`POST /ask`, `GET /health`).
- Dockerized deployment on AWS EC2.

## Stack

HuggingFace (`sentence-transformers`, `bge-small-en-v1.5`) for embeddings ·
FAISS for vector search · Groq for hosted LLM inference · PyMuPDF for PDF
extraction · pandas for metadata management.