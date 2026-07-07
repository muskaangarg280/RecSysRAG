"""
Step 1a - Collect paper metadata from arXiv.

Queries the arXiv API across a set of recommender-systems keyword searches
plus a small list of canonical named papers (BPR, NCF, SASRec, BERT4Rec,
Wide & Deep), merges and deduplicates the results by base arXiv ID, and saves
one row per paper to a metadata CSV. This CSV is the backbone of the whole
pipeline - every later step (download, extraction, indexing, citations) keys
off arxiv_base_id.

Usage:
    python collect_papers.py
"""

import re
import time
from pathlib import Path
from urllib.parse import urlencode

import feedparser
import pandas as pd
import requests


BASE_URL = "http://export.arxiv.org/api/query"

OUTPUT_PATH = Path("data/metadata/papers_metadata.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Cap on the final candidate set size. Applied only to the broad keyword
# queries below - named/canonical papers are never subject to this cap (see
# main()), so a well-known paper can't be silently dropped just because the
# keyword queries happened to return enough other results first.
MAX_CANDIDATE_PAPERS = 150

# Broad recommender-systems queries. Several smaller queries are used instead
# of one large one because arXiv's query syntax is sensitive to combining too
# many terms in a single request.
QUERIES = [
    'cat:cs.IR AND all:"recommender systems"',
    'cat:cs.IR AND all:"recommendation"',
    'cat:cs.IR AND all:"collaborative filtering"',
    'cat:cs.IR AND all:"implicit feedback"',
    'cat:cs.IR AND all:"matrix factorization"',
    'cat:cs.IR AND all:"cold start recommendation"',
    'cat:cs.IR AND all:"learning to rank"',
    'cat:cs.IR AND all:"neural recommendation"',
    'cat:cs.IR AND all:"sequential recommendation"',
    'cat:cs.IR AND all:"session based recommendation"',
    'cat:cs.IR AND all:"candidate generation"',
    'cat:cs.IR AND all:"two tower recommendation"',
    'cat:cs.LG AND all:"recommender systems"',
    'cat:cs.LG AND all:"neural collaborative filtering"',
    'cat:cs.LG AND all:"sequential recommendation"',
]

# Canonical papers that anchor the corpus and back the evaluation set's gold
# sources. These are searched by title and are exempt from MAX_CANDIDATE_PAPERS.
NAMED_QUERIES = [
    'all:"Bayesian Personalized Ranking"',
    'all:"Neural Collaborative Filtering"',
    'all:"Wide Deep Learning Recommender Systems"',
    'all:"Self-Attentive Sequential Recommendation"',
    'all:"BERT4Rec"',
    'all:"Matrix Factorization Techniques for Recommender Systems"',
]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_arxiv_id(entry_id: str) -> str:
    """entry_id looks like 'http://arxiv.org/abs/1708.05031v2'; keep the version."""
    return entry_id.split("/abs/")[-1].strip()


def get_pdf_url(entry) -> str:
    for link in entry.get("links", []):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            return link.get("href")
    return ""


def fetch_query(query: str, max_results: int = 30):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    url = f"{BASE_URL}?{urlencode(params)}"
    print(f"\nQuerying: {query}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    feed = feedparser.parse(response.text)
    results = []

    for entry in feed.entries:
        arxiv_id = normalize_arxiv_id(entry.id)
        arxiv_base_id = re.sub(r"v\d+$", "", arxiv_id)

        authors = [author.name for author in entry.get("authors", [])]
        categories = [tag["term"] for tag in entry.get("tags", [])]

        published = entry.get("published", "")
        year = published[:4] if published else ""

        results.append({
            "arxiv_id": arxiv_id,
            "arxiv_base_id": arxiv_base_id,
            "title": clean_text(entry.get("title", "")),
            "authors": "; ".join(authors),
            "year": year,
            "published": published,
            "updated": entry.get("updated", ""),
            "abstract": clean_text(entry.get("summary", "")),
            "categories": "; ".join(categories),
            "abs_url": entry.get("id", ""),
            "pdf_url": get_pdf_url(entry),
            "source_query": query,
            "pdf_path": "",
            "text_path": "",
            "download_status": "pending",
            "extract_status": "pending",
            "text_chars": 0,
        })

    print(f"Found {len(results)} results")
    return results


def main():
    named_rows = []
    for query in NAMED_QUERIES:
        named_rows.extend(fetch_query(query, max_results=5))
        time.sleep(3.2)  # arXiv asks for no more than one request every 3 seconds

    broad_rows = []
    for query in QUERIES:
        broad_rows.extend(fetch_query(query, max_results=20))
        time.sleep(3.2)

    named_df = pd.DataFrame(named_rows).drop_duplicates(subset=["arxiv_base_id"])
    broad_df = pd.DataFrame(broad_rows).drop_duplicates(subset=["arxiv_base_id"])

    if named_df.empty and broad_df.empty:
        print("No papers found. Check queries or network connection.")
        return

    # Named papers go in first and are never truncated, so a canonical paper
    # can't be dropped just because the broad queries filled the candidate
    # cap first. Broad results are then deduplicated against the named set
    # and capped to keep the corpus a manageable size.
    broad_df = broad_df[~broad_df["arxiv_base_id"].isin(named_df["arxiv_base_id"])]
    remaining_slots = max(MAX_CANDIDATE_PAPERS - len(named_df), 0)
    broad_df = broad_df.head(remaining_slots)

    df = pd.concat([named_df, broad_df], ignore_index=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} unique papers to {OUTPUT_PATH}")
    print(f"  named/canonical: {len(named_df)}")
    print(f"  broad keyword:   {len(broad_df)}")


if __name__ == "__main__":
    main()