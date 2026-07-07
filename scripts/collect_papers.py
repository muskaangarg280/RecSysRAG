import time
import re
from pathlib import Path
from urllib.parse import urlencode

import requests
import feedparser
import pandas as pd


BASE_URL = "http://export.arxiv.org/api/query"

OUTPUT_PATH = Path("data/metadata/papers_metadata.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Broad recommender-system queries.
# We use multiple smaller queries instead of one giant query because arXiv query syntax can be sensitive.
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

# Canonical papers / topics you want to force into the corpus if arXiv has them.
NAMED_QUERIES = [
    'all:"Bayesian Personalized Ranking"',
    'all:"Neural Collaborative Filtering"',
    'all:"Wide Deep Learning Recommender Systems"',
    'all:"Deep Neural Networks for YouTube Recommendations"',
    'all:"Self-Attentive Sequential Recommendation"',
    'all:"BERT4Rec"',
    'all:"Matrix Factorization Techniques for Recommender Systems"',
]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def normalize_arxiv_id(entry_id: str) -> str:
    """
    entry_id usually looks like:
    http://arxiv.org/abs/1708.05031v2
    We keep the versioned ID for exactness, but also create a base id later.
    """
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
    all_rows = []

    for query in QUERIES:
        rows = fetch_query(query, max_results=20)
        all_rows.extend(rows)
        time.sleep(3.2)  # arXiv asks for no more than one request every 3 seconds

    for query in NAMED_QUERIES:
        rows = fetch_query(query, max_results=5)
        all_rows.extend(rows)
        time.sleep(3.2)

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("No papers found. Check queries or network connection.")
        return

    # Deduplicate by base arXiv ID so v1/v2 don't duplicate the same paper.
    df = df.drop_duplicates(subset=["arxiv_base_id"]).reset_index(drop=True)

    # Keep a manageable candidate set.
    # You can increase this later.
    df = df.head(150)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} unique papers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()