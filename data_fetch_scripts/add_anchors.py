"""
Optional utility - fetch and append specific papers by exact arXiv ID.

Useful for guaranteeing a canonical paper is in the corpus (e.g. one that
doesn't reliably surface through keyword search) or for adding a paper found
after the initial collection pass. Papers already present in the metadata are
skipped rather than duplicated.

Usage:
    python add_anchor_papers.py
"""

import re
from pathlib import Path
from urllib.parse import urlencode

import feedparser
import pandas as pd
import requests


BASE_URL = "http://export.arxiv.org/api/query"
METADATA_PATH = Path("data/metadata/papers_metadata.csv")

# arXiv IDs to guarantee are in the corpus, with a short label for logging.
ANCHOR_IDS = {
    "1205.2618": "BPR: Bayesian Personalized Ranking",
    "1708.05031": "Neural Collaborative Filtering",
    "1606.07792": "Wide & Deep Learning for Recommender Systems",
    "1808.09781": "Self-Attentive Sequential Recommendation (SASRec)",
    "1904.06690": "BERT4Rec",
}
# Note: "Deep Neural Networks for YouTube Recommendations" (Covington et al.,
# 2016) is not on arXiv (ACM-only) and can't be fetched this way.


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_arxiv_id(entry_id: str) -> str:
    return entry_id.split("/abs/")[-1].strip()


def get_pdf_url(entry) -> str:
    for link in entry.get("links", []):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            return link.get("href")
    return ""


def fetch_by_ids(ids):
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    url = f"{BASE_URL}?{urlencode(params)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    rows = []
    for entry in feed.entries:
        arxiv_id = normalize_arxiv_id(entry.id)
        arxiv_base_id = re.sub(r"v\d+$", "", arxiv_id)
        authors = [a.name for a in entry.get("authors", [])]
        categories = [t["term"] for t in entry.get("tags", [])]
        published = entry.get("published", "")

        rows.append({
            "arxiv_id": arxiv_id,
            "arxiv_base_id": arxiv_base_id,
            "title": clean_text(entry.get("title", "")),
            "authors": "; ".join(authors),
            "year": published[:4] if published else "",
            "published": published,
            "updated": entry.get("updated", ""),
            "abstract": clean_text(entry.get("summary", "")),
            "categories": "; ".join(categories),
            "abs_url": entry.get("id", ""),
            "pdf_url": get_pdf_url(entry),
            "source_query": "anchor:id_list",
            "is_anchor": True,
            "pdf_path": "",
            "text_path": "",
            "download_status": "pending",
            "extract_status": "pending",
            "text_chars": 0,
        })
    return rows


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing metadata: {METADATA_PATH}. Run collect_papers.py first."
        )

    df = pd.read_csv(METADATA_PATH)
    if "is_anchor" not in df.columns:
        df["is_anchor"] = False

    print("Fetching anchor papers by arXiv ID...")
    anchor_df = pd.DataFrame(fetch_by_ids(list(ANCHOR_IDS.keys())))

    print("\nAnchors returned by arXiv:")
    for _, r in anchor_df.iterrows():
        print(f"  {r['arxiv_base_id']:<12} {r['title']}")

    returned_ids = set(anchor_df["arxiv_base_id"])
    missing = [i for i in ANCHOR_IDS if i not in returned_ids]
    if missing:
        print(f"\nWARNING: these anchor IDs did not come back: {missing}")

    existing_ids = set(df["arxiv_base_id"])
    df.loc[df["arxiv_base_id"].isin(returned_ids), "is_anchor"] = True
    new_anchors = anchor_df[~anchor_df["arxiv_base_id"].isin(existing_ids)]

    combined = pd.concat([df, new_anchors], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["arxiv_base_id"], keep="first"
    ).reset_index(drop=True)

    combined.to_csv(METADATA_PATH, index=False)

    print(f"\nAnchors already present: {len(returned_ids) - len(new_anchors)}")
    print(f"New anchors appended:    {len(new_anchors)}")
    print(f"Total papers now:        {len(combined)}")
    print("\nNext: re-run download_pdfs.py, then extract_text.py")


if __name__ == "__main__":
    main()