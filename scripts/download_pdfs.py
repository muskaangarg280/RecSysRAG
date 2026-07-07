import time
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm


METADATA_PATH = Path("data/metadata/papers_metadata.csv")
PDF_DIR = Path("data/raw/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(arxiv_base_id: str) -> str:
    return arxiv_base_id.replace("/", "_") + ".pdf"


def download_pdf(pdf_url: str, output_path: Path) -> bool:
    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        # Basic sanity check
        if "pdf" not in response.headers.get("content-type", "").lower():
            print(f"Warning: response may not be a PDF: {pdf_url}")

        output_path.write_bytes(response.content)
        return True

    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")
        return False


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    statuses = []
    pdf_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        arxiv_base_id = row["arxiv_base_id"]
        pdf_url = row.get("pdf_url", "")

        if not isinstance(pdf_url, str) or not pdf_url:
            statuses.append("missing_pdf_url")
            pdf_paths.append("")
            continue

        output_path = PDF_DIR / safe_filename(arxiv_base_id)

        if output_path.exists() and output_path.stat().st_size > 0:
            statuses.append("already_exists")
            pdf_paths.append(str(output_path))
            continue

        success = download_pdf(pdf_url, output_path)

        if success:
            statuses.append("downloaded")
            pdf_paths.append(str(output_path))
        else:
            statuses.append("failed")
            pdf_paths.append("")

        time.sleep(3.2)  # be gentle with arXiv

    df["download_status"] = statuses
    df["pdf_path"] = pdf_paths

    df.to_csv(METADATA_PATH, index=False)

    print("\nDownload summary:")
    print(df["download_status"].value_counts())
    print(f"\nUpdated metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()