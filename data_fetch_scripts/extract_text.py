"""
Step 1c - Extract clean text from each downloaded PDF.

Uses column-aware block extraction rather than PyMuPDF's simple top-to-bottom
sort, which interleaves left/right column lines on standard two-column
academic layouts (producing scrambled sentences). Also strips arXiv's margin
watermark and makes a best-effort attempt to trim the references section,
since that's mostly noise for retrieval.

Usage:
    python extract_text.py
"""

import re
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm


METADATA_PATH = Path("data/metadata/papers_metadata.csv")
TEXT_DIR = Path("data/processed/texts")
TEXT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS_FOR_SUCCESS = 1000

# arXiv stamps a vertical watermark down the left margin, e.g.
# "arXiv:2105.05008v1  [cs.IR]  11 May 2021". Column-aware extraction turns
# this into its own block rather than injecting it mid-word, but this still
# strips whatever remains.
ARXIV_WATERMARK = re.compile(
    r"arXiv:\d{4}\.\d{4,5}v?\d*\s*\[[a-z\-\.]+\]\s*\d{1,2}\s+\w+\s+\d{4}",
    re.IGNORECASE,
)


def safe_filename(arxiv_base_id: str) -> str:
    return arxiv_base_id.replace("/", "_") + ".txt"


def clean_extracted_text(text: str) -> str:
    text = ARXIV_WATERMARK.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Best-effort trim of the references section.
    patterns = [
        r"\nReferences\s*\n",
        r"\nREFERENCES\s*\n",
        r"\nBibliography\s*\n",
        r"\nBIBLIOGRAPHY\s*\n",
    ]
    cut_positions = [m.start() for p in patterns for m in [re.search(p, text)] if m]
    if cut_positions:
        text = text[: min(cut_positions)]

    return text.strip()


def extract_page_columns(page) -> str:
    """
    Read a page in column-aware order.

    page.get_text("blocks") returns tuples of (x0, y0, x1, y1, text, block_no,
    block_type). Text blocks (block_type == 0) are split into left/right by
    the page midpoint, and each column is read top-to-bottom, left column
    first. This avoids the line-by-line interleaving that full-width vertical
    sorting produces on two-column academic PDFs.
    """
    page_width = page.rect.width
    mid = page_width / 2.0

    blocks = [
        b for b in page.get_text("blocks")
        if b[6] == 0 and isinstance(b[4], str) and b[4].strip()
    ]
    if not blocks:
        return ""

    left, right = [], []
    for b in blocks:
        x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
        center = (x0 + x1) / 2.0
        width = x1 - x0

        # Full-width blocks (title band, spanning figures/tables) are kept in
        # the left stream so they stay in vertical order rather than being
        # forced into a column that doesn't match their layout.
        if width > 0.7 * page_width or center < mid:
            left.append((y0, btext))
        else:
            right.append((y0, btext))

    left.sort(key=lambda t: t[0])
    right.sort(key=lambda t: t[0])

    ordered = [t[1] for t in left] + [t[1] for t in right]
    return "\n".join(ordered)


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages_text = [extract_page_columns(page) for page in doc]
    doc.close()
    return clean_extracted_text("\n".join(pages_text))


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    extract_statuses = []
    text_paths = []
    text_chars_list = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        arxiv_base_id = row["arxiv_base_id"]
        pdf_path = row.get("pdf_path", "")

        if not isinstance(pdf_path, str) or not pdf_path or not Path(pdf_path).exists():
            extract_statuses.append("missing_pdf")
            text_paths.append("")
            text_chars_list.append(0)
            continue

        output_path = TEXT_DIR / safe_filename(arxiv_base_id)

        try:
            text = extract_text_from_pdf(Path(pdf_path))
            status = "extracted" if len(text) >= MIN_CHARS_FOR_SUCCESS else "too_short"
            output_path.write_text(text, encoding="utf-8")
            extract_statuses.append(status)
            text_paths.append(str(output_path))
            text_chars_list.append(len(text))
        except Exception as e:
            print(f"Failed to extract {pdf_path}: {e}")
            extract_statuses.append("failed")
            text_paths.append("")
            text_chars_list.append(0)

    df["extract_status"] = extract_statuses
    df["text_path"] = text_paths
    df["text_chars"] = text_chars_list

    df.to_csv(METADATA_PATH, index=False)

    print("\nExtraction summary:")
    print(df["extract_status"].value_counts())
    print(f"\nUpdated metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()