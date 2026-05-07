"""
scripts/download_docs.py
────────────────────────
Downloads freely available regulatory PDFs into data/docs/.

Documents included:
  - Basel III framework (BIS)
  - Basel III: International framework for liquidity risk (BIS)
  - BCBS: Core Principles for Effective Banking Supervision (BIS)

All hosted on bis.org which allows direct downloads.
"""

import sys
from pathlib import Path

import requests
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)

DOCS_DIR = Path("data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENTS = [
    {
        "name": "Basel_III_Framework.pdf",
        "url": "https://www.bis.org/publ/bcbs189.pdf",
        "description": "Basel III: A global regulatory framework for more resilient banks",
    },
    {
        "name": "Basel_III_Liquidity_Framework.pdf",
        "url": "https://www.bis.org/publ/bcbs188.pdf",
        "description": "Basel III: International framework for liquidity risk measurement",
    },
    {
        "name": "Basel_Core_Principles.pdf",
        "url": "https://www.bis.org/publ/bcbs230.pdf",
        "description": "Core Principles for Effective Banking Supervision (Basel Committee)",
    },
]


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        logger.info(f"  Already exists: {dest.name} — skipping.")
        return True

    logger.info(f"  Downloading: {dest.name}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-download/1.0)"}
        response = requests.get(url, timeout=60, stream=True, headers=headers)
        response.raise_for_status()

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = dest.stat().st_size / 1024
        logger.info(f"  ✓ Saved: {dest.name} ({size_kb:.0f} KB)")
        return True

    except requests.RequestException as e:
        logger.error(f"  ✗ Failed to download {dest.name}: {e}")
        logger.warning(
            f"  → You can manually download this document and place it in '{DOCS_DIR}':\n"
            f"    URL: {url}"
        )
        return False


def main() -> None:
    logger.info("=== Downloading regulatory documents ===")
    logger.info(f"Destination: {DOCS_DIR.resolve()}\n")

    success = 0
    for doc in DOCUMENTS:
        logger.info(f"Document: {doc['description']}")
        dest = DOCS_DIR / doc["name"]
        if download_file(doc["url"], dest):
            success += 1
        print()

    logger.info(f"Downloaded {success}/{len(DOCUMENTS)} documents.")

    if success == 0:
        logger.warning(
            "\nNo documents were downloaded automatically.\n"
            f"Please manually place PDF files in: {DOCS_DIR.resolve()}\n"
            "Then run: python -m app.core.ingestion"
        )
    else:
        logger.info("\nNext step: python -m app.core.ingestion")


if __name__ == "__main__":
    main()