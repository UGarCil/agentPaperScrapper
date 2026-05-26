"""
semantic_scholar_downloader.py
------------------------------
A script that searches Semantic Scholar for papers matching a query and
downloads their open-access PDFs to a specified output directory.

Key difference from ArXiv: Semantic Scholar is a metadata aggregator, not
a preprint server. Not every paper has a freely downloadable PDF. This
script filters for papers that expose an open-access PDF link and falls
back to ArXiv when a direct download is blocked by the publisher.

A metadata.json file is always saved to the output directory, recording
title, authors, year, DOI, ArXiv ID, citation count, and download status
for every paper — including those whose PDFs could not be retrieved.

Dependencies:
    pip install semanticscholar

Optional:
    A free Semantic Scholar API key raises the rate limit from ~1 req/s to
    ~100 req/s. Request one at: https://www.semanticscholar.org/product/api
    The script works without one but will be slower for large result sets.

Usage:
    Run the script directly and follow the interactive prompts.
    You will be asked for:
        - A search query       (e.g. "multi-agent reinforcement learning")
        - The number of papers to download
        - An output directory  (created automatically if it doesn't exist)
        - An optional API key  (press Enter to skip)
"""

import json
import os
import time
import urllib.request
import urllib.error

from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fields we request from the API. Asking only for what we need keeps
# responses small and reduces latency.
# 'openAccessPdf'  – direct PDF URL when freely available
# 'externalIds'    – contains ArXiv ID used as fallback download source
# 'citationCount'  – saved to metadata for reference
FIELDS = ["title", "authors", "year", "openAccessPdf", "externalIds", "citationCount"]

# Seconds to pause between individual PDF download requests.
# Raised from 2 s to 5 s to reduce HTTP 429 (Too Many Requests) errors
# from publisher servers, which are more aggressive than ArXiv.
DOWNLOAD_DELAY_SECONDS = 5.0

# Timeout (seconds) for the SemanticScholar API client.
API_TIMEOUT_SECONDS = 30

# ArXiv PDF URL template used as a fallback when the direct link is blocked.
# The ArXiv PDF endpoint is always open-access for papers hosted there.
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_client(api_key: str | None) -> SemanticScholar:
    """Create and return a configured SemanticScholar API client.

    The client has retry mode enabled by default, which automatically
    handles HTTP 429 responses with exponential back-off (up to 10 retries,
    starting at 5 s and doubling up to 60 s).

    Args:
        api_key: Optional Semantic Scholar API key. Passing one raises the
                 rate limit from ~1 request/s to ~100 requests/s. Pass
                 None to use the unauthenticated tier.

    Returns:
        A ready-to-use SemanticScholar client instance.
    """
    if api_key:
        return SemanticScholar(api_key=api_key, timeout=API_TIMEOUT_SECONDS)
    return SemanticScholar(timeout=API_TIMEOUT_SECONDS)


def search_papers(
    sch: SemanticScholar,
    query: str,
    max_results: int,
) -> list[Paper]:
    """Search Semantic Scholar for papers matching *query*.

    Uses open_access_pdf=True to restrict results to papers that have a
    known PDF link, and requests citationCount so metadata is complete.

    Args:
        sch:         An initialised SemanticScholar client.
        query:       Keyword query string. Supports Boolean operators:
                     use '+' for AND, '|' for OR, '-' to exclude a term.
                     Example: "transformer + language model - vision"
        max_results: Maximum number of open-access papers to retrieve.

    Returns:
        A list of Paper objects sorted by Semantic Scholar's relevance
        ranking (default), each expected to have an openAccessPdf field.
    """
    results = sch.search_paper(
        query,
        limit=min(max_results, 100),   # API hard cap is 100 per page
        fields=FIELDS,
        open_access_pdf=True,
    )

    # Exhaust the paginated generator up to max_results items.
    papers = []
    for paper in results:
        papers.append(paper)
        if len(papers) >= max_results:
            break

    return papers


def sanitize_filename(title: str) -> str:
    """Convert a paper title into a filesystem-safe filename.

    Args:
        title: The raw paper title string.

    Returns:
        A safe filename: spaces become underscores, special characters are
        stripped, and the result is capped at 100 characters.
    """
    safe = title.replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in ("_", "-", "."))
    return safe[:100]


def get_pdf_url(paper: Paper) -> str | None:
    """Extract the open-access PDF URL from a Paper object, if available.

    Args:
        paper: A semanticscholar Paper object.

    Returns:
        The PDF URL string, or None if no open-access PDF is listed.
    """
    oa = paper.openAccessPdf
    if not oa:
        return None
    # openAccessPdf is a dict: {'url': 'https://...', 'status': 'GREEN'/...}
    return oa.get("url") if isinstance(oa, dict) else None


def get_arxiv_id(paper: Paper) -> str | None:
    """Extract the ArXiv paper ID from a Paper's externalIds field.

    Args:
        paper: A semanticscholar Paper object.

    Returns:
        The ArXiv ID string (e.g. '2301.07041'), or None if not present.
    """
    ids = paper.externalIds
    if not ids or not isinstance(ids, dict):
        return None
    return ids.get("ArXiv")


def try_download(url: str, filepath: str) -> bool:
    """Attempt to download a PDF from *url* and save it to *filepath*.

    Args:
        url:      The URL of the PDF to download.
        filepath: The full local path to save the file to.

    Returns:
        True if the download succeeded, False otherwise.
    """
    try:
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception:  # noqa: BLE001
        # Remove any partial file left behind by a failed download
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def build_metadata_record(paper: Paper, status: str) -> dict:
    """Build a metadata dictionary for one paper to be saved in the JSON log.

    Captures all useful bibliographic fields plus the final download status,
    so papers that could not be downloaded can still be located manually.

    Args:
        paper:  A semanticscholar Paper object.
        status: A short string describing the outcome, one of:
                'downloaded_direct', 'downloaded_arxiv', 'skipped', or
                'already_exists'.

    Returns:
        A plain dict suitable for JSON serialisation.
    """
    # Flatten the authors list into a list of name strings
    authors = []
    if paper.authors:
        authors = [a.name for a in paper.authors if hasattr(a, "name")]

    # Pull DOI and ArXiv ID from externalIds if present
    ext = paper.externalIds or {}
    doi      = ext.get("DOI")
    arxiv_id = ext.get("ArXiv")

    return {
        "title":          paper.title,
        "authors":        authors,
        "year":           paper.year,
        "doi":            doi,
        "arxiv_id":       arxiv_id,
        "citation_count": getattr(paper, "citationCount", None),
        "status":         status,
    }


def save_metadata(records: list[dict], output_dir: str) -> None:
    """Write all paper metadata records to a JSON file in *output_dir*.

    The file is named 'metadata.json' and is always written, even when
    no PDFs were successfully downloaded. This lets the user manually
    locate papers that could not be retrieved automatically.

    Args:
        records:    List of metadata dicts produced by build_metadata_record.
        output_dir: Directory where metadata.json will be saved.
    """
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to: {metadata_path}")


def download_papers(papers: list[Paper], output_dir: str) -> list[dict]:
    """Download PDFs for each paper, falling back to ArXiv when blocked.

    Download strategy per paper:
      1. Try the direct open-access URL from Semantic Scholar.
      2. If that fails, check for an ArXiv ID and try arxiv.org/pdf/{id}.
      3. If both fail, mark the paper as skipped.

    A delay of DOWNLOAD_DELAY_SECONDS is applied between every download
    attempt to reduce HTTP 429 errors from publisher servers.

    Args:
        papers:     List of semanticscholar Paper objects to process.
        output_dir: Destination directory for PDFs. Created if absent.

    Returns:
        A list of metadata dicts (one per paper) recording the outcome,
        to be passed to save_metadata().
    """
    os.makedirs(output_dir, exist_ok=True)

    total      = len(papers)
    downloaded = 0
    skipped    = 0
    metadata_records = []

    for idx, paper in enumerate(papers, start=1):
        title = paper.title or f"paper_{idx}"
        print(f"[{idx}/{total}] {title}")

        filename = sanitize_filename(title) + ".pdf"
        filepath = os.path.join(output_dir, filename)

        # ── Already on disk ──────────────────────────────────────────────
        if os.path.exists(filepath):
            print(f"  -> Already exists, skipping: {filename}\n")
            metadata_records.append(build_metadata_record(paper, "already_exists"))
            skipped += 1
            # Still apply delay so subsequent downloads are spaced out
            if idx < total:
                time.sleep(DOWNLOAD_DELAY_SECONDS)
            continue

        pdf_url  = get_pdf_url(paper)
        arxiv_id = get_arxiv_id(paper)
        status   = "skipped"

        # ── Attempt 1: direct open-access URL ────────────────────────────
        if pdf_url:
            if try_download(pdf_url, filepath):
                print(f"  -> Saved (direct): {filename}\n")
                status = "downloaded_direct"
                downloaded += 1
            else:
                print("  -> Direct download failed, trying ArXiv fallback...")

        # ── Attempt 2: ArXiv fallback ─────────────────────────────────────
        # Reached when there was no direct URL, or the direct download failed.
        if status == "skipped" and arxiv_id:
            arxiv_url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
            if try_download(arxiv_url, filepath):
                print(f"  -> Saved (ArXiv fallback): {filename}\n")
                status = "downloaded_arxiv"
                downloaded += 1
            else:
                print("  -> ArXiv fallback also failed.\n")

        # ── No PDF available ──────────────────────────────────────────────
        if status == "skipped":
            if not pdf_url and not arxiv_id:
                print("  -> No PDF source available, skipping.\n")
            else:
                print("  -> Could not download from any source, skipping.\n")
            skipped += 1

        metadata_records.append(build_metadata_record(paper, status))

        # Pause between every paper to respect server rate limits
        if idx < total:
            time.sleep(DOWNLOAD_DELAY_SECONDS)

    print(f"\nSummary: {downloaded} downloaded, {skipped} skipped out of {total} papers.")
    return metadata_records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the interactive Semantic Scholar download workflow.

    Prompts for inputs, searches the API, downloads open-access PDFs with
    an ArXiv fallback, and saves a metadata.json for every paper.
    """
    print("=" * 55)
    print("       Semantic Scholar Paper Downloader")
    print("=" * 55)

    query      = "Multi agent systems MAS"
    num_papers = 50
    output_dir = "./semanticsholar"
    api_key    = "s2k-PvyJuBSdBjXMtnyUCXHcriER4MsRf4yu5aejR80X"

    sch = build_client(api_key)

    print(f'\nSearching Semantic Scholar for "{query}" '
          f'(up to {num_papers} open-access results)...\n')

    try:
        papers = search_papers(sch, query, max_results=num_papers)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Search failed — {exc}")
        print("Check your internet connection or try again in a moment.")
        return

    if not papers:
        print("No open-access papers found. Try a broader query.")
        return

    print(f"Found {len(papers)} paper(s). Starting downloads to '{output_dir}'...\n")

    # Download PDFs and collect metadata for every paper
    metadata_records = download_papers(papers, output_dir)

    # Always save metadata, even for papers that couldn't be downloaded
    save_metadata(metadata_records, output_dir)

    print("=" * 55)
    print(f"Done! PDFs saved to: {os.path.abspath(output_dir)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
