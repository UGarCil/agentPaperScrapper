"""
arxiv_downloader.py
-------------------
A script that searches ArXiv for papers matching a query and downloads
their PDFs to a specified output directory.

Dependencies:
    pip install arxiv

Usage:
    Run the script directly and follow the interactive prompts.
    You will be asked for:
        - A search query  (e.g. "transformer language models")
        - The number of papers to download
        - An output directory path (created automatically if it doesn't exist)
"""

import os
import time
import urllib.request
import arxiv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Sort papers by relevance to the query. Other options:
#   arxiv.SortCriterion.SubmittedDate  – newest first
#   arxiv.SortCriterion.LastUpdatedDate – most recently revised first
SORT_BY = arxiv.SortCriterion.Relevance

# Seconds to wait between successive API requests.
# ArXiv's fair-use guidelines ask for at least 3 s; we use 5 to be safe.
API_DELAY_SECONDS = 5.0

# How many times to retry the search if ArXiv returns HTTP 429 (rate-limited).
MAX_SEARCH_RETRIES = 4

# Starting back-off wait in seconds; doubles on each subsequent retry.
# Sequence: 10 s → 20 s → 40 s → give up
BACKOFF_BASE_SECONDS = 10


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def search_papers(query: str, max_results: int) -> list[arxiv.Result]:
    """Search ArXiv for papers matching *query* and return up to *max_results*.

    Creates a fresh client for each call so that page_size never exceeds
    max_results — requesting fewer results per page significantly reduces
    the chance of hitting ArXiv's rate-limiter (HTTP 429).

    Retries with exponential back-off if the API returns HTTP 429, waiting
    BACKOFF_BASE_SECONDS * 2^attempt seconds between each attempt.

    Args:
        query:       An ArXiv query string. Plain keywords work well
                     (e.g. "quantum computing error correction").
                     Advanced syntax is documented at:
                     https://arxiv.org/help/api/user-manual#query_details
        max_results: Maximum number of results to retrieve from the API.

    Returns:
        A list of arxiv.Result objects sorted by SORT_BY.

    Raises:
        arxiv.HTTPError: If the API keeps returning 429 after all retries.
    """
    # page_size = max_results avoids fetching an oversized first page.
    # For example, asking for 5 papers should not open with a 100-result
    # request — that's what triggered the 429 in the original version.
    client = arxiv.Client(
        page_size=max_results,        # fetch exactly as many as needed per page
        delay_seconds=API_DELAY_SECONDS,
        num_retries=1,                # we handle outer retries ourselves below
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=SORT_BY,
    )

    # Exponential back-off retry loop around the API call.
    for attempt in range(MAX_SEARCH_RETRIES):
        try:
            # client.results() returns a lazy generator; exhaust it into a
            # list so callers always receive a concrete collection.
            return list(client.results(search))

        except arxiv.HTTPError as exc:
            if exc.status != 429 or attempt == MAX_SEARCH_RETRIES - 1:
                # Non-rate-limit error, or we have exhausted retries — re-raise
                raise

            # Calculate how long to wait before the next attempt
            wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(
                f"  Rate-limited by ArXiv (HTTP 429). "
                f"Waiting {wait}s before retry {attempt + 1}/{MAX_SEARCH_RETRIES - 1}…"
            )
            time.sleep(wait)

    # Unreachable, but satisfies type checkers
    return []


def sanitize_filename(title: str) -> str:
    """Convert a paper title into a safe filename by removing special characters.

    Args:
        title: The raw paper title string.

    Returns:
        A filesystem-safe version of the title (spaces replaced with
        underscores, special characters stripped, length capped at 100 chars).
    """
    # Replace spaces with underscores for readability
    safe = title.replace(" ", "_")

    # Keep only alphanumerics, underscores, hyphens, and dots
    safe = "".join(c for c in safe if c.isalnum() or c in ("_", "-", "."))

    # Truncate to avoid hitting OS filename-length limits
    return safe[:100]


def download_papers(results: list[arxiv.Result], output_dir: str) -> None:
    """Download PDFs for each result in *results* to *output_dir*.

    Skips a paper if its PDF has already been downloaded (checks for an
    existing file with the same name). Prints progress and any errors
    to stdout so the user stays informed.

    Args:
        results:    List of arxiv.Result objects to download.
        output_dir: Path to the directory where PDFs will be saved.
                    Created automatically if it does not exist.
    """
    # Ensure the destination directory exists
    os.makedirs(output_dir, exist_ok=True)

    total = len(results)

    for idx, paper in enumerate(results, start=1):
        # Build a human-readable filename: "<sanitized_title>.pdf"
        filename = sanitize_filename(paper.title) + ".pdf"
        filepath = os.path.join(output_dir, filename)

        print(f"[{idx}/{total}] {paper.title}")

        # Skip if already downloaded to avoid redundant network requests
        if os.path.exists(filepath):
            print(f"  -> Already exists, skipping: {filename}\n")
            continue

        try:
            # paper.pdf_url is the direct link to the PDF and is available
            # across all versions of the arxiv library. We use urllib instead
            # of Result.download_pdf() to avoid version compatibility issues.
            urllib.request.urlretrieve(paper.pdf_url, filepath)
            print(f"  -> Saved: {filename}\n")
        except Exception as exc:  # noqa: BLE001
            # Catch-all so one failed download does not abort the whole run
            print(f"  -> ERROR downloading paper: {exc}\n")

        # Pause between downloads so back-to-back PDF requests don't trigger
        # ArXiv's rate-limiter. Skipped after the last paper to avoid a
        # pointless wait at the end of the run.
        if idx < total:
            time.sleep(API_DELAY_SECONDS)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the interactive ArXiv download workflow.

    Prompts the user for a query, paper count, and output directory, then
    searches ArXiv and downloads the matching PDFs.
    """
    print("=" * 50)
    print("         ArXiv Paper Downloader")
    print("=" * 50)

    # Collect inputs interactively
    query      = "Multi Agent Systems MAS"
    num_papers = 20
    output_dir = "./arxiv_papers"

    print(f'\nSearching ArXiv for "{query}" (up to {num_papers} results)...\n')

    # Fetch matching papers from the ArXiv API
    try:
        results = search_papers(query, max_results=num_papers)
    except arxiv.HTTPError as exc:
        print(f"ERROR: ArXiv API request failed after retries ({exc})")
        print("Try again in a few minutes — ArXiv may be temporarily throttling requests.")
        return

    if not results:
        print("No results found. Try broadening your search query.")
        return

    print(f"Found {len(results)} paper(s). Starting downloads to '{output_dir}'...\n")

    # Download PDFs
    download_papers(results, output_dir)

    print("=" * 50)
    print(f"Done! PDFs saved to: {os.path.abspath(output_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
