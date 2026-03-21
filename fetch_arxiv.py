"""Fetch recent cs.CL abstracts from arXiv and save as JSON for inference input."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


def _clean_abstract(text: str) -> str:
    return " ".join(text.strip().replace("\n", " ").split())


def _extract_paper_id(id_text: str) -> str:
    base = id_text.strip()
    base = re.sub(r"^https?://arxiv.org/abs/", "", base)
    base = re.sub(r"v\d+$", "", base)
    return base


def _fetch_batch(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_arxiv_abstracts(
    target_count: int = 500,
    start_year: int = 2024,
    end_year: int = 2025,
    output_path: str = "abstracts.json",
    delay_seconds: float = 3.0,
) -> list[dict]:
    """
    Fetch cs.CL abstracts from arXiv published between start_year and end_year.

    Args:
        target_count:   Number of unique abstracts to collect
        start_year:     Include papers published from this year (inclusive)
        end_year:       Include papers published up to this year (inclusive)
        output_path:    Where to save the output JSON file
        delay_seconds:  Sleep time between API requests (must be >= 3.0)

    Returns:
        List of {"paper_id": str, "text": str} dicts
    """
    delay_seconds = max(delay_seconds, 3.0)

    base_url = "http://export.arxiv.org/api/query"
    page_size = 100
    start = 0

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
    }

    seen_ids: set[str] = set()
    papers: list[dict] = []
    next_progress = 100

    while len(papers) < target_count:
        params = {
            "search_query": "cat:cs.CL",
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        response_text: str | None = None
        for attempt in range(2):
            try:
                response_text = _fetch_batch(url, timeout=30)
                break
            except Exception as exc:
                print(
                    f"Warning: request failed for start={start} "
                    f"(attempt {attempt + 1}/2): {exc}"
                )
                if attempt == 0:
                    time.sleep(10.0)

        if response_text is None:
            print(f"Skipping batch at start={start} after retry failure.")
            start += page_size
            time.sleep(delay_seconds)
            continue

        try:
            root = ET.fromstring(response_text)
        except ET.ParseError as exc:
            print(f"Warning: XML parse error at start={start}: {exc}")
            start += page_size
            time.sleep(delay_seconds)
            continue

        entries = root.findall("atom:entry", ns)
        if not entries:
            print("No more results returned by arXiv API; stopping.")
            break

        for entry in entries:
            id_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)
            summary_el = entry.find("atom:summary", ns)

            if id_el is None or published_el is None or summary_el is None:
                continue
            if id_el.text is None or published_el.text is None or summary_el.text is None:
                continue

            paper_id = _extract_paper_id(id_el.text)
            if not paper_id or paper_id in seen_ids:
                continue

            year_text = published_el.text[:4]
            if not year_text.isdigit():
                continue
            year = int(year_text)
            if not (start_year <= year <= end_year):
                continue

            abstract_text = _clean_abstract(summary_el.text)
            if not abstract_text:
                continue

            seen_ids.add(paper_id)
            papers.append({"paper_id": paper_id, "text": abstract_text})

            if len(papers) >= next_progress:
                print(f"Fetched {len(papers)} / {target_count} papers...")
                next_progress += 100

            if len(papers) >= target_count:
                break

        start += page_size
        time.sleep(delay_seconds)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"Final count: {len(papers)}")
    print(f"Saved output to: {output_path}")

    return papers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch cs.CL abstracts from arXiv API")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--output", type=str, default="abstracts.json")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--delay", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fetch_arxiv_abstracts(
        target_count=args.target,
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=args.output,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()
