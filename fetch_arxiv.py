"""Fetch cs.CL abstracts from arXiv and save as JSON for inference input."""

from __future__ import annotations

import argparse
import json
import os
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


def _format_date_for_arxiv(date_str: str, end_of_day: bool = False) -> str:
    """
    Convert YYYYMMDD to arXiv date format with time.
    
    Args:
        date_str: Date in YYYYMMDD format
        end_of_day: If True, use 2359 (end of day), else use 0000 (start of day)
    
    Returns:
        Formatted date string like "202401010000" or "202401012359"
    """
    time_part = "2359" if end_of_day else "0000"
    return f"{date_str}{time_part}"


def _date_in_range(date_str: str, date_from: str, date_to: str) -> bool:
    """
    Check if a date (YYYY-MM-DD format) falls within the range.
    
    Args:
        date_str: Date string in ISO format (YYYY-MM-DD)
        date_from: Start date in YYYYMMDD format
        date_to: End date in YYYYMMDD format
    
    Returns:
        True if date_str falls within the range
    """
    # Convert ISO date to YYYYMMDD for comparison
    iso_to_compact = date_str.replace("-", "")  # YYYY-MM-DD -> YYYYMMDD
    return date_from <= iso_to_compact <= date_to


def fetch_arxiv_abstracts(
    target_count: int = 500,
    start_year: int = 2024,
    end_year: int = 2025,
    output_path: str = "json/abstracts.json",
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

    os.makedirs("json", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"Final count: {len(papers)}")
    print(f"Saved output to: {output_path}")

    return papers


def fetch_date_slice(
    category: str,
    date_from: str,
    date_to: str,
    target_count: int = 1000,
    delay_seconds: float = 3.0,
) -> list[dict]:
    """
    Fetch up to target_count abstracts from a specific category and date range.
    
    Uses arXiv submittedDate range filter to enforce strict date bounds.
    
    Args:
        category:      arXiv category string, e.g. "cs.CL", "cs.LG", "cs.CV"
        date_from:     Start date in YYYYMMDD format, e.g. "20240101"
        date_to:       End date in YYYYMMDD format, e.g. "20241231"
        target_count:  Target number of abstracts to collect
        delay_seconds: Sleep time between API requests (must be >= 3.0)
    
    Returns:
        List of {"paper_id": str, "text": str} dicts, deduplicated by paper_id
    """
    delay_seconds = max(delay_seconds, 3.0)
    
    base_url = "http://export.arxiv.org/api/query"
    page_size = 100
    start = 0
    
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
    }
    
    # Build date range filter for arXiv API
    date_from_fmt = _format_date_for_arxiv(date_from, end_of_day=False)
    date_to_fmt = _format_date_for_arxiv(date_to, end_of_day=True)
    
    seen_ids: set[str] = set()
    papers: list[dict] = []
    next_progress = 100
    
    while len(papers) < target_count:
        # Combine category and date range filters with proper spacing
        search_query = f"cat:{category} AND submittedDate:[{date_from_fmt} TO {date_to_fmt}]"
        
        params = {
            "search_query": search_query,
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
                    f"[{category} {date_from}-{date_to}] Warning: request failed "
                    f"at start={start} (attempt {attempt + 1}/2): {exc}"
                )
                if attempt == 0:
                    time.sleep(10.0)
        
        if response_text is None:
            print(
                f"[{category} {date_from}-{date_to}] Skipping batch at start={start} "
                f"after retry failure."
            )
            start += page_size
            time.sleep(delay_seconds)
            continue
        
        try:
            root = ET.fromstring(response_text)
        except ET.ParseError as exc:
            print(
                f"[{category} {date_from}-{date_to}] Warning: XML parse error "
                f"at start={start}: {exc}"
            )
            start += page_size
            time.sleep(delay_seconds)
            continue
        
        entries = root.findall("atom:entry", ns)
        if not entries:
            print(
                f"[{category} {date_from}-{date_to}] No more results returned "
                f"by arXiv API; stopping."
            )
            break
        
        for entry in entries:
            id_el = entry.find("atom:id", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)
            
            if id_el is None or summary_el is None or published_el is None:
                continue
            if id_el.text is None or summary_el.text is None or published_el.text is None:
                continue
            
            # Extract and validate date (post-fetch filtering for reliability)
            date_text = published_el.text[:10]  # Extract YYYY-MM-DD
            if not _date_in_range(date_text, date_from, date_to):
                continue
            
            paper_id = _extract_paper_id(id_el.text)
            if not paper_id or paper_id in seen_ids:
                continue
            
            abstract_text = _clean_abstract(summary_el.text)
            if not abstract_text:
                continue
            
            seen_ids.add(paper_id)
            papers.append({"paper_id": paper_id, "text": abstract_text})
            
            if len(papers) >= next_progress:
                print(f"[{category} {date_from}-{date_to}] Fetched {len(papers)} papers...")
                next_progress += 100
            
            if len(papers) >= target_count:
                break
        
        start += page_size
        time.sleep(delay_seconds)
    
    print(f"[{category} {date_from}-{date_to}] Final count: {len(papers)}")
    return papers


def fetch_balanced(
    category: str,
    slices: list[tuple[str, str, int]],
    output_path: str,
    delay_seconds: float = 3.0,
) -> list[dict]:
    """
    Fetch multiple date slices and combine into one deduplicated output file.
    
    Args:
        category:      arXiv category string, e.g. "cs.CL"
        slices:        List of (date_from, date_to, target_count) tuples
                       e.g. [("20240101", "20241231", 1000),
                             ("20250101", "20251231", 1000)]
        output_path:   Where to save combined JSON
        delay_seconds: Sleep time between API requests (must be >= 3.0)
    
    Returns:
        Combined deduplicated list of {"paper_id": str, "text": str} dicts
    """
    all_papers: list[dict] = []
    seen_ids: set[str] = set()
    slice_counts: list[int] = []  # Track papers per slice
    
    for i, (date_from, date_to, target_count) in enumerate(slices):
        if i > 0:
            print(f"\nWaiting 5 seconds before fetching next slice...")
            time.sleep(5.0)
        
        print(f"\n--- Slice {i+1}/{len(slices)}: {date_from} to {date_to} ---")
        slice_papers = fetch_date_slice(
            category=category,
            date_from=date_from,
            date_to=date_to,
            target_count=target_count,
            delay_seconds=delay_seconds,
        )
        
        # Count new papers (not in global seen set before this slice)
        new_count = 0
        for paper in slice_papers:
            if paper["paper_id"] not in seen_ids:
                seen_ids.add(paper["paper_id"])
                all_papers.append(paper)
                new_count += 1
        
        slice_counts.append(new_count)
    
    # Save combined results
    os.makedirs("json", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Balanced fetch summary for {category}:")
    for i, (date_from, date_to, _) in enumerate(slices):
        print(f"  Slice {i+1} ({date_from}-{date_to}): {slice_counts[i]} papers")
    print(f"  Total combined: {len(all_papers)} papers")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}")
    
    return all_papers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch abstracts from arXiv by date range and category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single slice (2024 data)
  python fetch_arxiv.py \\
    --category cs.CL \\
    --date-from 20240101 \\
    --date-to 20241231 \\
    --target 500

  # Balanced two slices (2024 + 2025)
  python fetch_arxiv.py \\
    --category cs.CL \\
    --balanced \\
    --slice1-from 20240101 --slice1-to 20241231 --slice1-target 1000 \\
    --slice2-from 20250101 --slice2-to 20251231 --slice2-target 1000 \\
    --output abstracts_cscl_balanced.json
        """,
    )
    
    # Common arguments
    parser.add_argument(
        "--category",
        type=str,
        default="cs.CL",
        help="arXiv category (default: cs.CL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="json/abstracts.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Sleep time between API requests in seconds (default: 3.0)",
    )
    
    # Single slice mode arguments
    parser.add_argument(
        "--date-from",
        type=str,
        default="20250101",
        help="Start date in YYYYMMDD format (default: 20250101)",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default="20251231",
        help="End date in YYYYMMDD format (default: 20251231)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=500,
        help="Target count for single slice mode (default: 500)",
    )
    
    # Balanced mode flag
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced two-slice fetching mode",
    )
    
    # Slice 1 arguments (for balanced mode)
    parser.add_argument(
        "--slice1-from",
        type=str,
        default="20240101",
        help="Slice 1 start date in YYYYMMDD format (default: 20240101)",
    )
    parser.add_argument(
        "--slice1-to",
        type=str,
        default="20241231",
        help="Slice 1 end date in YYYYMMDD format (default: 20241231)",
    )
    parser.add_argument(
        "--slice1-target",
        type=int,
        default=1000,
        help="Slice 1 target count (default: 1000)",
    )
    
    # Slice 2 arguments (for balanced mode)
    parser.add_argument(
        "--slice2-from",
        type=str,
        default="20250101",
        help="Slice 2 start date in YYYYMMDD format (default: 20250101)",
    )
    parser.add_argument(
        "--slice2-to",
        type=str,
        default="20251231",
        help="Slice 2 end date in YYYYMMDD format (default: 20251231)",
    )
    parser.add_argument(
        "--slice2-target",
        type=int,
        default=1000,
        help="Slice 2 target count (default: 1000)",
    )
    
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    
    if args.balanced:
        # Balanced multi-slice mode
        slices = [
            (args.slice1_from, args.slice1_to, args.slice1_target),
            (args.slice2_from, args.slice2_to, args.slice2_target),
        ]
        fetch_balanced(
            category=args.category,
            slices=slices,
            output_path=args.output,
            delay_seconds=args.delay,
        )
    else:
        # Single slice mode
        papers = fetch_date_slice(
            category=args.category,
            date_from=args.date_from,
            date_to=args.date_to,
            target_count=args.target,
            delay_seconds=args.delay,
        )
        os.makedirs("json", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
