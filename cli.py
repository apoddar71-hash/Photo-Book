"""Command-line interface for building a 10x10 inch Photo Book from a folder of photos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from photobook.analyzer import DEFAULT_MODEL
from photobook.ingestion import MAX_ANALYSIS_PX, MIN_ANALYSIS_PX
from photobook.pipeline import build_photo_book


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Build a 10x10 inch Photo Book using Gemini 3.8 Flash (Medium effort)."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing your personal photos.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory root (defaults to ./output/<book_id>).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--thinking-level",
        default="medium",
        choices=["low", "medium", "high"],
        help="Gemini thinking effort level (default: medium).",
    )
    parser.add_argument(
        "--media-resolution",
        default="high",
        choices=["low", "medium", "high", "ultra_high"],
        help="Gemini per-image media resolution: 'high' (1,120 tokens) or 'ultra_high' (2,240 tokens).",
    )
    parser.add_argument(
        "--min-px",
        type=int,
        default=MIN_ANALYSIS_PX,
        help=f"Minimum longest-side pixel resolution for analysis (default: {MIN_ANALYSIS_PX}).",
    )
    parser.add_argument(
        "--max-px",
        type=int,
        default=MAX_ANALYSIS_PX,
        help=f"Maximum longest-side pixel resolution for analysis (default: {MAX_ANALYSIS_PX}).",
    )
    parser.add_argument(
        "--max-photos",
        type=int,
        default=90,
        help="Maximum curated photos to include in the book (0 = all unique photos, default: 90).",
    )

    args = parser.parse_args()
    folder_path = Path(args.folder).expanduser().resolve()

    def _progress(msg: str, pct: int, _total: int) -> None:
        print(f"[{pct:3d}%] {msg}")

    result = build_photo_book(
        folder_path=folder_path,
        output_root=args.output_dir,
        model=args.model,
        thinking_level=args.thinking_level,
        media_resolution=args.media_resolution,
        min_analysis_px=args.min_px,
        max_analysis_px=args.max_px,
        max_book_photos=args.max_photos,
        progress_cb=_progress,
    )

    print("\n" + "=" * 72)
    print(f"PHOTO BOOK BUILT SUCCESSFULLY: {result.title}")
    print("=" * 72)
    print(f"Model Used          : {result.model_used} (thinking_level={result.thinking_level}, resolution={result.media_resolution})")
    if result.model_note:
        print(f"Model Note          : {result.model_note}")
    print(f"Photos Scanned      : {result.total_scanned}")
    print(f"Photos Selected     : {result.total_selected} (in chronological order)")
    print(f"Excluded Duplicates : {result.total_excluded_duplicates}")
    print(f"Excluded Similar    : {result.total_excluded_similar}")
    print(f"Excluded Documents  : {result.total_excluded_documents}")
    print(f"Excluded Low-Qual   : {result.total_excluded_low_quality}")
    print(f"Hero Cover Photo    : {result.hero_cover_photo_id}")
    print(f"Clusters Created    : {len(result.clusters)}")
    print(f"10x10 Inch Pages    : {len(result.pages)} (Cover + {len(result.pages) - 1} interior pages)")
    print("-" * 72)
    print(f"10x10 PDF Book      : {result.pdf_path}")
    print(f"Interactive HTML    : {result.html_preview_path}")
    print(f"Analysis JSON       : {result.analysis_json_path}")
    print(f"Analysis Markdown   : {result.analysis_markdown_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
