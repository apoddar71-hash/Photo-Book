"""End-to-end orchestration pipeline for the 10x10 inch Photo Book Builder."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from photobook.analyzer import (
    DEFAULT_MODEL,
    analyze_photos_with_gemini,
    curate_clusters_and_pages,
    deduplicate_and_filter_photos,
    pre_analyze_all_photos,
)
from photobook.ingestion import MAX_ANALYSIS_PX, MIN_ANALYSIS_PX, scan_and_prepare_photos
from photobook.models import PhotoBookResult
from photobook.renderer import render_html_photobook, render_pdf_photobook, save_analysis_files


def build_photo_book(
    folder_path: str | Path,
    output_root: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    thinking_level: str = "medium",
    media_resolution: str = "high",
    min_analysis_px: int = MIN_ANALYSIS_PX,
    max_analysis_px: int = MAX_ANALYSIS_PX,
    max_book_photos: int = 90,
    api_key: Optional[str] = None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> PhotoBookResult:
    """Execute the complete Photo Book Builder pipeline on a photo folder.

    Strictly reads ONLY the image files in `folder_path` and saves all generated outputs
    into a newly created subfolder inside `/Users/amitpoddar/.gemini/antigravity/scratch/photo-book-builder/output`.
    """
    src_folder = Path(folder_path).expanduser().resolve()
    base_out = (
        Path(output_root).expanduser().resolve()
        if output_root
        else Path(__file__).resolve().parent.parent / "output"
    )
    folder_slug = src_folder.name.lower().replace(" ", "_")
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    book_hash = hashlib.sha1(f"{src_folder}-{datetime.now().isoformat()}".encode("utf-8")).hexdigest()[:6]
    book_id = f"{folder_slug}_{ts_str}_{book_hash}"
    book_dir = base_out / book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb("Step 1/5: Scanning folder (including .HEIC, .JPG, .PNG) & scaling to 1,120–2,240 px...", 3, 100)

    def _scan_progress(msg: str, idx: int, total: int) -> None:
        if progress_cb:
            pct = 3 + int(round((idx / max(1, total)) * 27))
            progress_cb(f"Step 1/5: {msg}", pct, 100)

    photos = scan_and_prepare_photos(
        folder_path=src_folder,
        work_dir=book_dir,
        min_px=min_analysis_px,
        max_px=max_analysis_px,
        progress_cb=_scan_progress,
    )

    def _pre_progress(msg: str, idx: int, total: int) -> None:
        if progress_cb:
            pct = 30 + int(round((idx / max(1, total)) * 20))
            progress_cb(f"Step 2/5: {msg}", pct, 100)

    pre_analyze_all_photos(
        photos=photos,
        progress_cb=_pre_progress,
    )

    if progress_cb:
        progress_cb(
            f"Step 3/5: Removing duplicates, similar bursts & documents across {len(photos)} photos...",
            52,
            100,
        )

    selected = deduplicate_and_filter_photos(photos, max_book_photos=max_book_photos)

    def _gemini_progress(msg: str, idx: int, total: int) -> None:
        if progress_cb:
            pct = 55 + int(round((idx / max(1, total)) * 22))
            progress_cb(f"Step 3/5: {msg}", pct, 100)

    model_used, model_note = analyze_photos_with_gemini(
        photos=selected,
        model=model,
        thinking_level=thinking_level,
        media_resolution=media_resolution,
        api_key=api_key,
        progress_cb=_gemini_progress,
    )

    # Filter out any photos newly marked as documents or blurry by Gemini 3.8 Flash
    post_selected = []
    for p in selected:
        if p.analysis.is_document_or_screenshot:
            p.status = "excluded_document"
            p.exclusion_reason = p.analysis.document_reason or "Detected document, receipt, or screenshot"
        elif p.analysis.is_blurry_or_flawed:
            p.status = "excluded_low_quality"
            p.exclusion_reason = f"Low technical quality or motion blur (sharpness {p.analysis.sharpness_score}/10)"
        else:
            post_selected.append(p)
    selected = post_selected

    if progress_cb:
        progress_cb(
            f"Step 4/5: Clustering {len(selected)} best chronological photos, selecting Hero cover & face-safe cropping...",
            80,
            100,
        )

    title, subtitle, hero_id, clusters, pages = curate_clusters_and_pages(
        selected_photos=selected,
        work_dir=book_dir,
        model=model_used,
        thinking_level=thinking_level,
        api_key=api_key,
    )

    analysis_json_path = book_dir / "photo_analysis.json"
    analysis_md_path = book_dir / "photo_analysis.md"
    html_preview_path = book_dir / "photobook.html"
    pdf_path = book_dir / "photobook_10x10.pdf"

    result = PhotoBookResult(
        book_id=book_id,
        title=title,
        subtitle=subtitle,
        folder_path=str(src_folder),
        created_at=datetime.now().isoformat(timespec="seconds"),
        model_requested=model,
        model_used=model_used,
        model_note=model_note,
        thinking_level=thinking_level,
        media_resolution=media_resolution,
        analysis_pixel_range=f"{min_analysis_px:,} to {max_analysis_px:,} px",
        book_size_inches=(10.0, 10.0),
        hero_cover_photo_id=hero_id,
        total_scanned=len(photos),
        total_selected=len(selected),
        total_excluded_duplicates=sum(1 for p in photos if p.status == "excluded_duplicate"),
        total_excluded_similar=sum(1 for p in photos if p.status == "excluded_similar"),
        total_excluded_documents=sum(1 for p in photos if p.status == "excluded_document"),
        total_excluded_low_quality=sum(1 for p in photos if p.status == "excluded_low_quality"),
        clusters=clusters,
        pages=pages,
        photos=photos,
        output_dir=str(book_dir),
        analysis_json_path=str(analysis_json_path),
        analysis_markdown_path=str(analysis_md_path),
        html_preview_path=str(html_preview_path),
        pdf_path=str(pdf_path),
    )

    if progress_cb:
        progress_cb("Step 5/5: Rendering 10×10 inch PDF, interactive HTML book & analysis files...", 92, 100)

    save_analysis_files(result)
    render_html_photobook(result, html_preview_path)
    render_pdf_photobook(result, pdf_path)

    if progress_cb:
        progress_cb("Complete! 10×10 inch Photo Book and Analysis Report ready.", 100, 100)

    return result
