"""End-to-end orchestration pipeline for the 10x10 inch Photo Book Builder."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from photobook.analyzer import (
    DEFAULT_MODEL,
    _safe_progress,
    analyze_photos_with_gemini,
    coarse_filter_photos,
    curate_clusters_and_pages,
    deduplicate_and_curate_with_gemini,
    deduplicate_and_filter_photos,
    is_sideways_or_inverted_photo,
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
    curation_model: str = "gemini-2.5-flash-lite",
    hero_model: str = "gemini-3.8-flash",
    layout_model: str = "local_geometric",
    batch_size: int = 6,
    concurrency: int = 12,
    media_resolution: str = "high",
    min_analysis_px: int = MIN_ANALYSIS_PX,
    max_analysis_px: int = MAX_ANALYSIS_PX,
    max_book_photos: int = 90,
    page_fill_mode: str = "high",
    crop_focus: str = "tight",
    caption_mode: str = "all",
    include_captions: bool = True,
    api_key: Optional[str] = None,
    progress_cb: Optional[Callable[..., None]] = None,
) -> PhotoBookResult:
    """Execute the complete Photo Book Builder pipeline on a photo folder.

    Strictly reads ONLY the image files in `folder_path` and saves all generated outputs
    into a newly created subfolder inside `/Users/amitpoddar/.gemini/antigravity/scratch/photo-book-builder/output`.
    """
    t_start = time.perf_counter()
    stage_timings: dict[str, float] = {}

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
        _safe_progress(progress_cb, "Step 1/5: Scanning folder (including .HEIC, .JPG, .PNG) & scaling to 1,120–2,240 px...", 3, 100)

    def _scan_progress(msg: str, idx: int, total: int, extra: Optional[dict] = None) -> None:
        if progress_cb:
            pct = 3 + int(round((idx / max(1, total)) * 27))
            _safe_progress(progress_cb, f"Step 1/5: {msg}", pct, 100, extra=extra)

    t0 = time.perf_counter()
    photos = scan_and_prepare_photos(
        folder_path=src_folder,
        work_dir=book_dir,
        min_px=min_analysis_px,
        max_px=max_analysis_px,
        progress_cb=_scan_progress,
    )
    stage_timings["scan_s"] = round(time.perf_counter() - t0, 2)

    def _pre_progress(msg: str, idx: int, total: int, extra: Optional[dict] = None) -> None:
        if progress_cb:
            pct = 22 + int(round((idx / max(1, total)) * 10))
            _safe_progress(progress_cb, f"Step 2/5 (Coarse Local Prune): {msg}", pct, 100, extra=extra)

    t1 = time.perf_counter()
    pre_analyze_all_photos(
        photos=photos,
        progress_cb=_pre_progress,
    )
    # Stage 2: Coarse Fast Pruning (drops exact duplicate files and clear receipts, keeps all unique photos for Gemini)
    coarse_survivors = coarse_filter_photos(photos)
    stage_timings["coarse_filter_s"] = round(time.perf_counter() - t1, 2)

    if progress_cb:
        _safe_progress(
            progress_cb,
            f"Step 3/5: Deep Gemini 3.8 Flash analysis on all {len(coarse_survivors)} surviving photos...",
            32,
            100,
        )

    def _gemini_progress(msg: str, idx: int, total: int, extra: Optional[dict] = None) -> None:
        if progress_cb:
            pct = 32 + int(round((idx / max(1, total)) * 38))
            _safe_progress(progress_cb, f"Step 3/5 (Gemini-First Vision): {msg}", pct, 100, extra=extra)

    t2 = time.perf_counter()
    model_used, model_note = analyze_photos_with_gemini(
        photos=coarse_survivors,
        model=model,
        thinking_level=thinking_level,
        media_resolution=media_resolution,
        batch_size=batch_size,
        concurrency=concurrency,
        api_key=api_key,
        progress_cb=_gemini_progress,
    )
    stage_timings["vision_s"] = round(time.perf_counter() - t2, 2)

    if progress_cb:
        _safe_progress(
            progress_cb,
            f"Step 4/5: Gemini-powered semantic burst curation & selection ({len(coarse_survivors)} → max {max_book_photos or 'all'})...",
            72,
            100,
        )

    t3 = time.perf_counter()
    selected = deduplicate_and_curate_with_gemini(
        photos=coarse_survivors,
        max_book_photos=max_book_photos,
    )
    stage_timings["dedup_s"] = round(time.perf_counter() - t3, 2)

    if progress_cb:
        _safe_progress(
            progress_cb,
            f"Step 5/5: Clustering {len(selected)} photos ({curation_model}), refining Hero cover ({hero_model}) & face-safe cropping...",
            84,
            100,
        )

    t4 = time.perf_counter()
    title, subtitle, hero_id, clusters, pages, book_location = curate_clusters_and_pages(
        selected_photos=selected,
        work_dir=book_dir,
        model=model_used,
        thinking_level=thinking_level,
        curation_model=curation_model,
        hero_model=hero_model,
        layout_model=layout_model,
        batch_size=batch_size,
        page_fill_mode=page_fill_mode,
        crop_focus=crop_focus,
        api_key=api_key,
    )
    stage_timings["curation_layout_s"] = round(time.perf_counter() - t4, 2)

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
        curation_model_used=curation_model,
        hero_model_used=hero_model,
        layout_model_used=layout_model,
        batch_size=batch_size,
        concurrency=concurrency,
        build_duration_seconds=round(time.perf_counter() - t_start, 2),
        stage_timings=stage_timings,
        model_note=model_note,
        thinking_level=thinking_level,
        media_resolution=media_resolution,
        analysis_pixel_range=f"{min_analysis_px:,} to {max_analysis_px:,} px",
        book_size_inches=(10.0, 10.0),
        page_fill_mode=page_fill_mode,
        crop_focus=crop_focus,
        caption_mode="none" if not include_captions else caption_mode,
        include_captions=include_captions and caption_mode != "none",
        location=book_location,
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
        _safe_progress(progress_cb, "Step 5/5: Rendering 10×10 inch PDF, interactive HTML book & analysis files...", 92, 100)

    t4 = time.perf_counter()
    render_html_photobook(result, html_preview_path)
    render_pdf_photobook(result, pdf_path)
    stage_timings["render_s"] = round(time.perf_counter() - t4, 2)
    result.build_duration_seconds = round(time.perf_counter() - t_start, 2)
    result.stage_timings = stage_timings
    save_analysis_files(result)

    if progress_cb:
        _safe_progress(
            progress_cb,
            f"Complete in {result.build_duration_seconds}s! 10×10 inch Photo Book and Analysis Report ready.",
            100,
            100,
        )

    return result
