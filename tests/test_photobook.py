"""Automated tests verifying all 10 Photo Book Builder requirements using ONLY raw photo files:
1. Input is a folder name containing personal photos (no external index or caption files)
2. Deep image analysis using 1,120 to 2,240 pixels resolution
3. Removes duplicates, similar burst shots, and photos of documents
4. Generates captions (1-2 lines max) for individual and grouped photos
4a. Selects best photos in chronological order
4b. Clusters photos into thematic groups dynamically from photo content
5. Selects a Hero image for the cover
6. Arranges 1, 2, or 3 photos max on each page
7. 10x10 inch photo book dimensions (720pt x 720pt PDF + 1:1 HTML)
8. Leverages Gemini 3.8 Flash (Medium effort) & upgrades deprecated gemini-2.5-flash
9. Saves photo analysis details including captions in output files (JSON + Markdown)
10. Aesthetic smart cropping that never cuts off faces
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import app
from photobook.analyzer import (
    clamp_caption_two_lines,
    coarse_filter_photos,
    deduplicate_and_curate_with_gemini,
    resolve_gemini_model,
)
from photobook.cropper import compute_aesthetic_crop, verify_faces_preserved
from photobook.ingestion import MAX_ANALYSIS_PX, MIN_ANALYSIS_PX, scale_to_analysis_resolution
from photobook.models import BoundingBox, GeminiPhotoAnalysis
from photobook.pipeline import build_photo_book


def _create_raw_photos_test_folder(folder: Path) -> Path:
    """Create a folder containing ONLY raw image files:
    - 4 unique photos across 2 days
    - 1 exact duplicate
    - 1 near-identical burst shot
    - 1 paper document/receipt image
    """
    folder.mkdir(parents=True, exist_ok=True)

    def _save_with_time(img: Image.Image, name: str, dt_str: str) -> Path:
        p = folder / name
        exif = Image.Exif()
        exif[272] = "iPhone 16 Pro"
        exif[306] = dt_str
        exif[36867] = dt_str
        img.save(p, format="JPEG", quality=92, exif=exif.tobytes())
        ts = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").timestamp()
        os.utime(p, (ts, ts))
        return p

    # 1. Day 1 Morning: Coastal Village Scene
    img1 = Image.new("RGB", (1600, 1200), (70, 145, 225))
    d1 = ImageDraw.Draw(img1)
    d1.rectangle((280, 360, 1320, 1060), fill=(242, 240, 236))
    d1.ellipse((550, 200, 1050, 480), fill=(35, 105, 205))
    _save_with_time(img1, "photo_01.jpg", "2026:08:10 09:15:00")

    # 2. Day 1 Afternoon: Portrait on Balcony
    img2 = Image.new("RGB", (1200, 1600), (130, 180, 230))
    d2 = ImageDraw.Draw(img2)
    d2.ellipse((460, 260, 740, 560), fill=(225, 175, 145))
    d2.rectangle((380, 560, 820, 1350), fill=(60, 110, 170))
    _save_with_time(img2, "photo_02.jpg", "2026:08:10 14:30:00")

    # 3. Day 1 Exact Duplicate of photo_01
    _save_with_time(img1, "photo_03_dup.jpg", "2026:08:10 09:15:05")

    # 4. Day 1 Similar Burst Shot of photo_01 (taken 30s later)
    img4 = img1.copy()
    d4 = ImageDraw.Draw(img4)
    d4.ellipse((150, 120, 210, 175), fill=(250, 245, 210))
    _save_with_time(img4, "photo_04_burst.jpg", "2026:08:10 09:15:35")

    # 5. Day 1 Document / Receipt Image (high-brightness white paper with dense dark text lines)
    img5 = Image.new("RGB", (1200, 1600), (248, 248, 246))
    d5 = ImageDraw.Draw(img5)
    for y in range(120, 1480, 32):
        for x in range(100, 1080, 24):
            if (x + y) % 48 == 0:
                d5.rectangle((x, y, x + 14, y + 10), fill=(20, 20, 20))
    _save_with_time(img5, "photo_05_receipt.jpg", "2026:08:10 16:00:00")

    # 6. Day 2 Afternoon: Coastal Sailing
    img6 = Image.new("RGB", (1800, 1200), (45, 115, 195))
    d6 = ImageDraw.Draw(img6)
    d6.polygon([(400, 850), (1400, 850), (1250, 1020), (550, 1020)], fill=(245, 245, 248))
    d6.polygon([(900, 220), (900, 820), (1280, 780)], fill=(250, 248, 240))
    _save_with_time(img6, "photo_06.jpg", "2026:08:11 16:20:00")

    # 7. Day 2 Sunset: Golden Sunset Over the Sea
    img7 = Image.new("RGB", (1800, 1200), (235, 130, 75))
    d7 = ImageDraw.Draw(img7)
    d7.ellipse((760, 420, 1040, 700), fill=(255, 220, 110))
    d7.rectangle((0, 720, 1800, 1200), fill=(45, 65, 110))
    _save_with_time(img7, "photo_07.jpg", "2026:08:11 19:45:00")

    return folder


def test_resolution_scaling_1120_to_2240():
    """Verify images smaller than 1120px or larger than 2240px are scaled into [1120, 2240]px."""
    small = Image.new("RGB", (600, 400), (100, 120, 140))
    scaled_up = scale_to_analysis_resolution(small)
    assert max(scaled_up.size) == MIN_ANALYSIS_PX

    huge = Image.new("RGB", (4000, 3000), (100, 120, 140))
    scaled_down = scale_to_analysis_resolution(huge)
    assert max(scaled_down.size) == MAX_ANALYSIS_PX


def test_deprecated_model_auto_upgrades_to_gemini_3_8_flash():
    """Verify gemini-2.5-flash is automatically upgraded to gemini-3.8-flash per SDK rules."""
    model_used, note = resolve_gemini_model("gemini-2.5-flash")
    assert model_used == "gemini-3.8-flash"
    assert note is not None and "gemini-3.8-flash" in note


def test_clamp_caption_two_lines():
    """Verify long captions are clamped to 1-2 lines cleanly."""
    long_text = (
        "Warm golden sunset over the calm sea, with several sailboats gently floating near the rugged cliffside, "
        "reflecting shimmering orange ripples across the water during early evening."
    )
    clamped = clamp_caption_two_lines(long_text)
    assert len(clamped) <= 145
    assert len(clamped.splitlines()) <= 2


def test_face_safe_smart_cropper_never_cuts_off_faces():
    """Verify the smart cropper shifts the crop window so faces near edges are never clipped."""
    width, height = 1600, 1200
    faces = [
        BoundingBox(ymin=80, xmin=180, ymax=340, xmax=380, label="face"),
        BoundingBox(ymin=100, xmin=620, ymax=360, xmax=820, label="face"),
    ]
    crop_px, crop_norm, faces_ok = compute_aesthetic_crop(
        width=width,
        height=height,
        target_aspect=1.0,
        faces_boxes=faces,
    )
    assert faces_ok is True
    assert verify_faces_preserved(crop_px, width, height, faces) is True
    assert 0 <= crop_norm[0] < crop_norm[2] <= 1000
    assert 0 <= crop_norm[1] < crop_norm[3] <= 1000


def test_end_to_end_pipeline_and_outputs(tmp_path: Path):
    """Run the full pipeline on a folder of raw photos and verify all requirements."""
    test_folder = _create_raw_photos_test_folder(tmp_path / "raw_photos")
    out_root = tmp_path / "output"

    result = build_photo_book(
        folder_path=test_folder,
        output_root=out_root,
        model="gemini-2.5-flash",
        thinking_level="medium",
        media_resolution="high",
    )

    assert result.model_used == "gemini-3.8-flash"
    assert result.thinking_level == "medium"

    for p in result.photos:
        longest = max(p.analysis_width, p.analysis_height)
        assert MIN_ANALYSIS_PX <= longest <= MAX_ANALYSIS_PX

    assert result.total_scanned == 7
    assert result.total_excluded_documents == 1
    assert result.total_excluded_duplicates == 1
    assert result.total_excluded_similar == 1
    assert result.total_selected == 4

    selected_photos = [p for p in result.photos if p.status == "selected"]
    timestamps = [p.captured_timestamp for p in selected_photos]
    assert timestamps == sorted(timestamps)

    assert len(result.clusters) == 2
    for cluster in result.clusters:
        assert cluster.group_caption
        assert len(cluster.group_caption.splitlines()) <= 2

    assert result.hero_cover_photo_id in {p.photo_id for p in selected_photos}
    assert result.pages[0].page_type == "cover"
    assert result.pages[0].slots[0].photo_id == result.hero_cover_photo_id

    assert result.book_size_inches == (10.0, 10.0)
    for page in result.pages[1:]:
        assert 1 <= len(page.slots) <= 3
        for slot in page.slots:
            assert slot.faces_preserved is True
            crop_file = Path(result.output_dir) / slot.cropped_image_relpath
            assert crop_file.exists()

    json_path = Path(result.analysis_json_path)
    md_path = Path(result.analysis_markdown_path)
    html_path = Path(result.html_preview_path)
    pdf_path = Path(result.pdf_path)

    assert json_path.exists()
    assert md_path.exists()
    assert html_path.exists()
    assert pdf_path.exists()

    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")


def test_gemini_interactions_api_call_structure(tmp_path: Path):
    """Verify that when GEMINI_API_KEY is provided, client.interactions.create is called
    with model='gemini-3.8-flash', generation_config={'thinking_level': 'medium'},
    and per-image resolution='high'."""
    folder = tmp_path / "one_photo"
    folder.mkdir()
    img = Image.new("RGB", (1500, 1200), (80, 140, 200))
    img.save(folder / "coastal_sunset.jpg", format="JPEG")

    mock_analysis = GeminiPhotoAnalysis(
        is_document_or_screenshot=False,
        aesthetic_score=9.2,
        sharpness_score=9.0,
        emotional_score=8.8,
        cover_worthiness_score=9.5,
        subject_summary="Golden sunset over calm coastal waters",
        location_or_landmark="Coastal Bay",
        event_theme="Coastal Sunset",
        caption="Golden sunset reflecting across calm coastal waters.",
        people_count=1,
        faces_boxes=[BoundingBox(ymin=200, xmin=400, ymax=450, xmax=600, label="face")],
    )

    mock_interaction = MagicMock()
    mock_interaction.output_text = mock_analysis.model_dump_json()

    with patch("photobook.analyzer.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction
        mock_client_cls.return_value = mock_client

        result = build_photo_book(
            folder_path=folder,
            output_root=tmp_path / "out",
            model="gemini-3.8-flash",
            thinking_level="medium",
            media_resolution="high",
            api_key="test-fake-key",
        )

        assert mock_client.interactions.create.called
        first_call_kwargs = mock_client.interactions.create.call_args_list[0].kwargs
        assert first_call_kwargs["model"] == "gemini-3.8-flash"
        assert first_call_kwargs["generation_config"] == {"thinking_level": "medium"}
        image_input = first_call_kwargs["input"][0]
        assert image_input["type"] == "image"
        assert image_input["resolution"] == "high"
        assert result.total_selected == 1


def test_web_ui_endpoints(tmp_path: Path):
    """Verify the FastAPI Web UI endpoints (/api/config, /api/build)."""
    test_folder = _create_raw_photos_test_folder(tmp_path / "ui_photos")
    client = TestClient(app)

    r_index = client.get("/")
    assert r_index.status_code == 200
    assert "10×10 Photo Book" in r_index.text

    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    assert r_cfg.json()["default_model"] == "gemini-3.8-flash"

    r_build = client.post(
        "/api/build",
        json={
            "folder_path": str(test_folder),
            "model": "gemini-3.8-flash",
            "thinking_level": "medium",
            "media_resolution": "high",
        },
    )
    assert r_build.status_code == 200
    book_data = r_build.json()
    assert book_data["total_selected"] == 4
    book_id = book_data["book_id"]

    r_pdf = client.get(f"/api/books/{book_id}/files/photobook_10x10.pdf")
    assert r_pdf.status_code == 200
    assert r_pdf.content.startswith(b"%PDF-")


def test_multi_model_and_batch_processing(tmp_path: Path):
    """Verify multi-photo batch vision processing and per-stage model parameters."""
    test_folder = _create_raw_photos_test_folder(tmp_path / "batch_photos")
    client = TestClient(app)

    r_build = client.post(
        "/api/build",
        json={
            "folder_path": str(test_folder),
            "model": "gemini-3.8-flash",
            "thinking_level": "low",
            "curation_model": "gemini-2.5-flash-lite",
            "hero_model": "none",
            "batch_size": 8,
            "concurrency": 16,
            "media_resolution": "low",
            "min_analysis_px": 512,
            "max_analysis_px": 768,
        },
    )
    assert r_build.status_code == 200
    book_data = r_build.json()
    assert book_data["curation_model_used"] == "gemini-2.5-flash-lite"
    assert book_data["hero_model_used"] == "none"
    assert book_data["batch_size"] == 8
    assert book_data["concurrency"] == 16
    assert book_data["total_selected"] == 4


def test_hero_cover_prefers_scenic_and_excludes_sideways_faces():
    """Verify scenic views are preferred over portraits for the hero cover (with fallback if none available)
    and sideways/upside-down face orientations are excluded."""
    from photobook.analyzer import deduplicate_and_filter_photos, select_hero_cover_photo
    from photobook.models import PhotoMetadata

    portrait_photo = PhotoMetadata(
        photo_id="p_portrait",
        filename="portrait.jpg",
        file_path="/tmp/portrait.jpg",
        original_width=1200,
        original_height=1600,
        analysis_width=1200,
        analysis_height=1600,
        analysis_path="/tmp/portrait.jpg",
        captured_at="2026-08-10T10:00:00",
        captured_timestamp=1700000000.0,
        dhash="0000000000000001",
        ahash="0000000000000001",
        color_signature=[0.5, 0.5, 0.5],
        analysis=GeminiPhotoAnalysis(
            aesthetic_score=8.8,
            sharpness_score=8.8,
            emotional_score=9.0,
            cover_worthiness_score=8.9,
            subject_summary="Close-up portrait of a smiling person",
            caption="Bright smiles in the morning sun.",
            people_count=1,
            faces_boxes=[BoundingBox(ymin=200, xmin=300, ymax=500, xmax=700, label="face")],
        ),
    )
    scenic_photo = PhotoMetadata(
        photo_id="p_scenic",
        filename="scenic.jpg",
        file_path="/tmp/scenic.jpg",
        original_width=1600,
        original_height=1200,
        analysis_width=1600,
        analysis_height=1200,
        analysis_path="/tmp/scenic.jpg",
        captured_at="2026-08-10T11:00:00",
        captured_timestamp=1700003600.0,
        dhash="ffffffff00000000",
        ahash="ffffffff00000000",
        color_signature=[0.2, 0.4, 0.8],
        analysis=GeminiPhotoAnalysis(
            aesthetic_score=8.2,
            sharpness_score=8.4,
            emotional_score=8.0,
            cover_worthiness_score=8.0,
            subject_summary="Panoramic caldera cliffside view under blue sky",
            caption="Sunlit rooftops overlooking the dramatic caldera slopes.",
            people_count=0,
            faces_boxes=[],
        ),
    )
    sideways_photo = PhotoMetadata(
        photo_id="p_sideways",
        filename="sideways.jpg",
        file_path="/tmp/sideways.jpg",
        original_width=1600,
        original_height=1200,
        analysis_width=1600,
        analysis_height=1200,
        analysis_path="/tmp/sideways.jpg",
        captured_at="2026-08-10T12:00:00",
        captured_timestamp=1700007200.0,
        dhash="0f0f0f0f0f0f0f0f",
        ahash="0f0f0f0f0f0f0f0f",
        color_signature=[0.8, 0.2, 0.2],
        analysis=GeminiPhotoAnalysis(
            aesthetic_score=7.5,
            sharpness_score=8.0,
            emotional_score=7.5,
            cover_worthiness_score=5.5,
            subject_summary="Sideways/portrait orientation of a young woman smiling on a rocky overlook",
            caption="Smiling at the overlook.",
            people_count=1,
            faces_boxes=[BoundingBox(ymin=300, xmin=300, ymax=600, xmax=600, label="face")],
        ),
    )

    kept = deduplicate_and_filter_photos([portrait_photo, scenic_photo, sideways_photo])
    assert [p.photo_id for p in kept] == ["p_portrait", "p_scenic"]
    assert sideways_photo.status == "excluded_low_quality"

    # Scenic photo is chosen even when portrait had a higher raw score
    hero = select_hero_cover_photo(kept)
    assert hero.photo_id == "p_scenic"

    # Fallback: when no scenic photo is available, chooses best portrait
    hero_fallback = select_hero_cover_photo([portrait_photo])
    assert hero_fallback.photo_id == "p_portrait"


def test_smart_hybrid_gemini_first_flow() -> None:
    """Verify the Gemini-first Smart Hybrid flow:
    1. Coarse local filter drops exact file duplicates, but passes burst variations through to Gemini.
    2. Deep Gemini vision analyses all coarse survivors.
    3. Gemini-powered deduplication picks the best shot (e.g., smiling vs blinking) using Gemini scores,
       and curates to max_book_photos using Gemini composite quality.
    """
    from photobook.models import PhotoMetadata

    # Photo A: Burst shot 1 - Awkward blink, high local edge sharpness from hair
    p_blink = PhotoMetadata(
        photo_id="burst_01",
        filename="burst_01.jpg",
        file_path="/tmp/burst_01.jpg",
        original_width=1600,
        original_height=1200,
        analysis_width=1600,
        analysis_height=1200,
        analysis_path="/tmp/burst_01.jpg",
        captured_at="2026-08-10T14:30:00",
        captured_timestamp=1700010000.0,
        dhash="aaaaaaaa00000000",
        ahash="aaaaaaaa00000000",
        color_signature=[0.57735, 0.57735, 0.57735],
        analysis=GeminiPhotoAnalysis(
            aesthetic_score=5.5,
            sharpness_score=8.5,  # high sharpness, but bad expression
            emotional_score=4.0,
            cover_worthiness_score=4.5,
            subject_summary="Woman with eyes closed mid-blink on balcony",
            caption="Mid-blink on the sunny balcony.",
            people_count=1,
            faces_boxes=[BoundingBox(ymin=200, xmin=300, ymax=500, xmax=600, label="face")],
        ),
    )

    # Photo B: Burst shot 2 (30s later) - Warm natural smile, lower edge contrast
    p_smile = PhotoMetadata(
        photo_id="burst_02",
        filename="burst_02.jpg",
        file_path="/tmp/burst_02.jpg",
        original_width=1600,
        original_height=1200,
        analysis_width=1600,
        analysis_height=1200,
        analysis_path="/tmp/burst_02.jpg",
        captured_at="2026-08-10T14:30:30",
        captured_timestamp=1700010030.0,
        dhash="aaaaaaaa00000002",
        ahash="aaaaaaaa00000002",
        color_signature=[0.57735, 0.57735, 0.57735],
        analysis=GeminiPhotoAnalysis(
            aesthetic_score=8.8,
            sharpness_score=7.6,
            emotional_score=9.2,
            cover_worthiness_score=8.5,
            subject_summary="Woman smiling warmly looking at camera on sunlit balcony",
            caption="Smiling warmly on the sunlit balcony.",
            people_count=1,
            faces_boxes=[BoundingBox(ymin=200, xmin=300, ymax=500, xmax=600, label="face")],
        ),
    )

    # Photo C: Exact byte/hash duplicate of Photo A
    p_exact_dup = PhotoMetadata(
        photo_id="burst_01_copy",
        filename="burst_01_copy.jpg",
        file_path="/tmp/burst_01_copy.jpg",
        original_width=1600,
        original_height=1200,
        analysis_width=1600,
        analysis_height=1200,
        analysis_path="/tmp/burst_01_copy.jpg",
        captured_at="2026-08-10T14:30:00",
        captured_timestamp=1700010000.0,
        dhash="aaaaaaaa00000000",
        ahash="aaaaaaaa00000000",
        color_signature=[0.57735, 0.57735, 0.57735],
        analysis=GeminiPhotoAnalysis(
            aesthetic_score=5.5,
            sharpness_score=8.5,
            emotional_score=4.0,
            cover_worthiness_score=4.5,
            subject_summary="Woman with eyes closed mid-blink on balcony",
            caption="Mid-blink on the sunny balcony.",
            people_count=1,
            faces_boxes=[],
        ),
    )

    # 1. Coarse filtering drops the exact duplicate, but keeps both burst shots for Gemini!
    coarse = coarse_filter_photos([p_blink, p_smile, p_exact_dup])
    assert len(coarse) == 2
    assert p_exact_dup.status == "excluded_duplicate"
    assert {p.photo_id for p in coarse} == {"burst_01", "burst_02"}

    # 2. Gemini-powered deduplication picks the smiling photo (burst_02) over the blinking photo (burst_01)
    curated = deduplicate_and_curate_with_gemini(coarse, max_book_photos=1)
    assert len(curated) == 1
    assert curated[0].photo_id == "burst_02"
    assert p_blink.status == "excluded_similar"
    assert p_blink.duplicate_of == "burst_02"


def test_page_fill_modes_and_aesthetic_crop_focus() -> None:
    """Verify page fill modes (high, medium, low) and aesthetic crop focus (tight, balanced, wide)."""
    from photobook.analyzer import get_page_templates

    tpls_high = get_page_templates("high")
    tpls_med = get_page_templates("medium")
    tpls_low = get_page_templates("low")

    # High fill mode occupies substantially more page area with minimal margins
    w_high = tpls_high["single_full_square"][0][2]
    h_high = tpls_high["single_full_square"][0][3]
    w_low = tpls_low["single_full_square"][0][2]
    h_low = tpls_low["single_full_square"][0][3]

    assert w_high > w_low
    assert h_high > h_low
    assert w_high == 9.20
    assert h_high == 7.50
    assert w_low == 8.10
    assert h_low == 6.15

    # Verify aesthetic crop focus: tight zooms in, wide preserves full span
    face = BoundingBox(ymin=350, xmin=400, ymax=550, xmax=600, label="face")  # central face
    salient = BoundingBox(ymin=300, xmin=350, ymax=650, xmax=650, label="subject")

    crop_tight, _, faces_ok_tight = compute_aesthetic_crop(
        width=2000,
        height=1500,
        target_aspect=1.0,
        faces_boxes=[face],
        salient_box=salient,
        crop_focus="tight",
    )
    crop_wide, _, faces_ok_wide = compute_aesthetic_crop(
        width=2000,
        height=1500,
        target_aspect=1.0,
        faces_boxes=[face],
        salient_box=salient,
        crop_focus="wide",
    )

    # Wide crop uses full available height (1500x1500)
    assert (crop_wide[2] - crop_wide[0]) == 1500
    assert (crop_wide[3] - crop_wide[1]) == 1500
    assert faces_ok_wide is True

    # Tight crop zooms in closer to the subject, removing dead space (smaller crop window than 1500)
    assert (crop_tight[2] - crop_tight[0]) < 1500
    assert (crop_tight[3] - crop_tight[1]) < 1500
    assert faces_ok_tight is True


def test_include_captions_toggle_in_pipeline_and_renderers(tmp_path: Path) -> None:
    """Verify that include_captions=False suppresses captions in PDF and HTML while keeping clean layout."""
    test_folder = _create_raw_photos_test_folder(tmp_path / "raw_photos")

    # Run with include_captions=False
    out_no_caps = tmp_path / "out_no_caps"
    res_no_caps = build_photo_book(
        folder_path=test_folder,
        output_root=out_no_caps,
        include_captions=False,
    )
    assert res_no_caps.include_captions is False
    html_no_caps = Path(res_no_caps.html_preview_path).read_text(encoding="utf-8")
    assert '<div class="slot-subcaption">' not in html_no_caps
    assert '<p class="page-group-caption">' not in html_no_caps
    assert '<p class="cover-hero-caption">' not in html_no_caps
    pdf_no_caps = Path(res_no_caps.pdf_path).read_bytes()
    assert pdf_no_caps.startswith(b"%PDF-")

    # Run with include_captions=True (default)
    out_caps = tmp_path / "out_caps"
    res_caps = build_photo_book(
        folder_path=test_folder,
        output_root=out_caps,
        include_captions=True,
    )
    assert res_caps.include_captions is True
    html_caps = Path(res_caps.html_preview_path).read_text(encoding="utf-8")
    assert '<p class="cover-hero-caption">' in html_caps or '<p class="page-group-caption">' in html_caps
    pdf_caps = Path(res_caps.pdf_path).read_bytes()
    assert pdf_caps.startswith(b"%PDF-")


def test_remove_curated_and_dates_and_include_location_on_cover(tmp_path: Path) -> None:
    """Verify that 'CURATED 10x10' and dates are removed from pages, and location is displayed on cover."""
    test_folder = _create_raw_photos_test_folder(tmp_path / "raw_photos")
    out_root = tmp_path / "out_loc"

    result = build_photo_book(
        folder_path=test_folder,
        output_root=out_root,
    )

    html_text = Path(result.html_preview_path).read_text(encoding="utf-8")

    # 1. Ensure "CURATED 10×10 INCH PHOTO BOOK" and "A Curated 10x10" are nowhere in the HTML
    assert "CURATED 10×10 INCH PHOTO BOOK" not in html_text
    assert "A Curated 10×10 Photo Book" not in html_text
    assert "A Curated 10x10 Photo Book" not in html_text

    # 2. Ensure date range string is not in chapter headers
    for p in result.pages:
        if p.chapter_title:
            assert "(" not in p.chapter_title or "2024" not in p.chapter_title

    # 3. Test strip_curated_and_dates helper
    from photobook.analyzer import strip_curated_and_dates
    assert strip_curated_and_dates("A Curated 10×10 Photo Book • 2024-04-12 to 2024-04-15") == ""
    assert strip_curated_and_dates("Day 1 (2024-04-12) — Tokyo Tower") == "Day 1 — Tokyo Tower"
    assert strip_curated_and_dates("Captured Moments (2024-04-12 to 2024-04-15)") == "Captured Moments"

    # 4. Test location detection on cover
    from photobook.models import GeminiPhotoAnalysis, PhotoMetadata
    p1 = PhotoMetadata(
        photo_id="test_loc",
        filename="test.jpg",
        file_path="/tmp/test.jpg",
        original_width=1600,
        original_height=1200,
        analysis_width=1600,
        analysis_height=1200,
        analysis_path="/tmp/test.jpg",
        captured_at="2026-08-10T14:30:00",
        captured_timestamp=1700010000.0,
        dhash="aaaaaaaa00000000",
        ahash="aaaaaaaa00000000",
        color_signature=[0.57735, 0.57735, 0.57735],
        analysis=GeminiPhotoAnalysis(
            location_or_landmark="Old San Juan, Puerto Rico",
            cover_worthiness_score=9.5,
            aesthetic_score=9.5,
        ),
    )
    from photobook.analyzer import extract_book_location
    detected_loc = extract_book_location([p1], hero_photo=p1)
    assert detected_loc == "Old San Juan, Puerto Rico"


