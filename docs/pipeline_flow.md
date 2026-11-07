# 10×10 Inch AI Photo Book Builder — End-to-End Pipeline & Architecture Flow

This document explains the complete end-to-end pipeline used by the **10×10 Inch AI Photo Book Builder** ([`photobook/pipeline.py`](../photobook/pipeline.py)), including how a large collection of 700+ raw photos is filtered, ranked, deduplicated, analyzed with **Gemini 3.8 Flash**, curated into chronological story chapters, cropped with 100% face-preservation, and rendered into a print-ready 10×10" PDF and interactive **3D Lay-Flat Flip Viewer**.

---

## High-Level Pipeline Overview

```
Raw Photo Folder (e.g., 700+ .JPG / .HEIC / .PNG / .WEBP files)
   │
   ├─► Stage 1: Ingestion, EXIF Orientation, Perceptual Hashes & Scaling (All 700+ Photos)
   │            [photobook/ingestion.py :: scan_and_prepare_photos]
   │
   ├─► Stage 2: Fast Local Pixel-Level Pre-Analysis & Initial Scoring (All 700+ Photos)
   │            [photobook/analyzer.py :: pre_analyze_all_photos -> _analyze_single_photo_pixels]
   │
   ├─► Stage 3: Deduplication, Burst Filtering & Chronological Time-Bucket Curation (700+ → 90 Photos)
   │            [photobook/analyzer.py :: deduplicate_and_filter_photos]
   │
   ├─► Stage 4: Deep Gemini 3.8 Flash Vision Analysis & Post-Filtering (Top 90 Curated Photos)
   │            [photobook/analyzer.py :: analyze_photos_with_gemini + pipeline.py post-filter]
   │
   ├─► Stage 5: Scenic Hero Cover Selection & Single-Photo Precision Refinement
   │            [photobook/analyzer.py :: select_hero_cover_photo]
   │
   ├─► Stage 6: Chronological Story Clustering & Chapter Titling (Gemini 2.5 Flash-Lite / 3.8 Flash)
   │            [photobook/analyzer.py :: curate_clusters_and_pages -> _default_clusters_and_pages]
   │
   ├─► Stage 7: 10×10" Page Template Selection & 100% Face-Safe Aesthetic Cropping
   │            [photobook/analyzer.py :: choose_best_page_template + photobook/cropping.py :: compute_aesthetic_crop]
   │
   └─► Stage 8: Output Rendering (10×10" Print PDF, 3D Lay-Flat HTML Book & Analysis Reports)
                [photobook/renderer.py :: render_pdf_photobook, render_html_photobook, save_analysis_files]
```

---

## Detailed Stage-by-Stage Breakdown

### Stage 1: Ingestion, EXIF Extraction, Perceptual Hashes & Scaling (All 700+ Photos)
**Entry point:** [`scan_and_prepare_photos`](../photobook/ingestion.py)

Before calling any AI API, every supported image file (`.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`, `.heif`, `.tiff`, `.bmp`) in the input folder is processed in parallel across CPU worker threads:

1. **Strict Input Isolation**:
   - Reads only image files (ignoring `.txt`, `.md`, `.json`, `.pdf`, or hidden files) so external notes or metadata never pollute the analysis.
2. **EXIF Capture Timestamp & Camera Metadata**:
   - Applies `ImageOps.exif_transpose` to honor EXIF rotation tags.
   - Extracts `DateTimeOriginal` / `DateTime` (`captured_at` and `captured_timestamp`) and camera `Make` / `Model` (`camera_model`).
3. **Resolution Normalization (`1,120–2,240 px` Default)**:
   - Scales the longest edge into the target analysis range (default `1,120` to `2,240 px`, or `384–512 px` in `low_384` fast mode) and caches the scaled JPEG under `output/.photo_analysis_cache/`.
4. **Perceptual Hashes & Color Signatures**:
   - **`dHash` (64-bit Difference Hash)**: Captures horizontal luminance gradients on a `9×8` grid for structural comparison.
   - **`aHash` (64-bit Average Hash)**: Captures `8×8` mean luminance structure.
   - **`color_signature` (24-Bin Normalized RGB Histogram)**: 8 bins per R/G/B channel used to compute cosine color similarity (`0.0` to `1.0`) between frames.

---

### Stage 2: Fast Local Pixel-Level Pre-Analysis & Initial Scoring (All 700+ Photos)
**Entry point:** [`pre_analyze_all_photos`](../photobook/analyzer.py) → [`_analyze_single_photo_pixels`](../photobook/analyzer.py)

To avoid spending 700+ multimodal API calls on blurry frames, screenshots, or redundant camera bursts, 8 CPU threads perform a fast pixel-level inspection on every scanned photo:

1. **Edge-Gradient Sharpness Score (`sharpness_score`)**:
   - Computes adjacent pixel luminance differences across a `256×256` grayscale thumbnail and averages the top 8.3% strongest edges (`top_edges`).
   - Maps to a `2.5–9.8` sharpness scale. Any photo with `sharpness < 3.6` is flagged immediately as motion-blurred or out-of-focus (`is_blurry_or_flawed = True`).
2. **Pure-Pixel Document & Screenshot Detection (`is_document_or_screenshot`)**:
   - Flags PNG files lacking camera EXIF (`is_png_without_camera`), filenames matching document/receipt keywords, or bright neutral pages (`avg_brightness > 205`, `color_saturation < 11.0`, `avg_std > 30.0`, and high bimodal text-edge transitions `sharp_text_jumps > 0.045`).
3. **YCbCr Skin-Tone Head/Face Cluster Detection**:
   - [`_detect_skin_head_boxes_from_pixels`](../photobook/analyzer.py) scans the upper 70% of the frame in `YCbCr` color space (`Y > 65`, `80 <= Cb <= 124`, `136 <= Cr <= 172`) to detect plausible portrait head clusters and estimate initial `[0..1000]` face bounding boxes.
4. **Initial Aesthetic, Emotional & Composite Pre-Score**:
   - Combines sharpness, color saturation (`color_saturation`), luminance dynamic range (`avg_std`), aspect ratio balance, and face presence into an initial `composite_score`:
     $$\text{composite\_score} = 0.45 \times \text{aesthetic} + 0.30 \times \text{sharpness} + 0.25 \times \text{emotional}$$

---

### Stage 3: Deduplication, Burst Filtering & Chronological Time-Bucket Curation (700+ → 90 Photos)
**Entry point:** [`deduplicate_and_filter_photos`](../photobook/analyzer.py)

This stage narrows a large 700+ photo collection down to the target `max_book_photos` (default `90`) **before** invoking the Gemini Vision API:

1. **Hard Quality & Document Exclusions**:
   - Excludes all photos flagged in Stage 2 as documents/screenshots (`excluded_document`), blurry/low-sharpness (`sharpness_score < 3.6`, `excluded_low_quality`), or sideways/inverted (`is_sideways_or_inverted_photo`).
2. **Quality-Ranked Duplicate & Camera-Burst Removal**:
   - Sorts all surviving candidates from **highest to lowest** `(composite_score, sharpness_score)`.
   - Iterates down the quality-ranked list and compares each candidate against already-kept higher-scoring photos:
     - **Exact / Near Duplicate (`excluded_duplicate`)**: `dHash distance == 0`, `aHash distance == 0`, and `color_similarity >= 0.998`.
     - **Perceptually Similar Shot (`excluded_similar`)**: `dHash <= 6`, `aHash <= 6`, and `color_similarity >= 0.985`.
     - **Camera Time-Burst Shot (`excluded_similar`)**: Captured within `Δt <= 3.0 minutes` with `color_similarity >= 0.94` and `dHash <= 20`.
     - **Semantic Burst Shot (`excluded_similar`)**: Captured within `Δt <= 2.0 hours` with `dHash <= 15`, `color_similarity >= 0.965`, and token Jaccard similarity `>= 0.55`.
   - Because candidates are evaluated in descending quality order, the sharpest, best-composed photo in every burst is automatically kept while lower-rated burst mates are excluded.
3. **Chronological Time-Bucket Curation to `max_book_photos` (e.g., 90)**:
   - If more than `max_book_photos` (e.g., 350 unique photos out of 700+) survive deduplication, they are sorted chronologically from the start to the end of the trip.
   - The timeline is divided into `max_book_photos` (90) equal chronological windows (`bucket_size = len(kept) / 90`).
   - Within each chronological window, the single photo with the highest `(composite_score, sharpness_score)` is selected (`best_in_window`), and the remaining photos in that window are marked `excluded_similar`.
   - **Why this works well:** It guarantees that the entire timeline of your trip (beginning, middle, and end) is evenly represented while selecting the sharpest, highest-quality frame from every time segment. *(Setting **Max Curated Photos in Book** to `0` disables this cap and sends all non-duplicate photos to Gemini.)*

---

### Stage 4: Deep Gemini 3.8 Flash Vision Analysis & Post-Filtering (Top 90 Curated Photos)
**Entry point:** [`analyze_photos_with_gemini`](../photobook/analyzer.py) & post-Gemini filter in [`build_photo_book`](../photobook/pipeline.py)

The curated photos (e.g., 90 photos) are analyzed by **Gemini 3.8 Flash** (`gemini-3.8-flash`):

1. **Persistent Disk Cache**:
   - Checks `output/.photo_analysis_cache/` for existing `.analysis.<model>.<resolution>.json` files so previously analyzed photos return instantaneously (`0ms`).
2. **Multi-Image Batch Vision Calls (`batch_size = 6` Default)**:
   - Groups uncached photos into batches of up to 6 images per request and executes parallel worker calls (`ThreadPoolExecutor(max_workers=6)`, HTTP timeout `88,000 ms`, batch worker timeout `96.0s`).
   - For each photo, Gemini returns structured JSON ([`GeminiPhotoAnalysis`](../photobook/models.py)):
     - `is_document_or_screenshot` & `is_blurry_or_flawed`
     - `has_sideways_or_inverted_orientation` (true if faces or the horizon are rotated 90° sideways or 180° upside down, such as older cameras lacking an EXIF orientation sensor)
     - `aesthetic_score`, `sharpness_score`, `emotional_score`, `cover_worthiness_score`
     - `location_or_landmark`, `event_theme`, `subject_summary`, and a 1–2 line `caption` (clamped to `<= 120` chars)
     - `faces_boxes`: Normalized `[0..1000]` bounding boxes covering every visible human head from the top of the hair/forehead to the chin
     - `salient_subject_box`: Normalized `[0..1000]` bounding box of the primary visual subject
3. **Post-Gemini Safety Filter**:
   - Immediately after Gemini Vision completes, [`build_photo_book`](../photobook/pipeline.py) inspects the updated Gemini analysis and removes any photo newly identified by Gemini as:
     - a document/screenshot (`is_document_or_screenshot`),
     - blurry/flawed (`is_blurry_or_flawed`), or
     - having a sideways or upside-down face/scene orientation ([`is_sideways_or_inverted_photo`](../photobook/analyzer.py)).

---

### Stage 5: Scenic Hero Cover Selection & Single-Photo Precision Refinement
**Entry point:** [`select_hero_cover_photo`](../photobook/analyzer.py) & [`curate_clusters_and_pages`](../photobook/analyzer.py)

1. **Scenic-First Hero Cover Selection**:
   - [`select_hero_cover_photo`](../photobook/analyzer.py) gives priority to **scenic landscape/square views** (`people_count == 0`, `len(faces_boxes) == 0`, `cover_worthiness_score >= 6.8`, `aesthetic_score >= 7.0`, balanced aspect ratio `0.70–2.15`, and no indoor/tabletop object keywords).
   - Scenic candidates are ranked by `cover_worthiness_score + 0.55 (landscape/square bonus)`, `aesthetic_score`, and `sharpness_score`.
   - **Fallback**: If no qualifying scenic view exists in the collection, it automatically falls back to the highest-rated overall photo (`cover_worthiness_score`, `composite_score`, `sharpness_score`).
2. **Hero Cover Single-Photo Precision Pass (`hero_model = gemini-3.8-flash`)**:
   - When `batch_size > 1`, the winning Hero Cover photo is sent in a dedicated single-image, high-resolution (`media_resolution="high"`, `thinking_level="medium"`, timeout `72,000 ms`) call to `gemini-3.8-flash` for ultra-precise cropping coordinates and cover captioning.

---

### Stage 6: Chronological Story Clustering & Chapter Titling
**Entry point:** [`_default_clusters_and_pages`](../photobook/analyzer.py) & Stage 2 Curation in [`curate_clusters_and_pages`](../photobook/analyzer.py)

1. **Deterministic Chronological Day/Theme Grouping**:
   - Groups selected photos in strict chronological order by capture date and visual theme (`event_theme` and `location_or_landmark`).
   - Synthesizes initial chapter titles and 1–2 line group captions.
2. **Fast LLM Book Title & Chapter Refinement (`curation_model = gemini-2.5-flash-lite`)**:
   - Sends a compact JSON summary of each cluster's date range, landmarks, themes, and sample captions to `gemini-2.5-flash-lite` (`thinking_budget=0`, timeout `40,000 ms`) or `gemini-3.8-flash` to generate a cohesive `book_title` and polished chapter `title` + `group_caption` for every chapter.

---

### Stage 7: 10×10" Page Template Selection & 100% Face-Safe Aesthetic Cropping
**Entry point:** [`choose_best_page_template`](../photobook/analyzer.py) & [`compute_aesthetic_crop`](../photobook/cropping.py)

1. **1 to 3 Photos Max per 10×10" Page**:
   - Each interior page holds **1, 2, or 3 photos maximum** across 11 editorial 10×10" templates (`PAGE_TEMPLATES` in [`photobook/analyzer.py`](../photobook/analyzer.py)):
     - **Cover (`cover_hero`)**: 1 hero slot (`7.8" × 5.85"` at `x=1.1", y=1.65"`).
     - **1-Photo Pages**: `single_full_square`, `single_editorial` (portrait), `single_panoramic` (wide landscape).
     - **2-Photo Pages**: `two_side_by_side`, `two_stacked_horizontal`, `two_asymmetric_left`, `two_asymmetric_right`.
     - **3-Photo Pages**: `three_hero_top`, `three_hero_left`, `three_columns`.
   - All template slot coordinates are positioned to guarantee zero overlap with top chapter headers, outer trim borders (`inset: 3.2%`), two-line individual slot subcaptions, and bottom group captions.
2. **100% Face-Preserving Aesthetic Crop (`compute_aesthetic_crop`)**:
   - Computes the union of all detected `faces_boxes` in pixel coordinates with a **12% safety margin above the head/forehead** (`pad_top = 0.12 * box_h`) and **8% margin below the chin and on the sides**.
   - Searches for the largest crop window matching the target slot aspect ratio centered on the salient subject while **strictly enclosing 100% of the padded face union**.
   - **Automatic Template Backoff (`_can_page_preserve_all_faces`)**: If a multi-photo template's slot aspect ratio is too narrow/wide to fit every face in a group photo without clipping, the layout engine automatically steps down from 3 photos → 2 photos → 1 photo on that page so no face is ever cut off.
   - Crops are rendered in parallel at print resolution (`200 DPI`) into `output/<book_id>/crops/`.

---

### Stage 8: Output Rendering (10×10" Print PDF, 3D Lay-Flat HTML Book & Analysis Reports)
**Entry point:** [`photobook/renderer.py`](../photobook/renderer.py) & [`static/index.html`](../static/index.html)

1. **10×10 Inch Print-Ready PDF (`photobook_10x10.pdf`)**:
   - Rendered via ReportLab (`720 × 720 pt`) in [`render_pdf_photobook`](../photobook/renderer.py) with dark charcoal & gold-bordered Hero Cover, warm cream interior pages (`#FAF7F2`), chapter headers, white photo frames, individual 2-line captions, group captions, and page numbers.
2. **Standalone 10×10" Lay-Flat HTML Photo Book (`photobook.html`)**:
   - Rendered in [`render_html_photobook`](../photobook/renderer.py) with self-contained Base64-embedded crops, **180° Lay-Flat 2-Page Spreads** (1px micro-fold seam, zero spine gutter shadow, continuous perimeter trim border), interactive **✨ 3D Lay-Flat Flip Viewer**, and **▶ Autoplay** controls.
3. **Interactive Web UI (`http://127.0.0.1:8088`)**:
   - Served by [`app.py`](../app.py) and [`static/index.html`](../static/index.html) with live SSE progress streaming, elapsed timer, stage-by-stage timing breakdown, **✨ 3D Lay-Flat Flip Viewer** (rigid board 3D page turn animation + mouse-hover 3D tilt), **▶ Autoplay** with adjustable speed (`2.5s` to `8.5s` per spread), Face-Safe Crop & Slot Inspector overlays, and full inclusion/exclusion audit tables.
4. **Structured Analysis Reports (`photo_analysis.json` & `photo_analysis.md`)**:
   - Saved by [`save_analysis_files`](../photobook/renderer.py) with every scanned photo's scores, face counts, capture timestamps, and exact exclusion reasons.
