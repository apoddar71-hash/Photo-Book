# 10×10 AI Photo Book Builder — `Gemini-first` Branch Documentation

This document provides a comprehensive overview of all architectural, algorithmic, UI, and test changes introduced in the **`Gemini-first`** branch of the **10×10 Inch AI Photo Book Builder**.

---

## 1. Executive Summary & Motivation

In the legacy pipeline (`main` branch), raw user photos passed through a local CPU-based heuristic filter (Laplacian edge sharpness, saturation, and color histogram comparisons) to compress photo batches down to the book budget (e.g. 90 photos) *before* invoking Gemini vision models.

While computationally inexpensive, this "local-filter-first" approach had fundamental limitations:
1. **Expression-blind burst selection**: Local edge sharpness often selected a photo where a subject blinked or made an awkward face simply because their hair, eyeglasses, or clothing had higher contrast edges than an adjacent smiling shot.
2. **Premature culling of gems**: Subtle, emotionally rich moments (soft lighting, heartwarming interactions) were pruned by edge-threshold heuristics before Gemini's semantic intelligence could evaluate them.
3. **Sideways/inverted photo leakage**: Sideways photos could survive local filtering and consume valuable page slots before Gemini detected their orientation.

The **`Gemini-first`** branch shifts the paradigm to a **Smart Hybrid Pipeline**, where **all unique surviving photos undergo deep multimodal evaluation by Gemini 3.8 Flash Vision first**, ensuring burst selection, timeline curation, and page assignment are driven by semantic understanding.

In addition to the Smart Hybrid pipeline, this branch introduces:
- **Configurable Page Layout Fill (`high`, `medium`, `low`)**: Allowing users to maximize photo coverage across the 10×10" page with tight margins (down to 0.25") while preventing header/caption overlap.
- **Tight Aesthetic Smart Cropping (`tight`, `balanced`, `wide`)**: Intelligently trimming peripheral dead space (sky, ground, walls) to emphasize human subjects while guaranteeing 100% face safety.
- **Photo Captions Toggle (`include_captions`)**: Allowing users to choose whether to render AI-generated story captions under photos or create a clean, minimalist photo-only gallery.
- **High-Throughput Parallel Concurrency (`concurrency`)**: Configurable parallel workers (up to 20/32 concurrent requests) and 10 photos/call batch processing, eliminating serial bottlenecks to drastically reduce end-to-end vision analysis time across large libraries (500+ photos).
- **Clean Styling & Dynamic Cover Location**: Removed rigid boilerplate labels (`"CURATED 10×10 INCH PHOTO BOOK"`, `"A Curated 10×10 Photo Book"`) and dates from all pages, and dynamically elevated recognized geographic locations (e.g. `📍 San Juan, Puerto Rico`) to the cover page eyebrow and subtitle.
- **Dashboard UI Controls**: Direct dropdowns and status badges on the web UI for fill modes, cropping focus, caption display, and parallel concurrency.

---

## 2. Legacy vs. Smart Hybrid (Gemini-First) Comparison

| Feature / Step | Legacy Flow (`main`) | Smart Hybrid Flow (`Gemini-first`) |
|---|---|---|
| **Early Pruning** | Pruned down to 90 photos on local CPU using edge sharpness & saturation. | **Coarse filter only** (<1s): drops exact binary duplicates (`dHash=0, aHash=0`) and corrupted files. Retains all valid photos. |
| **Gemini Vision Stage** | Evaluated only the 90 pre-filtered photos. | **Evaluates all surviving unique photos** (e.g., 500–700+) using parallel multi-image batching. |
| **Burst Deduplication** | Selected the sharpest pixel-edge frame in each time window. | **Gemini Expression & Aesthetic Rank**: Chooses the frame where people are genuinely smiling, eyes open, with high emotional score. |
| **Sideways Filtering** | Flagged after photo count was already reduced. | Filtered out during semantic curation before slot assignment. |
| **Chronological Downsampling** | Local CPU heuristic picks per bucket. | Divided into equal time buckets, picking the highest **Gemini composite quality** photo in each bucket. |
| **Page Layout Fill** | Fixed templates with large white margins (0.75"–1.00"). | **Selectable Fill Modes**: `high` (0.25"–0.35" gaps, 93% width), `medium`, or `low`. |
| **Cropping Intelligence** | Centered on face cluster without zooming out irrelevant background. | **Aesthetic Tight Cropping**: Zooms in on key subjects, removing irrelevant periphery with 100% face-safe margins. |

---

## 3. End-to-End Smart Hybrid Pipeline Architecture

```
Raw Photo Folder (e.g., 500+ .JPG / .HEIC / .PNG / .WEBP files)
   │
   ├─► Stage 1: Ingestion, EXIF Orientation, Perceptual Hashes & Scaling
   │            [photobook/ingestion.py :: scan_and_prepare_photos]
   │            • Reads personal photos, fixes EXIF rotation, generates 64-bit dHash/aHash and color histograms.
   │
   ├─► Stage 2: Coarse Fast Pruning on Local CPU (<1 second)
   │            [photobook/analyzer.py :: coarse_filter_photos]
   │            • Drops exact duplicates (dHash==0, aHash==0, color_sim >= 0.998).
   │            • Drops unreadable frames or extreme white paper receipts.
   │            • Preserves all legitimate scene and portrait photos without downsampling.
   │
   ├─► Stage 3: Deep Gemini 3.8 Flash Vision on ALL Surviving Photos
   │            [photobook/analyzer.py :: analyze_photos_with_gemini]
   │            • Processes uncached photos in parallel batches (batch_size=6, 6 parallel workers).
   │            • Extracts exact [0..1000] face boxes, emotional expressions, aesthetic scores,
   │              landmarks, sideways orientation, and 1-2 line captions.
   │            • Caches results to disk (.analysis.gemini-3.8-flash.json) for instant re-runs.
   │
   ├─► Stage 4: Gemini-Powered Semantic Deduplication & Chronological Curation
   │            [photobook/analyzer.py :: deduplicate_and_curate_with_gemini]
   │            • Excludes Gemini-flagged sideways/inverted images and documents.
   │            • In rapid bursts (Δt <= 3 min, visual similarity), keeps the best-expressed photo
   │              (smiling vs. blinking) based on Gemini emotional and aesthetic scores.
   │            • Downsamples across chronological timeline buckets using Gemini composite scores.
   │
   ├─► Stage 5: Scenic Hero Cover Selection & Precision Pass
   │            [photobook/analyzer.py :: select_hero_cover_photo]
   │            • Prioritizes panoramic/scenic views without people (cover_worthiness >= 6.8, aesthetic >= 7.0).
   │            • Automatic fallback to top overall photo.
   │
   ├─► Stage 6: Story Clustering & Chapter Titling
   │            [photobook/analyzer.py :: curate_clusters_and_pages]
   │            • Organizes photos into chronological narrative chapters with titles and captions.
   │
   ├─► Stage 7: 10×10" Page Layout Selection & Tight Aesthetic Smart Cropping
   │            [photobook/analyzer.py :: choose_best_page_template + photobook/cropper.py :: compute_aesthetic_crop]
   │            • Applies selected Fill Mode (high / medium / low) across 11 templates.
   │            • Applies Aesthetic Crop Focus (tight / balanced / wide) with 100% face-safe margins.
   │
   └─► Stage 8: Output Generation
                [photobook/renderer.py :: render_pdf_photobook, render_html_photobook, save_analysis_files]
                • Generates print-ready 10×10" PDF (ReportLab 720×720 pt).
                • Generates interactive 3D Lay-Flat HTML book viewer with autoplay.
                • Generates photo_analysis.json and photo_analysis.md audit reports.
```

---

## 4. Feature Detail: Page Layout Fill Options (`high`, `medium`, `low`)

### The Problem
In the initial release, 10×10" pages had fixed, conservative photo boxes with large gaps (0.75"–1.00"), leaving substantial unused white space around photos.

### The Solution: `PAGE_TEMPLATES_BY_FILL`
The layout engine in [`photobook/analyzer.py`](../photobook/analyzer.py) now provides three distinct fill tiers across all 11 page templates (1-photo full/hero, 2-photo side-by-side/stacked, 3-photo triptych/grid/featured):

```
+--------------------------------------------------------------+
|                        10" x 10" Page                        |
|   Header Area: y = 0.0" to 1.10" (Chapter title & subtitle)   |
|  +--------------------------------------------------------+  |
|  |                                                        |  |
|  |              PHOTO SLOT (High Fill Mode)               |  |
|  |       Width: 9.20" - 9.30" (92-93% of page width)      |  |
|  |       Height: 7.50" (75% of page height)               |  |
|  |       Margins / Gaps: 0.25" - 0.35"                    |  |
|  |                                                        |  |
|  +--------------------------------------------------------+  |
|   Caption Area: y = 8.60" to 10.0" (Photo captions & metadata)|
+--------------------------------------------------------------+
```

### Modes Breakdown:
1. **`high` Fill Mode**:
   - **Photo Coverage**: Photos span **92% to 93%** of page width (`9.20"–9.30"`) and up to **7.50"** height.
   - **Gaps**: Reduced from `0.75"` to **`0.25"–0.35"`**.
   - **Text Safety**: Leaves `1.10"` above for chapter headers and `1.40"` below for captions with zero overlap.
2. **`medium` Fill Mode**:
   - **Photo Coverage**: ~8.40" width, 6.80" height.
   - **Gaps**: `0.50"` between photos.
   - **Balance**: Elegant balance between image prominence and breathing room.
3. **`low` Fill Mode**:
   - **Photo Coverage**: 7.60"–8.00" width, 6.00" height.
   - **Gaps**: `0.75"–1.00"`.
   - **Aesthetic**: Classic fine-art gallery layout with generous white borders.

---

## 5. Feature Detail: Tight Aesthetic Smart Cropping (`tight`, `balanced`, `wide`)

### The Problem
Photos often contain extraneous peripheral dead space (empty sky, asphalt, blank walls, or background clutter) that detracts from the emotional core and subjects of the photo.

### The Solution: `crop_focus` in `compute_aesthetic_crop`
Located in [`photobook/cropper.py`](../photobook/cropper.py), the cropping engine supports three focus modes:

1. **`tight` Mode**:
   - Computes the bounding hull of all detected human faces and key subject features.
   - Trims away peripheral non-essential areas by applying a tighter crop window (zoom factor up to 1.35x).
   - Re-centers the crop frame on the emotional center of mass.
   - **100% Face Safety Guarantee**: Enforces that all face boxes (plus safety margin) remain completely inside the visible window. If a tight crop would clip any face, the algorithm automatically relaxes the crop boundary to preserve the face.
2. **`balanced` Mode** (Default):
   - Balances subject emphasis with surrounding context.
   - Standard 12% forehead margin and natural rule-of-thirds alignment.
3. **`wide` Mode**:
   - Keeps maximum environmental context, ideal for landscape architecture and wide travel horizons.

---

## 6. Feature Detail: Photo Captions Toggle (`include_captions`)

### The Purpose
Some users want a rich documentary narrative with AI-synthesized story lines and moment descriptions under each photo, while others prefer a minimalist, clean fine-art photography monograph with zero text beneath individual images.

### Implementation
- **Data Model**: Added `include_captions: bool = True` to `BuildRequest` ([`app.py`](../app.py)) and `PhotoBookResult` ([`photobook/models.py`](../photobook/models.py)).
- **PDF Generation**: In [`photobook/renderer.py`](../photobook/renderer.py), when `include_captions=False`, slot subcaptions, page group captions, and cover hero photo captions are cleanly omitted. Outer borders, chapter headers, and page numbering remain perfectly balanced.
- **Standalone HTML Book & 3D Viewer**: In [`photobook/renderer.py`](../photobook/renderer.py) and [`static/index.html`](../static/index.html), caption DOM nodes (`.slot-subcaption`, `.page-group-caption`, `.cover-hero-caption`, `.slot-individual-cap`) are suppressed dynamically, letting the imagery command full visual attention.
- **Audit Logs**: Deep Gemini semantic captioning is still preserved in `photo_analysis.json` and `photo_analysis.md` for auditing, searchability, and chronological clustering.

---

## 7. Feature Detail: Clean Styling & Dynamic Cover Location

### The Problem
Earlier builds rendered repetitive boilerplate headers (`"CURATED 10×10 INCH PHOTO BOOK"`, `"A Curated 10×10 Photo Book"`) and explicit ISO date ranges (`(2024-04-12 to 2024-04-15)`) across cover eyebrows, subtitles, and chapter headers. This looked mechanical and detracted from the destination and emotional themes.

### The Solution
1. **Purged Boilerplate & Dates**:
   - Removed `"CURATED 10×10 INCH PHOTO BOOK"` from the front cover and endpapers.
   - Removed `"A Curated 10×10 Photo Book"` from book subtitles.
   - Stripped raw date stamps (`(2024-04-12)`, `2024-04-12 to 2024-04-15`) from chapter titles, headers, subtitles, and captions across all pages via [`strip_curated_and_dates`](../photobook/analyzer.py).
   - Chapter headers now read cleanly as `Chapter 1 — Historic Old Town` or `Memories & Highlights`.
2. **Dynamic Geographic Location on Cover**:
   - Added [`extract_book_location`](../photobook/analyzer.py) to extract the primary recognized landmark/place name from the Hero cover photo and consensus across Gemini 3.8 Flash Vision photo tags.
   - On the cover page:
     - Top Eyebrow: Displays the uppercase destination (e.g. `SAN JUAN, PUERTO RICO`). If no location is detected, the eyebrow is cleanly omitted rather than showing placeholder text.
     - Cover Subtitle: Displays `📍 San Juan, Puerto Rico` or destination narrative.
   - Preserves `location: Optional[str]` on `PhotoBookResult` ([`photobook/models.py`](../photobook/models.py)) for downstream apps and UI badge display.

---

## 8. High-Throughput Parallel Concurrency & Ultra-Fast Batching

### The Problem
In earlier iterations, vision API requests ran with a hardcoded worker pool (`max_workers = min(6, len(batches))`) and executed a blocking serial probe batch upfront before launching remaining jobs. For large collections (e.g., 500–700+ photos), vision analysis accounted for over 85–90% of the total pipeline run time (~204 seconds out of 232 seconds across 112 batches).

### The Solution
1. **Configurable Concurrency Engine**:
   - [`analyze_photos_with_gemini`](../photobook/analyzer.py) accepts a `concurrency: int = 12` parameter (supporting 1 to 32 concurrent requests).
   - `ThreadPoolExecutor` dynamically scales to `min(effective_concurrency, len(batches))` for batch mode and `min(effective_concurrency, len(remaining_photos))` for single mode.
2. **Zero-Latency Dispatch (Probe Removal)**:
   - Eliminated the blocking serial probe batch; all worker requests launch concurrently in parallel from millisecond zero.
3. **Automatic Exponential Backoff on HTTP 429**:
   - Added automatic retry with exponential backoff on transient quota/rate limit errors (up to 3 attempts with 2.0s delay), preventing dropped batches under high parallel load.
4. **Ultra-Fast 10-Photo Batching Option**:
   - Added `10 Photos / Call (Ultra Fast)` preset alongside `6 Photos / Call (Standard)`, reducing HTTP round-trip overhead by ~40% for large libraries.
5. **Expected Speedups**:
   - 668 photos at batch size 6 (112 batches): with 20 parallel workers, runtime drops from ~204s to **~45–60s** (a ~3.5×–4× throughput boost).
   - At batch size 10 (67 batches) with 20 workers, runtime drops to **~30–40s** (a ~5×–6× throughput boost).

---

## 9. Live Customer Engagement Panel & Real-Time Discovery Feed

### The Problem
During generation across hundreds of photos (which can take 30–90+ seconds), traditional UIs only show a static percentage progress bar and text log. Customers had no visual feedback on what memories were being explored or how Gemini was interpreting their photos until the entire book finished rendering.

### The Solution
1. **Live Gemini Vision Discovery Panel**:
   - Added a prominent live curation card directly below the build controls in [`static/index.html`](../static/index.html).
   - Features an animated green beacon dot (`pulse-dot`) indicating active real-time AI perception.
   - Live discovery stats chips:
     - 📸 **Photos Explored**: Real-time counter of analyzed photos.
     - 🏛️ **Landmarks Discovered**: Dynamic count of recognized places/landmarks.
     - ⭐ **Avg Quality**: Live rolling average of aesthetic and composition scores.
     - 👥 **Portraits Safe**: Live tally of detected human faces safely framed.
2. **Featured Spotlight Showcase with CSS Flash Animation**:
   - A dedicated hero card dynamically flashes the newest photo analyzed by Gemini.
   - Includes a visual flash animation (`spotlight-active-flash`) simulating an optical flash upon analysis.
   - Displays a floating badge with aesthetic rating and detected setting (e.g. `✨ 9.3 • Sunset`).
   - Quotes the custom Gemini caption in an elegant serif styling: `“The afternoon sun casting long dramatic shadows across the cobblestones...”`
   - Highlights metadata tags: Landmark pills, event themes, quality scores, face-safe indicators, and camera models.
3. **Smooth Rotating Slideshow**:
   - As parallel batches complete, photos are queued and smoothly cycled every 1.8 seconds so the customer enjoys a non-stop, cinematic preview of their memories being curated.
4. **Interactive Discovery Filmstrip**:
   - A horizontal scrolling roll displays the recent 40–50 analyzed photo thumbnails.
   - Clicking any thumbnail immediately spotlights that photo, updates its caption card, and pauses auto-rotation for 6 seconds.
5. **Secure Image Preview API (`/api/photos/preview`)**:
   - Added in [`app.py`](../app.py) to stream lightweight, pre-scaled JPEG previews directly to the browser.
   - Supported by `_PHOTO_CACHE_MAP` and disk cache fallbacks with HTTP cache headers.

---

## 10. Dashboard & User Interface Enhancements

File: [`static/index.html`](../static/index.html)

1. **Live Customer Engagement Panel**:
   - Pulsing live status indicator, featured spotlight card, live captions, metadata tags, and clickable recent discoveries filmstrip.
   - Includes `Hide / Show Live Feed` toggle button.
2. **Parallel Concurrency Selector**:
   - Dropdown under Settings:
     - `20 Concurrent Requests (Blazing Fast)`
     - `16 Concurrent Requests (High Speed)`
     - `12 Concurrent Requests (Default / Balanced)`
     - `8 Concurrent Requests (Moderate)`
     - `4 Concurrent Requests (Conservative)`
     - `1 Request (Serial / Debug)`
3. **Batch Size Selector with 10-Photo Preset**:
   - Added `10 Photos / Call (Ultra Fast)` option alongside `6 Photos / Call (Standard)` and single-photo mode.
4. **Page Layout Fill Selector**:
   - Dropdown under Settings:
     - `High (Maximum photo coverage, minimal gaps)` *(Default)*
     - `Medium (Balanced spacing)`
     - `Low (Gallery style, wide white margins)`
5. **Aesthetic Crop Focus Selector**:
   - Dropdown under Settings:
     - `Tight (Aesthetic close-up, remove dead space)` *(Default)*
     - `Balanced (Standard balanced framing)`
     - `Wide (Maximum environmental context)`
6. **Photo Captions Selector**:
   - Dropdown under Settings:
     - `Include Captions (Story & notes under photos)` *(Default)*
     - `No Captions (Clean photo-only gallery)`
7. **Live Status Badges**:
   - Dashboard summary bar displays active configuration pills:
     - `Parallel: <N> reqs (batch <B>)`
     - `Fill: High` / `Fill: Medium` / `Fill: Low`
     - `Crop: Tight` / `Crop: Balanced` / `Crop: Wide`
     - `Captions: Yes` / `Captions: No`
     - `Location: <Recognized Destination>` (when available)
8. **Interactive 3D Viewer & Controls**:
   - Integrates with the 3D Lay-Flat viewer, book flipping, and autoplay features.

---

## 11. API and Data Model Changes

### `app.py`
- Updated `/api/build` and `/api/build-stream` endpoints:
  ```python
  class BuildRequest(BaseModel):
      photos_dir: Optional[str] = None
      folder_path: Optional[str] = None
      title: str = "Our Unforgettable Journey"
      subtitle: str = "A Visual Chronicle"
      target_pages: int = 40
      max_book_photos: int = 90
      page_fill_mode: str = "high"       # "low", "medium", "high"
      crop_focus: str = "tight"          # "tight", "balanced", "wide"
      include_captions: bool = True      # True = show captions, False = clean gallery
      concurrency: int = 12              # 1 to 32 concurrent vision workers
  ```
- Added `/api/photos/preview` endpoint for live streaming thumbnails:
  ```python
  @app.get("/api/photos/preview")
  def preview_photo(id: Optional[str] = None, file: Optional[str] = None) -> FileResponse:
      ...
  ```

### `photobook/models.py`
- Added `page_fill_mode: str = "high"`, `crop_focus: str = "tight"`, `include_captions: bool = True`, `location: Optional[str] = None`, and `concurrency: int = 12` to `PhotoBookResult`.

### `photobook/pipeline.py`
- `build_photo_book` accepts `concurrency: int = 12`, passes it to `analyze_photos_with_gemini`, and records it on `PhotoBookResult`.
- Streams `scanned_photos` and `analyzed_photos` events through `_safe_progress`.

---

## 12. Test Suite & Verification

All changes are covered by comprehensive automated tests in [`tests/test_photobook.py`](../tests/test_photobook.py):

1. **`test_smart_hybrid_gemini_first_flow`**:
   - Tests that Stage 2 coarse filtering eliminates 100% exact duplicate files without arbitrarily pruning good photos.
   - Verifies Stage 3 Gemini enrichment on all surviving candidates.
   - Verifies Stage 4 deduplication selects the smiling subject over the blinking subject in camera bursts.
   - Verifies chronological bucket downsampling accurately respects `max_book_photos`.
2. **`test_page_fill_modes_and_aesthetic_crop_focus`**:
   - Validates slot dimensions and margins for `low`, `medium`, and `high` fill modes across multiple templates.
   - Confirms that `high` fill slots span >= 9.20" width and >= 7.50" height.
   - Tests `tight`, `balanced`, and `wide` crop focus on images with face bounding boxes, confirming dead space removal and 100% face safety.
3. **`test_include_captions_toggle_in_pipeline_and_renderers`**:
   - Verifies that `include_captions=False` suppresses slot captions and footer captions in both PDF and HTML renders while preserving valid PDF output.
   - Verifies that `include_captions=True` renders story captions under photos.
4. **`test_remove_curated_and_dates_and_include_location_on_cover`**:
   - Verifies that `"CURATED 10×10 INCH PHOTO BOOK"` and date patterns are completely stripped from HTML and PDF pages.
   - Confirms dynamic detection and rendering of geographic locations on the cover.
5. **`test_multi_model_and_batch_processing`**:
   - Verifies batch analysis with custom `concurrency: 16` and confirms result retains configured concurrency metadata.
6. **`test_live_engagement_photo_preview_endpoint_and_events`**:
   - Verifies `/api/photos/preview` serves images with HTTP 200 and image/jpeg Content-Type.
   - Verifies `_format_photo_preview` data structures including captions, landmarks, quality scores, and face counts.

### Test Execution
```bash
uv run pytest
# Output: 14 passed in 1.73s
```

---

## 13. Summary of Modified Files on `Gemini-first`

| File | Primary Changes |
|---|---|
| [`photobook/analyzer.py`](../photobook/analyzer.py) | Added configurable `concurrency` pool, `_format_photo_preview`, `_safe_progress` streaming, removed blocking serial probe batch, added 429 exponential backoff, added `coarse_filter_photos`, `deduplicate_and_curate_with_gemini`, `extract_book_location`, `strip_curated_and_dates`, `PAGE_TEMPLATES_BY_FILL`, and `get_page_templates`. |
| [`photobook/ingestion.py`](../photobook/ingestion.py) | Added live scan preview streaming (`scanned_photos`) so photos immediately populate the UI during folder ingestion. |
| [`photobook/cropper.py`](../photobook/cropper.py) | Added `crop_focus` parameter (`tight`, `balanced`, `wide`) to `compute_aesthetic_crop` with dead-space removal and face preservation. |
| [`photobook/pipeline.py`](../photobook/pipeline.py) | Orchestrates Smart Hybrid pipeline, extracts location, wires `concurrency` parameter, and threads live event payloads through `_safe_progress`. |
| [`photobook/models.py`](../photobook/models.py) | Added `page_fill_mode`, `crop_focus`, `include_captions`, `location`, and `concurrency` fields to `PhotoBookResult`. |
| [`app.py`](../app.py) | Exposed `/api/photos/preview` endpoint, cached photo mapping in `_PHOTO_CACHE_MAP`, passed live preview payloads in `/api/build-stream`. |
| [`static/index.html`](../static/index.html) | Added Live Customer Engagement Panel with flashing spotlight photo, AI captions, landmark tags, live stats chips, and recent discoveries filmstrip; added parallel concurrency selector (up to 20 workers), 10-photo batch preset, fill mode, crop focus, captions toggle, and live status badges; removed boilerplate text and dates. |
| [`photobook/renderer.py`](../photobook/renderer.py) | Purged "CURATED 10x10" and dates from PDF and HTML; displays uppercase location on cover eyebrow and subtitle; logs concurrency in build reports. |
| [`tests/test_photobook.py`](../tests/test_photobook.py) | Added end-to-end unit tests covering all features, concurrency options, clean styling assertions, and live preview endpoints. |
| [`docs/pipeline_flow.md`](pipeline_flow.md) | Updated pipeline diagram and stage descriptions to reflect the Smart Hybrid model. |
| [`docs/gemini_first_branch_changes.md`](gemini_first_branch_changes.md) | Master documentation file detailing all branch changes. |
