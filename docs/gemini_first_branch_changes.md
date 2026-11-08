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
- **Dashboard UI Controls**: Direct dropdowns and status badges on the web UI for fill modes and cropping focus.

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

## 6. Dashboard & User Interface Enhancements

File: [`static/index.html`](../static/index.html)

1. **Page Layout Fill Selector**:
   - Dropdown under Settings:
     - `High (Maximum photo coverage, minimal gaps)` *(Default)*
     - `Medium (Balanced spacing)`
     - `Low (Gallery style, wide white margins)`
2. **Aesthetic Crop Focus Selector**:
   - Dropdown under Settings:
     - `Tight (Aesthetic close-up, remove dead space)` *(Default)*
     - `Balanced (Standard balanced framing)`
     - `Wide (Maximum environmental context)`
3. **Live Status Badges**:
   - Dashboard summary bar displays active configuration pills:
     - `Fill: High` / `Fill: Medium` / `Fill: Low`
     - `Crop: Tight` / `Crop: Balanced` / `Crop: Wide`
4. **Interactive 3D Viewer & Controls**:
   - Integrates with the 3D Lay-Flat viewer, book flipping, and autoplay features.

---

## 7. API and Data Model Changes

### `app.py`
- Updated `/api/build` endpoint:
  ```python
  class BuildRequest(BaseModel):
      photos_dir: str
      title: str = "Our Unforgettable Journey"
      subtitle: str = "A Curated Visual Chronicle"
      target_pages: int = 40
      max_book_photos: int = 90
      page_fill_mode: str = "high"     # "low", "medium", "high"
      crop_focus: str = "tight"        # "tight", "balanced", "wide"
  ```

### `photobook/models.py`
- Added `page_fill_mode: str = "high"` and `crop_focus: str = "tight"` to `PhotoBookResult` for persistence in report JSON and metadata.

### `photobook/pipeline.py`
- Pipeline coordinates the full Smart Hybrid flow:
  ```python
  def generate_photobook(
      ...,
      page_fill_mode: str = "high",
      crop_focus: str = "tight",
  ) -> PhotoBookResult:
  ```

---

## 8. Test Suite & Verification

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

### Test Execution
```bash
uv run pytest
# Output: 11 passed in 1.17s
```

---

## 9. Summary of Modified Files on `Gemini-first`

| File | Primary Changes |
|---|---|
| [`photobook/analyzer.py`](../photobook/analyzer.py) | Added `coarse_filter_photos`, `deduplicate_and_curate_with_gemini`, `PAGE_TEMPLATES_BY_FILL`, and `get_page_templates`. |
| [`photobook/cropper.py`](../photobook/cropper.py) | Added `crop_focus` parameter (`tight`, `balanced`, `wide`) to `compute_aesthetic_crop` with dead-space removal and face preservation. |
| [`photobook/pipeline.py`](../photobook/pipeline.py) | Implemented the 8-stage Smart Hybrid orchestration calling Gemini on all unique photos before downsampling. |
| [`photobook/models.py`](../photobook/models.py) | Added `page_fill_mode` and `crop_focus` fields to `PhotoBookResult`. |
| [`app.py`](../app.py) | Exposed `page_fill_mode` and `crop_focus` on `/api/build` REST API. |
| [`static/index.html`](../static/index.html) | Added Fill and Crop dropdown controls and live status badges to Dashboard UI. |
| [`photobook/renderer.py`](../photobook/renderer.py) | Documented fill mode and crop focus in generated markdown analysis logs. |
| [`tests/test_photobook.py`](../tests/test_photobook.py) | Added end-to-end tests for Smart Hybrid flow, fill modes, and aesthetic cropping. |
| [`docs/pipeline_flow.md`](pipeline_flow.md) | Updated pipeline diagram and stage descriptions to reflect the Smart Hybrid model. |
| [`docs/gemini_first_branch_changes.md`](gemini_first_branch_changes.md) | This master documentation file detailing all branch changes. |
