# 10×10 Inch AI Photo Book Builder — Pipeline Flow (Gemini-First "Smart Hybrid" Branch)

This document describes the **Smart Hybrid (Gemini-First)** pipeline on the `Gemini-first` branch of the **10×10 Inch AI Photo Book Builder** ([`photobook/pipeline.py`](../photobook/pipeline.py)).

In this architecture, **all unique surviving photos go through Gemini 3.8 Flash Vision first** before downsampling or curating to the final book size. This ensures burst selection, aesthetic curation, and chronological downsampling are driven by Gemini's deep semantic understanding (genuine smiles, open eyes, true composition, and landmark importance) rather than crude pixel-edge heuristics.

---

## High-Level "Smart Hybrid" Pipeline Overview

```
Raw Photo Folder (e.g., 700+ .JPG / .HEIC / .PNG / .WEBP files)
   │
   ├─► Stage 1: Ingestion, EXIF Orientation, Perceptual Hashes & Scaling (All 700+ Photos)
   │            [photobook/ingestion.py :: scan_and_prepare_photos]
   │
   ├─► Stage 2: Coarse Fast Pruning on Local CPU (<1s, Drops Exact Duplicate Files & Corrupt Receipts)
   │            [photobook/analyzer.py :: coarse_filter_photos]
   │            • Drops exact duplicates (dHash==0, aHash==0, color >= 0.998)
   │            • Drops extreme corrupt/blank frames and clear local receipts
   │            • DOES NOT compress or downsample to 90 photos here!
   │
   ├─► Stage 3: Deep Gemini 3.8 Flash Vision Analysis (ALL Surviving Unique Photos)
   │            [photobook/analyzer.py :: analyze_photos_with_gemini]
   │            • Multi-image parallel batches (batch_size=6, 6 parallel workers)
   │            • Persistent disk caching (.analysis.gemini-3.8-flash.json)
   │            • Extracts exact face boxes, emotional expressions, aesthetic scores,
   │              landmarks, sideways orientation, and 1-2 line captions
   │
   ├─► Stage 4: Gemini-Powered Semantic Deduplication & Chronological Curation (e.g., 600 → 90 Photos)
   │            [photobook/analyzer.py :: deduplicate_and_curate_with_gemini]
   │            • Filters out Gemini-detected documents, severe flaws, and sideways faces
   │            • In camera bursts, picks best-expressed shot (smiling vs blinking) using Gemini scores
   │            • Downsamples across chronological timeline buckets using true Gemini composite quality
   │
   ├─► Stage 5: Scenic Hero Cover Selection & Single-Photo Precision Pass
   │            [photobook/analyzer.py :: select_hero_cover_photo]
   │            • Scenic-first preference (landscape/square panoramic views without portraits)
   │            • Fallback to highest-rated overall photo if no scenic view exists
   │
   ├─► Stage 6: Chronological Story Clustering & Chapter Titling (Gemini 2.5 Flash-Lite / 3.8 Flash)
   │            [photobook/analyzer.py :: curate_clusters_and_pages]
   │
   ├─► Stage 7: 10×10" Page Template Selection & 100% Face-Safe Aesthetic Cropping
   │            [photobook/analyzer.py :: choose_best_page_template + photobook/cropping.py :: compute_aesthetic_crop]
   │
   └─► Stage 8: Output Rendering (10×10" Print PDF, 3D Lay-Flat HTML Book & Analysis Reports)
                [photobook/renderer.py :: render_pdf_photobook, render_html_photobook, save_analysis_files]
```

---

## Detailed Stage-by-Stage Breakdown

### Stage 1: Ingestion, EXIF Extraction, Perceptual Hashes & Scaling
**Entry point:** [`scan_and_prepare_photos`](../photobook/ingestion.py)

1. **Strict Personal Photo Ingestion**: Reads only personal image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`, `.heif`, `.tiff`, `.bmp`), ignoring non-image files.
2. **EXIF Transposition & Timestamping**: Handles EXIF rotation tags (`ImageOps.exif_transpose`) and parses exact capture timestamps (`DateTimeOriginal`).
3. **Resolution Normalization**: Pre-scales images into the target vision analysis window (default `1,120` to `2,240 px`).
4. **Perceptual Signatures**: Computes 64-bit `dHash`, `aHash`, and 24-bin L2-normalized RGB color histogram vectors (`color_signature`).

---

### Stage 2: Coarse Fast Pruning on Local CPU (<1 Second)
**Entry point:** [`coarse_filter_photos`](../photobook/analyzer.py)

Before dispatching network calls, a lightweight local check prunes redundant work:
1. **Drops 100% Exact Duplicate Files**: If two files share identical visual hashes (`dHash == 0, aHash == 0, color >= 0.998`), the duplicate is immediately marked `excluded_duplicate`.
2. **Drops Obvious White Paper Receipts / Extreme Blanks**: Drops non-photo digital graphics, white receipt paper, or unreadable frames (`sharpness < 2.5 and aesthetic < 3.0`).
3. **Preserves All Legitimate Photos**: Unlike the legacy flow, **no arbitrary downsampling (e.g. to 90 photos) is performed here**. All unique scene and personal photos pass through to Gemini.

---

### Stage 3: Deep Gemini 3.8 Flash Vision on All Surviving Photos
**Entry point:** [`analyze_photos_with_gemini`](../photobook/analyzer.py)

All unique photos from Stage 2 are analyzed by **Gemini 3.8 Flash**:
1. **Parallel Worker Batches**: Uncached photos are processed in batches of 6 images concurrently across 6 worker threads with live progress reporting.
2. **Full Semantic Extraction**:
   - `faces_boxes`: Exact `[0..1000]` bounding boxes for all visible human heads (hair to chin)
   - `has_sideways_or_inverted_orientation`: Identifies unrotated sideways or upside-down frames
   - `aesthetic_score`, `emotional_score`, `sharpness_score`, `cover_worthiness_score`
   - `location_or_landmark`, `event_theme`, `subject_summary`, and a 1–2 line `caption`
3. **Automatic Caching**: Results are cached on disk (`output/.photo_analysis_cache/`) so re-runs complete in `0ms`.

---

### Stage 4: Gemini-Powered Semantic Deduplication & Chronological Curation
**Entry point:** [`deduplicate_and_curate_with_gemini`](../photobook/analyzer.py)

Because Gemini has already analyzed all surviving photos, deduplication and downsampling decisions are now driven by AI:
1. **Sideways & Quality Purge**: Automatically excludes any photo Gemini flagged with sideways/inverted orientation, documents, or severe flaws.
2. **Quality-Ranked Burst Selection (Smiling vs. Blinking)**:
   - Candidates are ranked by Gemini's multimodal composite score.
   - In rapid bursts (`Δt <= 3.0 min`, `color_sim >= 0.94`, `d_dist <= 20`), the photo where people have the best expression and highest emotional/aesthetic score is kept, while inferior burst mates are excluded.
3. **Chronological Bucket Downsampling to `max_book_photos`**:
   - If more photos remain than `max_book_photos` (e.g. 90), the chronological timeline is divided into 90 equal windows.
   - In each window, the photo with the highest Gemini score is selected, guaranteeing uniform trip coverage with the absolute highest-quality frames.

---

### Stage 5: Scenic Hero Cover Selection & Precision Pass
**Entry point:** [`select_hero_cover_photo`](../photobook/analyzer.py)

1. Prioritizes sweeping scenic landscape/square views (`cover_worthiness >= 6.8`, `aesthetic >= 7.0`, `people_count == 0`).
2. Automatically falls back to the best overall photo if no qualifying scenic view exists.
3. Runs a dedicated high-resolution precision pass on the selected cover image for perfect cropping.

---

### Stage 6: Story Clustering & Chapter Titling
**Entry point:** [`curate_clusters_and_pages`](../photobook/analyzer.py)

Groups photos into chronological days and themes, and uses `gemini-2.5-flash-lite` or `gemini-3.8-flash` to craft cohesive chapter titles and warm 1–2 line group captions.

---

### Stage 7: 10×10" Layout & 100% Face-Safe Cropping
**Entry point:** [`choose_best_page_template`](../photobook/analyzer.py) & [`compute_aesthetic_crop`](../photobook/cropping.py)

1. Dynamically assigns 1, 2, or 3 photos per 10×10" page across 11 balanced templates with generous text padding and no overlap.
2. Applies face-safe bounding-box preservation with a 12% safety margin above foreheads and automatic fallback to fewer photos per page if a slot aspect ratio would cut off a face.

---

### Stage 8: Output Rendering
**Entry point:** [`photobook/renderer.py`](../photobook/renderer.py)

Generates:
1. Print-ready `photobook_10x10.pdf` (ReportLab 720×720 pt)
2. Standalone `photobook.html` with **180° Lay-Flat 2-Page Spreads**, **✨ 3D Lay-Flat Flip Viewer**, and **▶ Autoplay**
3. `photo_analysis.json` and human-readable `photo_analysis.md` audit logs
