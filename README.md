# 10×10 Inch AI Photo Book Builder

An end-to-end **10×10 inch Photo Book Builder** powered by **Google Gemini 3.8 Flash** (`gemini-3.8-flash`, Medium thinking effort) with an interactive **Side-by-Side 2-Page Open Book Spread** Web UI, printable 10×10" PDF generator, standalone HTML book exporter, and detailed Markdown/JSON photo analysis reports.

## Key Features
1. **Strict Photo-Only Input (`.HEIC`, `.JPG`, `.PNG`, `.WEBP`)**:
   - Scans a local folder of personal photos (including Apple `.HEIC`/`.HEIF`), extracts EXIF capture timestamps & camera metadata, and sorts chronologically.
2. **Configurable Analysis Resolution (`384 px` to `2,240 px`)**:
   - Supports **Very Low (`384–512 px`)**, **Low (`512–768 px`)**, **Medium (`768–1,120 px`)**, **High (`1,120–1,680 px`)**, and **Ultra High (`1,680–2,240 px`)** presets.
3. **Automatic Deduplication & Document Filtering**:
   - Removes exact/near duplicates (`dHash` + `aHash`), visually similar burst shots, blurry images, and photos of documents/receipts/screenshots.
4. **1–2 Line Captions & Chronological Story Clustering**:
   - Generates concise 1–2 line captions for both individual photos and thematic chapters/clusters.
5. **Hero Cover Selection & 1–3 Photos per 10×10" Page**:
   - Selects the highest cover-worthy photo for the front cover and lays out 1, 2, or 3 photos max per 10×10 inch page.
6. **Face-Safe Aesthetic Smart Cropping**:
   - Detects human face bounding boxes and slides/scales the crop window so **no face is ever cut off**.
7. **Side-by-Side Open Photo Book Preview + Printable Outputs**:
   - Interactive 2-page side-by-side spread viewer in the Web UI, plus `photobook_10x10.pdf`, `photobook.html`, `photo_analysis.json`, and `photo_analysis.md`.

## Quick Start

### 1. Configure Your Gemini API Key
Copy `.env.example` to `.env` and add your key from [Google AI Studio](https://aistudio.google.com/apikey):
```bash
cp .env.example .env
```

### 2. Run the Web UI
```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8088 --reload
```
Then open **http://127.0.0.1:8088** in your browser.

### 3. Or Run via CLI
```bash
uv run python cli.py /path/to/your/photos
```

### 4. Run Tests
```bash
uv run pytest -v
```
