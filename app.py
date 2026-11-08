"""FastAPI Web Application for the 10x10 Inch Photo Book Builder."""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from photobook.analyzer import DEFAULT_MODEL
from photobook.ingestion import MAX_ANALYSIS_PX, MIN_ANALYSIS_PX
from photobook.models import PhotoBookResult
from photobook.pipeline import build_photo_book

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_ROOT = BASE_DIR / "output"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="10×10 Photo Book Builder (Gemini 3.8 Flash)",
    version="0.1.0",
)

_BOOKS: dict[str, PhotoBookResult] = {}


class BuildRequest(BaseModel):
    folder_path: str = Field(description="Path to the local folder containing personal photos")
    model: str = Field(default=DEFAULT_MODEL, description="Stage 1: Photo Vision Analysis model ID")
    thinking_level: str = Field(default="medium", description="Thinking effort: low, medium, or high")
    curation_model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Stage 2: Fast model for chapter clustering, book title & group captions",
    )
    hero_model: str = Field(
        default="gemini-3.8-flash",
        description="Stage 3: Model for Hero Cover single-photo precision refinement ('none' to skip)",
    )
    layout_model: str = Field(
        default="local_geometric",
        description="Stage 4: Page layout engine ('local_geometric' or a Gemini model ID)",
    )
    batch_size: int = Field(
        default=6,
        ge=1,
        le=10,
        description="Number of photos sent per single vision API call (1 = single-photo mode, 4-8 = fast batch mode)",
    )
    media_resolution: str = Field(
        default="high",
        description="Photo analysis resolution preset: 'low_384', 'low', 'medium', 'high', or 'ultra_high'",
    )
    min_analysis_px: int = Field(default=MIN_ANALYSIS_PX, ge=256, le=2240)
    max_analysis_px: int = Field(default=MAX_ANALYSIS_PX, ge=384, le=4096)
    max_book_photos: int = Field(
        default=90,
        ge=0,
        le=1000,
        description="Maximum curated photos to include in the book (0 = include all non-duplicate photos)",
    )
    page_fill_mode: str = Field(
        default="high",
        description="Page layout coverage: 'low', 'medium', or 'high'",
    )
    crop_focus: str = Field(
        default="tight",
        description="Aesthetic crop focus: 'tight', 'balanced', or 'wide'",
    )
    api_key: Optional[str] = Field(default=None, description="Optional Gemini API key override")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend UI file not found")
    return HTMLResponse(
        index_file.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _persist_api_key_if_provided(api_key: Optional[str]) -> None:
    if api_key and api_key.strip() and api_key.strip() != "YOUR_GEMINI_API_KEY_HERE":
        clean_key = api_key.strip()
        os.environ["GEMINI_API_KEY"] = clean_key
        env_file = BASE_DIR / ".env"
        env_file.write_text(
            f'# Google Gemini API Key\nGEMINI_API_KEY="{clean_key}"\n',
            encoding="utf-8",
        )


@app.get("/api/config")
def get_config() -> dict:
    load_dotenv(BASE_DIR / ".env", override=True)
    env_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    has_env_key = bool(env_key and env_key != "YOUR_GEMINI_API_KEY_HERE")
    default_folder = "/Users/amitpoddar/ClaudeProjects/PyhtonLearning/photo_search/myphotos"
    return {
        "default_model": DEFAULT_MODEL,
        "default_thinking_level": "medium",
        "default_media_resolution": "high",
        "min_analysis_px": MIN_ANALYSIS_PX,
        "max_analysis_px": MAX_ANALYSIS_PX,
        "has_env_api_key": has_env_key,
        "default_folder_path": default_folder if Path(default_folder).is_dir() else "",
    }


@app.post("/api/build", response_model=PhotoBookResult)
def build_book_endpoint(req: BuildRequest) -> PhotoBookResult:
    try:
        _persist_api_key_if_provided(req.api_key)
        result = build_photo_book(
            folder_path=req.folder_path,
            output_root=OUTPUT_ROOT,
            model=req.model,
            thinking_level=req.thinking_level,
            curation_model=req.curation_model,
            hero_model=req.hero_model,
            layout_model=req.layout_model,
            batch_size=req.batch_size,
            media_resolution=req.media_resolution,
            min_analysis_px=req.min_analysis_px,
            max_analysis_px=req.max_analysis_px,
            max_book_photos=req.max_book_photos,
            page_fill_mode=req.page_fill_mode,
            crop_focus=req.crop_focus,
            api_key=req.api_key,
        )
        _BOOKS[result.book_id] = result
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build photo book: {exc}") from exc


@app.post("/api/build-stream")
def build_book_stream_endpoint(req: BuildRequest) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming endpoint for real-time progress across 1,000+ photos."""
    q: queue.Queue[dict] = queue.Queue()

    def _progress(msg: str, pct: int, total: int) -> None:
        q.put({"type": "progress", "message": msg, "percent": pct, "total": total})

    def _worker() -> None:
        try:
            _persist_api_key_if_provided(req.api_key)
            result = build_photo_book(
                folder_path=req.folder_path,
                output_root=OUTPUT_ROOT,
                model=req.model,
                thinking_level=req.thinking_level,
                curation_model=req.curation_model,
                hero_model=req.hero_model,
                layout_model=req.layout_model,
                batch_size=req.batch_size,
                media_resolution=req.media_resolution,
                min_analysis_px=req.min_analysis_px,
                max_analysis_px=req.max_analysis_px,
                max_book_photos=req.max_book_photos,
                page_fill_mode=req.page_fill_mode,
                crop_focus=req.crop_focus,
                api_key=req.api_key,
                progress_cb=_progress,
            )
            _BOOKS[result.book_id] = result
            q.put({"type": "complete", "book": json.loads(result.model_dump_json())})
        except Exception as exc:
            q.put({"type": "error", "detail": str(exc)})

    threading.Thread(target=_worker, daemon=True).start()

    def _event_gen() -> Iterator[str]:
        while True:
            item = q.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] in ("complete", "error"):
                break

    return StreamingResponse(_event_gen(), media_type="text/event-stream")


@app.get("/api/books/latest", response_model=PhotoBookResult)
def get_latest_book() -> PhotoBookResult:
    candidates: list[tuple[float, Path]] = []
    if OUTPUT_ROOT.exists():
        for d in OUTPUT_ROOT.iterdir():
            if not d.is_dir() or d.name.startswith(".") or d.name.startswith("ui_photos"):
                continue
            jf = d / "photo_analysis.json"
            if jf.exists():
                candidates.append((jf.stat().st_mtime, jf))
    if not candidates:
        raise HTTPException(status_code=404, detail="No generated photo books found yet")
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_json = candidates[0][1]
    loaded = PhotoBookResult.model_validate_json(latest_json.read_text(encoding="utf-8"))
    _BOOKS[loaded.book_id] = loaded
    return loaded


@app.get("/api/books/{book_id}", response_model=PhotoBookResult)
def get_book(book_id: str) -> PhotoBookResult:
    if book_id in _BOOKS:
        return _BOOKS[book_id]
    json_file = OUTPUT_ROOT / book_id / "photo_analysis.json"
    if json_file.exists():
        loaded = PhotoBookResult.model_validate_json(json_file.read_text(encoding="utf-8"))
        _BOOKS[book_id] = loaded
        return loaded
    raise HTTPException(status_code=404, detail=f"Book '{book_id}' not found")


@app.get("/api/books/{book_id}/files/{file_path:path}")
def serve_book_file(book_id: str, file_path: str) -> FileResponse:
    book_dir = (OUTPUT_ROOT / book_id).resolve()
    target = (book_dir / file_path).resolve()
    if not str(target).startswith(str(book_dir)) or not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Requested file not found")
    return FileResponse(
        target,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
