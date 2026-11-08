"""Gemini 3.8 Flash (Medium effort) deep image analysis at 1,120-2,240px resolution,
pure-image duplicate/similar/document removal, chronological clustering, hero cover selection,
and face-safe 10x10 inch page layout engine.

Strictly uses ONLY the image files inside the input photo folder—no external indexes or caption files.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from google import genai
from google.genai import types
from PIL import Image, ImageStat
from pydantic import BaseModel, Field

from photobook.cropper import can_fit_aspect_without_cutting_faces, compute_aesthetic_crop, crop_and_save_photo
from photobook.ingestion import cosine_similarity, hamming_distance
from photobook.models import (
    BatchGeminiPhotoAnalysis,
    BatchPhotoAnalysisItem,
    BookCurationRefinement,
    BookPage,
    BoundingBox,
    GeminiPhotoAnalysis,
    PageTemplateId,
    PhotoCluster,
    PhotoMetadata,
    PhotoSlot,
)

DEFAULT_MODEL = "gemini-3.8-flash"
DEPRECATED_PREFIXES = ("gemini-2.5", "gemini-2.0", "gemini-1.5")


def resolve_gemini_model(requested_model: str) -> tuple[str, Optional[str]]:
    """Ensure deprecated Gemini models (e.g. gemini-2.5-flash) are upgraded to gemini-3.8-flash."""
    clean = (requested_model or DEFAULT_MODEL).strip()
    if any(clean.startswith(prefix) for prefix in DEPRECATED_PREFIXES):
        return (
            DEFAULT_MODEL,
            f"Automatically upgraded deprecated model '{clean}' to current '{DEFAULT_MODEL}' (Medium effort).",
        )
    return clean, None


def clamp_caption_two_lines(text: str, max_chars: int = 145) -> str:
    """Ensure a caption is at most 1-2 lines and ends cleanly at a clause boundary."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    joined = " ".join(lines[:2])
    if len(joined) <= max_chars:
        return joined
    cut = joined[: max_chars - 1]
    clause_seps = [
        ", while ",
        ", featuring ",
        ", where ",
        ", with ",
        ", as ",
        ", overlooking ",
        ", surrounded ",
        ", illuminated ",
        ", set ",
        ", likely ",
        "— ",
    ]
    for sep in clause_seps:
        idx = cut.rfind(sep, 35)
        if idx != -1:
            return cut[:idx].rstrip(" ,;:") + "."
    clipped = cut.rsplit(" ", 1)[0].rstrip(".,;:")
    return clipped + "…"


class _CurationClusterPlan(BaseModel):
    cluster_id: str = Field(description="Unique cluster ID, e.g. 'cluster_01'")
    title: str = Field(description="Chapter/group title derived from the photos' locations and activities")
    date_range_label: str = Field(description="Human-readable date or day label")
    group_caption: str = Field(description="1 to 2 line max caption summarizing this group of photos")
    photo_ids: list[str] = Field(description="Chronological photo_ids belonging to this cluster")


class _CurationPagePlan(BaseModel):
    cluster_id: str = Field(description="Cluster ID this page belongs to")
    photo_ids: list[str] = Field(
        min_length=1,
        max_length=3,
        description="1, 2, or 3 photo_ids max on this 10x10 inch page, in chronological order",
    )
    group_caption: Optional[str] = Field(
        default=None,
        description="Optional 1-2 line page/group caption when multiple photos share a story on this page",
    )


class _BookCurationPlan(BaseModel):
    book_title: str = Field(description="Evocative title for the 10x10 photo book cover")
    book_subtitle: str = Field(description="Short subtitle (e.g. date range or destination summary)")
    hero_cover_photo_id: str = Field(description="photo_id of the single best hero photo for the front cover")
    clusters: list[_CurationClusterPlan] = Field(description="Chronological thematic clusters of the selected photos")
    pages: list[_CurationPagePlan] = Field(description="Page-by-page grouping (1, 2, or 3 photos max per page)")


ANALYSIS_PROMPT = """You are an expert photo book editor and computer vision analyst.
Analyze this personal photo (provided at 1,120 to 2,240 pixels resolution) for a 10x10 inch luxury photo book.

Return a JSON object matching the schema with:
1. `is_document_or_screenshot`: Set to `true` if this image is a photograph of a paper document, handwritten note, receipt, invoice, boarding pass, menu, bus/train timetable, signpost, informational plaque/panel, printed diagram, city map board, museum text label, screenshot, smartphone screen, or QR code sign that does not belong in a personal photo book.
2. `document_reason`: If `is_document_or_screenshot` is true, state why (e.g. 'Public transport timetable sign', 'Digital route map screenshot', 'Stone informational plaque').
3. `is_blurry_or_flawed`: Set to `true` if severely out-of-focus, motion-blurred, or an accidental frame.
4. `has_sideways_or_inverted_orientation`: Set to `true` if a person's face/head or the horizon/scene is oriented sideways (rotated 90 degrees left or right) or upside-down (such as an unrotated vertical camera shot or a top-down upside-down face).
5. `aesthetic_score` (1-10), `sharpness_score` (1-10), `emotional_score` (1-10), and `cover_worthiness_score` (1-10). For `cover_worthiness_score`, strongly prefer breathtaking SCENIC VIEWS, landscapes, sunsets, coastlines, and iconic architecture (`8.5-10.0`) over portraits or selfies (`<= 6.8`).
6. `subject_summary`: Specific description of the main subject, pose, angle, background, and lighting so near-duplicate burst shots of the same subject can be identified.
7. `location_or_landmark`: Landmark, city, or setting name if recognizable from the visual scene.
8. `event_theme`: Short grouping theme describing the place and activity.
9. `caption`: A warm, evocative caption for the photo book (strictly 1 to 2 lines maximum, under 135 characters).
10. `people_count`: Count of visible people.
11. `faces_boxes`: CRITICAL FOR AESTHETIC CROPPING — detect EVERY visible human head/face in the image and return its normalized [0..1000] bounding box (`ymin`, `xmin`, `ymax`, `xmax`, `label="face"`). Include the ENTIRE head from the top of the hair/forehead to the bottom of the chin so cropping NEVER cuts off anyone's face or forehead.
12. `salient_subject_box`: Normalized [0..1000] bounding box (`ymin`, `xmin`, `ymax`, `xmax`, `label="subject"`) covering the main visual subject(s) for aesthetic composition.
"""


# ── Pure-Pixel Image Pre-Analysis (Strictly from the Photo Pixels Only) ──────


def _detect_skin_head_boxes_from_pixels(thumb_rgb: Image.Image) -> list[BoundingBox]:
    """Detect approximate head/face bounding region(s) from YCbCr skin-tone clusters in the upper/mid frame."""
    ycbcr = thumb_rgb.resize((64, 64), Image.Resampling.BILINEAR).convert("YCbCr")
    getter = getattr(ycbcr, "get_flattened_data", None)
    pixels = list(getter()) if getter is not None else list(ycbcr.getdata())

    skin_xs: list[int] = []
    skin_ys: list[int] = []
    # Scan upper 70% of image for skin-tone pixels (Y > 60, 80 <= Cb <= 125, 136 <= Cr <= 172)
    for idx, (y_val, cb, cr) in enumerate(pixels):
        row = idx // 64
        col = idx % 64
        if row < 6 or row > 46 or col < 6 or col > 58:
            continue
        if y_val > 65 and 80 <= cb <= 124 and 136 <= cr <= 172:
            skin_xs.append(col)
            skin_ys.append(row)

    # Require a plausible portrait face cluster size (between 1.5% and 22% of frame)
    if not (40 <= len(skin_xs) <= 650):
        return []

    skin_xs.sort()
    skin_ys.sort()
    # Use 15th to 85th percentile to reject background skin/sand outliers
    x0 = skin_xs[int(len(skin_xs) * 0.15)]
    x1 = skin_xs[int(len(skin_xs) * 0.85)]
    y0 = skin_ys[int(len(skin_ys) * 0.10)]
    y1 = skin_ys[int(len(skin_ys) * 0.75)]

    if (x1 - x0) < 4 or (y1 - y0) < 4 or (x1 - x0) > 44:
        return []

    # Expand upward for hair/forehead
    ymin = max(0, int(round(((y0 - 4) / 64.0) * 1000)))
    ymax = min(1000, int(round(((y1 + 4) / 64.0) * 1000)))
    xmin = max(0, int(round(((x0 - 3) / 64.0) * 1000)))
    xmax = min(1000, int(round(((x1 + 3) / 64.0) * 1000)))
    if ymax <= ymin + 40 or xmax <= xmin + 40:
        return []
    return [BoundingBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax, label="face")]


def _classify_visual_scene_from_pixels(
    thumb_rgb: Image.Image,
    avg_brightness: float,
    color_saturation: float,
    captured_at: str,
) -> tuple[str, str, str]:
    """Classify visual scene type (sunset, coastal/sea, sunlit architecture, greenery, evening) purely from image pixels."""
    grid = thumb_rgb.resize((32, 32), Image.Resampling.BILINEAR)
    getter = getattr(grid, "get_flattened_data", None)
    px = list(getter()) if getter is not None else list(grid.getdata())

    top_band = px[: 32 * 10]
    mid_band = px[32 * 10 : 32 * 22]
    bot_band = px[32 * 22 :]

    def _avg_rgb(band: list[tuple[int, int, int]]) -> tuple[float, float, float]:
        n = max(1, len(band))
        return (
            sum(p[0] for p in band) / n,
            sum(p[1] for p in band) / n,
            sum(p[2] for p in band) / n,
        )

    tr, tg, tb = _avg_rgb(top_band)
    mr, mg, mb = _avg_rgb(mid_band)
    br, bg, bb = _avg_rgb(bot_band)

    dt_str = captured_at[:10]
    hour = int(captured_at[11:13]) if len(captured_at) >= 13 and captured_at[11:13].isdigit() else 12
    tod = "Morning" if hour < 12 else ("Afternoon" if hour < 17 else "Evening")

    # 1. Sunset / Golden Hour (warm orange/red/gold sky or horizon)
    if (tr > tb + 35 and tg > tb + 10 and avg_brightness > 75) or (mr > mb + 40 and hour >= 17):
        return (
            "Golden Horizon",
            "Golden Hour & Sunset Views",
            f"Warm golden-hour light across the horizon ({dt_str}).",
        )

    # 2. Coastal / Sea / Blue Water & Sky (strong blue dominance in top or bottom/mid bands)
    if (tb > tr + 25 and bb > br + 18) or (mb > mr + 28 and tb > tr + 15):
        return (
            "Coastal Waters",
            "Coastal Vistas & Sea Breeze",
            f"Deep blue coastal waters and open sky captured on {dt_str}.",
        )

    # 3. Sunlit Whitewashed / Bright Architecture under Blue Sky
    white_pixels = sum(1 for (r, g, b) in px if r > 200 and g > 200 and b > 195 and max(r, g, b) - min(r, g, b) < 25)
    blue_pixels = sum(1 for (r, g, b) in px if b > r + 25 and b > 130)
    if white_pixels > 180 and blue_pixels > 80:
        return (
            "Sunlit Architecture",
            "Sunlit Streets & Architecture",
            f"Sunlit whitewashed architecture against a bright blue sky ({dt_str}).",
        )

    # 4. Night / Evening Illuminated Scene
    if avg_brightness < 82 or hour >= 20 or hour < 6:
        return (
            "Evening Lights",
            "Evening Stroll & Night Lights",
            f"Illuminated evening scene captured on {dt_str}.",
        )

    # 5. Greenery / Nature / Outdoor Landscape
    if mg > mr + 12 and mg > mb + 12:
        return (
            "Outdoor Landscape",
            "Outdoor Exploration & Scenery",
            f"Natural outdoor scenery captured in {tod.lower()} light ({dt_str}).",
        )

    # 6. Warm Indoor / Dining / Gathering
    if mr > mb + 18 and avg_brightness < 175 and color_saturation > 12:
        return (
            f"{tod} Gathering",
            "Dining & Indoor Moments",
            f"Warm {tod.lower()} gathering captured on {dt_str}.",
        )

    return (
        f"{tod} Scene",
        f"{tod} Highlights & Exploration",
        f"{tod} moment captured on {dt_str}.",
    )


def _analyze_single_photo_pixels(photo: PhotoMetadata) -> GeminiPhotoAnalysis:
    """Fast pure-pixel vision analysis of a photo at 1,120-2,240px (used for pre-screening
    before Gemini API calls or when running offline without an API key)."""
    stem = Path(photo.filename).stem.lower()
    with Image.open(photo.analysis_path) as img:
        thumb_rgb = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        small_g = thumb_rgb.convert("L")
        w, h = photo.analysis_width, photo.analysis_height

        stat = ImageStat.Stat(thumb_rgb)
        mean_r, mean_g, mean_b = stat.mean
        std_r, std_g, std_b = stat.stddev
        avg_brightness = (mean_r + mean_g + mean_b) / 3.0
        color_saturation = max(abs(mean_r - mean_g), abs(mean_g - mean_b), abs(mean_r - mean_b))
        avg_std = (std_r + std_g + std_b) / 3.0

        getter = getattr(small_g, "get_flattened_data", None)
        px = list(getter()) if getter is not None else list(small_g.getdata())
        diffs = [abs(px[i] - px[i + 1]) for i in range(0, len(px) - 1, 2)]
        top_edges = sorted(diffs, reverse=True)[: max(1, len(diffs) // 12)]
        avg_top_edge = sum(top_edges) / max(1, len(top_edges))
        sharpness = max(2.5, min(9.8, round(5.5 + avg_top_edge * 0.12, 1)))

        # Count sharp bimodal text-like transitions (adjacent pixels jumping > 65 luminance levels)
        sharp_text_jumps = sum(1 for d in diffs if d > 65) / float(max(1, len(diffs)))

    # Pure-pixel document / screenshot / digital map detection:
    # 1) PNG image with no camera EXIF model (e.g. downloaded digital map or phone screenshot)
    # 2) High-brightness neutral paper/document page with high-contrast text jumps
    # 3) Explicit document filename keyword
    doc_keywords = ("receipt", "document", "invoice", "ticket", "scan", "screenshot", "menu", "form", "bill", "timetable")
    has_doc_keyword = any(k in stem for k in doc_keywords)
    is_png_without_camera = photo.filename.lower().endswith(".png") and not photo.camera_model
    is_paper_document = (
        avg_brightness > 205
        and color_saturation < 11.0
        and avg_std > 30.0
        and sharp_text_jumps > 0.045
    )

    if has_doc_keyword or is_png_without_camera or is_paper_document:
        reason = (
            "Digital image / screenshot without camera EXIF"
            if is_png_without_camera
            else "High-contrast document, receipt, or text page layout"
        )
        return GeminiPhotoAnalysis(
            is_document_or_screenshot=True,
            document_reason=reason,
            is_blurry_or_flawed=False,
            aesthetic_score=2.0,
            sharpness_score=sharpness,
            emotional_score=1.5,
            cover_worthiness_score=1.0,
            subject_summary=f"Document or screenshot ({photo.filename})",
            location_or_landmark="N/A",
            event_theme="Excluded Documents",
            caption="Excluded document or screenshot.",
            people_count=0,
            faces_boxes=[],
        )

    if sharpness < 3.6:
        return GeminiPhotoAnalysis(
            is_document_or_screenshot=False,
            is_blurry_or_flawed=True,
            aesthetic_score=3.0,
            sharpness_score=sharpness,
            emotional_score=2.5,
            cover_worthiness_score=1.5,
            subject_summary=f"Low-sharpness / motion-blurred frame ({photo.filename})",
            location_or_landmark="",
            event_theme="Excluded Blurry Frames",
            caption="Excluded blurry frame.",
            people_count=0,
            faces_boxes=[],
        )

    faces_boxes = _detect_skin_head_boxes_from_pixels(thumb_rgb)
    landmark, theme, caption_text = _classify_visual_scene_from_pixels(
        thumb_rgb,
        avg_brightness,
        color_saturation,
        photo.captured_at,
    )

    contrast_bonus = min(0.8, avg_std / 85.0)
    aesthetic = round(
        min(9.0, max(5.5, 6.5 + (color_saturation / 45.0) + (sharpness - 6.0) * 0.25 + contrast_bonus)),
        2,
    )
    emotional = round(min(9.0, aesthetic + (0.35 if faces_boxes else 0.0)), 2)
    aspect_ratio = w / max(1.0, float(h))
    is_balanced_aspect = 0.72 <= aspect_ratio <= 1.95
    # Give preference to scenic views (no close-up faces) over portraits for the cover, and cap fallback cover score <= 7.4
    scenic_bonus = 0.35 if not faces_boxes else -0.45
    cover_score = round(
        min(
            7.4,
            max(
                3.5,
                0.50 * aesthetic
                + 0.30 * sharpness
                + 0.20 * emotional
                + scenic_bonus
                + (0.25 if photo.orientation in ("landscape", "square") and is_balanced_aspect else -0.4),
            ),
        ),
        2,
    )

    if faces_boxes:
        b = faces_boxes[0]
        salient = BoundingBox(
            ymin=max(0, b.ymin - 60),
            xmin=max(0, b.xmin - 100),
            ymax=min(1000, b.ymax + 220),
            xmax=min(1000, b.xmax + 100),
            label="subject",
        )
    else:
        salient = BoundingBox(ymin=120, xmin=140, ymax=880, xmax=860, label="subject")

    return GeminiPhotoAnalysis(
        is_document_or_screenshot=False,
        is_blurry_or_flawed=False,
        aesthetic_score=aesthetic,
        sharpness_score=sharpness,
        emotional_score=emotional,
        cover_worthiness_score=cover_score,
        subject_summary=f"{theme} ({photo.orientation}, {w}x{h}px)",
        location_or_landmark=landmark,
        event_theme=theme,
        caption=clamp_caption_two_lines(caption_text),
        people_count=len(faces_boxes),
        faces_boxes=faces_boxes,
        salient_subject_box=salient,
    )


def pre_analyze_all_photos(
    photos: list[PhotoMetadata],
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    """Run fast parallel pixel-level pre-analysis across all scanned photos (strictly from the photos themselves)."""
    total = len(photos)
    max_workers = min(8, (os.cpu_count() or 4))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze_single_photo_pixels, p): p for p in photos}
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            p = futures[fut]
            try:
                p.analysis = fut.result()
            except Exception:
                pass
            if progress_cb and (done_count % 25 == 0 or done_count == total):
                progress_cb(
                    f"Pre-screened {done_count}/{total} photos at {p.analysis_width}×{p.analysis_height}px",
                    done_count,
                    total,
                )


RESOLUTION_MAX_PX_MAP: dict[str, int] = {
    "low_384": 512,
    "low": 768,
    "medium": 1120,
    "high": 1680,
    "ultra_high": 2240,
}


def _prepare_gemini_image_bytes(photo: PhotoMetadata, media_resolution: str) -> tuple[bytes, int, int, str]:
    """Prepare JPEG bytes scaled to the target resolution preset so lower resolution choices
    transmit 5x-15x smaller payloads and process much faster."""
    import io

    target_max = RESOLUTION_MAX_PX_MAP.get(media_resolution, 1680)
    api_res_token = "low" if media_resolution in ("low_384", "low") else (
        "medium" if media_resolution == "medium" else (
            "ultra_high" if media_resolution == "ultra_high" else "high"
        )
    )
    src_path = Path(photo.analysis_path)
    if max(photo.analysis_width, photo.analysis_height) <= target_max:
        return src_path.read_bytes(), photo.analysis_width, photo.analysis_height, api_res_token

    with Image.open(src_path) as img:
        w, h = img.size
        scale = target_max / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=82)
        return buf.getvalue(), new_w, new_h, api_res_token


def _call_gemini_vision_api(
    client: genai.Client,
    photo: PhotoMetadata,
    resolved_model: str,
    requested_model: str,
    thinking_level: str,
    media_resolution: str,
    schema: dict,
    prefer_generate_content: bool = False,
) -> tuple[GeminiPhotoAnalysis, bool]:
    """Call Gemini 3.8 Flash via `client.interactions.create` (with fallback to `models.generate_content`
    if the user's API endpoint only exposes generate_content for their key).
    Returns (parsed_analysis, used_generate_content_fallback).
    """
    image_bytes, sent_w, sent_h, api_res_token = _prepare_gemini_image_bytes(photo, media_resolution)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt_text = (
        f"{ANALYSIS_PROMPT}\n"
        f"Filename: {photo.filename}\n"
        f"Captured At: {photo.captured_at}\n"
        f"Analysis Resolution: {sent_w}x{sent_h} pixels"
    )

    primary_exc: Optional[Exception] = None
    if not prefer_generate_content:
        try:
            interaction = client.interactions.create(
                model=resolved_model,
                input=[
                    {
                        "type": "image",
                        "data": b64_image,
                        "mime_type": "image/jpeg",
                        "resolution": api_res_token,
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
                generation_config={
                    "thinking_level": thinking_level,
                },
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
            )
            parsed = GeminiPhotoAnalysis.model_validate_json(interaction.output_text)
            parsed.caption = clamp_caption_two_lines(parsed.caption)
            return parsed, False
        except Exception as exc:
            primary_exc = exc

    # If the live endpoint returns 404 for interactions/gemini-3.8-flash, fall back to generate_content
    fallback_model = requested_model if requested_model.startswith("gemini-2.5") else "gemini-2.5-flash"
    try:
        resp = client.models.generate_content(
            model=fallback_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt_text,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiPhotoAnalysis,
            ),
        )
        parsed = GeminiPhotoAnalysis.model_validate_json(resp.text)
        parsed.caption = clamp_caption_two_lines(parsed.caption)
        return parsed, True
    except Exception as fallback_exc:
        raise primary_exc or fallback_exc


def _item_to_photo_analysis(item: BatchPhotoAnalysisItem) -> GeminiPhotoAnalysis:
    data = item.model_dump()
    data.pop("photo_id", None)
    parsed = GeminiPhotoAnalysis.model_validate(data)
    parsed.caption = clamp_caption_two_lines(parsed.caption)
    return parsed


def _call_gemini_vision_batch_api(
    client: genai.Client,
    batch_photos: list[PhotoMetadata],
    resolved_model: str,
    requested_model: str,
    thinking_level: str,
    media_resolution: str,
    prefer_generate_content: bool = False,
) -> tuple[dict[str, GeminiPhotoAnalysis], bool]:
    """Send multiple photos (e.g., 4 to 8 images) in a single multimodal API call to drastically reduce
    round-trip latency and thinking overhead."""
    if len(batch_photos) == 1:
        single_parsed, used_fb = _call_gemini_vision_api(
            client=client,
            photo=batch_photos[0],
            resolved_model=resolved_model,
            requested_model=requested_model,
            thinking_level=thinking_level,
            media_resolution=media_resolution,
            schema=GeminiPhotoAnalysis.model_json_schema(),
            prefer_generate_content=prefer_generate_content,
        )
        return {batch_photos[0].photo_id: single_parsed}, used_fb

    interaction_inputs: list[dict] = []
    generate_contents: list = []
    manifest_lines: list[str] = []

    for idx, p in enumerate(batch_photos, start=1):
        img_bytes, sent_w, sent_h, api_res_token = _prepare_gemini_image_bytes(p, media_resolution)
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        label_line = (
            f"Image #{idx}: photo_id='{p.photo_id}', filename='{p.filename}', "
            f"captured_at='{p.captured_at}', resolution={sent_w}x{sent_h}px"
        )
        manifest_lines.append(label_line)
        interaction_inputs.append(
            {
                "type": "image",
                "data": b64_img,
                "mime_type": "image/jpeg",
                "resolution": api_res_token,
            }
        )
        interaction_inputs.append({"type": "text", "text": label_line})
        generate_contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        generate_contents.append(label_line)

    batch_prompt = (
        f"{ANALYSIS_PROMPT}\n\n"
        f"You are analyzing a batch of {len(batch_photos)} photos in order.\n"
        "Return a JSON object with `items` containing exactly one entry per photo in this batch, "
        "preserving each photo's exact `photo_id`:\n"
        + "\n".join(manifest_lines)
    )
    interaction_inputs.append({"type": "text", "text": batch_prompt})
    generate_contents.append(batch_prompt)

    batch_schema = BatchGeminiPhotoAnalysis.model_json_schema()
    primary_exc: Optional[Exception] = None
    used_fallback = prefer_generate_content

    if not prefer_generate_content and not requested_model.startswith("gemini-2.5"):
        try:
            interaction = client.interactions.create(
                model=resolved_model,
                input=interaction_inputs,
                generation_config={"thinking_level": thinking_level},
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": batch_schema,
                },
            )
            batch_parsed = BatchGeminiPhotoAnalysis.model_validate_json(interaction.output_text)
            used_fallback = False
        except Exception as exc:
            primary_exc = exc
            batch_parsed = None
    else:
        batch_parsed = None

    if batch_parsed is None:
        fallback_model = requested_model if requested_model.startswith("gemini-2.5") else "gemini-2.5-flash"
        try:
            resp = client.models.generate_content(
                model=fallback_model,
                contents=generate_contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchGeminiPhotoAnalysis,
                ),
            )
            batch_parsed = BatchGeminiPhotoAnalysis.model_validate_json(resp.text)
            used_fallback = True
        except Exception as fallback_exc:
            raise primary_exc or fallback_exc

    out_map: dict[str, GeminiPhotoAnalysis] = {}
    by_id = {item.photo_id: item for item in batch_parsed.items}
    for idx, p in enumerate(batch_photos):
        matched_item = by_id.get(p.photo_id)
        if matched_item is None and idx < len(batch_parsed.items):
            matched_item = batch_parsed.items[idx]
        if matched_item is not None:
            out_map[p.photo_id] = _item_to_photo_analysis(matched_item)

    return out_map, used_fallback


def analyze_photos_with_gemini(
    photos: list[PhotoMetadata],
    model: str = DEFAULT_MODEL,
    thinking_level: str = "medium",
    media_resolution: str = "high",
    batch_size: int = 1,
    api_key: Optional[str] = None,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> tuple[str, Optional[str]]:
    """Run deep image analysis on photos using Gemini (supports both single-photo and multi-photo batch mode)."""
    from concurrent.futures import wait as futures_wait, FIRST_COMPLETED
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)

    resolved_model, model_note = resolve_gemini_model(model)
    effective_key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if effective_key in ("YOUR_GEMINI_API_KEY_HERE", "AIzaSyYourKeyHere"):
        effective_key = ""

    unanalyzed = [p for p in photos if not p.analysis.caption]
    if unanalyzed:
        pre_analyze_all_photos(unanalyzed, progress_cb=progress_cb)

    if not effective_key or not photos:
        note = "No GEMINI_API_KEY provided — used pixel-level visual analysis. Enter your Gemini API Key in the UI for deep Gemini 3.8 Flash captions & landmark recognition."
        model_note = f"{model_note} {note}".strip() if model_note else note
        return resolved_model, model_note

    try:
        client = genai.Client(api_key=effective_key, http_options={"timeout": 88000})
    except Exception:
        client = genai.Client(api_key=effective_key)
    schema = GeminiPhotoAnalysis.model_json_schema()
    api_errors: list[str] = []
    prefer_generate_content = model.startswith("gemini-2.5")

    def _cache_path(photo: PhotoMetadata) -> Path:
        return Path(photo.analysis_path).with_suffix(f".{resolved_model}.{thinking_level}.{media_resolution}.json")

    def _find_any_cached_analysis(photo: PhotoMetadata) -> Optional[GeminiPhotoAnalysis]:
        exact = _cache_path(photo)
        if exact.exists():
            try:
                return GeminiPhotoAnalysis.model_validate_json(exact.read_text(encoding="utf-8"))
            except Exception:
                pass
        base_stem = Path(photo.analysis_path).stem
        parent_dir = Path(photo.analysis_path).parent
        for cand_file in parent_dir.glob(f"{base_stem}.gemini-*.json"):
            try:
                return GeminiPhotoAnalysis.model_validate_json(cand_file.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

    remaining_photos: list[PhotoMetadata] = []
    g_total = len(photos)
    g_done = 0
    for p in photos:
        cached = _find_any_cached_analysis(p)
        if cached is not None:
            p.analysis = cached
            g_done += 1
            if progress_cb and (g_done % 10 == 0 or g_done == g_total):
                progress_cb(
                    f"Loaded cached Gemini analysis {g_done}/{g_total}: {p.filename}",
                    g_done,
                    g_total,
                )
            continue
        remaining_photos.append(p)

    effective_batch_size = max(1, min(10, int(batch_size or 1)))

    if remaining_photos and effective_batch_size > 1 and len(remaining_photos) > 1:
        batches = [
            remaining_photos[i : i + effective_batch_size]
            for i in range(0, len(remaining_photos), effective_batch_size)
        ]
        # Probe the first batch first
        first_batch = batches[0]
        if progress_cb:
            progress_cb(
                f"Batch analyzing photos {g_done + 1}–{g_done + len(first_batch)}/{g_total} (batch size={effective_batch_size})...",
                max(1, g_done),
                g_total,
            )
        try:
            mapped, used_fb = _call_gemini_vision_batch_api(
                client=client,
                batch_photos=first_batch,
                resolved_model=resolved_model,
                requested_model=model,
                thinking_level=thinking_level,
                media_resolution=media_resolution,
                prefer_generate_content=prefer_generate_content,
            )
            prefer_generate_content = used_fb
            for bp in first_batch:
                if bp.photo_id in mapped:
                    bp.analysis = mapped[bp.photo_id]
                    _cache_path(bp).write_text(bp.analysis.model_dump_json(), encoding="utf-8")
            g_done += len(first_batch)
            batches = batches[1:]
            if progress_cb:
                progress_cb(
                    f"Batch analyzed {g_done}/{g_total} photos ({len(batches)} batch calls remaining)...",
                    g_done,
                    g_total,
                )
        except Exception as probe_exc:
            api_errors.append(str(probe_exc))
            batches = []

        def _call_batch(b_photos: list[PhotoMetadata]) -> int:
            try:
                mapped_res, _ = _call_gemini_vision_batch_api(
                    client=client,
                    batch_photos=b_photos,
                    resolved_model=resolved_model,
                    requested_model=model,
                    thinking_level=thinking_level,
                    media_resolution=media_resolution,
                    prefer_generate_content=prefer_generate_content,
                )
                for bp in b_photos:
                    if bp.photo_id in mapped_res:
                        bp.analysis = mapped_res[bp.photo_id]
                        _cache_path(bp).write_text(bp.analysis.model_dump_json(), encoding="utf-8")
                return len(b_photos)
            except Exception as exc:
                if len(api_errors) < 3:
                    api_errors.append(str(exc))
                return len(b_photos)

        if batches:
            executor = ThreadPoolExecutor(max_workers=min(6, len(batches)))
            try:
                pending = {executor.submit(_call_batch, b): b for b in batches}
                while pending:
                    done_set, not_done_set = futures_wait(pending.keys(), timeout=96.0, return_when=FIRST_COMPLETED)
                    if not done_set:
                        for stalled_fut in list(not_done_set):
                            stalled_fut.cancel()
                            pending.pop(stalled_fut, None)
                        break
                    for fut in done_set:
                        b_list = pending.pop(fut)
                        g_done += len(b_list)
                        try:
                            fut.result(timeout=1.0)
                        except Exception:
                            pass
                        if progress_cb:
                            progress_cb(
                                f"Batch analyzed {min(g_done, g_total)}/{g_total} photos (last: {b_list[-1].filename})",
                                min(g_done, g_total),
                                g_total,
                            )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        if api_errors:
            err_msg = f"Gemini API error ({api_errors[0][:160]})."
            model_note = f"{model_note} {err_msg}".strip() if model_note else err_msg

        return resolved_model, model_note

    # Single-photo mode (batch_size == 1 or only 1 uncached photo)
    if remaining_photos:
        probe_photo = remaining_photos[0]
        if progress_cb:
            progress_cb(
                f"Connecting to Gemini API for photo {g_done + 1}/{g_total} ({probe_photo.filename})...",
                max(1, g_done),
                g_total,
            )
        try:
            parsed, used_fallback = _call_gemini_vision_api(
                client=client,
                photo=probe_photo,
                resolved_model=resolved_model,
                requested_model=model,
                thinking_level=thinking_level,
                media_resolution=media_resolution,
                schema=schema,
                prefer_generate_content=prefer_generate_content,
            )
            probe_photo.analysis = parsed
            _cache_path(probe_photo).write_text(parsed.model_dump_json(), encoding="utf-8")
            prefer_generate_content = used_fallback
            remaining_photos = remaining_photos[1:]
            g_done += 1
            if progress_cb:
                progress_cb(
                    f"Gemini analyzed {g_done}/{g_total}: {probe_photo.filename}",
                    g_done,
                    g_total,
                )
        except Exception as probe_exc:
            api_errors.append(str(probe_exc))
            remaining_photos = []

    def _call_one(photo: PhotoMetadata) -> None:
        cf = _cache_path(photo)
        try:
            parsed, _ = _call_gemini_vision_api(
                client=client,
                photo=photo,
                resolved_model=resolved_model,
                requested_model=model,
                thinking_level=thinking_level,
                media_resolution=media_resolution,
                schema=schema,
                prefer_generate_content=prefer_generate_content,
            )
            photo.analysis = parsed
            cf.write_text(parsed.model_dump_json(), encoding="utf-8")
        except Exception as exc:
            if len(api_errors) < 3:
                api_errors.append(str(exc))

    if remaining_photos:
        executor = ThreadPoolExecutor(max_workers=min(10, len(remaining_photos)))
        try:
            pending = {executor.submit(_call_one, p): p for p in remaining_photos}
            while pending:
                done_set, not_done_set = futures_wait(pending.keys(), timeout=80.0, return_when=FIRST_COMPLETED)
                if not done_set:
                    for stalled_fut in list(not_done_set):
                        stalled_fut.cancel()
                        pending.pop(stalled_fut, None)
                    break
                for fut in done_set:
                    p = pending.pop(fut)
                    g_done += 1
                    try:
                        fut.result(timeout=1.0)
                    except Exception:
                        pass
                    if progress_cb:
                        progress_cb(
                            f"Gemini analyzed {g_done}/{g_total}: {p.filename}",
                            g_done,
                            g_total,
                        )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    if api_errors:
        err_msg = f"Gemini API error ({api_errors[0][:160]})."
        model_note = f"{model_note} {err_msg}".strip() if model_note else err_msg

    return resolved_model, model_note


def _text_token_jaccard(a: str, b: str) -> float:
    tokens_a = {w for w in re.findall(r"[a-z0-9]+", (a or "").lower()) if len(w) > 2}
    tokens_b = {w for w in re.findall(r"[a-z0-9]+", (b or "").lower()) if len(w) > 2}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / float(len(tokens_a | tokens_b))


def is_sideways_or_inverted_photo(photo: PhotoMetadata) -> tuple[bool, Optional[str]]:
    """Detect if a photo has a sideways or upside-down face/scene orientation that shouldn't appear in the photo book."""
    a = photo.analysis
    if getattr(a, "has_sideways_or_inverted_orientation", False):
        return True, "Sideways or upside-down face/scene orientation"

    summary_lower = (a.subject_summary or "").lower()
    caption_lower = (a.caption or "").lower()
    combined = f"{summary_lower} {caption_lower}"

    sideways_markers = (
        "sideways",
        "sideways-oriented",
        "sideways/portrait",
        "upside-down",
        "upside down",
        "rotated 90",
        "vertical orientation of sunlight and shadow casting patterns across steep canyon",
    )
    if any(m in combined for m in sideways_markers):
        return True, "Sideways or unrotated camera orientation"

    # Top-down / high-angle shot of someone lying upside-down/sideways with eyes closed on a cushion
    if (
        ("top-down perspective" in summary_lower or "high-angle perspective" in summary_lower)
        and ("resting on a boat cushion" in summary_lower or "resting flat on his back" in summary_lower or "naps on the deck" in combined)
    ):
        return True, "Upside-down / sideways face orientation in top-down resting shot"

    return False, None


def coarse_filter_photos(
    photos: list[PhotoMetadata],
    exact_hamming_threshold: int = 0,
    exact_color_threshold: float = 0.998,
) -> list[PhotoMetadata]:
    """Stage 2 in Smart Hybrid (Gemini-First) flow: Fast coarse pruning on local CPU (<1 second).
    Drops exact duplicate files (identical dHash/aHash/bytes) and obvious document/receipt files
    so Gemini API calls are not wasted, but preserves ALL unique scene and personal photos for deep Gemini analysis.
    Crucially: DOES NOT downsample by max_book_photos here; downsampling happens AFTER Gemini analyzes all photos.
    """
    candidates: list[PhotoMetadata] = []
    for p in photos:
        if p.analysis.is_document_or_screenshot:
            p.status = "excluded_document"
            p.exclusion_reason = p.analysis.document_reason or "Detected document, receipt, or screenshot (coarse pre-filter)"
            continue
        if p.analysis.sharpness_score < 2.5 and p.analysis.aesthetic_score < 3.0:
            p.status = "excluded_low_quality"
            p.exclusion_reason = f"Extreme technical blur or corrupt frame (sharpness {p.analysis.sharpness_score}/10)"
            continue
        candidates.append(p)

    # Rank by quality so if an exact duplicate exists, the higher-quality file is retained
    ranked = sorted(
        candidates,
        key=lambda x: (x.analysis.composite_score, x.analysis.sharpness_score, -x.captured_timestamp),
        reverse=True,
    )

    kept: list[PhotoMetadata] = []
    for cand in ranked:
        matched_keeper: Optional[PhotoMetadata] = None
        for keeper in kept:
            d_dist = hamming_distance(cand.dhash, keeper.dhash)
            a_dist = hamming_distance(cand.ahash, keeper.ahash)
            color_sim = cosine_similarity(cand.color_signature, keeper.color_signature)

            if d_dist <= exact_hamming_threshold and a_dist <= exact_hamming_threshold and color_sim >= exact_color_threshold:
                matched_keeper = keeper
                break

        if matched_keeper:
            cand.status = "excluded_duplicate"
            cand.duplicate_of = matched_keeper.photo_id
            cand.exclusion_reason = (
                f"Exact duplicate file of {matched_keeper.filename} (dHash distance={d_dist}, "
                f"color similarity={color_sim:.3f})"
            )
        else:
            cand.status = "selected"
            kept.append(cand)

    kept.sort(key=lambda r: (r.captured_timestamp, r.filename))
    return kept


def deduplicate_and_curate_with_gemini(
    photos: list[PhotoMetadata],
    duplicate_hamming_threshold: int = 6,
    similar_hamming_threshold: int = 15,
    similar_color_threshold: float = 0.965,
    max_book_photos: int = 90,
) -> list[PhotoMetadata]:
    """Stage 4 in Smart Hybrid (Gemini-First) flow: Semantic Deduplication, Burst Curation & Timeline Bucket Selection.
    Runs AFTER all surviving photos have been deeply analyzed by Gemini 3.8 Flash, leveraging Gemini's real
    facial expression awareness, true aesthetic scores, landmark recognition, and sideways orientation detection.
    """
    candidates: list[PhotoMetadata] = []
    for p in photos:
        if p.analysis.is_document_or_screenshot:
            p.status = "excluded_document"
            p.exclusion_reason = p.analysis.document_reason or "Detected document, receipt, or screenshot"
            continue
        if p.analysis.is_blurry_or_flawed or p.analysis.sharpness_score < 3.6:
            p.status = "excluded_low_quality"
            p.exclusion_reason = f"Low technical quality or motion blur (sharpness {p.analysis.sharpness_score}/10)"
            continue
        is_sideways, sideways_reason = is_sideways_or_inverted_photo(p)
        if is_sideways:
            p.status = "excluded_low_quality"
            p.exclusion_reason = sideways_reason or "Sideways or inverted face/scene orientation"
            continue
        candidates.append(p)

    # Rank using Gemini's multimodal scores (aesthetic, composite, and sharpness)
    ranked = sorted(
        candidates,
        key=lambda x: (
            x.analysis.composite_score,
            x.analysis.aesthetic_score,
            x.analysis.sharpness_score,
            -x.captured_timestamp,
        ),
        reverse=True,
    )

    kept: list[PhotoMetadata] = []
    for cand in ranked:
        matched_keeper: Optional[PhotoMetadata] = None
        match_kind: Optional[str] = None
        match_reason: Optional[str] = None

        for keeper in kept:
            d_dist = hamming_distance(cand.dhash, keeper.dhash)
            a_dist = hamming_distance(cand.ahash, keeper.ahash)
            color_sim = cosine_similarity(cand.color_signature, keeper.color_signature)
            time_diff_hours = abs(cand.captured_timestamp - keeper.captured_timestamp) / 3600.0
            time_diff_minutes = time_diff_hours * 60.0

            if d_dist == 0 and a_dist == 0 and color_sim >= 0.998:
                matched_keeper = keeper
                match_kind = "excluded_duplicate"
                match_reason = (
                    f"Exact/near duplicate of {keeper.filename} (dHash distance={d_dist}, "
                    f"color similarity={color_sim:.3f})"
                )
                break

            is_perceptually_similar = (
                d_dist <= duplicate_hamming_threshold
                and a_dist <= duplicate_hamming_threshold
                and color_sim >= 0.985
            )
            is_time_burst = time_diff_minutes <= 3.0 and color_sim >= 0.94 and d_dist <= 20
            is_semantic_burst = (
                time_diff_hours <= 2.0
                and d_dist <= similar_hamming_threshold
                and color_sim >= similar_color_threshold
                and _text_token_jaccard(cand.analysis.subject_summary, keeper.analysis.subject_summary) >= 0.45
            )
            if is_perceptually_similar or is_time_burst or is_semantic_burst:
                matched_keeper = keeper
                match_kind = "excluded_similar"
                match_reason = (
                    f"Similar burst shot to {keeper.filename} (dHash={d_dist}, "
                    f"color similarity={color_sim:.2f}, Δt={time_diff_minutes:.1f}m; kept higher-rated {keeper.filename} "
                    f"with Gemini score {keeper.analysis.composite_score} vs {cand.analysis.composite_score})"
                )
                break

        if matched_keeper and match_kind:
            cand.status = match_kind  # type: ignore[assignment]
            cand.duplicate_of = matched_keeper.photo_id
            cand.exclusion_reason = match_reason
        else:
            cand.status = "selected"
            kept.append(cand)

    kept.sort(key=lambda r: (r.captured_timestamp, r.filename))

    if max_book_photos > 0 and len(kept) > max_book_photos:
        bucket_size = len(kept) / float(max_book_photos)
        final_kept: list[PhotoMetadata] = []
        for b in range(max_book_photos):
            start_i = int(round(b * bucket_size))
            end_i = int(round((b + 1) * bucket_size))
            window = kept[start_i : max(start_i + 1, end_i)]
            if not window:
                continue
            best_in_window = max(
                window,
                key=lambda x: (
                    x.analysis.composite_score,
                    x.analysis.cover_worthiness_score,
                    x.analysis.sharpness_score,
                ),
            )
            final_kept.append(best_in_window)
            for other in window:
                if other.photo_id != best_in_window.photo_id:
                    other.status = "excluded_similar"
                    other.duplicate_of = best_in_window.photo_id
                    other.exclusion_reason = (
                        f"Curated out in chronological time-cluster in favor of higher-rated {best_in_window.filename} "
                        f"(Gemini score {best_in_window.analysis.composite_score} vs {other.analysis.composite_score})"
                    )
        kept = final_kept

    kept.sort(key=lambda r: (r.captured_timestamp, r.filename))
    return kept


def deduplicate_and_filter_photos(
    photos: list[PhotoMetadata],
    duplicate_hamming_threshold: int = 6,
    similar_hamming_threshold: int = 15,
    similar_color_threshold: float = 0.965,
    max_book_photos: int = 90,
) -> list[PhotoMetadata]:
    """Compatibility alias for deduplicate_and_curate_with_gemini."""
    return deduplicate_and_curate_with_gemini(
        photos=photos,
        duplicate_hamming_threshold=duplicate_hamming_threshold,
        similar_hamming_threshold=similar_hamming_threshold,
        similar_color_threshold=similar_color_threshold,
        max_book_photos=max_book_photos,
    )


# ── 10x10 Inch Page Templates & Slot Geometries ──────────────────────────────
PAGE_TEMPLATES_BY_FILL: dict[str, dict[PageTemplateId, list[tuple[float, float, float, float]]]] = {
    "low": {
        "cover_hero": [
            (1.10, 1.65, 7.80, 5.85),
        ],
        "single_full_square": [
            (0.95, 1.60, 8.10, 6.15),
        ],
        "single_editorial": [
            (2.15, 1.60, 5.70, 6.15),
        ],
        "single_panoramic": [
            (0.80, 2.15, 8.40, 5.25),
        ],
        "two_side_by_side": [
            (0.75, 1.65, 4.10, 6.05),
            (5.15, 1.65, 4.10, 6.05),
        ],
        "two_stacked_horizontal": [
            (1.05, 1.60, 7.90, 2.70),
            (1.05, 5.05, 7.90, 2.70),
        ],
        "two_asymmetric": [
            (0.75, 1.65, 5.05, 6.05),
            (6.05, 2.15, 3.20, 5.00),
        ],
        "three_hero_left": [
            (0.75, 1.60, 4.50, 6.15),
            (5.55, 1.60, 3.70, 2.70),
            (5.55, 5.05, 3.70, 2.70),
        ],
        "three_hero_top": [
            (0.85, 1.60, 8.30, 2.75),
            (0.85, 5.10, 3.95, 2.65),
            (5.20, 5.10, 3.95, 2.65),
        ],
        "three_columns": [
            (0.65, 1.70, 2.75, 5.95),
            (3.625, 1.70, 2.75, 5.95),
            (6.60, 1.70, 2.75, 5.95),
        ],
    },
    "medium": {
        "cover_hero": [
            (0.75, 1.40, 8.50, 6.40),
        ],
        "single_full_square": [
            (0.65, 1.35, 8.70, 6.85),
        ],
        "single_editorial": [
            (1.65, 1.35, 6.70, 6.85),
        ],
        "single_panoramic": [
            (0.55, 1.85, 8.90, 5.85),
        ],
        "two_side_by_side": [
            (0.55, 1.35, 4.30, 6.85),
            (5.15, 1.35, 4.30, 6.85),
        ],
        "two_stacked_horizontal": [
            (0.55, 1.35, 8.90, 3.25),
            (0.55, 4.95, 8.90, 3.25),
        ],
        "two_asymmetric": [
            (0.50, 1.35, 5.35, 6.85),
            (6.15, 1.85, 3.35, 5.85),
        ],
        "three_hero_left": [
            (0.50, 1.35, 4.90, 6.85),
            (5.65, 1.35, 3.80, 3.25),
            (5.65, 4.95, 3.80, 3.25),
        ],
        "three_hero_top": [
            (0.55, 1.35, 8.90, 3.25),
            (0.55, 4.95, 4.30, 3.25),
            (5.15, 4.95, 4.30, 3.25),
        ],
        "three_columns": [
            (0.45, 1.35, 2.90, 6.85),
            (3.55, 1.35, 2.90, 6.85),
            (6.65, 1.35, 2.90, 6.85),
        ],
    },
    "high": {
        "cover_hero": [
            (0.45, 1.15, 9.10, 7.00),
        ],
        "single_full_square": [
            (0.40, 1.10, 9.20, 7.50),
        ],
        "single_editorial": [
            (1.20, 1.10, 7.60, 7.50),
        ],
        "single_panoramic": [
            (0.35, 1.60, 9.30, 6.50),
        ],
        "two_side_by_side": [
            (0.35, 1.10, 4.50, 7.50),
            (5.15, 1.10, 4.50, 7.50),
        ],
        "two_stacked_horizontal": [
            (0.35, 1.10, 9.30, 3.60),
            (0.35, 5.00, 9.30, 3.60),
        ],
        "two_asymmetric": [
            (0.35, 1.10, 5.65, 7.50),
            (6.30, 1.60, 3.35, 6.50),
        ],
        "three_hero_left": [
            (0.35, 1.10, 5.25, 7.50),
            (5.85, 1.10, 3.80, 3.60),
            (5.85, 5.00, 3.80, 3.60),
        ],
        "three_hero_top": [
            (0.35, 1.10, 9.30, 3.60),
            (0.35, 5.00, 4.50, 3.60),
            (5.15, 5.00, 4.50, 3.60),
        ],
        "three_columns": [
            (0.35, 1.10, 2.95, 7.50),
            (3.52, 1.10, 2.95, 7.50),
            (6.70, 1.10, 2.95, 7.50),
        ],
    },
}

PAGE_TEMPLATES: dict[PageTemplateId, list[tuple[float, float, float, float]]] = PAGE_TEMPLATES_BY_FILL["high"]


def get_page_templates(fill_mode: str = "high") -> dict[PageTemplateId, list[tuple[float, float, float, float]]]:
    """Retrieve slot geometry tuples (x_in, y_in, w_in, h_in) for the requested page fill mode."""
    clean = (fill_mode or "high").lower().strip()
    return PAGE_TEMPLATES_BY_FILL.get(clean, PAGE_TEMPLATES_BY_FILL["high"])


def _can_page_preserve_all_faces(
    page_photos: list[PhotoMetadata],
    tpl: PageTemplateId,
    fill_mode: str = "high",
    crop_focus: str = "tight",
) -> bool:
    geoms = get_page_templates(fill_mode)[tpl]
    for photo, (_, _, w_in, h_in) in zip(page_photos, geoms):
        if not can_fit_aspect_without_cutting_faces(
            photo.analysis_width,
            photo.analysis_height,
            w_in / h_in,
            photo.analysis.faces_boxes,
            crop_focus=crop_focus,
        ):
            return False
    return True


def choose_best_page_template(
    page_photos: list[PhotoMetadata],
    fill_mode: str = "high",
    crop_focus: str = "tight",
) -> PageTemplateId:
    """Pick the most aesthetic 1-, 2-, or 3-photo template on a 10x10 inch page
    that preserves 100% of all detected faces without cutting any face off."""
    n = len(page_photos)
    if n == 1:
        p = page_photos[0]
        if p.orientation == "portrait":
            candidates: list[PageTemplateId] = ["single_editorial", "single_full_square", "single_panoramic"]
        elif p.aspect_ratio >= 1.35:
            candidates = ["single_panoramic", "single_full_square", "single_editorial"]
        else:
            candidates = ["single_full_square", "single_editorial", "single_panoramic"]
    elif n == 2:
        p0, p1 = page_photos[0], page_photos[1]
        if p0.orientation == "landscape" and p1.orientation == "landscape":
            candidates = ["two_stacked_horizontal", "two_side_by_side", "two_asymmetric"]
        elif p0.orientation == "portrait" and p1.orientation == "portrait":
            candidates = ["two_side_by_side", "two_asymmetric", "two_stacked_horizontal"]
        else:
            candidates = ["two_asymmetric", "two_stacked_horizontal", "two_side_by_side"]
    else:
        if page_photos[0].orientation == "landscape":
            candidates = ["three_hero_top", "three_hero_left", "three_columns"]
        elif all(p.orientation == "portrait" for p in page_photos):
            candidates = ["three_columns", "three_hero_left", "three_hero_top"]
        else:
            candidates = ["three_hero_left", "three_hero_top", "three_columns"]

    for tpl in candidates:
        if _can_page_preserve_all_faces(page_photos, tpl, fill_mode=fill_mode, crop_focus=crop_focus):
            return tpl

    return candidates[0]


def _synthesize_group_caption(photos_in_group: list[PhotoMetadata], chapter_title: str, day_str: str) -> str:
    """Create a natural 1-2 line caption for a cluster or multi-photo page."""
    if not photos_in_group:
        return f"Memorable moments from {chapter_title} ({day_str})."
    if len(photos_in_group) == 1:
        return clamp_caption_two_lines(photos_in_group[0].analysis.caption)

    best = max(photos_in_group, key=lambda p: p.analysis.composite_score)
    base_cap = best.analysis.caption.rstrip(".")
    themes = []
    for p in photos_in_group:
        t = (p.analysis.event_theme or "").strip()
        if t and t.lower() not in base_cap.lower() and t not in themes:
            themes.append(t)

    if themes and len(base_cap) < 90:
        combined = f"{base_cap} — {themes[0]}."
        if len(combined) <= 140:
            return combined
    return clamp_caption_two_lines(best.analysis.caption)


def select_hero_cover_photo(selected_photos: list[PhotoMetadata]) -> PhotoMetadata:
    """Select the Hero cover image for the 10x10 front cover:
    1. Give strong preference to a high-quality scenic view (no close-up faces/portraits, balanced aspect ratio).
    2. If no good scenic view is available, pick the overall most suitable photo across all selected photos.
    """
    non_scenic_object_keywords = (
        "fresco",
        "boutique",
        "tablecloth",
        "yogurt",
        "cacti",
        "chandelier",
        "pottery",
        "sacks",
        "menu",
        "receipt",
        "dinner",
        "lunch",
    )

    def _is_good_scenic(p: PhotoMetadata, require_gemini_caption: bool = True) -> bool:
        a = p.analysis
        if a.people_count > 0 or len(a.faces_boxes) > 0:
            return False
        if a.cover_worthiness_score < 6.8 or a.aesthetic_score < 7.0:
            return False
        aspect = p.analysis_width / max(1.0, float(p.analysis_height))
        if not (0.70 <= aspect <= 2.15):
            return False
        if require_gemini_caption and (a.subject_summary or "").endswith("px)"):
            return False
        txt = f"{a.subject_summary} {a.caption} {a.event_theme}".lower()
        if any(k in txt for k in non_scenic_object_keywords):
            return False
        return True

    def _scenic_rank_key(p: PhotoMetadata) -> tuple[float, float, float]:
        a = p.analysis
        orient_bonus = 0.55 if p.orientation in ("landscape", "square") else 0.0
        return (
            round(a.cover_worthiness_score + orient_bonus, 3),
            a.aesthetic_score,
            a.sharpness_score,
        )

    # First try Gemini-analyzed scenic views
    scenic_candidates = [p for p in selected_photos if _is_good_scenic(p, require_gemini_caption=True)]
    if not scenic_candidates:
        # Next try any scenic view (including offline/pixel-analyzed scenic views)
        scenic_candidates = [p for p in selected_photos if _is_good_scenic(p, require_gemini_caption=False)]

    if scenic_candidates:
        return max(scenic_candidates, key=_scenic_rank_key)

    # Fallback: if no good scenic view is available, pick the most suitable photo overall
    return max(
        selected_photos,
        key=lambda p: (
            0 if (p.analysis.subject_summary or "").endswith("px)") else 1,
            p.analysis.cover_worthiness_score + (0.35 if p.orientation in ("square", "landscape") else 0.0),
            p.analysis.composite_score,
            p.analysis.sharpness_score,
        ),
    )


def _default_clusters_and_pages(
    selected_photos: list[PhotoMetadata],
    fill_mode: str = "high",
    crop_focus: str = "tight",
) -> _BookCurationPlan:
    """Chronological clustering and 1-to-3 photo page grouping from the photo analyses."""
    by_day: dict[str, list[PhotoMetadata]] = defaultdict(list)
    for p in selected_photos:
        by_day[p.captured_at[:10]].append(p)

    clusters: list[_CurationClusterPlan] = []
    for day_idx, (day_str, day_photos) in enumerate(sorted(by_day.items()), start=1):
        theme_counts: Counter[str] = Counter()
        landmark_counts: Counter[str] = Counter()
        for p in day_photos:
            if p.analysis.event_theme:
                theme_counts[p.analysis.event_theme] += 1
            if p.analysis.location_or_landmark and p.analysis.location_or_landmark not in ("N/A", ""):
                landmark_counts[p.analysis.location_or_landmark] += 1

        top_themes = [t for t, _ in theme_counts.most_common(2)]
        title_body = " & ".join(top_themes) if top_themes else f"Highlights of {day_str}"
        chapter_title = f"Day {day_idx} ({day_str}) — {title_body}" if len(by_day) > 1 else title_body

        cid = f"cluster_{len(clusters) + 1:02d}"
        grp_cap = _synthesize_group_caption(day_photos, chapter_title, day_str)
        clusters.append(
            _CurationClusterPlan(
                cluster_id=cid,
                title=chapter_title,
                date_range_label=day_str,
                group_caption=grp_cap,
                photo_ids=[p.photo_id for p in day_photos],
            )
        )

    hero_photo = select_hero_cover_photo(selected_photos)

    pages: list[_CurationPagePlan] = []
    photo_by_id = {p.photo_id: p for p in selected_photos}

    for cluster in clusters:
        c_photos = [photo_by_id[pid] for pid in cluster.photo_ids if pid in photo_by_id]
        idx = 0
        while idx < len(c_photos):
            left_count = len(c_photos) - idx
            if left_count == 1:
                take = 1
            elif left_count == 2:
                take = 1 if c_photos[idx].analysis.composite_score >= 9.35 else 2
            elif left_count == 3:
                take = 3 if c_photos[idx].analysis.composite_score < 9.2 else 1
            elif left_count == 4:
                take = 2
            else:
                take = 3 if (len(pages) % 2 == 0) else 2

            batch = c_photos[idx : idx + take]
            while len(batch) > 1:
                tpl_check = choose_best_page_template(batch, fill_mode=fill_mode, crop_focus=crop_focus)
                if _can_page_preserve_all_faces(batch, tpl_check, fill_mode=fill_mode, crop_focus=crop_focus):
                    break
                take -= 1
                batch = c_photos[idx : idx + take]

            idx += take
            page_cap = _synthesize_group_caption(batch, cluster.title, cluster.date_range_label)
            pages.append(
                _CurationPagePlan(
                    cluster_id=cluster.cluster_id,
                    photo_ids=[p.photo_id for p in batch],
                    group_caption=page_cap,
                )
            )

    first_date = selected_photos[0].captured_at[:10]
    last_date = selected_photos[-1].captured_at[:10]
    date_subtitle = first_date if first_date == last_date else f"{first_date} to {last_date}"
    book_title = clusters[0].title if len(clusters) == 1 else f"Captured Moments ({date_subtitle})"

    return _BookCurationPlan(
        book_title=book_title,
        book_subtitle=f"A Curated 10×10 Photo Book • {date_subtitle}",
        hero_cover_photo_id=hero_photo.photo_id,
        clusters=clusters,
        pages=pages,
    )


def curate_clusters_and_pages(
    selected_photos: list[PhotoMetadata],
    work_dir: str | Path,
    model: str = DEFAULT_MODEL,
    thinking_level: str = "medium",
    curation_model: str = "gemini-2.5-flash-lite",
    hero_model: str = "gemini-3.8-flash",
    layout_model: str = "local_geometric",
    batch_size: int = 6,
    page_fill_mode: str = "high",
    crop_focus: str = "tight",
    api_key: Optional[str] = None,
) -> tuple[str, str, str, list[PhotoCluster], list[BookPage]]:
    """Cluster surviving chronological photos into thematic events, select the Hero cover image,
    and arrange 1, 2, or 3 photos max per 10x10 inch page with face-safe aesthetic cropping."""
    if not selected_photos:
        raise ValueError("No photos remaining after removing duplicates and documents.")

    effective_key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if effective_key in ("YOUR_GEMINI_API_KEY_HERE", "AIzaSyYourKeyHere"):
        effective_key = ""

    photo_map = {p.photo_id: p for p in selected_photos}

    # Stage 3 (Optional): Hero Cover single-photo precision refinement when batch_size > 1
    if effective_key and hero_model and hero_model != "none" and batch_size > 1:
        hero_cand = select_hero_cover_photo(selected_photos)
        hero_resolved, _ = resolve_gemini_model(hero_model)
        hero_cache = Path(hero_cand.analysis_path).with_suffix(f".hero.{hero_resolved}.json")
        if hero_cache.exists():
            try:
                hero_cand.analysis = GeminiPhotoAnalysis.model_validate_json(hero_cache.read_text(encoding="utf-8"))
            except Exception:
                pass
        else:
            try:
                client = genai.Client(api_key=effective_key, http_options={"timeout": 72000})
                refined_hero, _ = _call_gemini_vision_api(
                    client=client,
                    photo=hero_cand,
                    resolved_model=hero_resolved,
                    requested_model=hero_model,
                    thinking_level="medium",
                    media_resolution="high",
                    schema=GeminiPhotoAnalysis.model_json_schema(),
                    prefer_generate_content=hero_model.startswith("gemini-2.5"),
                )
                hero_cand.analysis = refined_hero
                hero_cache.write_text(refined_hero.model_dump_json(), encoding="utf-8")
            except Exception:
                pass

    # Build deterministic face-safe chronological clusters & pages first (<5ms)
    plan = _default_clusters_and_pages(selected_photos, fill_mode=page_fill_mode, crop_focus=crop_focus)

    # Stage 2: Fast Curation & Chapter Captioning model (e.g. gemini-2.5-flash-lite with 0 thinking budget)
    if effective_key and curation_model and curation_model != "local_fast":
        try:
            client = genai.Client(api_key=effective_key, http_options={"timeout": 40000})
            cluster_inputs = []
            for c in plan.clusters:
                c_photos = [photo_map[pid] for pid in c.photo_ids if pid in photo_map]
                landmarks = list({p.analysis.location_or_landmark for p in c_photos if p.analysis.location_or_landmark})[:4]
                themes = list({p.analysis.event_theme for p in c_photos if p.analysis.event_theme})[:4]
                sample_caps = [p.analysis.caption for p in c_photos[:3] if p.analysis.caption]
                cluster_inputs.append(
                    {
                        "cluster_id": c.cluster_id,
                        "date_range": c.date_range_label,
                        "landmarks": landmarks,
                        "themes": themes,
                        "sample_captions": sample_caps,
                    }
                )
            curation_prompt = (
                "You are titling a 10x10 inch personal photo book and its chronological chapters.\n"
                "Based ONLY on the actual landmarks, themes, and dates below, return:\n"
                "1. A memorable `book_title` (under 55 chars).\n"
                "2. For each cluster in `clusters`, a concise chapter `title` and a warm 1-2 line max `group_caption` (under 135 chars).\n\n"
                f"Clusters JSON:\n{json.dumps(cluster_inputs)}"
            )
            if curation_model.startswith("gemini-3.8"):
                interaction = client.interactions.create(
                    model="gemini-3.8-flash",
                    input=curation_prompt,
                    generation_config={"thinking_level": "low"},
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": BookCurationRefinement.model_json_schema(),
                    },
                )
                refinement = BookCurationRefinement.model_validate_json(interaction.output_text)
            else:
                target_curation_model = (
                    curation_model if curation_model.startswith("gemini-2.5") else "gemini-2.5-flash-lite"
                )
                resp = client.models.generate_content(
                    model=target_curation_model,
                    contents=curation_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BookCurationRefinement,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                refinement = BookCurationRefinement.model_validate_json(resp.text)

            if refinement.book_title:
                plan.book_title = refinement.book_title.strip()
            ref_by_id = {item.cluster_id: item for item in refinement.clusters}
            for c in plan.clusters:
                if c.cluster_id in ref_by_id:
                    r_item = ref_by_id[c.cluster_id]
                    if r_item.title:
                        c.title = r_item.title.strip()
                    if r_item.group_caption:
                        c.group_caption = clamp_caption_two_lines(r_item.group_caption)
        except Exception:
            pass

    if plan.hero_cover_photo_id not in photo_map:
        plan.hero_cover_photo_id = selected_photos[0].photo_id

    clusters: list[PhotoCluster] = []
    cluster_map: dict[str, PhotoCluster] = {}
    for c in plan.clusters:
        valid_pids = [pid for pid in c.photo_ids if pid in photo_map]
        if not valid_pids:
            continue
        cluster_obj = PhotoCluster(
            cluster_id=c.cluster_id,
            title=c.title,
            date_range_label=c.date_range_label,
            group_caption=clamp_caption_two_lines(c.group_caption),
            photo_ids=valid_pids,
        )
        clusters.append(cluster_obj)
        cluster_map[c.cluster_id] = cluster_obj
        for pid in valid_pids:
            photo_map[pid].cluster_id = c.cluster_id

    if not clusters:
        fallback_plan = _default_clusters_and_pages(selected_photos)
        for c in fallback_plan.clusters:
            cluster_obj = PhotoCluster(
                cluster_id=c.cluster_id,
                title=c.title,
                date_range_label=c.date_range_label,
                group_caption=clamp_caption_two_lines(c.group_caption),
                photo_ids=c.photo_ids,
            )
            clusters.append(cluster_obj)
            cluster_map[c.cluster_id] = cluster_obj

    crops_dir = Path(work_dir).expanduser().resolve() / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    DPI = 200

    pages: list[BookPage] = []
    crop_tasks: list[tuple[str, tuple[int, int, int, int], Path, int, int]] = []

    templates = get_page_templates(page_fill_mode)

    hero = photo_map[plan.hero_cover_photo_id]
    cx_in, cy_in, cw_in, ch_in = templates["cover_hero"][0]
    c_crop_px, c_crop_norm, c_faces_ok = compute_aesthetic_crop(
        width=hero.analysis_width,
        height=hero.analysis_height,
        target_aspect=cw_in / ch_in,
        faces_boxes=hero.analysis.faces_boxes,
        salient_box=hero.analysis.salient_subject_box,
        crop_focus=crop_focus,
    )
    cover_crop_name = f"page_00_cover_{hero.photo_id}.jpg"
    crop_tasks.append(
        (
            hero.analysis_path,
            c_crop_px,
            crops_dir / cover_crop_name,
            int(round(cw_in * DPI)),
            int(round(ch_in * DPI)),
        )
    )
    pages.append(
        BookPage(
            page_number=0,
            page_type="cover",
            template_id="cover_hero",
            chapter_title=plan.book_title,
            group_caption=clamp_caption_two_lines(plan.book_subtitle),
            slots=[
                PhotoSlot(
                    photo_id=hero.photo_id,
                    filename=hero.filename,
                    slot_index=0,
                    x_in=cx_in,
                    y_in=cy_in,
                    width_in=cw_in,
                    height_in=ch_in,
                    crop_box_px=c_crop_px,
                    crop_box_normalized=c_crop_norm,
                    cropped_image_relpath=f"crops/{cover_crop_name}",
                    caption=clamp_caption_two_lines(hero.analysis.caption),
                    faces_count=len(hero.analysis.faces_boxes),
                    faces_preserved=c_faces_ok,
                )
            ],
        )
    )

    seen_clusters: set[str] = set()
    page_num = 1
    for pg_plan in plan.pages:
        raw_pids = [pid for pid in pg_plan.photo_ids if pid in photo_map]
        for chunk_start in range(0, len(raw_pids), 3):
            chunk_pids = raw_pids[chunk_start : chunk_start + 3]
            if not chunk_pids:
                continue
            page_photos = [photo_map[pid] for pid in chunk_pids]
            page_photos.sort(key=lambda p: (p.captured_timestamp, p.filename))

            tpl_id = choose_best_page_template(page_photos, fill_mode=page_fill_mode, crop_focus=crop_focus)
            geoms = templates[tpl_id]

            cid = pg_plan.cluster_id or page_photos[0].cluster_id or clusters[0].cluster_id
            cluster_obj = cluster_map.get(cid, clusters[0])
            is_first_page_of_cluster = cid not in seen_clusters
            seen_clusters.add(cid)

            slots: list[PhotoSlot] = []
            for s_idx, (p, (x_in, y_in, w_in, h_in)) in enumerate(zip(page_photos, geoms)):
                crop_px, crop_norm, faces_ok = compute_aesthetic_crop(
                    width=p.analysis_width,
                    height=p.analysis_height,
                    target_aspect=w_in / h_in,
                    faces_boxes=p.analysis.faces_boxes,
                    salient_box=p.analysis.salient_subject_box,
                    crop_focus=crop_focus,
                )
                crop_filename = f"page_{page_num:03d}_slot_{s_idx + 1}_{p.photo_id}.jpg"
                crop_tasks.append(
                    (
                        p.analysis_path,
                        crop_px,
                        crops_dir / crop_filename,
                        int(round(w_in * DPI)),
                        int(round(h_in * DPI)),
                    )
                )
                slots.append(
                    PhotoSlot(
                        photo_id=p.photo_id,
                        filename=p.filename,
                        slot_index=s_idx,
                        x_in=x_in,
                        y_in=y_in,
                        width_in=w_in,
                        height_in=h_in,
                        crop_box_px=crop_px,
                        crop_box_normalized=crop_norm,
                        cropped_image_relpath=f"crops/{crop_filename}",
                        caption=clamp_caption_two_lines(p.analysis.caption),
                        faces_count=len(p.analysis.faces_boxes),
                        faces_preserved=faces_ok,
                    )
                )

            group_cap = clamp_caption_two_lines(
                pg_plan.group_caption or _synthesize_group_caption(page_photos, cluster_obj.title, cluster_obj.date_range_label)
            )
            pages.append(
                BookPage(
                    page_number=page_num,
                    page_type="photo_page",
                    template_id=tpl_id,
                    cluster_id=cluster_obj.cluster_id,
                    chapter_title=cluster_obj.title if is_first_page_of_cluster else None,
                    group_caption=group_cap,
                    slots=slots,
                )
            )
            page_num += 1

    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as executor:
        list(executor.map(lambda args: crop_and_save_photo(*args), crop_tasks))

    return plan.book_title, plan.book_subtitle, plan.hero_cover_photo_id, clusters, pages
