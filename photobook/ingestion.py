"""Photo folder ingestion, EXIF timestamp extraction, 1,120-2,240px resolution scaling,
HEIC/HEIF support, parallel processing, and perceptual hashing for duplicate/burst detection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pillow_heif
from PIL import ExifTags, Image, ImageOps

from photobook.models import PhotoMetadata

# Register HEIC/HEIF/AVIF support with Pillow globally
pillow_heif.register_heif_opener()

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
    ".avif",
    ".bmp",
    ".tiff",
    ".tif",
}
MIN_ANALYSIS_PX = 1120
MAX_ANALYSIS_PX = 2240

_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def _flat_pixels(img: Image.Image) -> list:
    getter = getattr(img, "get_flattened_data", None)
    return list(getter()) if getter is not None else list(img.getdata())


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a 64-bit difference hash (dHash) as a 16-char hex string."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = _flat_pixels(gray)
    bits = 0
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | (1 if pixels[row_offset + col] > pixels[row_offset + col + 1] else 0)
    return f"{bits:016x}"


def compute_ahash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a 64-bit average hash (aHash) as a 16-char hex string."""
    gray = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = _flat_pixels(gray)
    avg = sum(pixels) / len(pixels)
    bits = 0
    for px in pixels:
        bits = (bits << 1) | (1 if px >= avg else 0)
    return f"{bits:016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bitwise Hamming distance between two hex perceptual hashes."""
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def compute_color_signature(img: Image.Image, bins_per_channel: int = 8) -> list[float]:
    """Compute an L2-normalized RGB histogram vector for visual similarity checks."""
    rgb = img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    raw_hist = rgb.histogram()  # 768 values (256 R + 256 G + 256 B)
    step = 256 // bins_per_channel
    vec: list[float] = []
    for ch in range(3):
        ch_slice = raw_hist[ch * 256 : (ch + 1) * 256]
        for b in range(bins_per_channel):
            vec.append(float(sum(ch_slice[b * step : (b + 1) * step])))
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / norm, 5) for x in vec]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two L2-normalized histogram vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _parse_exif_datetime(raw: str) -> Optional[datetime]:
    cleaned = str(raw).strip().rstrip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(cleaned[:19], fmt)
        except ValueError:
            continue
    return None


def extract_exif_metadata(img: Image.Image, fallback_path: Path) -> tuple[datetime, Optional[str]]:
    """Extract capture timestamp and camera model from EXIF, falling back to file mtime."""
    captured_dt: Optional[datetime] = None
    camera_model: Optional[str] = None
    try:
        exif = img.getexif()
        if exif:
            model_val = exif.get(_EXIF_TAGS.get("Model", 272))
            if model_val:
                camera_model = str(model_val).strip().rstrip("\x00")
            for tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                tag_id = _EXIF_TAGS.get(tag_name)
                if tag_id and tag_id in exif:
                    captured_dt = _parse_exif_datetime(str(exif[tag_id]))
                    if captured_dt:
                        break
            if not captured_dt:
                exif_ifd = exif.get_ifd(0x8769)
                if exif_ifd:
                    for tag_name in ("DateTimeOriginal", "DateTimeDigitized"):
                        tag_id = _EXIF_TAGS.get(tag_name)
                        if tag_id and tag_id in exif_ifd:
                            captured_dt = _parse_exif_datetime(str(exif_ifd[tag_id]))
                            if captured_dt:
                                break
    except Exception:
        pass

    if not captured_dt:
        stat = fallback_path.stat()
        captured_dt = datetime.fromtimestamp(stat.st_mtime)

    return captured_dt, camera_model


def scale_to_analysis_resolution(
    img: Image.Image,
    min_px: int = MIN_ANALYSIS_PX,
    max_px: int = MAX_ANALYSIS_PX,
) -> Image.Image:
    """Ensure the image's longest side is within [1120, 2240] pixels for deep analysis."""
    w, h = img.size
    longest = max(w, h)
    if longest > max_px:
        scale = max_px / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    if longest < min_px:
        scale = min_px / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img.copy()


def _process_single_photo(
    idx: int,
    file_path: Path,
    cache_dir: Path,
    min_px: int,
    max_px: int,
) -> Optional[PhotoMetadata]:
    """Worker function to decode, orient, scale to 1,120-2,240px, and hash one image (with disk cache)."""
    try:
        stat = file_path.stat()
        cache_key = hashlib.sha1(
            f"{file_path}:{stat.st_size}:{int(stat.st_mtime)}:{min_px}:{max_px}".encode("utf-8")
        ).hexdigest()[:16]
        short_hash = hashlib.sha1(f"{file_path.name}-{idx}".encode("utf-8")).hexdigest()[:8]
        photo_id = f"photo_{idx:04d}_{short_hash}"

        cached_img_path = cache_dir / f"{cache_key}.jpg"
        cached_meta_path = cache_dir / f"{cache_key}.json"

        if cached_img_path.exists() and cached_meta_path.exists():
            try:
                meta = json.loads(cached_meta_path.read_text(encoding="utf-8"))
                return PhotoMetadata(
                    photo_id=photo_id,
                    filename=file_path.name,
                    file_path=str(file_path),
                    original_width=meta["original_width"],
                    original_height=meta["original_height"],
                    analysis_width=meta["analysis_width"],
                    analysis_height=meta["analysis_height"],
                    analysis_path=str(cached_img_path),
                    captured_at=meta["captured_at"],
                    captured_timestamp=meta["captured_timestamp"],
                    camera_model=meta.get("camera_model"),
                    dhash=meta["dhash"],
                    ahash=meta["ahash"],
                    color_signature=meta["color_signature"],
                )
            except Exception:
                pass

        with Image.open(file_path) as raw_img:
            captured_dt, camera_model = extract_exif_metadata(raw_img, file_path)
            raw_w, raw_h = raw_img.size
            # Check orientation tag (5,6,7,8 swap width/height)
            exif_orient = raw_img.getexif().get(274, 1) if hasattr(raw_img, "getexif") else 1
            orig_w, orig_h = (raw_h, raw_w) if exif_orient in (5, 6, 7, 8) else (raw_w, raw_h)

            # Hint decoder for faster downscaling on large HEIC/JPEG files
            try:
                raw_img.draft("RGB", (max_px, max_px))
            except Exception:
                pass

            img = ImageOps.exif_transpose(raw_img).convert("RGB")
            analysis_img = scale_to_analysis_resolution(img, min_px=min_px, max_px=max_px)
            ana_w, ana_h = analysis_img.size

            # Preserve embedded photobook_hint if present
            hint = raw_img.info.get("photobook_hint")
            if not cached_img_path.exists():
                analysis_img.save(cached_img_path, format="JPEG", quality=88)

            dhash = compute_dhash(analysis_img)
            ahash = compute_ahash(analysis_img)
            color_sig = compute_color_signature(analysis_img)

            meta_payload = {
                "original_width": orig_w,
                "original_height": orig_h,
                "analysis_width": ana_w,
                "analysis_height": ana_h,
                "captured_at": captured_dt.isoformat(timespec="seconds"),
                "captured_timestamp": captured_dt.timestamp(),
                "camera_model": camera_model,
                "dhash": dhash,
                "ahash": ahash,
                "color_signature": color_sig,
                "photobook_hint": hint,
            }
            cached_meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")

            return PhotoMetadata(
                photo_id=photo_id,
                filename=file_path.name,
                file_path=str(file_path),
                original_width=orig_w,
                original_height=orig_h,
                analysis_width=ana_w,
                analysis_height=ana_h,
                analysis_path=str(cached_img_path),
                captured_at=captured_dt.isoformat(timespec="seconds"),
                captured_timestamp=captured_dt.timestamp(),
                camera_model=camera_model,
                dhash=dhash,
                ahash=ahash,
                color_signature=color_sig,
            )
    except Exception:
        return None


def scan_and_prepare_photos(
    folder_path: str | Path,
    work_dir: str | Path,
    min_px: int = MIN_ANALYSIS_PX,
    max_px: int = MAX_ANALYSIS_PX,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> list[PhotoMetadata]:
    """Scan a folder (and subfolders) for photos (including .HEIC/.HEIF, .JPG, .PNG, .WEBP),
    normalize EXIF orientation, scale to 1,120-2,240px in parallel, and sort chronologically."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Photo folder not found: {folder}")

    # Shared cache directory across runs so re-builds on 1,000+ photos are near-instant
    work_path = Path(work_dir).expanduser().resolve()
    cache_dir = work_path.parent / ".photo_analysis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith(".")
        and ".photo_analysis_cache" not in p.parts
    )
    if not candidates:
        raise ValueError(
            f"No supported image files ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found in {folder}"
        )

    total = len(candidates)
    max_workers = min(8, (os.cpu_count() or 4))
    records: list[PhotoMetadata] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_photo, idx, fpath, cache_dir, min_px, max_px): fpath
            for idx, fpath in enumerate(candidates, start=1)
        }
        for fut in as_completed(futures):
            completed += 1
            rec = fut.result()
            if rec is not None:
                records.append(rec)
            if progress_cb and (completed % 10 == 0 or completed == total):
                progress_cb(
                    f"Scanned & scaled {completed}/{total} photos to 1,120–2,240px ({futures[fut].name})",
                    completed,
                    total,
                )

    if not records:
        raise ValueError(f"Could not read any valid photos from {folder}")

    # Sort chronologically by capture timestamp, then filename for deterministic ties
    records.sort(key=lambda r: (r.captured_timestamp, r.filename))
    # Reassign clean sequential photo_ids after chronological sort
    for idx, rec in enumerate(records, start=1):
        short_hash = hashlib.sha1(f"{rec.filename}-{idx}".encode("utf-8")).hexdigest()[:8]
        rec.photo_id = f"photo_{idx:04d}_{short_hash}"
    return records
