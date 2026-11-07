"""Face-safe aesthetic smart cropping engine.

Ensures every photo cropped into a 10x10 inch page slot:
1. Preserves 100% of all detected human faces/heads (with headroom padding so
   foreheads, chins, and ears are never cut off).
2. Centers aesthetically on the salient subject and rule-of-thirds focal point.
3. Signals when a narrow slot aspect ratio cannot contain wide multi-person faces
   so the page layout engine can automatically switch to a wider slot template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pillow_heif
from PIL import Image, ImageOps

from photobook.models import BoundingBox

pillow_heif.register_heif_opener()


def pad_face_box(
    box: BoundingBox,
    top_pad_ratio: float = 0.18,
    bottom_pad_ratio: float = 0.12,
    side_pad_ratio: float = 0.12,
) -> BoundingBox:
    """Expand a normalized [0..1000] face bounding box with extra headroom and side margin."""
    y0, y1 = min(box.ymin, box.ymax), max(box.ymin, box.ymax)
    x0, x1 = min(box.xmin, box.xmax), max(box.xmin, box.xmax)
    bh = max(10, y1 - y0)
    bw = max(10, x1 - x0)
    return BoundingBox(
        ymin=max(0, int(round(y0 - bh * top_pad_ratio))),
        xmin=max(0, int(round(x0 - bw * side_pad_ratio))),
        ymax=min(1000, int(round(y1 + bh * bottom_pad_ratio))),
        xmax=min(1000, int(round(x1 + bw * side_pad_ratio))),
        label=box.label,
    )


def union_boxes(boxes: list[BoundingBox]) -> Optional[BoundingBox]:
    """Return the normalized [0..1000] union bounding box of a non-empty list of boxes."""
    if not boxes:
        return None
    return BoundingBox(
        ymin=min(min(b.ymin, b.ymax) for b in boxes),
        xmin=min(min(b.xmin, b.xmax) for b in boxes),
        ymax=max(max(b.ymin, b.ymax) for b in boxes),
        xmax=max(max(b.xmin, b.xmax) for b in boxes),
        label="faces_union",
    )


def verify_faces_preserved(
    crop_box_px: tuple[int, int, int, int],
    width: int,
    height: int,
    faces_boxes: list[BoundingBox],
    tolerance_px: int = 2,
) -> bool:
    """Verify that every raw face box lies inside crop_box_px = (left, top, right, bottom)."""
    if not faces_boxes:
        return True
    c_left, c_top, c_right, c_bottom = crop_box_px
    for face in faces_boxes:
        f_left, f_top, f_right, f_bottom = face.to_pixels(width, height)
        if (
            f_left < c_left - tolerance_px
            or f_top < c_top - tolerance_px
            or f_right > c_right + tolerance_px
            or f_bottom > c_bottom + tolerance_px
        ):
            return False
    return True


def compute_aesthetic_crop(
    width: int,
    height: int,
    target_aspect: float,
    faces_boxes: Optional[list[BoundingBox]] = None,
    salient_box: Optional[BoundingBox] = None,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], bool]:
    """Compute a face-safe, aesthetically framed crop rectangle.

    Returns:
      - crop_box_px: (left, top, right, bottom) in pixel coordinates
      - crop_box_normalized: (ymin, xmin, ymax, xmax) in 0..1000 coordinates
      - faces_preserved: True if 100% of all faces are preserved inside the crop
    """
    faces_boxes = faces_boxes or []
    padded_faces = [pad_face_box(b) for b in faces_boxes]
    face_union = union_boxes(padded_faces)
    raw_face_union = union_boxes(faces_boxes)

    # Determine largest possible crop window of target_aspect inside (width, height)
    img_aspect = width / max(1.0, float(height))
    if img_aspect >= target_aspect:
        # Image is wider than slot -> full height, crop width
        crop_h = height
        crop_w = max(1, min(width, int(round(height * target_aspect))))
    else:
        # Image is taller than slot -> full width, crop height
        crop_w = width
        crop_h = max(1, min(height, int(round(width / target_aspect))))

    # Determine desired focal center (cx, cy) in pixels
    if face_union is not None:
        fl, ft, fr, fb = face_union.to_pixels(width, height)
        face_cx = 0.5 * (fl + fr)
        # Rule-of-thirds bias: place faces slightly in upper 42% of frame when cropping vertically
        face_cy = 0.5 * (ft + fb) + 0.08 * crop_h
        if salient_box is not None:
            sl, st, sr, sb = salient_box.to_pixels(width, height)
            sal_cx = 0.5 * (sl + sr)
            sal_cy = 0.5 * (st + sb)
            cx = 0.78 * face_cx + 0.22 * sal_cx
            cy = 0.78 * face_cy + 0.22 * sal_cy
        else:
            cx, cy = face_cx, face_cy
    elif salient_box is not None:
        sl, st, sr, sb = salient_box.to_pixels(width, height)
        cx = 0.5 * (sl + sr)
        cy = 0.5 * (st + sb)
    else:
        cx = width * 0.5
        cy = height * 0.5

    # Initial crop box centered on (cx, cy), clamped to image bounds
    left = int(round(cx - crop_w * 0.5))
    top = int(round(cy - crop_h * 0.5))
    left = max(0, min(width - crop_w, left))
    top = max(0, min(height - crop_h, top))
    right = left + crop_w
    bottom = top + crop_h

    # Enforce strict face protection: shift crop window so padded face_union (or raw_face_union) stays inside
    for guard_box in (face_union, raw_face_union):
        if guard_box is None:
            continue
        gl, gt, gr, gb = guard_box.to_pixels(width, height)
        gw = gr - gl
        gh = gb - gt

        if gw <= crop_w:
            if gl < left:
                left = gl
                right = left + crop_w
            elif gr > right:
                right = gr
                left = right - crop_w
        else:
            # Faces span wider than crop_w: center horizontally on the faces
            left = max(0, min(width - crop_w, int(round(0.5 * (gl + gr) - 0.5 * crop_w))))
            right = left + crop_w

        if gh <= crop_h:
            if gt < top:
                top = gt
                bottom = top + crop_h
            elif gb > bottom:
                bottom = gb
                top = bottom - crop_h
        else:
            # Faces span taller than crop_h: center vertically on the faces
            top = max(0, min(height - crop_h, int(round(0.5 * (gt + gb) - 0.5 * crop_h))))
            bottom = top + crop_h

    crop_box_px = (left, top, right, bottom)
    faces_ok = verify_faces_preserved(crop_box_px, width, height, faces_boxes)

    norm_ymin = max(0, min(1000, int(round(top / max(1.0, float(height)) * 1000))))
    norm_xmin = max(0, min(1000, int(round(left / max(1.0, float(width)) * 1000))))
    norm_ymax = max(0, min(1000, int(round(bottom / max(1.0, float(height)) * 1000))))
    norm_xmax = max(0, min(1000, int(round(right / max(1.0, float(width)) * 1000))))

    return crop_box_px, (norm_ymin, norm_xmin, norm_ymax, norm_xmax), faces_ok


def can_fit_aspect_without_cutting_faces(
    width: int,
    height: int,
    target_aspect: float,
    faces_boxes: list[BoundingBox],
) -> bool:
    """Return True if cropping (width, height) to target_aspect preserves all faces."""
    if not faces_boxes:
        return True
    _, _, faces_ok = compute_aesthetic_crop(
        width=width,
        height=height,
        target_aspect=target_aspect,
        faces_boxes=faces_boxes,
    )
    return faces_ok


def crop_and_save_photo(
    source_path: str | Path,
    crop_box_px: tuple[int, int, int, int],
    out_path: str | Path,
    target_width_px: int,
    target_height_px: int,
) -> tuple[int, int]:
    """Crop the original image using crop_box_px (after EXIF transpose) and resize to high-DPI slot dimensions."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
        w, h = img.size
        left, top, right, bottom = crop_box_px
        left = max(0, min(w - 1, left))
        top = max(0, min(h - 1, top))
        right = max(left + 1, min(w, right))
        bottom = max(top + 1, min(h, bottom))
        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize(
            (max(1, target_width_px), max(1, target_height_px)),
            Image.Resampling.LANCZOS,
        )
        resized.save(out_file, format="JPEG", quality=94)
        return resized.size
