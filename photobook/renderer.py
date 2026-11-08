"""Renderers for the 10x10 inch PDF Photo Book, standalone HTML Photo Book,
and JSON + Markdown Photo Analysis output files."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from photobook.models import BookPage, PhotoBookResult, PhotoMetadata


def _wrap_text_lines(c: canvas.Canvas, text: str, font_name: str, font_size: float, max_width_pt: float, max_lines: int = 2) -> list[str]:
    words = (text or "").strip().split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        candidate = f"{current} {w}"
        if c.stringWidth(candidate, font_name, font_size) <= max_width_pt:
            current = candidate
        else:
            lines.append(current)
            current = w
            if len(lines) >= max_lines - 1:
                break
    remaining_idx = sum(len(ln.split()) for ln in lines)
    tail_words = words[remaining_idx:]
    tail = " ".join(tail_words)
    while tail and c.stringWidth(tail, font_name, font_size) > max_width_pt and len(tail) > 4:
        tail = tail[:-2].rstrip() + "…"
    if tail:
        lines.append(tail)
    return lines[:max_lines]


def render_pdf_photobook(result: PhotoBookResult, out_pdf_path: str | Path) -> Path:
    """Render a print-ready 10x10 inch (720pt x 720pt) PDF photo book."""
    pdf_path = Path(out_pdf_path).expanduser().resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(result.output_dir).expanduser().resolve()

    PAGE_W = 10.0 * inch  # 720 pt
    PAGE_H = 10.0 * inch  # 720 pt

    c = canvas.Canvas(str(pdf_path), pagesize=(PAGE_W, PAGE_H))
    c.setTitle(f"{result.title} — 10x10 Photo Book")
    c.setAuthor("Gemini 3.8 Flash Photo Book Builder")

    for page in result.pages:
        if page.page_type == "cover":
            # Dark charcoal luxury cover background
            c.setFillColor(HexColor("#17191E"))
            c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

            # Subtle gold inner border frame (0.32in inset = 3.2%)
            c.setStrokeColor(HexColor("#C5A46D"))
            c.setLineWidth(1.0)
            c.rect(0.32 * inch, 0.32 * inch, 9.36 * inch, 9.36 * inch, stroke=1, fill=0)

            # Eyebrow header (well inside the 0.32in top border)
            c.setFillColor(HexColor("#C5A46D"))
            c.setFont("Helvetica-Bold", 8.5)
            c.drawCentredString(PAGE_W / 2.0, PAGE_H - 0.68 * inch, "CURATED 10×10 INCH PHOTO BOOK")

            # Book Title (1-2 lines max, well above y_in=1.65)
            c.setFillColor(HexColor("#FAF7F2"))
            c.setFont("Times-Bold", 17)
            title_lines = _wrap_text_lines(c, page.chapter_title or result.title, "Times-Bold", 17, 8.2 * inch, max_lines=2)
            cur_title_y = PAGE_H - 0.94 * inch
            for t_ln in title_lines:
                c.drawCentredString(PAGE_W / 2.0, cur_title_y, t_ln)
                cur_title_y -= 0.23 * inch

            # Hero image slot
            if page.slots:
                slot = page.slots[0]
                img_file = out_dir / slot.cropped_image_relpath
                x_pt = slot.x_in * inch
                y_pt = PAGE_H - (slot.y_in + slot.height_in) * inch
                w_pt = slot.width_in * inch
                h_pt = slot.height_in * inch

                # White mat frame around hero image
                c.setFillColor(HexColor("#FAF7F2"))
                c.rect(x_pt - 5, y_pt - 5, w_pt + 10, h_pt + 10, stroke=0, fill=1)
                if img_file.exists():
                    c.drawImage(str(img_file), x_pt, y_pt, width=w_pt, height=h_pt, preserveAspectRatio=False)

                # Hero caption & book subtitle below hero image
                c.setFillColor(HexColor("#E8E2D5"))
                c.setFont("Times-Italic", 11)
                cap_lines = _wrap_text_lines(c, slot.caption, "Times-Italic", 11, 7.8 * inch, max_lines=2)
                cur_y = y_pt - 0.32 * inch
                for ln in cap_lines:
                    c.drawCentredString(PAGE_W / 2.0, cur_y, ln)
                    cur_y -= 0.19 * inch

            # Subtitle at bottom of cover (well inside the 0.32in bottom border)
            c.setFillColor(HexColor("#C5A46D"))
            c.setFont("Helvetica", 9.5)
            c.drawCentredString(PAGE_W / 2.0, 0.58 * inch, page.group_caption or result.subtitle)
            c.showPage()
            continue

        # Interior 10x10 page: warm gallery cream background
        c.setFillColor(HexColor("#FAF7F2"))
        c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

        # Subtle outer trim border (0.32in inset)
        c.setStrokeColor(HexColor("#E5DEC9"))
        c.setLineWidth(0.6)
        c.rect(0.32 * inch, 0.32 * inch, 9.36 * inch, 9.36 * inch, stroke=1, fill=0)

        # Chapter / Cluster Header (1-2 lines max, always finishes above 1.38in; slots start >= 1.60in)
        if page.chapter_title:
            c.setFillColor(HexColor("#8C6D3F"))
            c.setFont("Helvetica-Bold", 8.0)
            c.drawString(0.75 * inch, PAGE_H - 0.62 * inch, "CHAPTER")
            c.setFillColor(HexColor("#1E232A"))
            c.setFont("Times-Bold", 13.0)
            ch_lines = _wrap_text_lines(c, page.chapter_title, "Times-Bold", 13.0, 8.5 * inch, max_lines=2)
            ch_y = PAGE_H - 0.84 * inch
            for ch_ln in ch_lines:
                c.drawString(0.75 * inch, ch_y, ch_ln)
                ch_y -= 0.20 * inch
            c.setStrokeColor(HexColor("#D4C7B0"))
            c.setLineWidth(0.75)
            div_y = ch_y + 0.11 * inch
            c.line(0.75 * inch, div_y, PAGE_W - 0.75 * inch, div_y)

        # Draw each photo slot (1, 2, or 3 max)
        for slot in page.slots:
            img_file = out_dir / slot.cropped_image_relpath
            x_pt = slot.x_in * inch
            y_pt = PAGE_H - (slot.y_in + slot.height_in) * inch
            w_pt = slot.width_in * inch
            h_pt = slot.height_in * inch

            # Subtle white photo mat + border
            c.setFillColor(HexColor("#FFFFFF"))
            c.setStrokeColor(HexColor("#DDD5C7"))
            c.setLineWidth(0.7)
            c.rect(x_pt - 3, y_pt - 3, w_pt + 6, h_pt + 6, stroke=1, fill=1)

            if img_file.exists():
                c.drawImage(str(img_file), x_pt, y_pt, width=w_pt, height=h_pt, preserveAspectRatio=False)

            # Individual slot caption (up to 2 lines; fits cleanly inside the 0.75in = 54pt gap)
            if len(page.slots) > 1 and slot.caption:
                c.setFillColor(HexColor("#4A4F55"))
                c.setFont("Times-Italic", 8.5)
                lines = _wrap_text_lines(c, slot.caption, "Times-Italic", 8.5, w_pt - 4, max_lines=2)
                sub_y = y_pt - 13
                for sub_ln in lines:
                    c.drawString(x_pt + 2, sub_y, sub_ln)
                    sub_y -= 10.5

        # Page / Group Caption (1-2 lines max) at bottom of 10x10 page
        footer_caption = page.group_caption or (page.slots[0].caption if page.slots else "")
        if footer_caption:
            c.setFillColor(HexColor("#22262C"))
            c.setFont("Times-Roman", 11)
            cap_lines = _wrap_text_lines(c, footer_caption, "Times-Roman", 11, 8.2 * inch, max_lines=2)
            base_y = 0.92 * inch if len(cap_lines) > 1 else 0.78 * inch
            for ln in cap_lines:
                c.drawCentredString(PAGE_W / 2.0, base_y, ln)
                base_y -= 0.20 * inch

        # Page number in bottom-right corner
        c.setFillColor(HexColor("#8A8478"))
        c.setFont("Helvetica", 8.5)
        c.drawRightString(PAGE_W - 0.65 * inch, 0.52 * inch, f"Page {page.page_number}")

        c.showPage()

    c.save()
    return pdf_path


def save_analysis_files(result: PhotoBookResult) -> tuple[Path, Path]:
    """Save the photo analysis details (including captions, scores, face boxes, and exclusions)
    to both JSON and Markdown output files."""
    json_path = Path(result.analysis_json_path).expanduser().resolve()
    md_path = Path(result.analysis_markdown_path).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # Build human-readable Markdown report
    lines = [
        f"# Photo Book Analysis Report — {result.title}",
        "",
        f"- **Input Folder:** `{result.folder_path}`",
        f"- **Generated At:** `{result.created_at}`",
        f"- **Total Build Time:** `{result.build_duration_seconds}s` ({result.stage_timings})",
        f"- **Stage Models Used:** Vision=`{result.model_used}` (batch={result.batch_size}, Thinking=`{result.thinking_level}`, Resolution=`{result.media_resolution}` / `{result.analysis_pixel_range}`), Curation=`{result.curation_model_used}`, Hero=`{result.hero_model_used}`, Layout=`{result.layout_model_used}`",
    ]
    if result.model_note:
        lines.append(f"- **Model Note:** {result.model_note}")
    lines += [
        f"- **Book Dimensions:** `10.0 × 10.0 inches` (Page Fill: `{result.page_fill_mode}`, Crop Focus: `{result.crop_focus}`)",
        f"- **Hero Cover Photo:** `{result.hero_cover_photo_id}`",
        f"- **Summary Counts:** {result.total_scanned} scanned → **{result.total_selected} selected**, "
        f"{result.total_excluded_duplicates} duplicates removed, {result.total_excluded_similar} similar burst shots removed, "
        f"{result.total_excluded_documents} documents removed, {result.total_excluded_low_quality} low-quality removed",
        "",
        "---",
        "",
        "## 1. Chronological Clusters & Group Captions",
        "",
    ]
    for cl in result.clusters:
        lines.append(f"### {cl.title} (`{cl.date_range_label}`)")
        lines.append(f"- **Group Caption:** {cl.group_caption}")
        lines.append(f"- **Photos ({len(cl.photo_ids)}):** {', '.join(f'`{pid}`' for pid in cl.photo_ids)}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 2. 10×10 Inch Page Layout Plan (1 to 3 Photos Max per Page)",
        "",
        "| Page | Type | Template | Chapter / Cluster | Photos on Page | Face-Safe Crop Verified | Page / Group Caption |",
        "|---|---|---|---|---|---|---|",
    ]
    for pg in result.pages:
        slot_summary = ", ".join(f"`{s.filename}`" for s in pg.slots)
        faces_verified = "Yes (100%)" if all(s.faces_preserved for s in pg.slots) else "Adjusted"
        lines.append(
            f"| {pg.page_number} | `{pg.page_type}` | `{pg.template_id}` | {pg.chapter_title or '—'} | "
            f"{slot_summary} | {faces_verified} | {pg.group_caption or ''} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Selected Photos (Chronological Order)",
        "",
        "| Photo ID | Filename | Captured At | Analysis Px | Score | Faces | Landmark / Theme | Individual Caption (1–2 Lines) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in result.photos:
        if p.status != "selected":
            continue
        lines.append(
            f"| `{p.photo_id}` | `{p.filename}` | `{p.captured_at}` | `{p.analysis_width}×{p.analysis_height}` | "
            f"**{p.analysis.composite_score}** | {len(p.analysis.faces_boxes)} | "
            f"{p.analysis.location_or_landmark or p.analysis.event_theme} | {p.analysis.caption} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Excluded Photos (Duplicates, Similar Shots & Documents)",
        "",
        "| Photo ID | Filename | Status | Reason |",
        "|---|---|---|---|",
    ]
    excluded = [p for p in result.photos if p.status != "selected"]
    if not excluded:
        lines.append("| — | — | — | No photos were excluded |")
    else:
        for p in excluded:
            lines.append(f"| `{p.photo_id}` | `{p.filename}` | `{p.status}` | {p.exclusion_reason or ''} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def render_html_photobook(result: PhotoBookResult, out_html_path: str | Path) -> Path:
    """Render a standalone, print-ready 10x10 inch HTML photo book with embedded base64 cropped images."""
    html_path = Path(out_html_path).expanduser().resolve()
    html_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(result.output_dir).expanduser().resolve()

    def _img_data_uri(relpath: str) -> str:
        p = out_dir / relpath
        if not p.exists():
            return ""
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    page_cards_html: list[str] = []
    for pg in result.pages:
        if pg.page_type == "cover":
            slot = pg.slots[0]
            uri = _img_data_uri(slot.cropped_image_relpath)
            left_pct = slot.x_in * 10.0
            top_pct = slot.y_in * 10.0
            width_pct = slot.width_in * 10.0
            height_pct = slot.height_in * 10.0
            page_cards_html.append(
                f"""
                <section class="book-page cover-page" data-page="{pg.page_number}">
                  <div class="cover-border"></div>
                  <div class="cover-header">
                    <div class="cover-eyebrow">CURATED 10×10 INCH PHOTO BOOK</div>
                    <h1 class="cover-title">{html.escape(pg.chapter_title or result.title)}</h1>
                  </div>
                  <div class="photo-slot cover-slot" style="left:{left_pct:.2f}%;top:{top_pct:.2f}%;width:{width_pct:.2f}%;height:{height_pct:.2f}%;">
                    <img src="{uri}" alt="{html.escape(slot.filename)}" />
                  </div>
                  <div class="cover-footer">
                    <p class="cover-hero-caption">{html.escape(slot.caption)}</p>
                    <div class="cover-subtitle">{html.escape(pg.group_caption or result.subtitle)}</div>
                  </div>
                </section>
                """
            )
            continue

        slots_html: list[str] = []
        for slot in pg.slots:
            uri = _img_data_uri(slot.cropped_image_relpath)
            left_pct = slot.x_in * 10.0
            top_pct = slot.y_in * 10.0
            width_pct = slot.width_in * 10.0
            height_pct = slot.height_in * 10.0
            subcap = (
                f'<div class="slot-subcaption">{html.escape(slot.caption)}</div>'
                if len(pg.slots) > 1 and slot.caption
                else ""
            )
            slots_html.append(
                f"""
                <div class="photo-slot" style="left:{left_pct:.2f}%;top:{top_pct:.2f}%;width:{width_pct:.2f}%;height:{height_pct:.2f}%;">
                  <img src="{uri}" alt="{html.escape(slot.filename)}" />
                  {subcap}
                </div>
                """
            )

        chapter_html = (
            f"""
            <div class="page-chapter-header">
              <span class="chapter-tag">CHAPTER</span>
              <h2>{html.escape(pg.chapter_title)}</h2>
            </div>
            """
            if pg.chapter_title
            else ""
        )
        footer_cap = pg.group_caption or (pg.slots[0].caption if pg.slots else "")
        page_cards_html.append(
            f"""
            <section class="book-page interior-page" data-page="{pg.page_number}">
              <div class="interior-border"></div>
              {chapter_html}
              {''.join(slots_html)}
              <div class="page-footer">
                <p class="page-group-caption">{html.escape(footer_cap)}</p>
                <span class="page-num">Page {pg.page_number}</span>
              </div>
            </section>
            """
        )

    cover_html = page_cards_html[0] if page_cards_html else ""
    interior_cards = page_cards_html[1:]
    spreads_html: list[str] = []
    if cover_html:
        spreads_html.append(
            f"""
            <div class="book-spread">
              <section class="book-page endpaper-page">
                <div class="interior-border"></div>
                <div class="endpaper-content">
                  <div class="endpaper-eyebrow">10×10 INCH HARDCOVER EDITION</div>
                  <div class="endpaper-title">{html.escape(result.title)}</div>
                  <div class="endpaper-sub">{html.escape(result.subtitle)}</div>
                </div>
              </section>
              {cover_html}
              <div class="spine-crease"></div>
            </div>
            """
        )
    for i in range(0, len(interior_cards), 2):
        left_leaf = interior_cards[i]
        right_leaf = (
            interior_cards[i + 1]
            if i + 1 < len(interior_cards)
            else """
            <section class="book-page endpaper-page">
              <div class="interior-border"></div>
              <div class="endpaper-content">
                <div class="endpaper-sub">— Finis —</div>
              </div>
            </section>
            """
        )
        spreads_html.append(
            f"""
            <div class="book-spread">
              {left_leaf}
              {right_leaf}
              <div class="spine-crease"></div>
            </div>
            """
        )

    doc_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{html.escape(result.title)} — 10×10 Inch Photo Book</title>
<style>
  @page {{ size: 10in 10in; margin: 0; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 32px 16px 64px;
    background: #111318;
    color: #f4f1ea;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .top-banner {{
    max-width: 1180px;
    margin: 0 auto 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    background: #1c2029;
    border: 1px solid #2e3442;
    padding: 14px 20px;
    border-radius: 10px;
  }}
  .top-banner h2 {{ margin: 0 0 4px 0; font-size: 16px; color: #faf7f2; }}
  .top-banner p {{ margin: 0; font-size: 12.5px; color: #a8b0c0; }}
  .viewer-toolbar {{
    max-width: 1180px;
    margin: 0 auto 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    background: #171a22;
    border: 1px solid #2e3442;
    padding: 10px 18px;
    border-radius: 10px;
    position: relative;
    overflow: hidden;
  }}
  .toolbar-btns {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }}
  .nav-btn {{
    background: #222734;
    color: #f4f1ea;
    border: 1px solid #384052;
    padding: 7px 14px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
  }}
  .nav-btn:hover {{ border-color: #c5a46d; color: #c5a46d; }}
  .btn-autoplay {{
    background: rgba(197, 164, 109, 0.16);
    color: #c5a46d;
    border: 1px solid #c5a46d;
    padding: 7px 14px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 700;
    cursor: pointer;
  }}
  .btn-autoplay.playing {{
    background: #c5a46d;
    color: #14161b;
    box-shadow: 0 0 14px rgba(197, 164, 109, 0.45);
  }}
  .speed-select {{
    background: #222734;
    color: #f4f1ea;
    border: 1px solid #384052;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
  }}
  .print-btn {{
    background: #c5a46d;
    color: #14161b;
    border: none;
    padding: 9px 16px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
  }}
  .autoplay-progress {{
    position: absolute;
    left: 0;
    bottom: 0;
    height: 3px;
    width: 0%;
    background: linear-gradient(90deg, #c5a46d, #f3dc99);
    transition: width 0.1s linear;
  }}
  .stage-3d-wrap {{
    max-width: 1180px;
    margin: 0 auto;
    perspective: 2200px;
    padding: 28px 16px 48px;
    display: flex;
    justify-content: center;
    align-items: center;
  }}
  .stage-3d-wrap .book-spread {{
    transform-style: preserve-3d;
    transform: rotateX(11deg) rotateY(0deg);
    transition: transform 0.25s ease-out;
    box-shadow:
      0 42px 80px rgba(0, 0, 0, 0.88),
      0 14px 0 -2px #ded6c6,
      0 18px 0 -3px #c9bfa9,
      0 23px 0 -4px #181512,
      0 28px 30px rgba(0, 0, 0, 0.75);
  }}
  .turning-leaf-3d {{
    position: absolute;
    top: 8px;
    bottom: 8px;
    width: calc(50% - 8px);
    transform-style: preserve-3d;
    z-index: 25;
    pointer-events: none;
  }}
  .turning-leaf-3d.turn-next {{
    right: 8px;
    transform-origin: left center;
    animation: flipNext3D 0.78s cubic-bezier(0.645, 0.045, 0.355, 1.000) forwards;
  }}
  .turning-leaf-3d.turn-prev {{
    left: 8px;
    transform-origin: right center;
    animation: flipPrev3D 0.78s cubic-bezier(0.645, 0.045, 0.355, 1.000) forwards;
  }}
  .turning-face {{
    position: absolute;
    inset: 0;
    backface-visibility: hidden;
    -webkit-backface-visibility: hidden;
    overflow: hidden;
    border-left: 1px solid #dfd7c6;
    border-right: 1px solid #dfd7c6;
  }}
  .turning-face.face-back {{
    transform: rotateY(180deg);
  }}
  @keyframes flipNext3D {{
    0% {{ transform: rotateY(0deg); box-shadow: 0 0 0 rgba(0,0,0,0); }}
    50% {{ transform: rotateY(-90deg); box-shadow: -22px 14px 36px rgba(0,0,0,0.45); }}
    100% {{ transform: rotateY(-180deg); box-shadow: 0 0 0 rgba(0,0,0,0); }}
  }}
  @keyframes flipPrev3D {{
    0% {{ transform: rotateY(0deg); box-shadow: 0 0 0 rgba(0,0,0,0); }}
    50% {{ transform: rotateY(90deg); box-shadow: 22px 14px 36px rgba(0,0,0,0.45); }}
    100% {{ transform: rotateY(180deg); box-shadow: 0 0 0 rgba(0,0,0,0); }}
  }}
  .pages-stack {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 42px;
  }}
  .book-spread {{
    position: relative;
    display: grid;
    grid-template-columns: 1fr 1fr;
    width: min(95vw, 1180px);
    aspect-ratio: 2 / 1;
    background: #1a1714;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid #3c352c;
    box-shadow: 0 26px 60px rgba(0, 0, 0, 0.72);
  }}
  /* Hairline 1px micro-fold seam for 180-degree lay-flat binding (no gutter shadow) */
  .spine-crease {{
    position: absolute;
    top: 8px;
    bottom: 8px;
    left: 50%;
    width: 1px;
    transform: translateX(-50%);
    background: rgba(175, 163, 142, 0.42);
    box-shadow: 1px 0 0 rgba(255, 255, 255, 0.45);
    pointer-events: none;
    z-index: 15;
  }}
  /* Continuous trim border across 20x10 lay-flat interior spreads */
  .book-spread > .interior-page:first-of-type .interior-border {{
    right: 0;
    border-right: none;
  }}
  .book-spread > .interior-page:nth-of-type(2) .interior-border {{
    left: 0;
    border-left: none;
  }}
  .book-page {{
    position: relative;
    width: 100%;
    height: 100%;
    aspect-ratio: 1 / 1;
    overflow: hidden;
    page-break-after: always;
  }}
  .endpaper-page {{
    background: #f2ece1;
    color: #7c7262;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    font-family: Georgia, "Times New Roman", serif;
  }}
  .endpaper-eyebrow {{
    font-size: 10px;
    letter-spacing: 0.16em;
    color: #9b8e78;
    margin-bottom: 6px;
  }}
  .endpaper-title {{
    font-size: 20px;
    font-weight: 700;
    color: #4a4338;
  }}
  .endpaper-sub {{
    font-style: italic;
    font-size: 13px;
    color: #827766;
    margin-top: 6px;
  }}
  .cover-page {{
    background: #17191e;
    color: #faf7f2;
  }}
  .cover-border {{
    position: absolute;
    inset: 3.2%;
    border: 1px solid #c5a46d;
    pointer-events: none;
  }}
  .cover-header {{
    position: absolute;
    top: 5.2%;
    left: 8%;
    right: 8%;
    text-align: center;
    z-index: 5;
  }}
  .cover-eyebrow {{
    font-size: 8.5px;
    letter-spacing: 0.16em;
    color: #c5a46d;
    font-weight: 700;
    margin-bottom: 2px;
  }}
  .cover-title {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 16px;
    line-height: 1.2;
    margin: 0;
    font-weight: 700;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .cover-footer {{
    position: absolute;
    bottom: 5.2%;
    left: 10%;
    right: 10%;
    text-align: center;
    z-index: 5;
  }}
  .cover-hero-caption {{
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic;
    font-size: 11px;
    line-height: 1.25;
    color: #e8e2d5;
    margin: 0 0 5px 0;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .cover-subtitle {{
    font-size: 10px;
    letter-spacing: 0.06em;
    color: #c5a46d;
  }}
  .interior-page {{
    background: #faf7f2;
    color: #1e232a;
  }}
  .interior-border {{
    position: absolute;
    inset: 3.2%;
    border: 1px solid #e5dec9;
    pointer-events: none;
  }}
  .page-chapter-header {{
    position: absolute;
    top: 4.8%;
    left: 7.5%;
    right: 7.5%;
    border-bottom: 1px solid #d4c7b0;
    padding-bottom: 3px;
    z-index: 5;
  }}
  .chapter-tag {{
    font-size: 8.5px;
    font-weight: 700;
    letter-spacing: 0.14em;
    line-height: 1.1;
    color: #8c6d3f;
    display: block;
  }}
  .page-chapter-header h2 {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 13px;
    line-height: 1.2;
    margin: 2px 0 0 0;
    color: #1e232a;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .page-running-head {{
    position: absolute;
    top: 4.5%;
    left: 7.5%;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic;
    font-size: 10.5px;
    color: #8a8478;
    z-index: 5;
  }}
  .photo-slot {{
    position: absolute;
    background: #fff;
    padding: 4px;
    border: 1px solid #ddd5c7;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    z-index: 2;
  }}
  .cover-slot {{
    background: #faf7f2;
    padding: 6px;
    border: none;
  }}
  .photo-slot img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }}
  .slot-subcaption {{
    position: absolute;
    left: 2px;
    right: 2px;
    top: calc(100% + 4px);
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic;
    font-size: 9.5px;
    line-height: 1.18;
    color: #4a4f55;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    z-index: 6;
    pointer-events: none;
  }}
  .page-footer {{
    position: absolute;
    bottom: 4.8%;
    left: 8%;
    right: 8%;
    text-align: center;
    z-index: 5;
  }}
  .page-group-caption {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 11px;
    line-height: 1.3;
    color: #22262c;
    margin: 0;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .page-num {{
    position: absolute;
    right: 0;
    bottom: -4px;
    font-size: 10px;
    color: #8a8478;
  }}
  @media print {{
    body {{ background: none; padding: 0; }}
    .top-banner, .viewer-toolbar, .stage-3d-wrap, .spine-crease, .endpaper-page {{ display: none !important; }}
    .pages-stack {{ display: flex !important; gap: 0; }}
    .book-spread {{ display: block; width: auto; aspect-ratio: auto; background: none; padding: 0; border: none; box-shadow: none; }}
    .book-page {{ width: 10in; height: 10in; box-shadow: none; }}
  }}
</style>
</head>
<body>
  <div class="top-banner">
    <div>
      <h2>{html.escape(result.title)} (10×10 Inch Lay-Flat Open-Spread Photo Book)</h2>
      <p>{result.total_selected} curated photos across {len(result.pages)} pages • 180° Lay-Flat 2-Page Spreads • Model: {html.escape(result.model_used)}</p>
    </div>
    <button class="print-btn" onclick="window.print()">Print / Save 10×10 PDF</button>
  </div>
  <div class="viewer-toolbar">
    <div class="toolbar-btns">
      <button type="button" class="nav-btn" id="prevBtn">← Prev Spread</button>
      <span id="spreadIndicator" style="font-size:13px;font-weight:700;color:#faf7f2;">Spread 1</span>
      <button type="button" class="nav-btn" id="nextBtn">Next Spread →</button>
      <button type="button" class="btn-autoplay" id="autoplayBtn">▶ Autoplay</button>
      <select class="speed-select" id="speedSelect">
        <option value="2500">2.5s / spread (Fast)</option>
        <option value="4000" selected>4.0s / spread (Normal)</option>
        <option value="6000">6.0s / spread (Relaxed)</option>
        <option value="8500">8.5s / spread (Slow)</option>
      </select>
    </div>
    <div class="toolbar-btns">
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;color:#c5a46d;font-weight:700;font-size:13px;">
        <input type="checkbox" id="mode3dToggle" checked />
        ✨ 3D Lay-Flat Flip Viewer
      </label>
    </div>
    <div class="autoplay-progress" id="autoplayProgress"></div>
  </div>
  <div class="stage-3d-wrap" id="stage3dWrap"></div>
  <div class="pages-stack" id="pagesStack" style="display:none;">
    {''.join(spreads_html)}
  </div>
  <script>
    (function() {{
      const stackSpreads = Array.from(document.querySelectorAll('#pagesStack .book-spread'));
      const stageWrap = document.getElementById('stage3dWrap');
      const pagesStack = document.getElementById('pagesStack');
      const mode3dToggle = document.getElementById('mode3dToggle');
      const prevBtn = document.getElementById('prevBtn');
      const nextBtn = document.getElementById('nextBtn');
      const autoplayBtn = document.getElementById('autoplayBtn');
      const speedSelect = document.getElementById('speedSelect');
      const spreadIndicator = document.getElementById('spreadIndicator');
      const autoplayProgress = document.getElementById('autoplayProgress');

      let curIdx = 0;
      let isFlipping = false;
      let autoTimer = null;
      let progTimer = null;
      let startTs = 0;

      function getSpreadPages(idx) {{
        const sp = stackSpreads[idx];
        const pages = sp ? sp.querySelectorAll('.book-page') : [];
        return {{
          leftHtml: pages[0] ? pages[0].outerHTML : '',
          rightHtml: pages[1] ? pages[1].outerHTML : ''
        }};
      }}

      function renderStage() {{
        if (!stackSpreads.length) return;
        spreadIndicator.textContent = `Spread ${{curIdx + 1}} of ${{stackSpreads.length}}`;
        if (!mode3dToggle.checked) {{
          stageWrap.style.display = 'none';
          pagesStack.style.display = 'flex';
          return;
        }}
        stageWrap.style.display = 'flex';
        pagesStack.style.display = 'none';
        const {{ leftHtml, rightHtml }} = getSpreadPages(curIdx);
        stageWrap.innerHTML = `
          <div class="book-spread" id="activeSpread3D">
            ${{leftHtml}}
            ${{rightHtml}}
            <div class="spine-crease"></div>
          </div>
        `;
      }}

      function flipTo(targetIdx, dir) {{
        if (!stackSpreads.length) return;
        const nextIdx = (targetIdx + stackSpreads.length) % stackSpreads.length;
        if (nextIdx === curIdx) return;
        if (!mode3dToggle.checked || isFlipping) {{
          curIdx = nextIdx;
          renderStage();
          if (!mode3dToggle.checked && stackSpreads[curIdx]) {{
            stackSpreads[curIdx].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
          }}
          return;
        }}
        isFlipping = true;
        const curP = getSpreadPages(curIdx);
        const nxtP = getSpreadPages(nextIdx);
        const spreadEl = document.getElementById('activeSpread3D');
        if (!spreadEl) {{
          curIdx = nextIdx;
          isFlipping = false;
          renderStage();
          return;
        }}
        if (dir === 'next') {{
          spreadEl.innerHTML = `
            ${{curP.leftHtml}}
            ${{nxtP.rightHtml}}
            <div class="turning-leaf-3d turn-next">
              <div class="turning-face face-front">${{curP.rightHtml}}</div>
              <div class="turning-face face-back">${{nxtP.leftHtml}}</div>
            </div>
            <div class="spine-crease"></div>
          `;
        }} else {{
          spreadEl.innerHTML = `
            ${{nxtP.leftHtml}}
            ${{curP.rightHtml}}
            <div class="turning-leaf-3d turn-prev">
              <div class="turning-face face-front">${{curP.leftHtml}}</div>
              <div class="turning-face face-back">${{nxtP.rightHtml}}</div>
            </div>
            <div class="spine-crease"></div>
          `;
        }}
        setTimeout(() => {{
          curIdx = nextIdx;
          isFlipping = false;
          renderStage();
        }}, 780);
      }}

      function stopAutoplay() {{
        if (autoTimer) clearInterval(autoTimer);
        if (progTimer) clearInterval(progTimer);
        autoTimer = null;
        progTimer = null;
        autoplayBtn.classList.remove('playing');
        autoplayBtn.textContent = '▶ Autoplay';
        autoplayProgress.style.width = '0%';
      }}

      function startAutoplay() {{
        stopAutoplay();
        const ms = parseInt(speedSelect.value, 10) || 4000;
        autoplayBtn.classList.add('playing');
        autoplayBtn.textContent = '⏸ Pause Autoplay';
        startTs = Date.now();
        progTimer = setInterval(() => {{
          const pct = Math.min(100, ((Date.now() - startTs) / ms) * 100);
          autoplayProgress.style.width = pct + '%';
        }}, 80);
        autoTimer = setInterval(() => {{
          startTs = Date.now();
          flipTo(curIdx + 1, 'next');
        }}, ms);
      }}

      prevBtn.addEventListener('click', () => flipTo(curIdx - 1, 'prev'));
      nextBtn.addEventListener('click', () => flipTo(curIdx + 1, 'next'));
      autoplayBtn.addEventListener('click', () => {{
        if (autoTimer) stopAutoplay();
        else startAutoplay();
      }});
      speedSelect.addEventListener('change', () => {{
        if (autoTimer) startAutoplay();
      }});
      mode3dToggle.addEventListener('change', () => renderStage());

      stageWrap.addEventListener('mousemove', (e) => {{
        if (!mode3dToggle.checked || isFlipping) return;
        const spreadEl = document.getElementById('activeSpread3D');
        if (!spreadEl) return;
        const r = stageWrap.getBoundingClientRect();
        const nx = ((e.clientX - r.left) / Math.max(1, r.width) - 0.5) * 2;
        const ny = ((e.clientY - r.top) / Math.max(1, r.height) - 0.5) * 2;
        spreadEl.style.transform = `rotateX(${{(11 - ny * 5.5).toFixed(2)}}deg) rotateY(${{(nx * 7.5).toFixed(2)}}deg)`;
      }});
      stageWrap.addEventListener('mouseleave', () => {{
        const spreadEl = document.getElementById('activeSpread3D');
        if (spreadEl) spreadEl.style.transform = 'rotateX(11deg) rotateY(0deg)';
      }});
      stageWrap.addEventListener('click', (e) => {{
        if (isFlipping) return;
        const r = stageWrap.getBoundingClientRect();
        if (e.clientX - r.left > r.width / 2) flipTo(curIdx + 1, 'next');
        else flipTo(curIdx - 1, 'prev');
      }});
      document.addEventListener('keydown', (e) => {{
        if (e.key === 'ArrowRight') flipTo(curIdx + 1, 'next');
        else if (e.key === 'ArrowLeft') flipTo(curIdx - 1, 'prev');
        else if (e.key === ' ') {{
          e.preventDefault();
          if (autoTimer) stopAutoplay();
          else startAutoplay();
        }}
      }});

      renderStage();
    }})();
  </script>
</body>
</html>
"""
    html_path.write_text(doc_html, encoding="utf-8")
    return html_path
