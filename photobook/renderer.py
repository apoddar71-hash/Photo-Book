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

            # Subtle gold inner border frame
            c.setStrokeColor(HexColor("#C5A46D"))
            c.setLineWidth(1.0)
            c.rect(0.45 * inch, 0.45 * inch, 9.1 * inch, 9.1 * inch, stroke=1, fill=0)

            # Eyebrow header
            c.setFillColor(HexColor("#C5A46D"))
            c.setFont("Helvetica-Bold", 9.5)
            c.drawCentredString(PAGE_W / 2.0, PAGE_H - 0.82 * inch, "CURATED 10×10 INCH PHOTO BOOK")

            # Book Title
            c.setFillColor(HexColor("#FAF7F2"))
            c.setFont("Times-Bold", 24)
            title_lines = _wrap_text_lines(c, page.chapter_title or result.title, "Times-Bold", 24, 8.2 * inch, max_lines=1)
            if title_lines:
                c.drawCentredString(PAGE_W / 2.0, PAGE_H - 1.18 * inch, title_lines[0])

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
                c.setFont("Times-Italic", 12)
                cap_lines = _wrap_text_lines(c, slot.caption, "Times-Italic", 12, 7.8 * inch, max_lines=2)
                cur_y = y_pt - 0.38 * inch
                for ln in cap_lines:
                    c.drawCentredString(PAGE_W / 2.0, cur_y, ln)
                    cur_y -= 0.22 * inch

            # Subtitle at bottom of cover
            c.setFillColor(HexColor("#C5A46D"))
            c.setFont("Helvetica", 10)
            c.drawCentredString(PAGE_W / 2.0, 0.72 * inch, page.group_caption or result.subtitle)
            c.showPage()
            continue

        # Interior 10x10 page: warm gallery cream background
        c.setFillColor(HexColor("#FAF7F2"))
        c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

        # Subtle outer trim border
        c.setStrokeColor(HexColor("#E5DEC9"))
        c.setLineWidth(0.6)
        c.rect(0.35 * inch, 0.35 * inch, 9.3 * inch, 9.3 * inch, stroke=1, fill=0)

        # Chapter / Cluster Header
        if page.chapter_title:
            c.setFillColor(HexColor("#8C6D3F"))
            c.setFont("Helvetica-Bold", 8.5)
            c.drawString(0.75 * inch, PAGE_H - 0.68 * inch, "CHAPTER")
            c.setFillColor(HexColor("#1E232A"))
            c.setFont("Times-Bold", 16.5)
            c.drawString(0.75 * inch, PAGE_H - 0.96 * inch, page.chapter_title)
            c.setStrokeColor(HexColor("#D4C7B0"))
            c.setLineWidth(0.75)
            c.line(0.75 * inch, PAGE_H - 1.08 * inch, PAGE_W - 0.75 * inch, PAGE_H - 1.08 * inch)
        else:
            c.setFillColor(HexColor("#8A8478"))
            c.setFont("Times-Italic", 10)
            c.drawString(0.75 * inch, PAGE_H - 0.85 * inch, result.title)

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

            # Individual slot caption if multi-slot or single
            if len(page.slots) > 1 and slot.caption:
                c.setFillColor(HexColor("#4A4F55"))
                c.setFont("Times-Italic", 8.5)
                lines = _wrap_text_lines(c, slot.caption, "Times-Italic", 8.5, w_pt - 4, max_lines=1)
                if lines:
                    c.drawString(x_pt + 2, y_pt - 13, lines[0])

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
        f"- **Model Used:** `{result.model_used}` (Thinking Level: `{result.thinking_level}`, Media Resolution: `{result.media_resolution}` / `{result.analysis_pixel_range}`)",
    ]
    if result.model_note:
        lines.append(f"- **Model Note:** {result.model_note}")
    lines += [
        f"- **Book Dimensions:** `10.0 × 10.0 inches`",
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
            else f'<div class="page-running-head">{html.escape(result.title)}</div>'
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
    margin: 0 auto 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #1c2029;
    border: 1px solid #2e3442;
    padding: 14px 20px;
    border-radius: 10px;
  }}
  .top-banner h2 {{ margin: 0 0 4px 0; font-size: 16px; color: #faf7f2; }}
  .top-banner p {{ margin: 0; font-size: 12.5px; color: #a8b0c0; }}
  .print-btn {{
    background: #c5a46d;
    color: #14161b;
    border: none;
    padding: 9px 16px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
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
  .spine-crease {{
    position: absolute;
    top: 8px;
    bottom: 8px;
    left: 50%;
    width: 28px;
    transform: translateX(-50%);
    background: linear-gradient(
      90deg,
      rgba(0, 0, 0, 0) 0%,
      rgba(0, 0, 0, 0.18) 38%,
      rgba(15, 12, 8, 0.48) 50%,
      rgba(255, 255, 255, 0.12) 54%,
      rgba(0, 0, 0, 0.16) 65%,
      rgba(0, 0, 0, 0) 100%
    );
    pointer-events: none;
    z-index: 15;
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
    inset: 4.5%;
    border: 1px solid #c5a46d;
    pointer-events: none;
  }}
  .cover-header {{
    position: absolute;
    top: 6.2%;
    left: 8%;
    right: 8%;
    text-align: center;
  }}
  .cover-eyebrow {{
    font-size: 10px;
    letter-spacing: 0.16em;
    color: #c5a46d;
    font-weight: 700;
    margin-bottom: 4px;
  }}
  .cover-title {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 24px;
    margin: 0;
    font-weight: 700;
  }}
  .cover-footer {{
    position: absolute;
    bottom: 6.2%;
    left: 10%;
    right: 10%;
    text-align: center;
  }}
  .cover-hero-caption {{
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic;
    font-size: 13px;
    color: #e8e2d5;
    margin: 0 0 8px 0;
  }}
  .cover-subtitle {{
    font-size: 11px;
    letter-spacing: 0.06em;
    color: #c5a46d;
  }}
  .interior-page {{
    background: #faf7f2;
    color: #1e232a;
  }}
  .interior-border {{
    position: absolute;
    inset: 3.5%;
    border: 1px solid #e5dec9;
    pointer-events: none;
  }}
  .page-chapter-header {{
    position: absolute;
    top: 5.2%;
    left: 7.5%;
    right: 7.5%;
    border-bottom: 1px solid #d4c7b0;
    padding-bottom: 6px;
  }}
  .chapter-tag {{
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: #8c6d3f;
    display: block;
  }}
  .page-chapter-header h2 {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 18px;
    margin: 2px 0 0 0;
    color: #1e232a;
  }}
  .page-running-head {{
    position: absolute;
    top: 6.2%;
    left: 7.5%;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic;
    font-size: 11px;
    color: #8a8478;
  }}
  .photo-slot {{
    position: absolute;
    background: #fff;
    padding: 4px;
    border: 1px solid #ddd5c7;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
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
    bottom: -18px;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic;
    font-size: 10px;
    color: #4a4f55;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .page-footer {{
    position: absolute;
    bottom: 5.2%;
    left: 8%;
    right: 8%;
    text-align: center;
  }}
  .page-group-caption {{
    font-family: Georgia, "Times New Roman", serif;
    font-size: 12.5px;
    line-height: 1.35;
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
    .top-banner, .spine-crease, .endpaper-page {{ display: none; }}
    .pages-stack {{ gap: 0; }}
    .book-spread {{ display: block; width: auto; aspect-ratio: auto; background: none; padding: 0; border: none; box-shadow: none; }}
    .book-page {{ width: 10in; height: 10in; box-shadow: none; }}
  }}
</style>
</head>
<body>
  <div class="top-banner">
    <div>
      <h2>{html.escape(result.title)} (10×10 Inch Open-Spread Photo Book)</h2>
      <p>{result.total_selected} curated photos across {len(result.pages)} pages • Side-by-Side 2-Page Spreads • Model: {html.escape(result.model_used)}</p>
    </div>
    <button class="print-btn" onclick="window.print()">Print / Save 10×10 PDF</button>
  </div>
  <div class="pages-stack">
    {''.join(spreads_html)}
  </div>
</body>
</html>
"""
    html_path.write_text(doc_html, encoding="utf-8")
    return html_path
