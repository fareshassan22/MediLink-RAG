#!/usr/bin/env python3
"""Render the final report (A4) and presentation (16:9 slides) to PDF.

Uses python-markdown + weasyprint (no headless browser needed, which the
server cannot launch). Run: python3 docs/build_pdfs.py
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown
from weasyprint import HTML

DOCS = Path(__file__).resolve().parent
MD_EXT = [
    "tables",
    "fenced_code",
    "sane_lists",
    "pymdownx.tilde",      # ~~strike~~
    "attr_list",
]

# ── Shared report (A4 portrait) styling ─────────────────────────────────
REPORT_CSS = """
@page { size: A4; margin: 22mm 20mm; @bottom-center {
  content: counter(page) " / " counter(pages); font-size: 9px; color: #888; } }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", "Segoe UI", Arial, sans-serif;
  font-size: 10.5px; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 23px; color: #0b3d66; border-bottom: 3px solid #0b3d66;
  padding-bottom: 6px; margin-top: 4px; }
h2 { font-size: 16px; color: #0b3d66; margin-top: 22px;
  border-bottom: 1px solid #cdd9e3; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 13px; color: #14507f; margin-top: 16px; page-break-after: avoid; }
p, li { text-align: justify; }
em { color: #444; }
code { font-family: "DejaVu Sans Mono", monospace; background: #eef2f6;
  padding: 1px 4px; border-radius: 3px; font-size: 9.5px; }
pre { background: #0f2030; color: #e6edf3; padding: 11px 13px;
  border-radius: 6px; font-size: 8.8px; line-height: 1.4; overflow-x: auto;
  page-break-inside: avoid; }
pre code { background: none; color: inherit; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 12px 0;
  font-size: 9.3px; page-break-inside: avoid; }
th { background: #0b3d66; color: #fff; text-align: left; padding: 6px 8px; }
td { border: 1px solid #d4dde6; padding: 5px 8px; vertical-align: top; }
tr:nth-child(even) td { background: #f4f8fb; }
blockquote { border-left: 4px solid #e0a800; background: #fffbe9;
  margin: 12px 0; padding: 8px 14px; color: #5a4a00; }
hr { border: none; border-top: 1px solid #d4dde6; margin: 20px 0; }
strong { color: #0b2a45; }
a { color: #14507f; text-decoration: none; }
"""

# ── Presentation (16:9 landscape, one slide per page) styling ───────────
SLIDE_CSS = """
@page { size: 254mm 143mm; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: "DejaVu Sans", "Segoe UI", Arial, sans-serif;
  color: #15212e; }
.slide { width: 254mm; height: 143mm; padding: 16mm 20mm; page-break-after: always;
  position: relative; display: flex; flex-direction: column; justify-content: flex-start;
  background: #ffffff; }
.slide.title { justify-content: center; align-items: flex-start;
  background: linear-gradient(135deg, #0b3d66 0%, #14507f 100%); color: #fff; }
.slide.title h1 { color: #fff; border: none; font-size: 46px; margin: 0 0 6px; }
.slide.title h2 { color: #cfe3f5; border: none; font-size: 24px; margin: 0 0 18px; }
.slide.title p, .slide.title em { color: #d7e7f5; font-size: 14px; }
.slide.section { justify-content: center; background:
  linear-gradient(135deg, #11324d 0%, #0b3d66 100%); color: #fff; }
.slide.section h1 { color: #fff; border: none; font-size: 40px; }
.slide.section h2 { color: #ffd45e; border: none; font-size: 22px; }
.slide.section p, .slide.section em { color: #d7e7f5; }
h1 { font-size: 28px; color: #0b3d66; margin: 0 0 4px; }
h2 { font-size: 21px; color: #0b3d66; margin: 0 0 12px; }
h3 { font-size: 17px; color: #14507f; margin: 10px 0 6px; }
p, li { font-size: 15px; line-height: 1.5; }
ul { margin: 4px 0 4px 4px; }
li { margin: 5px 0; }
code { font-family: "DejaVu Sans Mono", monospace; background: #eef2f6;
  padding: 1px 5px; border-radius: 3px; font-size: 13px; }
pre { background: #0f2030; color: #e6edf3; padding: 12px 16px; border-radius: 8px;
  font-size: 12px; line-height: 1.45; }
pre code { background: none; color: inherit; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 13px; }
th { background: #0b3d66; color: #fff; text-align: left; padding: 6px 9px; }
td { border: 1px solid #d4dde6; padding: 5px 9px; }
tr:nth-child(even) td { background: #f4f8fb; }
blockquote { border-left: 5px solid #e0a800; background: #fffbe9; margin: 10px 0;
  padding: 8px 16px; color: #5a4a00; font-size: 14px; }
strong { color: #0b2a45; }
hr { display: none; }
.pagenum { position: absolute; bottom: 7mm; right: 12mm; font-size: 11px; color: #9bb0c4; }
"""


def render_report(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=MD_EXT)
    html = f"<html><head><meta charset='utf-8'><style>{REPORT_CSS}</style></head><body>{body}</body></html>"
    HTML(string=html).write_pdf(str(dst))
    print(f"  report  -> {dst}")


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        # remove the first YAML frontmatter block only
        m = re.match(r"^---\n.*?\n---\n", text, flags=re.DOTALL)
        if m:
            return text[m.end():]
    return text


def render_slides(src: Path, dst: Path) -> None:
    text = _strip_frontmatter(src.read_text(encoding="utf-8"))
    # Marp slide separator is a line containing only ---
    raw_slides = re.split(r"\n---\s*\n", text)
    slides_html = []
    n = 0
    for raw in raw_slides:
        raw = raw.strip()
        if not raw:
            continue
        n += 1
        # detect directive comments like <!-- _paginate: false -->
        no_page = "_paginate: false" in raw
        raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL).strip()
        body = markdown.markdown(raw, extensions=MD_EXT)
        # classify slide type by content
        cls = "slide"
        low = body.lower()
        if n == 1 or "thank you" in low:
            cls = "slide title"
        elif "the honest part" in low or "questions & discussion" in low:
            cls = "slide section"
        pagenum = "" if no_page else f"<div class='pagenum'>{n}</div>"
        slides_html.append(f"<section class='{cls}'>{body}{pagenum}</section>")
    html = (f"<html><head><meta charset='utf-8'><style>{SLIDE_CSS}</style></head>"
            f"<body>{''.join(slides_html)}</body></html>")
    # weasyprint paginates on .slide page-break; wrap sections as .slide blocks
    html = html.replace("<section class=", "<div class=").replace("</section>", "</div>")
    HTML(string=html).write_pdf(str(dst))
    print(f"  slides  -> {dst}  ({n} slides)")


if __name__ == "__main__":
    render_report(DOCS / "final_report_book.md", DOCS / "MediLink_Final_Report.pdf")
    render_slides(DOCS / "final_presentation.md", DOCS / "MediLink_Final_Presentation.pdf")
    print("Done.")
