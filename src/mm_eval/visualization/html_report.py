"""Minimal HTML report helper."""
from __future__ import annotations
from pathlib import Path

def markdown_to_basic_html(markdown: str, output_html: str | Path) -> Path:
    """Write a very small HTML wrapper around markdown text."""
    p=Path(output_html); p.parent.mkdir(parents=True,exist_ok=True)
    html="<html><body><pre>"+markdown.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")+"</pre></body></html>"
    p.write_text(html,encoding="utf-8"); return p
