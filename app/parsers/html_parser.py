# app/parsers/html_parser.py
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.parsers.base_parser import BaseDocumentParser


class HtmlDocumentParser(BaseDocumentParser):
    def parse(self, path: Path) -> str:
        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        return "\n".join(soup.stripped_strings).strip()