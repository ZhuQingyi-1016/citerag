# app/parsers/text_parser.py
from __future__ import annotations

from pathlib import Path

from app.parsers.base_parser import BaseDocumentParser


class TextDocumentParser(BaseDocumentParser):
    def parse(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")