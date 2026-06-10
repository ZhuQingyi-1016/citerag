# app/parsers/docx_parser.py
from __future__ import annotations

from pathlib import Path

from docx import Document

from app.parsers.base_parser import BaseDocumentParser


class DocxDocumentParser(BaseDocumentParser):
    def parse(self, path: Path) -> str:
        doc = Document(str(path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(parts).strip()