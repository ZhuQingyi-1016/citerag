# app/parsers/pdf_parser.py
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.parsers.base_parser import BaseDocumentParser


class PdfDocumentParser(BaseDocumentParser):
    def parse(self, path: Path) -> str:
        reader = PdfReader(str(path))
        parts: list[str] = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)

        return "\n\n".join(parts).strip()