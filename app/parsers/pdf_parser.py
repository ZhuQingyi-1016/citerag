# app/parsers/pdf_parser.py
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from app.parsers.base_parser import BaseDocumentParser


def _clean_pdf_text(text: str) -> str:
    """Normalize PDF text extraction artifacts before indexing.

    pypdf text extraction can preserve visual line breaks that are not real
    paragraph boundaries. Cleaning these artifacts here makes downstream
    chunks, hits, and citations easier to read.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenated words split across lines: de-\nenergized -> de-energized
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1-\2", text)

    # Fix common technical tokens split by line breaks:
    # CO\n2 -> CO2, 10.6 μ\nm -> 10.6 μm
    text = re.sub(r"([A-Za-z])\s*\n\s*(\d)", r"\1\2", text)
    text = re.sub(r"(\d)\s*\n\s*([A-Za-zμµ])", r"\1 \2", text)
    text = re.sub(r"([μµ])\s*\n\s*([A-Za-z])", r"\1\2", text)

    # Merge single line breaks inside paragraphs while preserving blank lines.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize repeated spaces and excessive blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Add light spacing between Chinese and ASCII tokens for readability.
    text = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", text)

    return text.strip()


class PdfDocumentParser(BaseDocumentParser):
    def parse(self, path: Path) -> str:
        reader = PdfReader(str(path))
        parts: list[str] = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                # Older pypdf versions do not support extraction_mode.
                raw_text = page.extract_text() or ""

            text = _clean_pdf_text(raw_text)
            if text:
                parts.append(f"[Page {page_number}]\n{text}")

        return "\n\n".join(parts).strip()