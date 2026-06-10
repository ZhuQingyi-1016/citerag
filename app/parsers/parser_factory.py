from pathlib import Path

from app.parsers.docx_parser import DocxDocumentParser
from app.parsers.html_parser import HtmlDocumentParser
from app.parsers.pdf_parser import PdfDocumentParser
from app.parsers.text_parser import TextDocumentParser


def get_document_parser(file_path: Path):
    ext = file_path.suffix.lower()

    if ext in {".txt", ".md"}:
        return TextDocumentParser()
    if ext == ".html":
        return HtmlDocumentParser()
    if ext == ".pdf":
        return PdfDocumentParser()
    if ext == ".docx":
        return DocxDocumentParser()

    raise ValueError(f"Unsupported file type: {ext}")