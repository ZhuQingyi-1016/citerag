from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from opentelemetry import trace
from rank_bm25 import BM25Okapi

from app.core.observability import (
    INDEX_COUNT,
    INDEX_LATENCY,
    elapsed_ms,
    log_event,
    now_perf,
)
from app.parsers.parser_factory import get_document_parser
from app.repositories.file_repository import FileRepository
from app.repositories.index_repository import InMemoryIndexRepository
from app.repositories.sqlite_repository import SQLiteRepository

tracer = trace.get_tracer(__name__)


@dataclass
class Chunk:
    chunk_id: int
    start: int
    end: int
    text: str


def simple_tokenize(text: str) -> list[str]:
    normalized = text.lower()
    english_tokens = re.findall(r"[a-z0-9]+", normalized)

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    chinese_tokens: list[str] = []
    if chinese_chars:
        chinese_text = "".join(chinese_chars)
        chinese_tokens.append(chinese_text)
        chinese_tokens.extend(chinese_chars)
        chinese_tokens.extend(
            chinese_text[i : i + 2]
            for i in range(max(len(chinese_text) - 1, 0))
        )
        chinese_tokens.extend(
            chinese_text[i : i + 3]
            for i in range(max(len(chinese_text) - 2, 0))
        )

    return english_tokens + chinese_tokens


def normalize_document_text(text: str) -> str:
    """Clean parser output before chunking.

    PDF/DOCX extraction often leaves artificial line breaks, split chemical
    formulas such as CO\n2, broken units such as μ\nm, and hyphenated words
    across lines. Cleaning here improves both retrieval chunks and citations.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Keep paragraph boundaries, but normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Restore hyphenated English words split by PDF line breaks:
    # de-\nenergized -> de-energized
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1-\2", text)

    # Restore common technical tokens split across lines:
    # CO\n2 -> CO2, 10.6 μ\nm -> 10.6 μm
    text = re.sub(r"([A-Za-z])\s*\n\s*(\d)", r"\1\2", text)
    text = re.sub(r"(\d)\s*\n\s*([A-Za-zμµ])", r"\1 \2", text)
    text = re.sub(r"([μµ])\s*\n\s*([A-Za-z])", r"\1\2", text)

    # Merge single line breaks inside a paragraph. Keep blank lines as
    # paragraph separators.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize spaces and add light spacing between CJK and ASCII tokens.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", text)

    return text.strip()


def _split_long_text_by_sentence(text: str, chunk_size: int) -> list[str]:
    """Split long text into sentence-like units for Chinese and English."""
    if len(text) <= chunk_size:
        return [text]

    # Split after common sentence punctuation. The English rule avoids
    # splitting decimals by requiring a following space and capital/number.
    sentences = re.split(
        r"(?<=[。！？!?；;])\s*|(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= chunk_size:
            units.append(sentence)
        else:
            units.extend(
                sentence[i : i + chunk_size]
                for i in range(0, len(sentence), chunk_size)
            )
    return units


def _sentence_aligned_tail(text: str, overlap: int) -> str:
    """Return overlap text while avoiding starts in the middle of words."""
    if overlap <= 0 or len(text) <= overlap:
        return text.strip()

    tail = text[-overlap:]

    # Prefer starting after a sentence boundary inside the overlap window.
    boundary_matches = list(re.finditer(r"[。！？!?；;.]\s+", tail))
    for match in boundary_matches:
        candidate = tail[match.end() :].strip()
        if len(candidate) >= 30:
            return candidate

    # Otherwise start at a whitespace boundary.
    whitespace = re.search(r"\s+", tail)
    if whitespace and whitespace.end() < len(tail) - 20:
        return tail[whitespace.end() :].strip()

    return tail.strip()


def _build_chunks_from_pieces(
    text: str,
    pieces: list[str],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        if not current:
            current = piece
            continue

        separator = "\n\n" if "\n" in current or "\n" in piece else " "
        candidate = f"{current}{separator}{piece}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(Chunk(chunk_id=len(chunks), start=0, end=0, text=current.strip()))
            tail = _sentence_aligned_tail(current, overlap)
            if tail and len(f"{tail} {piece}".strip()) <= chunk_size:
                current = f"{tail} {piece}".strip()
            else:
                current = piece

    if current.strip():
        chunks.append(Chunk(chunk_id=len(chunks), start=0, end=0, text=current.strip()))

    # Map chunk text back to normalized-document offsets where possible.
    cursor = 0
    mapped: list[Chunk] = []
    for chunk in chunks:
        probe = chunk.text[: min(80, len(chunk.text))]
        start = text.find(probe, cursor) if probe else -1
        if start == -1 and probe:
            start = text.find(probe)
        if start == -1:
            start = cursor
        end = min(len(text), start + len(chunk.text))
        mapped.append(
            Chunk(
                chunk_id=len(mapped),
                start=start,
                end=end,
                text=chunk.text,
            )
        )
        cursor = max(cursor, end)

    return mapped


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = normalize_document_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_split_long_text_by_sentence(paragraph, chunk_size))

    return _build_chunks_from_pieces(
        text=text,
        pieces=pieces,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def chunk_text_recursive(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = normalize_document_text(text)
    if not text:
        return []

    # Recursive mode should prioritize natural document boundaries instead of
    # raw character offsets, so citations do not start halfway through a
    # sentence. Split by paragraphs first, then by sentence-like units for
    # long paragraphs.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_long_text_by_sentence(paragraph, chunk_size))

    return _build_chunks_from_pieces(
        text=text,
        pieces=pieces,
        chunk_size=chunk_size,
        overlap=overlap,
    )


class BM25Index:
    def __init__(self, file_id: str, chunks: list[Chunk], filename: str):
        self.file_id = file_id
        self.filename = filename
        self.chunks = chunks
        self.tokenized = [simple_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None
        self.indexed_at = datetime.now(timezone.utc).isoformat()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0 or self.bm25 is None or not self.chunks:
            return []

        q_tokens = simple_tokenize(query)
        if not q_tokens:
            return []

        q_token_set = set(q_tokens)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        hits = []
        for i in ranked:
            score = float(scores[i])
            c = self.chunks[i]
            has_lexical_overlap = bool(q_token_set & set(self.tokenized[i]))
            if score <= 0 and not has_lexical_overlap:
                continue

            hits.append(
                {
                    "file_id": self.file_id,
                    "filename": self.filename,
                    "chunk_id": c.chunk_id,
                    "score": score,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                }
            )
            if len(hits) >= top_k:
                break

        return hits


class IndexService:
    def __init__(
        self,
        file_repo: FileRepository,
        index_repo: InMemoryIndexRepository,
        sqlite_repo: SQLiteRepository,
    ):
        self.file_repo = file_repo
        self.index_repo = index_repo
        self.sqlite_repo = sqlite_repo

    def build_index_for_file(
        self,
        file_id: str,
        chunk_size: int = 800,
        overlap: int = 100,
        chunk_method: str = "fixed",
    ) -> dict:
        start = now_perf()

        try:
            with tracer.start_as_current_span("index.build_index_for_file") as span:
                span.set_attribute("citerag.file_id", file_id)
                span.set_attribute("citerag.chunk_method", chunk_method)
                span.set_attribute("citerag.chunk_size", chunk_size)
                span.set_attribute("citerag.overlap", overlap)

                p = self.file_repo.find_path_by_file_id(file_id)
                parser = get_document_parser(p)

                text = parser.parse(p)

                if not text.strip():
                    raise ValueError("parsed document is empty")

                if chunk_method == "recursive":
                    chunks = chunk_text_recursive(
                        text,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                else:
                    chunks = chunk_text(
                        text,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )

                if not chunks:
                    raise ValueError("no chunks generated from document")

                filename = p.name.split("__", 1)[1] if "__" in p.name else p.name

                index_obj = BM25Index(file_id=file_id, chunks=chunks, filename=filename)
                self.index_repo.set_index(file_id, index_obj)

                self.sqlite_repo.replace_chunks(
                    file_id=file_id,
                    chunks=[
                        {
                            "chunk_id": c.chunk_id,
                            "start": c.start,
                            "end": c.end,
                            "text": c.text,
                        }
                        for c in chunks
                    ],
                )

                existing_status = self.sqlite_repo.get_index_status(file_id)
                self.sqlite_repo.upsert_index_status(
                    file_id=file_id,
                    bm25_indexed=True,
                    vector_indexed=(
                        False
                        if existing_status is None
                        else bool(existing_status["vector_indexed"])
                    ),
                    indexed_at=index_obj.indexed_at,
                )

                span.set_attribute("citerag.filename", filename)
                span.set_attribute("citerag.chunks_count", len(chunks))

                duration_ms = elapsed_ms(start)
                INDEX_COUNT.labels(status="indexed", chunk_method=chunk_method).inc()
                INDEX_LATENCY.labels(chunk_method=chunk_method).observe(duration_ms / 1000)

                log_event(
                    "build_index",
                    file_id=file_id,
                    filename=filename,
                    chunk_method=chunk_method,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    chunks_count=len(chunks),
                    duration_ms=duration_ms,
                )

                return {
                    "file_id": file_id,
                    "chunks_count": len(chunks),
                    "indexed_at": index_obj.indexed_at,
                }

        except Exception as e:
            duration_ms = elapsed_ms(start)
            INDEX_COUNT.labels(status="failed", chunk_method=chunk_method).inc()
            INDEX_LATENCY.labels(chunk_method=chunk_method).observe(duration_ms / 1000)

            current_span = trace.get_current_span()
            if current_span is not None:
                current_span.record_exception(e)
                current_span.set_attribute("citerag.error", True)

            log_event(
                "build_index_failed",
                file_id=file_id,
                chunk_method=chunk_method,
                chunk_size=chunk_size,
                overlap=overlap,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise

    def rebuild_bm25_from_sqlite(self) -> dict:
        rows = self.sqlite_repo.list_indexed_files()

        rebuilt = []
        skipped = []

        for row in rows:
            file_id = row["file_id"]

            if not row["bm25_indexed"]:
                skipped.append(
                    {
                        "file_id": file_id,
                        "reason": "bm25_indexed is false in sqlite",
                    }
                )
                continue

            try:
                chunk_rows = self.sqlite_repo.list_chunks_for_file(file_id)
                if not chunk_rows:
                    skipped.append(
                        {
                            "file_id": file_id,
                            "reason": "no chunks found in sqlite",
                        }
                    )
                    continue

                chunks = [
                    Chunk(
                        chunk_id=r["chunk_id"],
                        start=r["start_pos"],
                        end=r["end_pos"],
                        text=r["text"],
                    )
                    for r in chunk_rows
                ]

                p = self.file_repo.find_path_by_file_id(file_id)
                filename = p.name.split("__", 1)[1] if "__" in p.name else p.name

                index_obj = BM25Index(file_id=file_id, chunks=chunks, filename=filename)
                self.index_repo.set_index(file_id, index_obj)

                rebuilt.append(file_id)

            except Exception as e:
                skipped.append(
                    {
                        "file_id": file_id,
                        "reason": str(e),
                    }
                )

        return {
            "rebuilt_bm25_files": rebuilt,
            "skipped": skipped,
        }

    def search_in_file(
        self,
        file_id: str,
        question: str,
        top_k: int = 5,
    ) -> list[dict]:
        index = self.index_repo.get_index(file_id)
        if index is None:
            raise KeyError("not_indexed")
        return index.search(question, top_k=top_k)

    def search_all_files(self, question: str, top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            return []

        all_hits = []

        for file_id in self.index_repo.list_file_ids():
            hits = self.search_in_file(file_id, question, top_k=top_k)
            all_hits.extend(hits)

        all_hits.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return all_hits[:top_k]

    def has_index(self, file_id: str) -> bool:
        return self.index_repo.has_index(file_id)

    def has_any_index(self) -> bool:
        return self.index_repo.has_any_index()

    def get_index(self, file_id: str):
        idx = self.index_repo.get_index(file_id)
        if idx is None:
            raise ValueError(f"index not found for file_id={file_id}")
        return idx
