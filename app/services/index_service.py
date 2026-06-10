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
    text = text.lower()
    pattern = r"""
        [a-zA-Z0-9_\-{}<>/:.]+   |
        [\u4e00-\u9fff]+
    """
    tokens = re.findall(pattern, text, re.VERBOSE)
    return [tok.strip() for tok in tokens if tok.strip()]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    chunk_id = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(Chunk(chunk_id=chunk_id, start=start, end=end, text=chunk))
        chunk_id += 1
        if end == n:
            break
        start = end - overlap

    return chunks


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

    separators = [
        "\n\n",
        "\n",
        "。", ". ",
        "；", ";",
        "：", ":",
        "，", ",",
        "|",
        "\t",
        " ",
        "",
    ]

    def split_recursively(piece: str, seps: list[str]) -> list[str]:
        if len(piece) <= chunk_size:
            return [piece]

        if not seps:
            return [piece[i:i + chunk_size] for i in range(0, len(piece), chunk_size)]

        sep = seps[0]

        if sep == "":
            return [piece[i:i + chunk_size] for i in range(0, len(piece), chunk_size)]

        parts = piece.split(sep)

        if len(parts) == 1:
            return split_recursively(piece, seps[1:])

        merged = []
        current = ""

        for part in parts:
            candidate = part if not current else current + sep + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                if len(part) <= chunk_size:
                    current = part
                else:
                    merged.extend(split_recursively(part, seps[1:]))
                    current = ""

        if current:
            merged.append(current)

        return merged

    pieces = split_recursively(text, separators)

    chunks: list[Chunk] = []
    cursor = 0

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        start = text.find(piece, cursor)
        if start == -1:
            start = cursor
        end = start + len(piece)

        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                start=start,
                end=end,
                text=piece,
            )
        )
        cursor = end

    if overlap > 0 and chunks:
        overlapped_chunks: list[Chunk] = []
        for i, c in enumerate(chunks):
            start = c.start
            end = c.end
            if i > 0:
                start = max(0, start - overlap)

            overlapped_chunks.append(
                Chunk(
                    chunk_id=i,
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
        return overlapped_chunks

    return chunks


class BM25Index:
    def __init__(self, file_id: str, chunks: list[Chunk], filename: str):
        self.file_id = file_id
        self.filename = filename
        self.chunks = chunks
        self.tokenized = [simple_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        self.indexed_at = datetime.now(timezone.utc).isoformat()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for i in ranked:
            c = self.chunks[i]
            hits.append(
                {
                    "file_id": self.file_id,
                    "filename": self.filename,
                    "chunk_id": c.chunk_id,
                    "score": float(scores[i]),
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                }
            )
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

            span.record_exception(e)
            span.set_attribute("citerag.error", True)

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
        all_hits = []

        for file_id in self.index_repo.list_file_ids():
            hits = self.search_in_file(file_id, question, top_k=top_k)
            all_hits.extend(hits)

        all_hits.sort(key=lambda x: x["score"], reverse=True)
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
