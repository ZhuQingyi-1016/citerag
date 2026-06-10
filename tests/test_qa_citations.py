from app.schemas import AskRequest
from app.services.index_service import BM25Index, Chunk
from app.services.qa_service import QAService
from app.services.retrieval_service import RetrievalService


class _StubIndexService:
    def __init__(self, hits):
        self._hits = hits

    def has_index(self, _file_id: str) -> bool:
        return True

    def has_any_index(self) -> bool:
        return True

    def search_in_file(self, _file_id: str, _question: str, top_k: int = 5):
        return self._hits[:top_k]

    def search_all_files(self, _question: str, top_k: int = 5):
        return self._hits[:top_k]


class _StubRetrievalService:
    def __init__(self, hits):
        self._hits = hits
        self.rerank_provider = None

    def search(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
        file_id=None,
        question_type: str = "general",
    ):
        return self._hits[:top_k]

    def rerank_hits(self, question: str, hits, final_top_k: int, rerank_mode: str = "none"):
        return hits[:final_top_k]


class _StubGenerator:
    def generate(self, question: str, hits, question_type: str = "general"):
        return {
            "answer": "ok",
            "used_chunk_ids": [7],  # legacy field kept for compatibility
            "used_chunk_refs": [{"file_id": "f2", "chunk_id": 7}],
            "usage": {},
        }


def test_citation_bind_by_file_and_chunk():
    hits = [
        {
            "file_id": "f1",
            "filename": "a.txt",
            "chunk_id": 7,
            "score": 1.0,
            "start": 0,
            "end": 100,
            "text": "A",
        },
        {
            "file_id": "f2",
            "filename": "b.txt",
            "chunk_id": 7,
            "score": 0.9,
            "start": 0,
            "end": 100,
            "text": "B",
        },
    ]

    svc = QAService(
        index_service=_StubIndexService(hits),
        retrieval_service=_StubRetrievalService(hits),
        generator=_StubGenerator(),
    )

    res = svc.ask_debug(
        AskRequest(
            question="q",
            retrieval_mode="bm25",
            rerank_mode="none",
            file_id="f-any",
            top_k=2,
            retrieve_top_k=2,
        )
    )
    assert len(res.citations) == 1
    assert res.citations[0].filename == "b.txt"
    assert res.citations[0].chunk_id == 7


def test_hits_not_truncated():
    long_text = "field: " + ("value-" * 120)
    idx = BM25Index(
        file_id="f1",
        filename="doc.txt",
        chunks=[Chunk(chunk_id=0, start=0, end=len(long_text), text=long_text)],
    )

    hits = idx.search("field", top_k=1)
    assert len(hits) == 1
    assert hits[0]["text"] == long_text
    assert len(hits[0]["text"]) > 400


def test_rerank_none_diversity():
    svc = RetrievalService(
        index_service=None,
        embedding_provider=None,
        vector_repo=None,
        sqlite_repo=None,
    )

    hits = [
        {"file_id": "f1", "chunk_id": 1, "text": "alpha beta gamma", "score": 1.0},
        {"file_id": "f1", "chunk_id": 2, "text": "alpha beta gamma", "score": 0.99},
        {"file_id": "f1", "chunk_id": 3, "text": "delta epsilon zeta", "score": 0.98},
    ]

    out = svc.rerank_hits(
        question="q",
        hits=hits,
        final_top_k=2,
        rerank_mode="none",
    )

    assert len(out) == 2
    assert out[0]["chunk_id"] == 1
    assert out[1]["chunk_id"] == 3
