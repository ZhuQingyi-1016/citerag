from app.providers.llm_provider import build_prompt
from app.schemas import AskRequest
from app.services.qa_service import QAService


class _NoopIndexService:
    def has_index(self, _file_id: str) -> bool:
        return True

    def has_any_index(self) -> bool:
        return True


class _ProbeRetrievalService:
    def __init__(self):
        self.last_top_k = None
        self.last_question_type = None
        self.rerank_provider = None
        self.sqlite_repo = None

    def search(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
        file_id=None,
        question_type: str = "general",
    ):
        self.last_top_k = top_k
        self.last_question_type = question_type
        return []


class _NoopGenerator:
    def generate(self, question: str, hits, question_type: str = "general"):
        return {
            "answer": "ok",
            "used_chunk_ids": [],
            "used_chunk_refs": [],
            "usage": {},
        }


def test_detect_process_question_type():
    svc = QAService(
        index_service=_NoopIndexService(),
        retrieval_service=_ProbeRetrievalService(),
        generator=_NoopGenerator(),
    )
    qtype = svc._detect_question_type("激光等离子体LSP的发展过程是什么？")
    assert qtype == "process"


def test_process_question_raises_effective_retrieve_top_k():
    retrieval = _ProbeRetrievalService()
    svc = QAService(
        index_service=_NoopIndexService(),
        retrieval_service=retrieval,
        generator=_NoopGenerator(),
    )
    _ = svc.ask_debug(
        AskRequest(
            question="激光等离子体LSP的发展过程是什么？",
            retrieval_mode="hybrid",
            rerank_mode="none",
            retrieve_top_k=10,
        )
    )

    assert retrieval.last_question_type == "process"
    assert retrieval.last_top_k == 15


def test_build_prompt_for_process_contains_abstain_instruction():
    prompt = build_prompt(
        question="激光等离子体LSP的发展过程是什么？（用中文回答）",
        hits=[{"chunk_id": 1, "text": "LSP spectrum details only."}],
        question_type="process",
    )

    assert "development-process question" in prompt
    assert "do not fabricate one" in prompt
    assert "Answer in Chinese." in prompt
