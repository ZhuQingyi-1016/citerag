from app.providers.embedding_provider import OpenAIEmbeddingProvider
from app.providers.llm_provider import get_answer_generator
from app.providers.rerank_provider import (
    BaseRerankProvider,
    CohereRerankProvider,
    NoopRerankProvider,
)
from app.repositories.file_repository import FileRepository
from app.repositories.index_repository import InMemoryIndexRepository
from app.repositories.sqlite_repository import SQLiteRepository
from app.repositories.vector_repository import InMemoryVectorRepository
from app.services.file_service import FileService
from app.services.index_service import IndexService
from app.services.qa_service import QAService
from app.services.retrieval_service import RetrievalService

file_repo = FileRepository()
index_repo = InMemoryIndexRepository()
vector_repo = InMemoryVectorRepository()
sqlite_repo = SQLiteRepository()


def get_file_repository() -> FileRepository:
    return file_repo


def get_index_repository() -> InMemoryIndexRepository:
    return index_repo


def get_vector_repository() -> InMemoryVectorRepository:
    return vector_repo


def get_sqlite_repository() -> SQLiteRepository:
    return sqlite_repo


def get_embedding_provider() -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider()


def get_index_service() -> IndexService:
    return IndexService(
        file_repo=get_file_repository(),
        index_repo=get_index_repository(),
        sqlite_repo=get_sqlite_repository(),
    )


def get_retrieval_service() -> RetrievalService:
    return RetrievalService(
        index_service=get_index_service(),
        embedding_provider=get_embedding_provider(),
        vector_repo=get_vector_repository(),
        sqlite_repo=get_sqlite_repository(),
        rerank_provider=get_rerank_provider(),
    )
    

def get_file_service() -> FileService:
    return FileService(
        file_repo=get_file_repository(),
        index_service=get_index_service(),
        retrieval_service=get_retrieval_service(),
        sqlite_repo=get_sqlite_repository(),
    )


def get_qa_service() -> QAService:
    return QAService(
        index_service=get_index_service(),
        retrieval_service=get_retrieval_service(),
        generator=get_answer_generator(),
    )

def get_rerank_provider(mode: str = "none") -> BaseRerankProvider:
    if mode == "cohere":
        return CohereRerankProvider()
    return NoopRerankProvider()