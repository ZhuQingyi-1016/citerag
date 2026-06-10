from typing import List, Literal

from pydantic import BaseModel, Field


class EchoRequest(BaseModel):
    message: str

class EchoResponse(BaseModel):
    message: str
    server_time: str

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    display_name: str | None = None
    group_id: str | None = None
    size_bytes: int
    indexed: bool = False
    indexed_status: str | None = None
    indexed_message: str | None = None

class FileItem(BaseModel):
    file_id: str
    filename: str
    display_name: str | None = None
    group_id: str | None = None
    deleted_at: str | None = None
    size_bytes: int
    modified_time: str
    bm25_indexed: bool = False
    vector_indexed: bool = False
    indexed_at: str | None = None

class FilesResponse(BaseModel):
    files: List[FileItem]


class GroupItem(BaseModel):
    group_id: str
    name: str
    created_at: str
    updated_at: str


class GroupsResponse(BaseModel):
    groups: list[GroupItem]


class CreateGroupRequest(BaseModel):
    name: str


class RenameFileRequest(BaseModel):
    display_name: str | None = None


class UpdateFileGroupRequest(BaseModel):
    group_id: str | None = None


class FileManageResponse(BaseModel):
    file_id: str
    filename: str
    display_name: str | None = None
    group_id: str | None = None
    updated_at: str


class DeleteFileCleanupItem(BaseModel):
    bm25_removed: bool
    vector_removed: bool
    chunks_removed: bool
    index_status_removed: bool


class DeleteFileResponse(BaseModel):
    file_id: str
    deleted: bool
    deleted_at: str
    index_cleanup: DeleteFileCleanupItem

class IndexRequest(BaseModel):
    chunk_size: int = Field(800, gt=0)
    overlap: int = Field(100, ge=0)
    chunk_method: Literal["fixed", "recursive"] = "recursive"

class IndexResponse(BaseModel):
    file_id: str
    chunks_count: int
    indexed_at: str

class IndexAllItem(BaseModel):
    file_id: str
    filename: str
    indexed: bool
    indexed_status: str
    indexed_message: str | None = None
    chunks_count: int | None = None


class IndexAllResponse(BaseModel):
    total_files: int
    indexed_files: int
    skipped_files: int
    results: list[IndexAllItem]

class HitItem(BaseModel):
    file_id: str
    filename: str
    chunk_id: int
    score: float
    start: int
    end: int
    text: str
    rerank_score: float | None = None

class CitationItem(BaseModel):
    filename: str
    chunk_id: int
    start: int
    end: int
    text: str

class AskRequest(BaseModel):
    question: str
    file_id: str | None = None
    top_k: int = Field(3, ge=1, le=10) #第二阶段重排后，最终保留多少给 generator
    retrieve_top_k: int = Field(10, ge=1, le=30) # 第一阶段先拿多少候选
    retrieval_mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    rerank_mode: Literal["none", "cohere"] = "none"

class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]

class AskDebugResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]
    hits: list[HitItem]
    question_type: str | None = None
    effective_retrieve_top_k: int | None = None
