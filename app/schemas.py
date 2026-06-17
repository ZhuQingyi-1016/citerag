from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EchoRequest(BaseModel):
    message: str = Field(..., description="Message to echo back.")


class EchoResponse(BaseModel):
    message: str = Field(..., description="Echoed message.")
    server_time: str = Field(..., description="Server timestamp in ISO format.")


class UploadResponse(BaseModel):
    file_id: str = Field(..., description="Internal file identifier.")
    filename: str = Field(..., description="Original uploaded filename.")
    display_name: str | None = Field(None, description="User-facing display name.")
    group_id: str | None = Field(None, description="Optional document group id.")
    size_bytes: int = Field(..., ge=0, description="Uploaded file size in bytes.")
    indexed: bool = Field(False, description="Whether auto-indexing succeeded.")
    indexed_status: str | None = Field(
        None,
        description="Indexing status, such as indexed or failed.",
    )
    indexed_message: str | None = Field(
        None,
        description="Optional indexing message or error detail.",
    )


class FileItem(BaseModel):
    file_id: str = Field(..., description="Internal file identifier.")
    filename: str = Field(..., description="Original stored filename.")
    display_name: str | None = Field(None, description="User-facing display name.")
    group_id: str | None = Field(None, description="Optional document group id.")
    deleted_at: str | None = Field(None, description="Soft-delete timestamp, if deleted.")
    size_bytes: int = Field(..., ge=0, description="File size in bytes.")
    modified_time: str = Field(..., description="Last modified timestamp.")
    bm25_indexed: bool = Field(False, description="Whether BM25 index exists for this file.")
    vector_indexed: bool = Field(False, description="Whether vector index exists for this file.")
    indexed_at: str | None = Field(None, description="Last successful indexing timestamp.")


class FilesResponse(BaseModel):
    files: list[FileItem]


class GroupItem(BaseModel):
    group_id: str = Field(..., description="Document group identifier.")
    name: str = Field(..., description="Document group name.")
    created_at: str = Field(..., description="Creation timestamp.")
    updated_at: str = Field(..., description="Last update timestamp.")


class GroupsResponse(BaseModel):
    groups: list[GroupItem]


class CreateGroupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80, description="Document group name.")


class RenameFileRequest(BaseModel):
    display_name: str | None = Field(
        None,
        min_length=1,
        max_length=120,
        description="Optional user-facing file display name.",
    )


class UpdateFileGroupRequest(BaseModel):
    group_id: str | None = Field(None, description="Target group id. Use null to remove grouping.")


class FileManageResponse(BaseModel):
    file_id: str
    filename: str
    display_name: str | None = None
    group_id: str | None = None
    updated_at: str


class DeleteFileCleanupItem(BaseModel):
    bm25_removed: bool = Field(..., description="Whether BM25 in-memory index was removed.")
    vector_removed: bool = Field(..., description="Whether vector index entries were removed.")
    chunks_removed: bool = Field(..., description="Whether stored chunks were removed.")
    index_status_removed: bool = Field(
        ...,
        description="Whether index status metadata was removed.",
    )


class DeleteFileResponse(BaseModel):
    file_id: str
    deleted: bool
    deleted_at: str
    index_cleanup: DeleteFileCleanupItem


class IndexRequest(BaseModel):
    chunk_size: int = Field(
        800,
        ge=200,
        le=3000,
        description=(
            "Maximum characters per chunk before overlap. Larger chunks "
            "preserve context but may reduce retrieval precision."
        ),
    )
    overlap: int = Field(
        100,
        ge=0,
        le=800,
        description=(
            "Maximum overlap characters between adjacent chunks. "
            "Must be smaller than chunk_size."
        ),
    )
    chunk_method: Literal["fixed", "recursive"] = Field(
        "recursive",
        description="Chunking strategy. recursive prioritizes paragraph and sentence boundaries.",
    )

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class IndexResponse(BaseModel):
    file_id: str
    chunks_count: int = Field(..., ge=0)
    indexed_at: str


class IndexAllItem(BaseModel):
    file_id: str
    filename: str
    indexed: bool
    indexed_status: str
    indexed_message: str | None = None
    chunks_count: int | None = Field(None, ge=0)


class IndexAllResponse(BaseModel):
    total_files: int = Field(..., ge=0)
    indexed_files: int = Field(..., ge=0)
    skipped_files: int = Field(..., ge=0)
    results: list[IndexAllItem]


class HitItem(BaseModel):
    file_id: str = Field(..., description="Source file id.")
    filename: str = Field(..., description="Source filename or display name.")
    chunk_id: int = Field(..., ge=0, description="Chunk id within the source file.")
    source_ref: str | None = Field(
        None,
        description="Stable debug reference in file_id:chunk_id format.",
    )
    score: float = Field(..., description="Retrieval or hybrid score.")
    start: int = Field(..., ge=0, description="Character start offset in normalized document text.")
    end: int = Field(..., ge=0, description="Character end offset in normalized document text.")
    text: str = Field(..., description="Raw retrieved chunk text for debugging.")
    rerank_score: float | None = Field(None, description="Optional external rerank score.")

    @model_validator(mode="after")
    def fill_source_ref(self):
        if self.source_ref is None:
            self.source_ref = f"{self.file_id}:{self.chunk_id}"
        return self


class CitationItem(BaseModel):
    file_id: str = Field(..., description="Source file id.")
    filename: str = Field(..., description="Source filename or display name.")
    chunk_id: int = Field(..., ge=0, description="Chunk id within the source file.")
    source_ref: str | None = Field(
        None,
        description="Stable citation reference in file_id:chunk_id format.",
    )
    start: int = Field(..., ge=0, description="Character start offset in normalized document text.")
    end: int = Field(..., ge=0, description="Character end offset in normalized document text.")
    text: str = Field(..., description="Clean evidence snippet shown to users.")

    @model_validator(mode="after")
    def fill_source_ref(self):
        if self.source_ref is None:
            self.source_ref = f"{self.file_id}:{self.chunk_id}"
        return self


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User question to answer from uploaded documents.",
        examples=["README 里有哪些 retrieval modes？"],
    )
    file_id: str | None = Field(
        None,
        description="Optional file id. When omitted, search across all indexed files.",
    )
    top_k: int = Field(
        3,
        ge=1,
        le=10,
        description=(
            "Final number of chunks passed to the generator after retrieval "
            "and optional rerank."
        ),
    )
    retrieve_top_k: int = Field(
        10,
        ge=1,
        le=30,
        description="First-stage candidate count before final top_k selection.",
    )
    retrieval_mode: Literal["bm25", "vector", "hybrid"] = Field(
        "hybrid",
        description="Retrieval strategy.",
    )
    rerank_mode: Literal["none", "cohere"] = Field(
        "none",
        description="Optional rerank strategy. none is the default validated baseline.",
    )

    @model_validator(mode="after")
    def validate_top_k(self):
        if self.top_k > self.retrieve_top_k:
            raise ValueError("top_k must be less than or equal to retrieve_top_k")
        return self


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]


class AskDebugResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]
    hits: list[HitItem]
    question_type: str | None = Field(
        None,
        description="Detected question type used for retrieval tuning.",
    )
    effective_retrieve_top_k: int | None = Field(
        None,
        description="Actual first-stage retrieval count after question-type adjustment.",
    )
