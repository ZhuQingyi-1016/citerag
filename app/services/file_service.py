import uuid
from datetime import datetime, timezone

from fastapi import UploadFile

from app.core.observability import (
    UPLOAD_COUNT,
    UPLOAD_LATENCY,
    elapsed_ms,
    log_event,
    now_perf,
)
from app.repositories.file_repository import FileRepository
from app.repositories.sqlite_repository import SQLiteRepository
from app.schemas import (
    DeleteFileCleanupItem,
    DeleteFileResponse,
    FileItem,
    FileManageResponse,
    FilesResponse,
    GroupItem,
    GroupsResponse,
    UploadResponse,
)
from app.services.index_service import IndexService
from app.services.retrieval_service import RetrievalService


class FileService:
    def __init__(
        self,
        file_repo: FileRepository,
        index_service: IndexService,
        retrieval_service: RetrievalService,
        sqlite_repo: SQLiteRepository,
    ):
        self.file_repo = file_repo
        self.index_service = index_service
        self.retrieval_service = retrieval_service
        self.sqlite_repo = sqlite_repo

    def upload_file(
        self,
        file: UploadFile,
        auto_index: bool = False,
        group_id: str | None = None,
    ) -> UploadResponse:
        start = now_perf()

        if group_id is not None and self.sqlite_repo.get_group(group_id) is None:
            raise ValueError("group_id not found")

        file_id, filename, size_bytes = self.file_repo.save_upload(file)
        path = self.file_repo.find_path_by_file_id(file_id)
        content_hash = self.file_repo.compute_file_hash(path)

        existing = self.sqlite_repo.find_file_by_content_hash(content_hash)
        if existing is not None:
            self.file_repo.delete_file_by_path(path)

            existing_file_id = existing["file_id"]
            status = self.sqlite_repo.get_index_status(existing_file_id)
            already_indexed = bool(status["bm25_indexed"]) if status else False

            if auto_index and not already_indexed:
                try:
                    self.index_service.build_index_for_file(
                        file_id=existing_file_id,
                        chunk_size=800,
                        overlap=100,
                        chunk_method="recursive",
                    )
                    self.retrieval_service.index_file_embeddings(existing_file_id)

                    duration_ms = elapsed_ms(start)
                    UPLOAD_COUNT.labels(status="duplicate_reused_and_indexed").inc()
                    UPLOAD_LATENCY.observe(duration_ms / 1000)

                    log_event(
                        "upload_file",
                        file_id=existing_file_id,
                        filename=existing["filename"],
                        status="duplicate_reused_and_indexed",
                        auto_index=auto_index,
                        duration_ms=duration_ms,
                    )

                    return UploadResponse(
                        file_id=existing["file_id"],
                        filename=existing["filename"],
                        display_name=existing.get("display_name"),
                        group_id=existing.get("group_id"),
                        size_bytes=existing["size_bytes"],
                        indexed=True,
                        indexed_status="duplicate_reused_and_indexed",
                        indexed_message=(
                            "Duplicate file detected. Reused existing file and "
                            "built missing indexes."
                        ),
                    )
                except Exception as e:
                    duration_ms = elapsed_ms(start)
                    UPLOAD_COUNT.labels(status="duplicate_reused_index_failed").inc()
                    UPLOAD_LATENCY.observe(duration_ms / 1000)

                    log_event(
                        "upload_file",
                        file_id=existing_file_id,
                        filename=existing["filename"],
                        status="duplicate_reused_index_failed",
                        auto_index=auto_index,
                        error=str(e),
                        duration_ms=duration_ms,
                    )

                    return UploadResponse(
                        file_id=existing["file_id"],
                        filename=existing["filename"],
                        display_name=existing.get("display_name"),
                        group_id=existing.get("group_id"),
                        size_bytes=existing["size_bytes"],
                        indexed=False,
                        indexed_status="duplicate_reused_index_failed",
                        indexed_message=(
                            f"Duplicate file detected. Reused existing file, but "
                            f"indexing failed: {str(e)}"
                        ),
                    )

            duration_ms = elapsed_ms(start)
            UPLOAD_COUNT.labels(status="duplicate_reused").inc()
            UPLOAD_LATENCY.observe(duration_ms / 1000)

            log_event(
                "upload_file",
                file_id=existing_file_id,
                filename=existing["filename"],
                status="duplicate_reused",
                auto_index=auto_index,
                indexed=already_indexed,
                duration_ms=duration_ms,
            )

            return UploadResponse(
                file_id=existing["file_id"],
                filename=existing["filename"],
                display_name=existing.get("display_name"),
                group_id=existing.get("group_id"),
                size_bytes=existing["size_bytes"],
                indexed=already_indexed,
                indexed_status="duplicate_reused",
                indexed_message="Duplicate file detected. Reused existing file and index state.",
            )

        uploaded_at = datetime.now(timezone.utc).isoformat()

        self.sqlite_repo.upsert_file(
            file_id=file_id,
            filename=filename,
            display_name=filename,
            group_id=group_id,
            size_bytes=size_bytes,
            modified_time=uploaded_at,
            uploaded_at=uploaded_at,
            content_hash=content_hash,
        )

        indexed = False
        indexed_status = "skipped"
        indexed_message = "File uploaded successfully, indexing not requested."

        if auto_index:
            try:
                self.index_service.build_index_for_file(
                    file_id=file_id,
                    chunk_size=800,
                    overlap=100,
                    chunk_method="recursive",
                )
                self.retrieval_service.index_file_embeddings(file_id)

                indexed = True
                indexed_status = "indexed"
                indexed_message = "File uploaded and indexed successfully."
            except Exception as e:
                indexed = False
                indexed_status = "index_failed"
                indexed_message = f"File uploaded, but indexing failed: {str(e)}"

        duration_ms = elapsed_ms(start)
        UPLOAD_COUNT.labels(status=indexed_status).inc()
        UPLOAD_LATENCY.observe(duration_ms / 1000)

        log_event(
            "upload_file",
            file_id=file_id,
            filename=filename,
            size_bytes=size_bytes,
            auto_index=auto_index,
            indexed=indexed,
            indexed_status=indexed_status,
            duration_ms=duration_ms,
        )

        return UploadResponse(
            file_id=file_id,
            filename=filename,
            display_name=filename,
            group_id=group_id,
            size_bytes=size_bytes,
            indexed=indexed,
            indexed_status=indexed_status,
            indexed_message=indexed_message,
        )

    def list_uploaded_files(
        self,
        group_id: str | None = None,
        include_deleted: bool = False,
    ) -> FilesResponse:
        rows = self.sqlite_repo.list_files_with_status_filtered(
            group_id=group_id,
            include_deleted=include_deleted,
        )

        items = [
            FileItem(
                file_id=row["file_id"],
                filename=row["filename"],
                display_name=row.get("display_name"),
                group_id=row.get("group_id"),
                deleted_at=row.get("deleted_at"),
                size_bytes=row["size_bytes"],
                modified_time=row["modified_time"] or row["uploaded_at"],
                bm25_indexed=bool(row["bm25_indexed"]),
                vector_indexed=bool(row["vector_indexed"]),
                indexed_at=row.get("indexed_at"),
            )
            for row in rows
        ]
        return FilesResponse(files=items)

    def list_groups(self) -> GroupsResponse:
        rows = self.sqlite_repo.list_groups()
        return GroupsResponse(groups=[GroupItem(**row) for row in rows])

    def create_group(self, name: str) -> GroupItem:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("group name cannot be empty")
        if self.sqlite_repo.find_group_by_name(clean_name) is not None:
            raise ValueError("group name already exists")

        now = datetime.now(timezone.utc).isoformat()
        group = self.sqlite_repo.create_group(
            group_id=f"g_{uuid.uuid4().hex}",
            name=clean_name,
            created_at=now,
            updated_at=now,
        )
        return GroupItem(**group)

    def rename_file(self, file_id: str, display_name: str | None) -> FileManageResponse:
        existing = self.sqlite_repo.get_file(file_id=file_id, include_deleted=False)
        if existing is None:
            raise ValueError("file not found")

        now = datetime.now(timezone.utc).isoformat()
        clean_name = None if display_name is None else display_name.strip()
        if clean_name == "":
            clean_name = None

        updated = self.sqlite_repo.rename_file(
            file_id=file_id,
            display_name=clean_name,
            updated_at=now,
        )
        if updated is None:
            raise ValueError("file not found")

        return FileManageResponse(
            file_id=updated["file_id"],
            filename=updated["filename"],
            display_name=updated.get("display_name"),
            group_id=updated.get("group_id"),
            updated_at=now,
        )

    def move_file_group(self, file_id: str, group_id: str | None) -> FileManageResponse:
        existing = self.sqlite_repo.get_file(file_id=file_id, include_deleted=False)
        if existing is None:
            raise ValueError("file not found")

        if group_id is not None and self.sqlite_repo.get_group(group_id) is None:
            raise ValueError("group_id not found")

        now = datetime.now(timezone.utc).isoformat()
        updated = self.sqlite_repo.update_file_group(
            file_id=file_id,
            group_id=group_id,
            updated_at=now,
        )
        if updated is None:
            raise ValueError("file not found")

        return FileManageResponse(
            file_id=updated["file_id"],
            filename=updated["filename"],
            display_name=updated.get("display_name"),
            group_id=updated.get("group_id"),
            updated_at=now,
        )

    def delete_file(self, file_id: str) -> DeleteFileResponse:
        existing = self.sqlite_repo.get_file(file_id=file_id, include_deleted=False)
        if existing is None:
            raise ValueError("file not found")

        deleted_at = datetime.now(timezone.utc).isoformat()
        updated = self.sqlite_repo.soft_delete_file(file_id=file_id, deleted_at=deleted_at)
        if updated is None:
            raise ValueError("file not found")

        # Cleanup all retrieval/index artifacts so deleted files are never surfaced.
        self.index_service.index_repo.clear_file(file_id)
        self.retrieval_service.vector_repo.clear_file(file_id)
        self.sqlite_repo.clear_chunks_for_file(file_id)
        self.sqlite_repo.clear_index_status(file_id)

        try:
            path = self.file_repo.find_path_by_file_id(file_id)
            self.file_repo.delete_file_by_path(path)
        except Exception:
            # Metadata cleanup is authoritative; missing local file is non-fatal.
            pass

        return DeleteFileResponse(
            file_id=file_id,
            deleted=True,
            deleted_at=deleted_at,
            index_cleanup=DeleteFileCleanupItem(
                bm25_removed=True,
                vector_removed=True,
                chunks_removed=True,
                index_status_removed=True,
            ),
        )
