from __future__ import annotations

import sqlite3
from typing import Any

from app.settings import STORAGE_DIR


class SQLiteRepository:
    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            db_path = str(STORAGE_DIR / "citerag.db")

        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_time TEXT,
                    uploaded_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    start_pos INTEGER NOT NULL,
                    end_pos INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    UNIQUE(file_id, chunk_id)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_status (
                    file_id TEXT PRIMARY KEY,
                    bm25_indexed INTEGER NOT NULL DEFAULT 0,
                    vector_indexed INTEGER NOT NULL DEFAULT 0,
                    indexed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS groups (
                    group_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.commit()

        self._ensure_column("files", "content_hash", "content_hash TEXT")
        self._ensure_column("files", "display_name", "display_name TEXT")
        self._ensure_column("files", "group_id", "group_id TEXT")
        self._ensure_column("files", "deleted_at", "deleted_at TEXT")

        with self._connect() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_files_content_hash")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_files_group_id
                ON files(group_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_files_deleted_at
                ON files(deleted_at)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_files_content_hash
                ON files(content_hash)
                WHERE content_hash IS NOT NULL AND deleted_at IS NULL
                """
            )
            conn.execute(
                """
                UPDATE files
                SET display_name = filename
                WHERE display_name IS NULL
                """
            )
            conn.commit()


    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        with self._connect() as conn:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            existing = {row["name"] for row in rows}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
                conn.commit()

    # ---------- files ----------
    def upsert_file(
        self,
        file_id: str,
        filename: str,
        size_bytes: int,
        modified_time: str | None,
        uploaded_at: str,
        content_hash: str | None = None,
        display_name: str | None = None,
        group_id: str | None = None,
        deleted_at: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (
                    file_id,
                    filename,
                    size_bytes,
                    modified_time,
                    uploaded_at,
                    content_hash,
                    display_name,
                    group_id,
                    deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    filename = excluded.filename,
                    size_bytes = excluded.size_bytes,
                    modified_time = excluded.modified_time,
                    uploaded_at = excluded.uploaded_at,
                    content_hash = excluded.content_hash,
                    display_name = excluded.display_name,
                    group_id = excluded.group_id,
                    deleted_at = excluded.deleted_at
                """,
                (
                    file_id,
                    filename,
                    size_bytes,
                    modified_time,
                    uploaded_at,
                    content_hash,
                    display_name,
                    group_id,
                    deleted_at,
                ),
            )
            conn.commit()

    def list_files(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    file_id,
                    filename,
                    display_name,
                    group_id,
                    deleted_at,
                    size_bytes,
                    modified_time,
                    uploaded_at,
                    content_hash
                FROM files
                ORDER BY uploaded_at DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    # ---------- chunks ----------
    def replace_chunks(self, file_id: str, chunks: list[dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            conn.executemany(
                """
                INSERT INTO chunks (file_id, chunk_id, start_pos, end_pos, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        file_id,
                        chunk["chunk_id"],
                        chunk["start"],
                        chunk["end"],
                        chunk["text"],
                    )
                    for chunk in chunks
                ],
            )
            conn.commit()

    def list_chunks_for_file(self, file_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT file_id, chunk_id, start_pos, end_pos, text
                FROM chunks
                WHERE file_id = ?
                ORDER BY chunk_id ASC
                """,
                (file_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # ---------- index status ----------
    def upsert_index_status(
        self,
        file_id: str,
        bm25_indexed: bool,
        vector_indexed: bool,
        indexed_at: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_status (file_id, bm25_indexed, vector_indexed, indexed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    bm25_indexed = excluded.bm25_indexed,
                    vector_indexed = excluded.vector_indexed,
                    indexed_at = excluded.indexed_at
                """,
                (file_id, int(bm25_indexed), int(vector_indexed), indexed_at),
            )
            conn.commit()

    def get_index_status(self, file_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT file_id, bm25_indexed, vector_indexed, indexed_at
                FROM index_status
                WHERE file_id = ?
                """,
                (file_id,),
            ).fetchone()
            return dict(row) if row else None
        
    def list_files_with_status(self) -> list[dict[str, Any]]:
        return self.list_files_with_status_filtered(group_id=None, include_deleted=False)

    def list_files_with_status_filtered(
        self,
        group_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        where_clauses: list[str] = []
        params: list[Any] = []

        if not include_deleted:
            where_clauses.append("f.deleted_at IS NULL")
        if group_id is not None:
            where_clauses.append("f.group_id = ?")
            params.append(group_id)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    f.file_id,
                    f.filename,
                    f.display_name,
                    f.group_id,
                    f.deleted_at,
                    f.size_bytes,
                    f.modified_time,
                    f.uploaded_at,
                    COALESCE(s.bm25_indexed, 0) AS bm25_indexed,
                    COALESCE(s.vector_indexed, 0) AS vector_indexed,
                    s.indexed_at AS indexed_at
                FROM files f
                LEFT JOIN index_status s
                ON f.file_id = s.file_id
                {where_sql}
                ORDER BY f.uploaded_at DESC
                """
                ,
                params,
            ).fetchall()
            return [dict(row) for row in rows]
    
    def list_indexed_files(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    f.file_id,
                    f.filename,
                    f.size_bytes,
                    f.modified_time,
                    f.uploaded_at,
                    COALESCE(s.bm25_indexed, 0) AS bm25_indexed,
                    COALESCE(s.vector_indexed, 0) AS vector_indexed,
                    s.indexed_at
                FROM files f
                JOIN index_status s
                ON f.file_id = s.file_id
                WHERE (s.bm25_indexed = 1 OR s.vector_indexed = 1)
                  AND f.deleted_at IS NULL
                ORDER BY f.uploaded_at DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]


    def clear_index_status(self, file_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM index_status
                WHERE file_id = ?
                """,
                (file_id,),
            )
            conn.commit()

    def clear_chunks_for_file(self, file_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM chunks
                WHERE file_id = ?
                """,
                (file_id,),
            )
            conn.commit()

    def find_file_by_content_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    file_id,
                    filename,
                    display_name,
                    group_id,
                    deleted_at,
                    size_bytes,
                    modified_time,
                    uploaded_at,
                    content_hash
                FROM files
                WHERE content_hash = ?
                  AND deleted_at IS NULL
                """,
                (content_hash,),
            ).fetchone()
            return dict(row) if row else None

    def get_file(self, file_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        with self._connect() as conn:
            if include_deleted:
                row = conn.execute(
                    """
                    SELECT
                        file_id,
                        filename,
                        display_name,
                        group_id,
                        deleted_at,
                        size_bytes,
                        modified_time,
                        uploaded_at,
                        content_hash
                    FROM files
                    WHERE file_id = ?
                    """,
                    (file_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT
                        file_id,
                        filename,
                        display_name,
                        group_id,
                        deleted_at,
                        size_bytes,
                        modified_time,
                        uploaded_at,
                        content_hash
                    FROM files
                    WHERE file_id = ?
                      AND deleted_at IS NULL
                    """,
                    (file_id,),
                ).fetchone()
            return dict(row) if row else None

    def rename_file(self, file_id: str, display_name: str | None, updated_at: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE files
                SET display_name = ?,
                    modified_time = ?
                WHERE file_id = ?
                  AND deleted_at IS NULL
                """,
                (display_name, updated_at, file_id),
            )
            conn.commit()
        return self.get_file(file_id=file_id, include_deleted=False)

    def update_file_group(self, file_id: str, group_id: str | None, updated_at: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE files
                SET group_id = ?,
                    modified_time = ?
                WHERE file_id = ?
                  AND deleted_at IS NULL
                """,
                (group_id, updated_at, file_id),
            )
            conn.commit()
        return self.get_file(file_id=file_id, include_deleted=False)

    def soft_delete_file(self, file_id: str, deleted_at: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE files
                SET deleted_at = ?,
                    modified_time = ?
                WHERE file_id = ?
                  AND deleted_at IS NULL
                """,
                (deleted_at, deleted_at, file_id),
            )
            conn.commit()
        return self.get_file(file_id=file_id, include_deleted=True)

    # ---------- groups ----------
    def list_groups(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT group_id, name, created_at, updated_at
                FROM groups
                ORDER BY name ASC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def get_group(self, group_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT group_id, name, created_at, updated_at
                FROM groups
                WHERE group_id = ?
                """,
                (group_id,),
            ).fetchone()
            return dict(row) if row else None

    def create_group(
        self,
        group_id: str,
        name: str,
        created_at: str,
        updated_at: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO groups (group_id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (group_id, name, created_at, updated_at),
            )
            conn.commit()
        out = self.get_group(group_id)
        assert out is not None
        return out

    def find_group_by_name(self, name: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT group_id, name, created_at, updated_at
                FROM groups
                WHERE name = ?
                """,
                (name,),
            ).fetchone()
            return dict(row) if row else None
