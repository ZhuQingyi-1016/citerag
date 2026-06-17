import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.core.exceptions import DuplicateFileIdError, FileNotFoundError
from app.settings import STORAGE_DIR

_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._()\-\u4e00-\u9fff]+")


def _safe_upload_filename(filename: str | None) -> str:
    raw_name = Path(filename or "uploaded_document").name.strip()
    safe_name = _SAFE_FILENAME_PATTERN.sub("_", raw_name).strip("._ ")
    if not safe_name:
        safe_name = "uploaded_document"
    return safe_name[:180]

class FileRepository:
    def ensure_storage_dir(self) -> None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file: UploadFile) -> tuple[str, str, int]:
        self.ensure_storage_dir()

        safe_name = _safe_upload_filename(file.filename)
        file_id = uuid.uuid4().hex
        out_path = STORAGE_DIR / f"{file_id}__{safe_name}"
        tmp_path = STORAGE_DIR / f".{file_id}__{safe_name}.tmp"

        size = 0
        try:
            file.file.seek(0)
            with tmp_path.open("wb") as f:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    f.write(chunk)
            os.replace(tmp_path, out_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if out_path.exists():
                out_path.unlink()
            raise

        return file_id, safe_name, size

    def list_files(self) -> list[dict]:
        self.ensure_storage_dir()

        items = []
        for p in STORAGE_DIR.iterdir():
            if not p.is_file():
                continue
            if p.name.startswith(".") or p.name.endswith(".tmp"):
                continue

            name = p.name
            if "__" in name:
                file_id, filename = name.split("__", 1)
            else:
                file_id, filename = "unknown", name

            stat = p.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            items.append(
                {
                    "file_id": file_id,
                    "filename": filename,
                    "size_bytes": stat.st_size,
                    "modified_time": mtime,
                }
            )

        items.sort(key=lambda x: x["modified_time"], reverse=True)
        return items

    def find_path_by_file_id(self, file_id: str) -> Path:
        self.ensure_storage_dir()
        matches = list(STORAGE_DIR.glob(f"{file_id}__*"))

        if not matches:
            raise FileNotFoundError("file_id not found")
        if len(matches) > 1:
            raise DuplicateFileIdError("multiple files match this file_id")

        return matches[0]

    def compute_file_hash(self, path: Path) -> str:
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256")
        return digest.hexdigest()

    def delete_file_by_path(self, path: Path) -> None:
        if path.exists():
            path.unlink()