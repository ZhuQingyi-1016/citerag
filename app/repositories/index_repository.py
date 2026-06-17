from threading import RLock
from typing import Any


class InMemoryIndexRepository:
    def __init__(self) -> None:
        self._index: dict[str, Any] = {}
        self._lock = RLock()

    def set_index(self, file_id: str, index_obj: Any) -> None:
        with self._lock:
            self._index[file_id] = index_obj

    def get_index(self, file_id: str) -> Any | None:
        with self._lock:
            return self._index.get(file_id)

    def has_index(self, file_id: str) -> bool:
        with self._lock:
            return file_id in self._index

    def list_file_ids(self) -> list[str]:
        with self._lock:
            return list(self._index.keys())

    def has_any_index(self) -> bool:
        with self._lock:
            return bool(self._index)

    def clear_file(self, file_id: str) -> None:
        with self._lock:
            self._index.pop(file_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._index.clear()
