from typing import Any, Dict


class InMemoryIndexRepository:
    def __init__(self) -> None:
        self._index: Dict[str, Any] = {}

    def set_index(self, file_id: str, index_obj: Any) -> None:
        self._index[file_id] = index_obj

    def get_index(self, file_id: str) -> Any | None:
        return self._index.get(file_id)

    def has_index(self, file_id: str) -> bool:
        return file_id in self._index

    def list_file_ids(self) -> list[str]:
        return list(self._index.keys())

    def has_any_index(self) -> bool:
        return len(self._index) > 0

    def clear_file(self, file_id: str) -> None:
        if file_id in self._index:
            del self._index[file_id]
