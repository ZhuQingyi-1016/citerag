# app/parsers/base_parser.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDocumentParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> str:
        raise NotImplementedError