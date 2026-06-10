from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import FileNotFoundError, UnsupportedFileTypeError
from app.dependencies import get_index_service, get_retrieval_service
from app.schemas import IndexAllResponse, IndexRequest, IndexResponse
from app.services.index_service import IndexService
from app.services.retrieval_service import RetrievalService

router = APIRouter(tags=["index"])


@router.post("/index/{file_id}", response_model=IndexResponse)
def index_file(
    file_id: str,
    req: IndexRequest,
    index_service: Annotated[IndexService, Depends(get_index_service)] = None,
):
    try:
        out = index_service.build_index_for_file(
            file_id=file_id,
            chunk_size=req.chunk_size,
            overlap=req.overlap,
            chunk_method=req.chunk_method,
        )
        return IndexResponse(**out)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=415, detail=str(e))


@router.post("/index_all", response_model=IndexAllResponse)
def index_all(
    req: IndexRequest,
    index_service: Annotated[IndexService, Depends(get_index_service)] = None,
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)] = None,
):
    return index_service.build_index_for_all_files(
        chunk_size=req.chunk_size,
        overlap=req.overlap,
        post_index_hook=retrieval_service.index_file_embeddings,
        chunk_method=req.chunk_method,
    )