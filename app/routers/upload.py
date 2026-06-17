from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.dependencies import get_file_service
from app.schemas import (
    CreateGroupRequest,
    DeleteFileResponse,
    FileManageResponse,
    FilesResponse,
    GroupItem,
    GroupsResponse,
    RenameFileRequest,
    UpdateFileGroupRequest,
    UploadResponse,
)
from app.services.file_service import FileService

router = APIRouter(tags=["files"])


def _value_error_to_http_exception(error: ValueError) -> HTTPException:
    msg = str(error)
    lowered = msg.lower()

    if "not found" in lowered:
        return HTTPException(status_code=404, detail=msg)
    if "already exists" in lowered:
        return HTTPException(status_code=409, detail=msg)
    if "empty" in lowered or "unsupported" in lowered or "invalid" in lowered:
        return HTTPException(status_code=400, detail=msg)
    return HTTPException(status_code=400, detail=msg)


@router.post("/upload", response_model=UploadResponse)
def upload(
    file: UploadFile = File(...),
    auto_index: bool = Query(
        False,
        description="Whether to build index immediately after upload",
    ),
    group_id: str | None = Query(
        None,
        description="Optional group id for uploaded file",
    ),
    file_service: Annotated[FileService, Depends(get_file_service)] = None,
):
    try:
        return file_service.upload_file(file=file, auto_index=auto_index, group_id=group_id)
    except ValueError as e:
        raise _value_error_to_http_exception(e)


@router.get("/files", response_model=FilesResponse)
def files(
    group_id: str | None = Query(None, description="Optional group filter"),
    include_deleted: bool = Query(False, description="Whether to include deleted files"),
    file_service: Annotated[FileService, Depends(get_file_service)] = None,
):
    return file_service.list_uploaded_files(
        group_id=group_id,
        include_deleted=include_deleted,
    )


@router.get("/groups", response_model=GroupsResponse)
def groups(
    file_service: Annotated[FileService, Depends(get_file_service)] = None,
):
    return file_service.list_groups()


@router.post("/groups", response_model=GroupItem)
def create_group(
    req: CreateGroupRequest,
    file_service: Annotated[FileService, Depends(get_file_service)] = None,
):
    try:
        return file_service.create_group(name=req.name)
    except ValueError as e:
        raise _value_error_to_http_exception(e)


@router.patch("/files/{file_id}/rename", response_model=FileManageResponse)
def rename_file(
    file_id: str,
    req: RenameFileRequest,
    file_service: Annotated[FileService, Depends(get_file_service)] = None,
):
    try:
        return file_service.rename_file(file_id=file_id, display_name=req.display_name)
    except ValueError as e:
        raise _value_error_to_http_exception(e)


@router.patch("/files/{file_id}/group", response_model=FileManageResponse)
def update_file_group(
    file_id: str,
    req: UpdateFileGroupRequest,
    file_service: Annotated[FileService, Depends(get_file_service)] = None,
):
    try:
        return file_service.move_file_group(file_id=file_id, group_id=req.group_id)
    except ValueError as e:
        raise _value_error_to_http_exception(e)


@router.delete("/files/{file_id}", response_model=DeleteFileResponse)
def delete_file(
    file_id: str,
    file_service: Annotated[FileService, Depends(get_file_service)] = None,
):
    try:
        return file_service.delete_file(file_id=file_id)
    except ValueError as e:
        raise _value_error_to_http_exception(e)
