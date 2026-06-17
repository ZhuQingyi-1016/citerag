from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_qa_service
from app.schemas import AskDebugResponse, AskRequest, AskResponse
from app.services.qa_service import QAService

router = APIRouter(tags=["qa"])


def _qa_error_to_http_exception(error: Exception) -> HTTPException:
    if isinstance(error, HTTPException):
        return error

    msg = str(error)
    lowered = msg.lower()

    if "not indexed" in lowered or "no indexed files" in lowered:
        return HTTPException(status_code=409, detail=msg)
    if "not found" in lowered:
        return HTTPException(status_code=404, detail=msg)
    if "empty" in lowered or "invalid" in lowered or "unsupported" in lowered:
        return HTTPException(status_code=400, detail=msg)
    return HTTPException(status_code=500, detail="failed to answer question")


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, qa_service: QAService = Depends(get_qa_service)):
    try:
        return qa_service.ask(req)
    except Exception as e:
        raise _qa_error_to_http_exception(e)


@router.post("/ask_debug", response_model=AskDebugResponse)
def ask_debug(req: AskRequest, qa_service: QAService = Depends(get_qa_service)):
    try:
        return qa_service.ask_debug(req)
    except Exception as e:
        raise _qa_error_to_http_exception(e)