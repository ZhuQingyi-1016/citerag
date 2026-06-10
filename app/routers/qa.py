from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_qa_service
from app.schemas import AskDebugResponse, AskRequest, AskResponse
from app.services.qa_service import QAService

router = APIRouter(tags=["qa"])


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, qa_service: QAService = Depends(get_qa_service)):
    try:
        return qa_service.ask(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ask_debug", response_model=AskDebugResponse)
def ask_debug(req: AskRequest, qa_service: QAService = Depends(get_qa_service)):
    try:
        return qa_service.ask_debug(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))