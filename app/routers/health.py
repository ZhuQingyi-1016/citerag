from datetime import datetime, timezone

from fastapi import APIRouter

from app.schemas import EchoRequest, EchoResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/echo", response_model=EchoResponse)
def echo(req: EchoRequest):
    if not req.message.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="message cannot be empty")

    return EchoResponse(
        message=req.message,
        server_time=datetime.now(timezone.utc).isoformat(),
    )