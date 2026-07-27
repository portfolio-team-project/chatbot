import json
import os

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag import answer, answer_stream
from routers.token_header import verify_token_header

APP_ENV = os.environ.get("APP_ENV", "prod")

_dependencies = [Depends(verify_token_header)] if APP_ENV == "prod" else []

router = APIRouter(prefix="/api", tags=["API"], dependencies=_dependencies)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    result = answer(request.app.state.llm, request.app.state.retriever, req.query)
    return ChatResponse(answer=result)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    def event_generator():
        for piece in answer_stream(request.app.state.llm, request.app.state.retriever, req.query):
            yield f"data: {json.dumps(piece, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: null\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
