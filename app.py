import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from llama_cpp import Llama

from rag import CHAT_FORMAT, DATASET_PATH, MODEL_PATH, Retriever, load_dataset
from routers.chat import router as chat_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = os.environ["HOST"]
PORT = int(os.environ["PORT"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application starting up")
    entries = load_dataset(DATASET_PATH)
    app.state.retriever = Retriever(entries)
    app.state.llm = Llama(model_path=MODEL_PATH, chat_format=CHAT_FORMAT, verbose=False)
    logger.info("FastAPI application startup complete")
    yield
    logger.info("FastAPI application shutting down")


app = FastAPI(
    lifespan=lifespan,
    title="챗봇 1.0 API",
    summary="포트폴리오 사이트 FAQ 챗봇 API",
    description=(
        "팀 포트폴리오 사이트의 Q&A 데이터셋을 BM25로 검색하고, "
        "검색된 내용만 근거로 LLM이 답변을 생성하는 RAG 챗봇 API입니다. "
        "데이터셋에서 근거를 찾지 못하면 모른다고 답합니다."
    ),
    version="1.0.0",
)

app.include_router(chat_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=HOST, port=PORT)
