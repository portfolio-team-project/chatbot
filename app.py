import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from llama_cpp import Llama
from pydantic import BaseModel

from rag import CHAT_FORMAT, DATASET_PATH, MODEL_PATH, Retriever, answer, load_dataset

HOST = os.environ["HOST"]
PORT = int(os.environ["PORT"])

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    entries = load_dataset(DATASET_PATH)
    state["retriever"] = Retriever(entries)
    state["llm"] = Llama(model_path=MODEL_PATH, chat_format=CHAT_FORMAT, verbose=False)
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = answer(state["llm"], state["retriever"], req.query)
    return ChatResponse(answer=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=HOST, port=PORT)
