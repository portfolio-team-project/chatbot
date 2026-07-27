import json
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
from llama_cpp import Llama
from rank_bm25 import BM25Okapi

from prompts import build_system_prompt

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

DATASET_PATH = os.environ["DATASET_PATH"]
MODEL_PATH = os.environ["MODEL_PATH"]
CHAT_FORMAT = os.environ.get("CHAT_FORMAT", "gemma")
TOP_K = int(os.environ.get("RAG_TOP_K", 3))
SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", 1.0))

_TOKEN_RE = re.compile(r"[가-힣a-zA-Z0-9]+")

# llama.cpp는 하나의 Llama 인스턴스에 동시 요청이 들어오는 걸 안전하게 처리하지 못해서
# (컨테이너가 죽거나 연결이 끊기는 원인이 됨), 요청을 한 번에 하나씩만 처리하도록 직렬화한다.
_llm_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in _TOKEN_RE.findall(text.lower()):
        if len(word) <= 2:
            tokens.append(word)
            continue
        tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return tokens


def load_dataset(path: str) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class Retriever:
    def __init__(self, entries: list[dict]):
        self.entries = entries
        corpus = [tokenize(e["instruction"]) for e in entries]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[dict, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.entries, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def build_context(matches: list[tuple[dict, float]]) -> str:
    lines = []
    for entry, _ in matches:
        lines.append(f"Q: {entry['instruction']}\nA: {entry['output']}")
    return "\n\n".join(lines)


UNKNOWN_ANSWER = "죄송해요, 해당 내용은 갖고 있는 정보에 없어요."


def answer(llm: Llama, retriever: Retriever, query: str) -> str:
    matches = retriever.search(query)
    matches = [(e, s) for e, s in matches if s >= SCORE_THRESHOLD]

    if not matches:
        return UNKNOWN_ANSWER

    context = build_context(matches)
    prompt = build_system_prompt(context) + f"\n\n[사용자 질문]\n{query}"

    with _llm_lock:
        response = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
    result = response["choices"][0]["message"]["content"]

    # 작은 모델은 거절 문구를 그대로 재현하지 못하고 "~없어요" 대신 "~없어"처럼
    # 반말로 바꾸거나 미묘하게 다르게 낼 때가 있어서, "답변 전체가 거절"인 경우만
    # 정해진 문구로 통일한다. 일부만 거절하는 부분 답변(예: "A는 ~예요. B는 정보에 없어요.")까지
    # 뭉개면 안 되므로, 응답이 짧게 거절 문구 하나로만 이루어진 경우만 대상으로 한다.
    is_short_refusal = len(result) <= 40 and ("정보에 없" in result or "갖고 있지 않" in result)
    if is_short_refusal:
        return UNKNOWN_ANSWER
    return result
