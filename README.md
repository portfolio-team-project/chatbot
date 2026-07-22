# chatbot

Portfolio team project.

팀 포트폴리오 사이트용 FAQ 챗봇입니다. 사전에 준비된 Q&A 데이터셋에서 BM25로 관련 질문을 검색하고, 검색된 내용만 근거로 LLM(Gemma-2-2B, llama.cpp)이 답변을 생성합니다. 데이터셋에 없는 질문은 모른다고 답합니다. 라즈베리파이4(8GB) 환경에서 동작하는 것을 목표로 합니다.

## 구조

- `app.py` — FastAPI 진입점. 서버 시작 시 모델/BM25 인덱스를 한 번만 로드하고 `POST /chat`을 제공합니다.
- `rag.py` — 데이터 로드, BM25 검색, LLM 답변 생성 등 RAG 핵심 로직.
- `prompts.py` — LLM에 전달하는 프롬프트 템플릿.
- `dataset/` — Q&A 데이터셋(`.jsonl`). git에는 포함되지 않으며 팀 내부에서 별도 공유합니다.
- `models/` — GGUF 모델 파일. git에는 포함되지 않으며 팀 내부에서 별도 공유합니다.

## 준비

1. `chatbot_env`(또는 원하는 이름) 가상환경 생성 후 활성화
2. 의존성 설치
   ```
   pip install -r requirements.txt
   ```
   (오프라인 설치가 필요하면 `wheels/` 디렉터리를 팀 내부에서 공유받아 `pip install --no-index --find-links=wheels -r requirements.txt`)
3. `dataset/`, `models/`, `.env`를 팀 내부 공유 채널에서 받아 프로젝트 루트에 배치

## 실행

```
python app.py
```

또는

```
uvicorn app:app --host $HOST --port $PORT
```

## API

메인 백엔드에서 서버 대 서버로 호출하는 구조이며, 브라우저에서 직접 호출하지 않습니다.

| 항목 | 내용 |
|---|---|
| Method | POST |
| Endpoint | `/chat` |
| Content-Type | `application/json` |

**Request**

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| query | string | Y | 사용자 질문 |

**Response**

| 필드 | 타입 | 설명 |
|---|---|---|
| answer | string | 챗봇 답변. 데이터셋에서 근거를 찾지 못하면 모른다는 안내 문구가 옵니다. |
