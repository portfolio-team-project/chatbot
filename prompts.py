SYSTEM_PROMPT_TEMPLATE = (
    "너는 포트폴리오 사이트 안내 챗봇이야. 아래 참고 자료에 있는 내용만 근거로 답변해. "
    "참고 자료로 답할 수 없는 질문이면 모른다고 답해.\n\n"
    "[참고 자료]\n{context}"
)


def build_system_prompt(context: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(context=context)
