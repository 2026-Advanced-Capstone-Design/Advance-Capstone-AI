import json
from openai import OpenAI
from config import OPENAI_API_KEY, GPT_MINI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# Zero-shot + Constraint 프롬프트
_SYSTEM_PROMPT = """당신은 뉴스 기사를 분석하는 전문가입니다.
주어진 기사를 읽고 아래 형식으로 정확하게 응답하세요.

[제약 조건]
- 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.
- 특정 정치적 방향으로 해석하거나 유도하지 마세요.
- 사실에 근거하여 객관적으로 작성하세요.
- 한국어로 작성하세요.

[응답 형식]
{
  "one_line_summary": "기사 전체를 한 문장으로 요약 (30자 이내)",
  "key_facts": ["핵심 사실 1", "핵심 사실 2", "핵심 사실 3"],
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "topic": "기사의 주요 이슈/주제 (예: 총선, 물가, 외교)"
}"""

_USER_PROMPT_TEMPLATE = """다음 뉴스 기사를 분석해주세요.

[기사 본문]
{text}"""


def summarize(text: str) -> dict:
    """
    Zero-shot + Constraint 프롬프트로 기사 요약/키워드/주제를 생성합니다.
    """
    # 토큰 절약을 위해 본문 앞 3000자만 사용
    truncated = text[:3000] if len(text) > 3000 else text

    response = client.chat.completions.create(
        model=GPT_MINI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=truncated)},
        ],
        temperature=0.2,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)
