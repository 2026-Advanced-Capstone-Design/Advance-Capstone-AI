import json
from openai import OpenAI
from config import OPENAI_API_KEY, GPT_MINI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = """당신은 뉴스 기사를 분석하는 전문가입니다.
주어진 기사를 읽고 아래 형식으로 정확하게 응답하세요.

[제약 조건]
- 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.
- 특정 정치적 방향으로 해석하거나 유도하지 마세요.
- 사실에 근거하여 객관적으로 작성하세요.
- 한국어로 작성하세요.

[응답 형식]
{
  "key_facts": ["핵심 사실 1 (1~2문장)", "핵심 사실 2 (1~2문장)", "핵심 사실 3 (1~2문장)"],
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "topic": "기사의 주요 이슈/주제 (예: 총선, 물가, 외교)"
}

[key_facts 작성 기준]
- 기사에서 가장 중요한 사실 2~3개를 각각 1~2문장으로 요약
- 광고성/홍보성 내용 제외
- 사실 기반으로 객관적으로 작성"""

_USER_PROMPT_TEMPLATE = """다음 뉴스 기사를 분석해주세요.

[기사 본문]
{text}"""


def summarize(text: str) -> dict:
    truncated = text[:5000] if len(text) > 5000 else text

    response = client.chat.completions.create(
        model=GPT_MINI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=truncated)},
        ],
        temperature=0.2,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)
