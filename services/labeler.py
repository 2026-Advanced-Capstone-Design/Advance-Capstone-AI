import json
from openai import OpenAI
from config import OPENAI_API_KEY, GPT_MINI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = """당신은 한국 뉴스 기사의 정치적 편향을 분류하는 전문가입니다.
기사를 읽고 아래 기준에 따라 분류하세요.

[분류 기준]
- progressive (진보): 복지 확대, 재벌 규제, 노동자 권리, 정부 개입 강조, 진보 정당·인물에 우호적 프레이밍
- conservative (보수): 시장 자유, 안보 강조, 기업 친화, 전통 가치, 보수 정당·인물에 우호적 프레이밍
- neutral (중립): 특정 방향으로 치우치지 않고 사실 위주로 보도

[제약 조건]
- 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.
- confidence는 0.0~1.0 사이 실수입니다.

[응답 형식]
{
  "label": "progressive" | "conservative" | "neutral",
  "confidence": 0.0~1.0,
  "reason": "판단 근거를 2~3문장으로 설명"
}"""

# Few-shot 예시: 진보 / 보수 / 중립 각 1개
_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """[기사]
정부가 대기업 규제를 강화하고 중소기업과 노동자 보호를 위한 새로운 법안을 발의했다.
시민단체는 "오랫동안 기다려온 개혁"이라며 환영했고, 경제계는 "기업 경쟁력을 훼손한다"며 반발했다.
전문가들은 이번 법안이 분배 구조를 개선하는 데 기여할 것으로 기대하고 있다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "label": "progressive",
            "confidence": 0.82,
            "reason": (
                "재벌 규제·노동자 보호 강화를 긍정적으로 프레이밍하고, "
                "시민단체 환영 의견을 경제계 반발보다 앞에 배치했습니다. "
                "'분배 구조 개선'이라는 표현은 진보적 경제 관점을 반영합니다."
            ),
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """[기사]
한미 연합훈련이 재개된 가운데 정부는 북핵 억지력을 강화하고 동맹을 공고히 하겠다고 밝혔다.
야당은 "긴장 고조를 부추긴다"며 비판했지만, 여당은 "안보는 양보할 수 없다"고 맞받았다.
전문가들은 현 안보 위협에 대응하기 위한 불가피한 선택이라고 평가했다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "label": "conservative",
            "confidence": 0.79,
            "reason": (
                "한미 동맹 강화와 군사 억지력을 긍정적으로 서술하고, "
                "야당 비판을 여당 반박으로 즉시 상쇄시키는 구조입니다. "
                "'안보는 양보할 수 없다'는 프레임을 전문가 평가로 뒷받침해 보수 관점을 강조합니다."
            ),
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """[기사]
통계청이 발표한 자료에 따르면 지난달 소비자물가지수가 전년 동월 대비 2.3% 상승했다.
식료품과 에너지 가격이 상승을 주도했으며, 서비스 물가는 소폭 하락했다.
정부는 물가 안정 대책을 검토 중이라고 밝혔다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "label": "neutral",
            "confidence": 0.91,
            "reason": (
                "공식 통계 수치를 그대로 인용하고 특정 정치 세력에 유리한 프레이밍이 없습니다. "
                "원인 분석도 경제 지표 중심으로 서술되어 있으며, "
                "정부 대응을 단순 사실로만 전달하고 있습니다."
            ),
        }, ensure_ascii=False),
    },
]

_USER_PROMPT_TEMPLATE = """[기사]
{text}"""


def label(text: str) -> dict:
    """
    Few-shot 프롬프트로 기사의 정치적 편향 레이블을 분류합니다.
    confidence < 0.5이면 label을 'uncertain'으로 덮어씁니다.

    반환: {"label": str, "confidence": float, "reason": str}
    """
    truncated = text[:2000] if len(text) > 2000 else text

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *_FEW_SHOT_EXAMPLES,
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=truncated)},
    ]

    response = client.chat.completions.create(
        model=GPT_MINI_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=256,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    if result.get("confidence", 0) < 0.5:
        result["label"] = "uncertain"

    return result
