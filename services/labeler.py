import json
from openai import OpenAI
from config import OPENAI_API_KEY, GPT_MINI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = """당신은 한국 뉴스 기사의 편향을 분석하는 전문가입니다.
기사를 주제별 섹션으로 나누고, 각 섹션을 반드시 아래 3단계 Chain-of-Thought로 분석하세요.

[섹션 분류 기준]
- 정치: 선거, 정당, 국회, 대통령, 장관, 정부 정책, 탄핵, 정치인
- 경제: 경제정책, 기업, 금융, 부동산, 주식, 무역, 세금, 고용
- 사회: 복지, 교육, 환경, 범죄, 의료, 노동, 젠더, 인권
- 국제: 외교, 안보, 전쟁, 국제관계, 북한, 미중관계
- 기타: 문화, 스포츠, 연예, 과학기술, 날씨 등 위 항목에 해당하지 않는 내용

[섹션별 편향 레이블]
▸ 정치 섹션: progressive | conservative | neutral
  - progressive : 민주당·이재명·조국 우호 / 검찰개혁·적폐청산·촛불·복지 강조 /
                  극좌 주장 비판 없이 인용 / 보수 정권을 내란·독재·친일로 표현
  - conservative: 국민의힘·윤석열·한동훈 우호 / 윤어게인·태극기·자유민주주의 수호 강조 /
                  한미동맹·국방 강화 일방 지지 / 극우 주장 비판 없이 인용 /
                  진보 진영을 종북·친중·빨갱이로 표현

▸ 경제 섹션: pro_labor | pro_business | neutral
  - pro_labor   : 노동자 권리·최저임금 인상·재벌 규제·불평등 해소·노조 활동 긍정 강조
  - pro_business: 규제 완화·감세·기업 경쟁력·시장 자유·친기업 정책 강조

▸ 사회 섹션: progressive | conservative | neutral
  - progressive : 복지 확대·젠더 평등·소수자 권리·환경 규제·다양성 강조
  - conservative: 전통 가치·가족 제도·종교적 관점·법질서·반이민 강조

▸ 국제 섹션: pro_us | pro_china | balanced | neutral
  - pro_us      : 한미동맹·대중 강경·북한 압박 일방 강조
  - pro_china   : 대중 협력·미국 비판·대북 유화 강조
  - balanced    : 미중 양측을 균형 있게 서술

▸ 기타 섹션: neutral 고정

[3단계 Chain-of-Thought 분석 방법]
각 섹션에 대해 레이블을 결정하기 전에 반드시 아래 순서로 분석하세요.

Step 1 (편향 표현 수집): 해당 섹션에서 특정 진영에 유리하거나 감정적·편향적인
  단어·표현·프레이밍을 구체적으로 나열하세요. 없으면 빈 배열.

Step 2 (균형 표현 확인): 반대 관점의 입장, 중립적 서술, 양측 의견이 균형 있게
  제시된 표현을 나열하세요. 없으면 빈 배열.

Step 3 (종합 판단): Step 1과 Step 2를 비교하여 편향 방향과 정도를 판단하고
  bias_label과 confidence를 결정하세요. Step 1이 많고 Step 2가 적을수록 confidence↑.

[주의 사항]
- 섹션은 1개 이상이어야 하며, 기사 전체 내용을 빠짐없이 분류하세요.
- 특정 인물·정당이 등장해도 비판적·중립적으로 다루면 neutral로 분류하세요.
- overall_label은 섹션들의 편향을 종합해 가장 지배적인 방향으로 결정하세요.
- 반드시 JSON만 출력하세요. 다른 텍스트는 절대 포함하지 마세요.

[응답 형식]
{
  "sections": [
    {
      "topic": "정치 | 경제 | 사회 | 국제 | 기타",
      "step1_biased_expressions": ["편향 표현1", "편향 표현2"],
      "step2_neutral_expressions": ["균형 표현1"],
      "step3_judgment": "Step1·Step2 비교 후 편향 방향과 정도 서술 (1~2문장)",
      "bias_label": "레이블",
      "confidence": 0.0~1.0,
      "reason": "최종 판단 요약 (1문장)"
    }
  ],
  "overall_label": "progressive | conservative | pro_labor | pro_business | pro_us | pro_china | balanced | neutral",
  "overall_confidence": 0.0~1.0,
  "overall_reason": "전체 기사 편향 방향 2~3문장"
}"""

# Few-shot 예시 (CoT 형식)
_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """[기사]
국민의힘이 이재명 대표의 사법 리스크를 집중 공략하며 야당을 압박하고 있다.
당 대변인은 "법 앞에 예외는 없다"며 수사 정당성을 강조했고, 윤어게인 집회 참가자들은
태극기를 흔들며 자유민주주의 수호를 외쳤다.

한편 정부는 반도체 기업에 대한 세제 혜택을 확대하고 규제를 완화하는 방향으로
경제 정책을 추진하겠다고 밝혔다. 재계는 이를 환영했으나 노동계는 "노동자를 외면한
친기업 정책"이라며 강하게 반발했다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "sections": [
                {
                    "topic": "정치",
                    "step1_biased_expressions": [
                        "'윤어게인' — 보수 집회 구호를 비판 없이 인용",
                        "'태극기를 흔들며 자유민주주의 수호' — 보수 프레이밍 그대로 전달",
                        "'사법 리스크 집중 공략' — 여당 공세를 긍정적 어조로 서술",
                    ],
                    "step2_neutral_expressions": [],
                    "step3_judgment": (
                        "Step1에서 보수 진영 핵심 표현이 3개 확인되었고 "
                        "Step2에서 야당 입장이나 반론이 전혀 없다. "
                        "여당 관점만 일방적으로 전달하는 conservative 성향이 강하다."
                    ),
                    "bias_label": "conservative",
                    "confidence": 0.87,
                    "reason": "보수 진영 표현만 비판 없이 전달하고 반론이 전무해 보수 편향이 뚜렷하다.",
                },
                {
                    "topic": "경제",
                    "step1_biased_expressions": [
                        "'재계는 이를 환영' — 친기업 반응을 먼저, 긍정적으로 서술",
                        "세제 혜택·규제 완화를 정책 방향으로 확정적으로 제시",
                    ],
                    "step2_neutral_expressions": [
                        "'노동계는 강하게 반발' — 반대 입장도 언급",
                    ],
                    "step3_judgment": (
                        "Step1에서 친기업 프레이밍 2개, Step2에서 노동계 반발 1개가 확인된다. "
                        "반발이 언급됐으나 재계 환영이 앞에 배치되어 pro_business 기조가 우세하다."
                    ),
                    "bias_label": "pro_business",
                    "confidence": 0.78,
                    "reason": "친기업 서술이 우선 배치되고 노동계 반발은 부가적으로 언급되어 친기업 관점이 우세하다.",
                },
            ],
            "overall_label": "conservative",
            "overall_confidence": 0.83,
            "overall_reason": (
                "정치 섹션에서 보수 진영 프레이밍이 강하게 나타나고, "
                "경제 섹션도 친기업 기조로 보수적 경제관을 반영합니다. "
                "전반적으로 보수 성향 독자층을 겨냥한 보도 방식입니다."
            ),
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """[기사]
통계청이 발표한 자료에 따르면 지난달 소비자물가지수가 전년 동월 대비 2.3% 상승했다.
식료품과 에너지 가격이 상승을 주도했으며, 서비스 물가는 소폭 하락했다.

외교부는 한미 정상회담 결과를 발표하며 양국이 반도체·AI 분야 협력을 강화하기로 했다고 밝혔다.
중국 측은 이에 대해 우려 입장을 표명했으며, 정부는 한중 관계도 안정적으로 관리하겠다고 했다.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "sections": [
                {
                    "topic": "경제",
                    "step1_biased_expressions": [],
                    "step2_neutral_expressions": [
                        "통계청 공식 수치 인용 — 사실 기반 서술",
                        "상승 품목과 하락 품목 모두 언급 — 균형 서술",
                    ],
                    "step3_judgment": (
                        "Step1에서 편향 표현이 없고 Step2에서 공식 통계와 균형 서술이 확인된다. "
                        "특정 세력에 유리한 프레이밍이 전혀 없으므로 neutral이 적절하다."
                    ),
                    "bias_label": "neutral",
                    "confidence": 0.92,
                    "reason": "통계 수치 중심의 사실 보도로 편향 표현이 없다.",
                },
                {
                    "topic": "국제",
                    "step1_biased_expressions": [
                        "'반도체·AI 협력 강화' — 한미 동맹을 긍정적으로 서술",
                    ],
                    "step2_neutral_expressions": [
                        "'중국 측 우려 입장 표명' — 중국 반응도 균형 있게 전달",
                        "'한중 관계도 안정적으로 관리' — 어느 한쪽 편들지 않음",
                    ],
                    "step3_judgment": (
                        "Step1에서 친미 표현 1개가 있으나 Step2에서 중국 입장과 한중 관계 관리가 균형 있게 서술된다. "
                        "미중 어느 쪽도 일방적으로 지지하지 않아 balanced가 적절하다."
                    ),
                    "bias_label": "balanced",
                    "confidence": 0.81,
                    "reason": "한미 협력 언급과 함께 중국 입장도 균형 있게 전달해 어느 한쪽에 치우치지 않는다.",
                },
            ],
            "overall_label": "neutral",
            "overall_confidence": 0.87,
            "overall_reason": (
                "경제 섹션은 통계 기반 사실 보도이고, 국제 섹션도 미중 양측을 균형 있게 다루고 있습니다. "
                "전체적으로 특정 정치 성향으로 치우치지 않은 중립적 보도입니다."
            ),
        }, ensure_ascii=False),
    },
]

_USER_PROMPT_TEMPLATE = """[기사]
{text}"""


def label(text: str) -> dict:
    """
    CoT 3단계 (편향 표현 수집 → 균형 표현 확인 → 종합 판단) + Few-shot으로
    섹션별 편향을 분석합니다.

    반환:
      sections         - topic / step1~3 CoT / bias_label / confidence / reason
      overall_label    - 전체 기사 편향 레이블
      overall_confidence
      overall_reason
    """
    truncated = text[:2500] if len(text) > 2500 else text

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *_FEW_SHOT_EXAMPLES,
        {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=truncated)},
    ]

    response = client.chat.completions.create(
        model=GPT_MINI_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    if result.get("overall_confidence", 0) < 0.5:
        result["overall_label"] = "uncertain"

    return result
