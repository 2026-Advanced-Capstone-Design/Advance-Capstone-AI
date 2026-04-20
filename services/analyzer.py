import json
from openai import OpenAI
from config import OPENAI_API_KEY, GPT_MINI_MODEL, GPT_STRONG_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Step 1: Generated Knowledge (gpt-4o-mini) ─────────────────────────────
_KNOWLEDGE_SYSTEM = """당신은 한국 시사 전문가입니다.
주어진 주제에 대해 진보·보수 양측 관점의 배경 지식을 간략히 서술하세요.
200자 이내로 작성하고, 어느 한쪽 편을 들지 마세요."""

_KNOWLEDGE_USER = "주제: {topic}\n키워드: {keywords}"

# ── Step 2: CoT 편향 분석 (gpt-4o) — Prompt Merging ─────────────────────
_ANALYSIS_SYSTEM = """당신은 뉴스 편향 분석 전문가입니다.
아래 배경 지식을 참고하여 기사를 4단계 Chain-of-Thought로 분석하세요.

[배경 지식]
{background}

[분석 4단계]
1. 어휘 선택 (vocab): 감정적·편향적 단어 사용 여부
2. 프레이밍 (framing): 특정 관점을 부각하거나 약화시키는 구조 여부
3. 인용 편향 (citation): 인용 출처가 한쪽에 치우쳤는지 여부
4. 정보 생략 (omission): 반대 관점의 중요 사실을 누락했는지 여부

[제약 조건]
- 각 단계별 score는 0.0(편향 없음)~1.0(강한 편향) 실수
- bias_direction: "left" | "center" | "right"
- spectrum_label: "진보" | "중립" | "보수"
- 반드시 JSON만 출력하세요.

[응답 형식]
{{
  "step1_vocab":    {{"score": 0.0, "reason": "..."}},
  "step2_framing":  {{"score": 0.0, "reason": "..."}},
  "step3_citation": {{"score": 0.0, "reason": "..."}},
  "step4_omission": {{"score": 0.0, "reason": "..."}},
  "bias_direction": "center",
  "spectrum_label": "중립"
}}"""

_ANALYSIS_USER = "[기사]\n{text}"


def _generate_background(topic: str, keywords: list[str]) -> str:
    """Model Tiering: gpt-4o-mini로 배경 지식 생성 (Generated Knowledge)"""
    kw_str = ", ".join(keywords) if keywords else topic
    response = client.chat.completions.create(
        model=GPT_MINI_MODEL,
        messages=[
            {"role": "system", "content": _KNOWLEDGE_SYSTEM},
            {"role": "user", "content": _KNOWLEDGE_USER.format(topic=topic, keywords=kw_str)},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def _run_cot_analysis(text: str, background: str) -> dict:
    """Model Tiering: gpt-4o로 CoT 4단계 분석 (Prompt Merging: 배경 지식 주입 + 분석 통합)"""
    system = _ANALYSIS_SYSTEM.format(background=background)
    response = client.chat.completions.create(
        model=GPT_STRONG_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _ANALYSIS_USER.format(text=text[:2500])},
        ],
        temperature=0.1,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def _compute_scores(cot: dict, label: str) -> dict:
    """CoT 결과 → 4대 지표 + 종합 점수 계산"""
    vocab    = cot.get("step1_vocab",    {}).get("score", 0.5)
    framing  = cot.get("step2_framing",  {}).get("score", 0.5)
    citation = cot.get("step3_citation", {}).get("score", 0.5)
    omission = cot.get("step4_omission", {}).get("score", 0.5)

    emotion_neutrality = round(1.0 - vocab,    3)
    fact_ratio         = round(1.0 - citation, 3)
    source_balance     = round(1.0 - framing,  3)
    bias_score         = round((vocab + framing + citation + omission) / 4, 3)
    total_score        = int((emotion_neutrality + fact_ratio + source_balance + (1 - bias_score)) / 4 * 100)

    direction_map = {"progressive": "left", "conservative": "right"}
    bias_direction = direction_map.get(label, "center")

    spectrum_map = {"progressive": "진보", "conservative": "보수"}
    spectrum_label = spectrum_map.get(label, "중립")

    return {
        "emotion_neutrality": emotion_neutrality,
        "fact_ratio":         fact_ratio,
        "source_balance":     source_balance,
        "bias_score":         bias_score,
        "total_score":        total_score,
        "bias_direction":     cot.get("bias_direction", bias_direction),
        "spectrum_label":     cot.get("spectrum_label", spectrum_label),
        "cot_vocab_reason":   cot.get("step1_vocab",    {}).get("reason", ""),
        "cot_framing_reason": cot.get("step2_framing",  {}).get("reason", ""),
        "cot_citation_reason":cot.get("step3_citation", {}).get("reason", ""),
        "cot_omission_reason":cot.get("step4_omission", {}).get("reason", ""),
    }


def analyze(text: str, topic: str, keywords: list[str], bias_label: str) -> dict:
    """
    Generated Knowledge → CoT 4단계 (Prompt Merging) → 점수 계산
    Model Tiering: 배경 지식=gpt-4o-mini / CoT 분석=gpt-4o
    """
    background = _generate_background(topic, keywords)
    cot = _run_cot_analysis(text, background)
    scores = _compute_scores(cot, bias_label)
    return {"background": background, **scores}
