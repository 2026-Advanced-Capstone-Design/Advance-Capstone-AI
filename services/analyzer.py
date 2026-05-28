import json
from openai import OpenAI
from config import OPENAI_API_KEY, GPT_MINI_MODEL, GPT_STRONG_MODEL
from services.cache import get_background, set_background

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

[분석 2단계]
1. 어휘 선택 (vocab): 감정적·편향적 단어 사용 여부
2. 사실 기반도 (fact_basis): 기사의 주장이 검증 가능한 데이터·통계·공식 출처에 근거하는지 여부. score는 사실 근거가 부족할수록 1.0에 가깝게 설정하세요.

[제약 조건]
- 각 단계별 score는 0.0(문제 없음)~1.0(심각한 문제) 실수
- 반드시 JSON만 출력하세요.

[응답 형식]
{{
  "step1_vocab":      {{"score": 0.0, "reason": "..."}},
  "step2_fact_basis": {{"score": 0.0, "reason": "이 기사의 주장이 사실에 근거하는 정도와 그 이유를 서술하세요."}}
}}"""

_ANALYSIS_USER = "[기사]\n{text}"

# ── Step 3: 문장 하이라이팅 (gpt-4o-mini) ────────────────────────────────
_HIGHLIGHT_SYSTEM = """당신은 뉴스 편향 분석 전문가입니다.
아래 문장 목록에서 편향 또는 사실성 문제가 드러나는 문장을 찾아 JSON으로 반환하세요.

[분류 기준 — 3가지 유형]
- fact        : 사실 근거가 부족하거나 검증되지 않은 주장을 담은 문장 (수치·출처 없음, 과장된 단언 등)
- emotion     : 감정적·선동적 단어나 표현을 사용해 독자의 감정을 자극하는 문장
- section_bias: 특정 토픽 섹션에서 한쪽 관점만 강화하거나 반대 관점을 배제하는 문장

[제약 조건]
- 편향이 명확한 문장만 선택하세요 (최대 7개)
- 편향이 없으면 highlighted_sentences를 빈 배열로 반환하세요
- sentence 값은 아래 문장 목록의 원문을 그대로 사용하세요
- score: 0.0(약함)~1.0(강함)
- reason: 이 문장이 선택된 구체적인 이유를 1~2문장으로 서술하세요
- 반드시 JSON만 출력하세요

[응답 형식]
{{
  "highlighted_sentences": [
    {{"sentence": "문장 원문", "type": "emotion",      "score": 0.85, "reason": "감정적 단어 '반드시'를 사용해 독자를 선동함"}},
    {{"sentence": "문장 원문", "type": "fact",         "score": 0.72, "reason": "구체적 수치나 출처 없이 주장만 제시함"}},
    {{"sentence": "문장 원문", "type": "section_bias", "score": 0.68, "reason": "경제 토픽에서 보수 측 주장만 인용하고 반론을 누락함"}}
  ]
}}"""

_HIGHLIGHT_USER = """[CoT 분석 결과]
감정 중립성 근거: {vocab_reason}
사실 기반도 근거: {fact_basis_reason}

[섹션별 편향 요약]
{sections_summary}

[문장 목록]
{sentences}"""


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


def _format_sections_summary(sections: list[dict]) -> str:
    if not sections:
        return "섹션 정보 없음"
    lines = []
    for s in sections:
        topic = s.get("topic", "")
        label = s.get("bias_label", "")
        reason = s.get("reason", "")
        lines.append(f"- {topic}: {label} — {reason}")
    return "\n".join(lines)


def _run_highlight_analysis(sentences: list[str], cot: dict, sections: list[dict] = None) -> list[dict]:
    """gpt-4o-mini로 편향 문장 하이라이팅 (최대 30문장 처리)"""
    if not sentences:
        return []

    sentence_text = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(sentences[:30])
    )

    user = _HIGHLIGHT_USER.format(
        vocab_reason=cot.get("step1_vocab", {}).get("reason", ""),
        fact_basis_reason=cot.get("step2_fact_basis", {}).get("reason", ""),
        sections_summary=_format_sections_summary(sections or []),
        sentences=sentence_text,
    )

    response = client.chat.completions.create(
        model=GPT_MINI_MODEL,
        messages=[
            {"role": "system", "content": _HIGHLIGHT_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result.get("highlighted_sentences", [])


def _compute_scores(cot: dict, label: str) -> dict:
    """CoT 결과 → 2대 지표 + 종합 점수 계산"""
    vocab      = cot.get("step1_vocab",      {}).get("score", 0.5)
    fact_basis = cot.get("step2_fact_basis", {}).get("score", 0.5)

    # 각 점수를 반전 → 2대 독립 지표 (높을수록 좋음)
    emotion_neutrality = round(1.0 - vocab,      3)
    fact_ratio         = round(1.0 - fact_basis, 3)

    bias_score = round((vocab + fact_basis) / 2, 3)

    return {
        "emotion_neutrality":    emotion_neutrality,
        "fact_ratio":            fact_ratio,
        "bias_score":            bias_score,
        "cot_emotion_reason": cot.get("step1_vocab", {}).get("reason", ""),
    }


def _make_topic_key(topic: str, keywords: list[str]) -> str:
    return f"{topic.strip().lower()}:{','.join(sorted(k.strip().lower() for k in keywords))}"


def precompute_background(topic: str, keywords: list[str]) -> None:
    """summarize 완료 직후 호출 — background를 캐시에 미리 적재"""
    topic_key = _make_topic_key(topic, keywords)
    if get_background(topic_key) is None:
        background = _generate_background(topic, keywords)
        set_background(topic_key, background)


def analyze(text: str, topic: str, keywords: list[str], bias_label: str,
            sentences: list[str] = None, sections: list[dict] = None) -> dict:
    """
    Generated Knowledge → CoT 2단계 (Prompt Merging) → 문장 하이라이팅 → 점수 계산
    Model Tiering: 배경 지식·하이라이팅=gpt-4o-mini / CoT 분석=gpt-4o
    Topic-based Caching: 동일 이슈 재요청 시 GPT 호출 없이 캐시 재사용
    """
    topic_key = _make_topic_key(topic, keywords)
    background = get_background(topic_key)
    if background is None:
        background = _generate_background(topic, keywords)
        set_background(topic_key, background)

    cot = _run_cot_analysis(text, background)
    highlighted = _run_highlight_analysis(sentences or [], cot, sections or [])
    scores = _compute_scores(cot, bias_label)
    return {"background": background, "highlighted_sentences": highlighted, **scores}
