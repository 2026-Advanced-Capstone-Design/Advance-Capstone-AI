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

[분석 3단계]
1. 어휘 선택 (vocab): 감정적·편향적 단어 사용 여부
2. 사실 기반도 (fact_basis): 기사의 주장이 검증 가능한 데이터·통계·공식 출처에 근거하는지 여부. score는 사실 근거가 부족할수록 1.0에 가깝게 설정하세요.
3. 정보 생략 (omission): 반대 관점의 중요 사실을 누락했는지 여부

[제약 조건]
- 각 단계별 score는 0.0(문제 없음)~1.0(심각한 문제) 실수
- bias_direction: "left" | "center" | "right"
- spectrum_label: "진보" | "중립" | "보수"
- 반드시 JSON만 출력하세요.

[응답 형식]
{{
  "step1_vocab":      {{"score": 0.0, "reason": "..."}},
  "step2_fact_basis": {{"score": 0.0, "reason": "이 기사의 주장이 사실에 근거하는 정도와 그 이유를 서술하세요."}},
  "step3_omission":   {{"score": 0.0, "reason": "..."}},
  "bias_direction": "center",
  "spectrum_label": "중립"
}}"""

_ANALYSIS_USER = "[기사]\n{text}"

# ── Step 3: 문장 하이라이팅 (gpt-4o-mini) ────────────────────────────────
_HIGHLIGHT_SYSTEM = """당신은 뉴스 편향 분석 전문가입니다.
아래 문장 목록에서 편향이 드러나는 문장을 찾아 JSON으로 반환하세요.

[편향 유형]
- vocab    : 감정적·편향적 단어를 사용한 문장
- framing  : 특정 관점을 부각하거나 약화시키는 구조의 문장
- citation : 한쪽에 치우친 인용이 포함된 문장
- omission : 반대 관점을 무시하거나 중요한 사실을 누락한 문장

[제약 조건]
- 편향이 명확한 문장만 선택하세요 (최대 7개)
- 편향이 없으면 highlighted_sentences를 빈 배열로 반환하세요
- sentence 값은 아래 문장 목록의 원문을 그대로 사용하세요
- score: 0.0(약함)~1.0(강함)
- 반드시 JSON만 출력하세요

[응답 형식]
{{
  "highlighted_sentences": [
    {{"sentence": "문장 원문 그대로", "type": "vocab",    "score": 0.85}},
    {{"sentence": "문장 원문 그대로", "type": "framing",  "score": 0.72}}
  ]
}}"""

_HIGHLIGHT_USER = """[분석된 편향 근거]
어휘 선택: {vocab_reason}
사실 기반도: {fact_basis_reason}
정보 생략: {omission_reason}

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


def _run_highlight_analysis(sentences: list[str], cot: dict) -> list[dict]:
    """gpt-4o-mini로 편향 문장 하이라이팅 (최대 30문장 처리)"""
    if not sentences:
        return []

    # 번호를 붙여서 GPT가 문장을 구분하기 쉽게 구성
    sentence_text = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(sentences[:30])
    )

    user = _HIGHLIGHT_USER.format(
        vocab_reason=cot.get("step1_vocab", {}).get("reason", ""),
        fact_basis_reason=cot.get("step2_fact_basis", {}).get("reason", ""),
        omission_reason=cot.get("step3_omission", {}).get("reason", ""),
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
    """CoT 결과 → 4대 지표 + 종합 점수 계산 (각 25% 균등 가중치)"""
    vocab      = cot.get("step1_vocab",      {}).get("score", 0.5)
    fact_basis = cot.get("step2_fact_basis", {}).get("score", 0.5)
    omission   = cot.get("step3_omission",   {}).get("score", 0.5)

    # 각 점수를 반전 → 3대 독립 지표 (높을수록 좋음)
    emotion_neutrality  = round(1.0 - vocab,      3)
    fact_ratio          = round(1.0 - fact_basis, 3)
    omission_neutrality = round(1.0 - omission,   3)

    bias_score = round((vocab + fact_basis + omission) / 3, 3)
    # total_score는 routes/analyze.py에서 섹션별 편향도 포함하여 최종 계산

    direction_map = {"progressive": "left", "conservative": "right"}
    bias_direction = direction_map.get(label, "center")

    spectrum_map = {"progressive": "진보", "conservative": "보수"}
    spectrum_label = spectrum_map.get(label, "중립")

    return {
        "emotion_neutrality":    emotion_neutrality,
        "fact_ratio":            fact_ratio,
        "omission_neutrality":   omission_neutrality,
        "bias_score":            bias_score,
        "bias_direction":        cot.get("bias_direction", bias_direction),
        "spectrum_label":        cot.get("spectrum_label", spectrum_label),
        "cot_emotion_reason":    cot.get("step1_vocab",      {}).get("reason", ""),
        "cot_fact_ratio_reason": cot.get("step2_fact_basis", {}).get("reason", ""),
    }


def _make_topic_key(topic: str, keywords: list[str]) -> str:
    return f"{topic.strip().lower()}:{','.join(sorted(k.strip().lower() for k in keywords))}"


def analyze(text: str, topic: str, keywords: list[str], bias_label: str, sentences: list[str] = None) -> dict:
    """
    Generated Knowledge → CoT 4단계 (Prompt Merging) → 문장 하이라이팅 → 점수 계산
    Model Tiering: 배경 지식·하이라이팅=gpt-4o-mini / CoT 분석=gpt-4o
    Topic-based Caching: 동일 이슈 재요청 시 GPT 호출 없이 캐시 재사용
    """
    topic_key = _make_topic_key(topic, keywords)
    background = get_background(topic_key)
    if background is None:
        background = _generate_background(topic, keywords)
        set_background(topic_key, background)

    cot = _run_cot_analysis(text, background)
    highlighted = _run_highlight_analysis(sentences or [], cot)
    scores = _compute_scores(cot, bias_label)
    return {"background": background, "highlighted_sentences": highlighted, **scores}
