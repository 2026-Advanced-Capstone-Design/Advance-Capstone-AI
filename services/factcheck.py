import json
import logging
import requests
from openai import OpenAI
from config import GOOGLE_FACTCHECK_API_KEY, OPENAI_API_KEY, GPT_MINI_MODEL

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

_RATING_MAP = {
    "true": 1.0, "사실": 1.0, "correct": 1.0, "accurate": 1.0,
    "mostly true": 0.75, "대체로 사실": 0.75,
    "half true": 0.5, "mixture": 0.5,
    "misleading": 0.3,
    "mostly false": 0.25, "대체로 거짓": 0.25,
    "false": 0.0, "거짓": 0.0, "incorrect": 0.0, "inaccurate": 0.0,
}

_SYSTEM = """당신은 팩트체크 전문가입니다.
주어진 주장 목록을 분석하여 기사 전체의 사실성 점수와 근거를 판단하세요.

[판단 기준]
- Google Fact Check 검증 결과가 있으면 해당 결과를 우선 반영하세요.
- Google 결과가 없으면 주장 자체의 구체성, 검증 가능성, 출처 명시 여부를 기준으로 판단하세요.
  예) 날짜·수치·기관명 등 구체적 사실이 포함된 주장 → 높은 점수
      모호하거나 과장된 주장, 출처 불명 → 낮은 점수

[응답 형식 — 반드시 JSON만 출력]
{{
  "fact_ratio": 0.0~1.0,
  "fact_check_reason": "전체 주장들의 사실성에 대한 종합 근거를 2~3문장으로 설명"
}}"""


def _rating_to_score(rating: str) -> float | None:
    if not rating:
        return None
    normalized = rating.strip().lower()
    for key, score in _RATING_MAP.items():
        if key in normalized:
            return score
    return None


def _fetch_google_result(fact: str) -> dict | None:
    """Google Fact Check API 조회 — 유용한 필드 모두 추출"""
    try:
        resp = requests.get(
            _API_URL,
            params={"query": fact, "key": GOOGLE_FACTCHECK_API_KEY},
            timeout=5,
        )
        claims = resp.json().get("claims", [])
        if not claims:
            return None

        claim = claims[0]
        reviews = claim.get("claimReview", [])
        if not reviews:
            return None

        review = reviews[0]
        rating = review.get("textualRating", "")
        score = _rating_to_score(rating)
        if score is None:
            return None

        return {
            "query": fact,
            "matched_claim": claim.get("text", ""),
            "claimant": claim.get("claimant", ""),
            "claim_date": claim.get("claimDate", ""),
            "rating": rating,
            "score": score,
            "title": review.get("title", ""),
            "publisher": review.get("publisher", {}).get("name", ""),
            "publisher_site": review.get("publisher", {}).get("site", ""),
            "review_date": review.get("reviewDate", ""),
            "url": review.get("url", ""),
        }
    except Exception:
        return None


def _format_google_context(google_results: list[dict]) -> str:
    """GPT에게 전달할 Google 검증 결과 포맷"""
    if not google_results:
        return "Google Fact Check 검증 결과 없음"
    lines = []
    for r in google_results:
        line = f"- 주장: {r['query']}"
        if r.get("matched_claim") and r["matched_claim"] != r["query"]:
            line += f"\n  (Google 매칭 원문: {r['matched_claim']})"
        if r.get("claimant"):
            line += f"\n  주장 주체: {r['claimant']}"
        if r.get("claim_date"):
            line += f" ({r['claim_date'][:10]})"
        line += f"\n  판정: {r['rating']} (점수: {r['score']:.2f})"
        if r.get("title"):
            line += f"\n  팩트체크 기사: {r['title']}"
        if r.get("publisher"):
            line += f"\n  검증 기관: {r['publisher']}"
            if r.get("review_date"):
                line += f" ({r['review_date'][:10]})"
        lines.append(line)
    return "\n".join(lines)


def _gpt_check(key_facts: list[str], google_results: list[dict]) -> dict:
    """Google 결과를 컨텍스트로 제공해 GPT가 종합 판단. Google 결과 없으면 GPT 단독 분석."""
    facts_text = "\n".join(f"- {f}" for f in key_facts[:3])
    google_context = _format_google_context(google_results)

    try:
        response = client.chat.completions.create(
            model=GPT_MINI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": (
                    f"[검증 대상 주장]\n{facts_text}\n\n"
                    f"[Google Fact Check 결과]\n{google_context}"
                )},
            ],
            temperature=0.1,
            max_tokens=250,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("[factcheck] GPT 호출 실패: %s", e, exc_info=True)
        if google_results:
            scores = [r["score"] for r in google_results]
            return {
                "fact_ratio": round(sum(scores) / len(scores), 3),
                "fact_check_reason": f"Google Fact Check 결과 {len(google_results)}건을 바탕으로 산출된 점수입니다.",
            }
        return {"fact_ratio": None, "fact_check_reason": None}


def check_facts(key_facts: list[str]) -> dict:
    """
    Google Fact Check API로 각 주장 검증 후 GPT가 종합 점수 + 근거 생성.

    반환:
      fact_ratio        - 사실성 점수 (0.0~1.0)
      fact_check_reason - 사실성 판단 근거 요약 (2~3문장)
    """
    if not key_facts:
        return {"fact_ratio": None, "fact_check_reason": ""}

    google_results = []
    if GOOGLE_FACTCHECK_API_KEY:
        for fact in key_facts[:3]:
            result = _fetch_google_result(fact)
            if result:
                google_results.append(result)

    result = _gpt_check(key_facts, google_results)
    return {
        "fact_ratio": result.get("fact_ratio"),
        "fact_check_reason": result.get("fact_check_reason", ""),
    }
