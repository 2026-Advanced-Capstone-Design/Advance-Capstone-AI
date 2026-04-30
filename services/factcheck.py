import requests
from config import GOOGLE_FACTCHECK_API_KEY

_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# textualRating 문자열 → 0.0~1.0 점수 매핑
_RATING_MAP = {
    "true": 1.0,
    "사실": 1.0,
    "correct": 1.0,
    "accurate": 1.0,
    "mostly true": 0.75,
    "대체로 사실": 0.75,
    "half true": 0.5,
    "mixture": 0.5,
    "misleading": 0.3,
    "mostly false": 0.25,
    "대체로 거짓": 0.25,
    "false": 0.0,
    "거짓": 0.0,
    "incorrect": 0.0,
    "inaccurate": 0.0,
}


def _rating_to_score(rating: str) -> float | None:
    if not rating:
        return None
    normalized = rating.strip().lower()
    for key, score in _RATING_MAP.items():
        if key in normalized:
            return score
    return None


def check_facts(key_facts: list[str]) -> dict:
    """
    Google Fact Check API로 key_facts를 검증합니다 (최대 3개 조회).

    반환:
      fact_ratio  - 검증된 주장의 평균 사실 점수 (0.0~1.0), API 결과 없으면 None
      results     - 각 주장별 팩트체크 상세 결과
      checked     - 조회 시도한 주장 수
      found       - 실제 결과가 있었던 주장 수
    """
    if not key_facts or not GOOGLE_FACTCHECK_API_KEY:
        return {"fact_ratio": None, "results": [], "checked": 0, "found": 0}

    scores = []
    results = []

    for fact in key_facts[:3]:
        try:
            resp = requests.get(
                _API_URL,
                params={"query": fact, "key": GOOGLE_FACTCHECK_API_KEY},
                timeout=5,
            )
            data = resp.json()
            claims = data.get("claims", [])

            if not claims:
                results.append({"fact": fact, "found": False})
                continue

            reviews = claims[0].get("claimReview", [])
            if not reviews:
                results.append({"fact": fact, "found": False})
                continue

            rating = reviews[0].get("textualRating", "")
            score = _rating_to_score(rating)

            if score is not None:
                scores.append(score)

            results.append({
                "fact": fact,
                "found": True,
                "rating": rating,
                "score": score,
                "publisher": reviews[0].get("publisher", {}).get("name", ""),
                "url": reviews[0].get("url", ""),
            })

        except Exception as e:
            results.append({"fact": fact, "found": False, "error": str(e)})

    fact_ratio = round(sum(scores) / len(scores), 3) if scores else None

    return {
        "fact_ratio": fact_ratio,
        "results": results,
        "checked": len(key_facts[:3]),
        "found": len(scores),
    }
