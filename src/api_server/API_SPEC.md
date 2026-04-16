# News Lens AI Server - API 명세

## 엔드포인트

### GET `/analyze/youtube/{video_id}`

Spring에서 유튜브 영상 ID를 전달하면 댓글 100개를 관련도 순으로 수집하여 감정 분석 + 봇 판별 + 여론 요약 결과를 반환합니다.

**요청 예시**
```
GET http://localhost:8000/analyze/youtube/dQw4w9WgXcQ
```

---

### POST `/analyze/comments`

유튜브 없이 댓글 목록을 직접 전달하여 분석합니다.

**요청 예시**
```json
POST http://localhost:8000/analyze/comments

[
  { "text": "진짜 편파보도 너무하다 ㅋㅋ", "likes": 10 },
  { "text": "좋은 기사 감사합니다!", "likes": 5 }
]
```

---

## 응답 예시

```json
{
  "video_title": "영상 제목",
  "video_comment_count": "12345",
  "total": 98,
  "positive": 40,
  "negative": 20,
  "neutral": 38,
  "positive_pct": 40.8,
  "negative_pct": 20.4,
  "neutral_pct": 38.8,
  "bot_count": 5,
  "bot_pct": 5.1,
  "summary": "전반적 여론: 비판적 댓글이 다수를 차지하며 부정적 여론이 지배적입니다.\n지배 감정: 부정 (40.8%) — 정치·언론에 대한 불신과 분노가 주를 이룸\n핵심 주제: 편파보도, 정치 비판, 경제 불만\n주목할 댓글: \"진짜 편파보도 너무하다\" — 언론 신뢰도에 대한 불만을 직접적으로 표현\n특이사항: 동일 문구 반복 댓글 3건 봇 의심",
  "comments": [
    {
      "text": "댓글 내용",
      "likes": 123,
      "sentiment": "긍정",
      "sentiment_score": 0.87,
      "bot_score": 10,
      "is_bot": false,
      "bot_reasons": []
    },
    {
      "text": "반드시 촉구해야 합니다. 올바른 방향으로",
      "likes": 0,
      "sentiment": "중립",
      "sentiment_score": 0.71,
      "bot_score": 75,
      "is_bot": true,
      "bot_reasons": ["선동 키워드", "딱딱한 문체"]
    }
  ]
}
```

---

## 필드 설명

### 영상/분석 요약

| 필드 | 타입 | 설명 |
|------|------|------|
| video_title | string | 유튜브 영상 제목 (POST /analyze/comments 사용 시 "직접 입력") |
| video_comment_count | string | 영상의 전체 댓글 수 |
| total | int | 실제 분석된 댓글 수 (최대 100) |
| positive / negative / neutral | int | 긍정 / 부정 / 중립 댓글 수 |
| positive_pct / negative_pct / neutral_pct | float | 각 비율 (%) |
| bot_count | int | 봇으로 판정된 댓글 수 |
| bot_pct | float | 봇 비율 (%) |
| summary | string | GPT가 생성한 여론 요약 (줄바꿈 포함 멀티라인) |

### comments 배열 각 항목

| 필드 | 타입 | 설명 |
|------|------|------|
| text | string | 댓글 원문 |
| likes | int | 댓글 좋아요 수 |
| sentiment | string | 긍정 / 부정 / 중립 |
| sentiment_score | float | 감정 신뢰도 (0.0 ~ 1.0) |
| bot_score | int | 봇 의심 점수 (0 ~ 100, 50 이상이면 봇 판정) |
| is_bot | boolean | true = 봇 판정 (bot_score >= 50) |
| bot_reasons | string[] | 봇 판정 근거 (없으면 빈 배열) |

### bot_reasons 가능한 값

| 값 | 설명 |
|----|------|
| 딱딱한 문체 | 구어체·이모지·슬랭이 전혀 없는 형식적 문체 |
| 선동 키워드 | 반드시, 촉구, 규탄, 꼭 공유 등 선동성 키워드 다수 포함 |
| 균일한 문장 | 보도자료·성명서처럼 균일한 문장 길이와 형식적 구조 |
| 극단적 편향 | 특정 정치인·정당을 일방적으로 극단 옹호하거나 극단 비난 |
| 주제 무관 반복 | 뉴스 주제와 무관하게 정치 메시지만 반복 |
| 유사댓글 N개 | TF-IDF 유사도 0.8 초과 댓글이 N개 존재 |

---

## 에러 응답

```json
{ "detail": "에러 메시지" }
```

| HTTP 상태코드 | 상황 |
|--------------|------|
| 400 | 댓글 목록이 비어있음 |
| 403 | 댓글이 비활성화된 영상 |
| 404 | 영상 없음 또는 댓글 없음 |
| 503 | YOUTUBE_API_KEY 미설정 |
| 500 | 서버 내부 오류 |

---

## 서버 실행

```bash
uvicorn src.api_server.main:app --host 0.0.0.0 --port 8000
```

### GET `/health`

서버 상태 확인용

```json
{ "status": "ok", "engine": "gpt-4o-mini" }
```
