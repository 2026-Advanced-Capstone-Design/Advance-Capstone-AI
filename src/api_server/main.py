import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import AsyncOpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import AsyncGenerator, List, Optional, Tuple
import os


load_dotenv()

app = FastAPI(title="News Lens AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 클라이언트 초기화 =====
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY) if YOUTUBE_API_KEY else None

BATCH_SIZE = 25  # GPT 병렬 호출당 댓글 수

# ===== 요청/응답 스키마 =====
class CommentItem(BaseModel):
    text: str
    likes: Optional[int] = 0
    author_name: Optional[str] = None
    author_id: Optional[str] = None

class SentimentResult(BaseModel):
    text: str
    likes: Optional[int] = 0
    author_name: Optional[str] = None
    author_id: Optional[str] = None
    sentiment: str
    sentiment_score: float
    bot_score: int
    is_bot: bool
    bot_reasons: List[str]

class YoutubeResult(BaseModel):
    video_title: str
    channel_name: Optional[str] = None
    view_count: Optional[int] = None
    published_at: Optional[str] = None
    video_comment_count: str
    total: int
    positive: int
    negative: int
    neutral: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    bot_count: int
    bot_pct: float
    summary: str
    comments: List[SentimentResult]

# ===== 감정분류 + 봇탐지 통합 프롬프트 =====
ANALYZE_SYSTEM_PROMPT = """당신은 한국어 뉴스·정치·경제 댓글의 감정을 분류하고 봇 여부를 판별하는 전문가입니다.

━━━ 감정 분류 기준 ━━━

[라벨 정의]
- 부정: 분노, 실망, 비판, 욕설, 혐오, 불만, 냉소, 불신, 우려, 반대, 조롱, 위협 인식
- 긍정: 지지, 칭찬, 공감, 감사, 기대, 응원, 놀라움(긍정적), 희망
- 중립: 감정 없이 사실만 전달하거나 순수하게 판단을 보류

[반드시 부정으로 분류할 패턴]
1. 인터넷 슬랭 충격/경악 — 부정적 맥락에서 사용 시
   - "ㄷㄷ", "ㄷㄷㄷ", "덜덜" + (세금낭비/비리/재난 등)
   - 예: "140조 ㄷㄷㄷ 그것도 증액" → 부정
2. 신조어·합성어 비판
   - 인물+히틀러/나치, 정당+비하어 합성
   - 예: "또람프틀러", "틀딱", "꼰대정치" → 부정
3. 암묵적 부정 — 기도/소원 형식이지만 비판 의도
   - "하루속히 끌어내려주소서", "제발 사라져줬으면" → 부정
4. 불신·의심: "안 믿는다", "말이 되냐", "황당하다", 음모론 암시
5. 위기·전쟁·재난을 부정 시각으로 서술: "전쟁 재개", "민간인 살상", "나라 망한다"
6. 기관·언론 직접 비판: "기레기", "적폐", "편파보도", "어용 언론"
7. 강한 요구/주장 — 비판적 맥락: "탄핵해야", "심판해야", "막아야 한다"

[중립 판단 주의]
겉으로 사실 서술처럼 보여도 비판·우려·불신 의도가 읽히면 부정입니다.

━━━ 봇 판별 기준 ━━━

아래 중 2개 이상 해당하면 봇의심, 1개 이하면 정상입니다.

[봇의심 신호]
1. 자연스러운 구어체·이모지·인터넷 슬랭이 전혀 없는 딱딱한 문체
2. "반드시", "촉구", "규탄", "국민을 위한", "올바른 방향", "꼭 공유", "퍼뜨려야" 등 선동성 키워드 다수 포함
3. 마치 보도자료·성명서처럼 균일한 문장 길이와 형식적 구조
4. 특정 정치인·정당을 일방적으로 극단 옹호하거나 극단 비난하는 패턴
5. 댓글 내용이 뉴스 주제와 무관하게 정치 메시지만 반복

[정상 댓글 특징]
- 감탄사, 줄임말, 이모지, 맞춤법 오류 등 자연스러운 구어체
- 기사 내용에 맞는 맥락적 반응
- 개인적 경험이나 감정 표현 포함

━━━ 출력 형식 — 엄격히 준수 ━━━

번호|감정라벨|신뢰도|봇점수|봇근거

- 감정라벨: 부정 / 긍정 / 중립
- 신뢰도: 0.00~1.00
- 봇점수: 0~100 정수 (봇일 가능성. 신호 없음=0~20, 1개=30~50, 2개=60~80, 3개이상=90~100)
- 봇근거: 봇점수 50 이상일 때만 해당 신호 번호 나열 (예: 1,2), 미만이면 "-"
- 다른 설명·부가 텍스트 절대 금지

예시:
1|부정|0.92|10|-
2|중립|0.78|75|1,3
3|긍정|0.85|0|-"""

# 봇 근거 번호 → 한국어 이유 매핑
BOT_REASON_MAP = {
    "1": "딱딱한 문체",
    "2": "선동 키워드",
    "3": "균일한 문장",
    "4": "극단적 편향",
    "5": "주제 무관 반복",
}


def _parse_gpt_lines(lines: List[str], count: int) -> List[Tuple[str, float, int, bool, List[str]]]:
    """GPT 응답 라인을 파싱해서 튜플 리스트로 반환."""
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) == 5:
            _, label, score_str, bot_label, bot_reasons_str = parts
            label = label.strip()
            try:
                score = round(float(score_str.strip()), 2)
            except ValueError:
                score = 0.8
            try:
                bot_score = max(0, min(100, int(bot_label.strip())))
            except ValueError:
                bot_score = 0
            is_bot = bot_score >= 50
            reasons = [
                BOT_REASON_MAP[r.strip()]
                for r in bot_reasons_str.split(",")
                if r.strip() in BOT_REASON_MAP
            ]
            results.append((label if label in ("부정", "긍정", "중립") else "중립",
                             score, bot_score, is_bot, reasons))
        else:
            results.append(("중립", 0.8, 0, False, []))

    while len(results) < count:
        results.append(("중립", 0.8, 0, False, []))
    return results[:count]


def _apply_tfidf(
    results: List[Tuple],
    all_texts: List[str],
    sim_matrix,
    global_offset: int,
) -> List[Tuple]:
    """TF-IDF 유사도 기반 중복 댓글 보완. global_offset은 전체 texts에서의 시작 인덱스."""
    final = []
    for local_i, (label, score, bot_score, is_bot, reasons) in enumerate(results):
        i = global_offset + local_i
        if sim_matrix is not None:
            similar_count = sum(
                1 for j in range(len(all_texts))
                if i != j and sim_matrix[i][j] > 0.8
            )
            if similar_count > 0:
                is_bot = True
                bot_score = max(bot_score, 80)
                if "유사댓글" not in " ".join(reasons):
                    reasons = [f"유사댓글 {similar_count}개"] + reasons
        final.append((label, score, bot_score, is_bot, reasons))
    return final


async def _call_gpt_batch(texts: List[str]) -> List[Tuple[str, float, int, bool, List[str]]]:
    """단일 배치를 GPT에 비동기로 호출."""
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
                {"role": "user",   "content": f"댓글 목록:\n{numbered}"}
            ],
            temperature=0,
            max_tokens=len(texts) * 20,
        )
        lines = response.choices[0].message.content.strip().splitlines()
        return _parse_gpt_lines(lines, len(texts))
    except Exception as e:
        print(f"[GPT 배치 오류] {e}")
        return [("중립", 0.8, 0, False, [])] * len(texts)


async def analyze_all_parallel(
    texts: List[str],
    sim_matrix,
) -> List[Tuple[str, float, int, bool, List[str]]]:
    """BATCH_SIZE씩 나눠 병렬 GPT 호출 → 결과 합산."""
    batches = [texts[i:i+BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    batch_results = await asyncio.gather(*[_call_gpt_batch(b) for b in batches])

    raw = []
    for batch_res in batch_results:
        raw.extend(batch_res)

    # TF-IDF 유사도 보완 (전체 인덱스 기준)
    return _apply_tfidf(raw, texts, sim_matrix, global_offset=0)


# ===== 요약 프롬프트 =====
SUMMARY_SYSTEM_PROMPT = """당신은 한국어 뉴스 댓글 여론 분석 전문가입니다.
댓글 감정 분류 결과와 대표 댓글을 받아 아래 형식으로 요약하세요.

[출력 형식]
전반적 여론: (한 문장으로 전체 분위기 요약)
지배 감정: (부정/긍정/중립 중 가장 많은 것과 비율, 그 이유 한 줄)
핵심 주제: (댓글에서 자주 언급된 주제 또는 키워드 2~3가지)
주목할 댓글: (가장 대표적인 댓글 1개 인용 후 한 줄 해석)
특이사항: (봇 의심 댓글, 선동 패턴, 여론 양극화 등 눈에 띄는 패턴 — 없으면 "없음")"""


async def summarize_comments_async(texts: List[str], analysis: List[Tuple]) -> str:
    total = len(analysis)
    neg       = sum(1 for a in analysis if a[0] == "부정")
    pos       = sum(1 for a in analysis if a[0] == "긍정")
    neu       = sum(1 for a in analysis if a[0] == "중립")
    bot_count = sum(1 for a in analysis if a[3])

    samples = {"부정": [], "긍정": [], "중립": []}
    for text, item in zip(texts, analysis):
        label = item[0]
        if len(samples[label]) < 4:
            samples[label].append(text[:60])

    sample_str = ""
    for label, txts in samples.items():
        if txts:
            sample_str += f"\n[{label} 대표 댓글]\n" + "\n".join(f"- {t}" for t in txts)

    user_content = f"""총 {total}개 댓글 분석 결과:
- 부정: {neg}개 ({neg/total*100:.1f}%)
- 긍정: {pos}개 ({pos/total*100:.1f}%)
- 중립: {neu}개 ({neu/total*100:.1f}%)
- 봇 의심: {bot_count}개 ({bot_count/total*100:.1f}%)
{sample_str}"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT 요약 오류] {e}")
        return "(요약 실패)"


# ===== 핵심 분석 로직 (비동기) =====
async def _analyze_comments(comments: List[CommentItem]):
    texts = [c.text for c in comments]

    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
    except Exception:
        sim_matrix = None

    # 병렬 배치 GPT 호출 + 요약 동시 시작
    analysis = await analyze_all_parallel(texts, sim_matrix)
    summary  = await summarize_comments_async(texts, analysis)

    results = []
    positive = negative = neutral = bot_count = 0

    for i, comment in enumerate(comments):
        label, score, bot_score, is_bot, reasons = analysis[i]

        if label == "긍정":   positive += 1
        elif label == "부정": negative += 1
        else:                 neutral  += 1
        if is_bot: bot_count += 1

        results.append(SentimentResult(
            text=comment.text,
            likes=comment.likes,
            author_name=comment.author_name,
            author_id=comment.author_id,
            sentiment=label,
            sentiment_score=score,
            bot_score=bot_score,
            is_bot=is_bot,
            bot_reasons=reasons,
        ))

    return results, positive, negative, neutral, bot_count, len(results), summary


def _fetch_youtube_comments(video_id: str):
    """YouTube API 댓글 수집 (동기)."""
    video_response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    if not video_response["items"]:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")

    video = video_response["items"][0]
    video_title = video["snippet"]["title"]
    channel_name = video["snippet"]["channelTitle"]
    published_at = video["snippet"]["publishedAt"][:10]
    view_count = int(video["statistics"].get("viewCount", 0))
    comment_count_str = video["statistics"].get("commentCount", "?")

    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText",
        order="relevance",
    ).execute()

    comments = []
    for item in response.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        text = snippet["textDisplay"].strip()
        if len(text) > 2:
            author_channel = snippet.get("authorChannelId", {})
            comments.append(CommentItem(
                text=text,
                likes=snippet["likeCount"],
                author_name=snippet.get("authorDisplayName"),
                author_id=author_channel.get("value") if author_channel else None,
            ))

    return video_title, channel_name, view_count, published_at, comment_count_str, comments


# ===== SSE 헬퍼 =====
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_analysis(
    comments: List[CommentItem],
    video_title: str,
    video_comment_count: str,
    channel_name: Optional[str] = None,
    view_count: Optional[int] = None,
    published_at: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    texts = [c.text for c in comments]
    total = len(texts)

    # 메타 정보 즉시 전송
    yield _sse("meta", {
        "video_title": video_title,
        "channel_name": channel_name,
        "view_count": view_count,
        "published_at": published_at,
        "video_comment_count": video_comment_count,
        "total": total,
        "batch_size": BATCH_SIZE,
        "batch_count": (total + BATCH_SIZE - 1) // BATCH_SIZE,
    })

    # TF-IDF (빠름, 동기)
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
    except Exception:
        sim_matrix = None

    # 배치 분할
    batches = [texts[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    offsets = list(range(0, total, BATCH_SIZE))

    # 각 배치를 태스크로 만들어 완료 순서대로 스트리밍
    tasks = {
        asyncio.ensure_future(_call_gpt_batch(batch)): (offset, batch)
        for batch, offset in zip(batches, offsets)
    }

    all_analysis: List[Optional[Tuple]] = [None] * total
    completed_batches = 0

    pending = set(tasks.keys())
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for fut in done:
            offset, batch = tasks[fut]
            raw_results = fut.result()
            adjusted = _apply_tfidf(raw_results, texts, sim_matrix, global_offset=offset)

            for local_i, result in enumerate(adjusted):
                global_i = offset + local_i
                all_analysis[global_i] = result
                comment = comments[global_i]
                label, score, bot_score, is_bot, reasons = result

                yield _sse("comment", {
                    "index": global_i,
                    "text": comment.text,
                    "likes": comment.likes,
                    "author_name": comment.author_name,
                    "author_id": comment.author_id,
                    "sentiment": label,
                    "sentiment_score": score,
                    "bot_score": bot_score,
                    "is_bot": is_bot,
                    "bot_reasons": reasons,
                })

            completed_batches += 1
            yield _sse("progress", {
                "completed_batches": completed_batches,
                "total_batches": len(batches),
                "processed": min(offset + BATCH_SIZE, total),
                "total": total,
            })

    # 모든 배치 완료 → 요약 생성
    summary = await summarize_comments_async(texts, all_analysis)

    positive  = sum(1 for a in all_analysis if a[0] == "긍정")
    negative  = sum(1 for a in all_analysis if a[0] == "부정")
    neutral   = sum(1 for a in all_analysis if a[0] == "중립")
    bot_count = sum(1 for a in all_analysis if a[3])

    yield _sse("summary", {"summary": summary})
    yield _sse("stats", {
        "total": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_pct": round(positive / total * 100, 1),
        "negative_pct": round(negative / total * 100, 1),
        "neutral_pct":  round(neutral  / total * 100, 1),
        "bot_count": bot_count,
        "bot_pct": round(bot_count / total * 100, 1),
    })
    yield _sse("done", {})


# ===== API 엔드포인트 =====

@app.get("/health")
def health():
    return {"status": "ok", "engine": "gpt-4o-mini"}


# ── 기존 방식 (단일 JSON 응답) ──────────────────────────────────────────────

@app.get("/analyze/youtube/{video_id}", response_model=YoutubeResult)
async def analyze_youtube_by_id(video_id: str):
    if not youtube:
        raise HTTPException(status_code=503, detail="YOUTUBE_API_KEY가 설정되지 않았습니다.")
    try:
        video_title, channel_name, view_count, published_at, comment_count_str, comments = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_youtube_comments, video_id
        )
    except HTTPException:
        raise
    except HttpError as e:
        if "commentsDisabled" in str(e):
            raise HTTPException(status_code=403, detail="댓글이 비활성화된 영상입니다.")
        raise HTTPException(status_code=500, detail=f"YouTube API 오류: {str(e)}")

    if not comments:
        raise HTTPException(status_code=404, detail="수집된 댓글이 없습니다.")

    try:
        results, positive, negative, neutral, bot_count, total, summary = await _analyze_comments(comments)
        return YoutubeResult(
            video_title=video_title,
            channel_name=channel_name,
            view_count=view_count,
            published_at=published_at,
            video_comment_count=comment_count_str,
            total=total,
            positive=positive,
            negative=negative,
            neutral=neutral,
            positive_pct=round(positive / total * 100, 1),
            negative_pct=round(negative / total * 100, 1),
            neutral_pct=round(neutral  / total * 100, 1),
            bot_count=bot_count,
            bot_pct=round(bot_count / total * 100, 1),
            summary=summary,
            comments=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/comments", response_model=YoutubeResult)
async def analyze_comments_direct(comments: List[CommentItem]):
    if not comments:
        raise HTTPException(status_code=400, detail="댓글이 없습니다.")
    try:
        results, positive, negative, neutral, bot_count, total, summary = await _analyze_comments(comments)
        return YoutubeResult(
            video_title="직접 입력",
            video_comment_count=str(total),
            total=total,
            positive=positive,
            negative=negative,
            neutral=neutral,
            positive_pct=round(positive / total * 100, 1),
            negative_pct=round(negative / total * 100, 1),
            neutral_pct=round(neutral  / total * 100, 1),
            bot_count=bot_count,
            bot_pct=round(bot_count / total * 100, 1),
            summary=summary,
            comments=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── SSE 스트리밍 방식 ──────────────────────────────────────────────────────

@app.get("/analyze/youtube/{video_id}/stream")
async def stream_youtube(video_id: str):
    """
    SSE 스트리밍 엔드포인트.
    댓글이 분석되는 즉시 배치 단위로 클라이언트에 전송합니다.

    이벤트 종류:
      meta     — 영상 정보 및 총 댓글 수
      comment  — 개별 댓글 분석 결과 (index 순서대로)
      progress — 배치 완료 진행률
      summary  — GPT 여론 요약
      stats    — 최종 통계
      done     — 완료 신호
    """
    if not youtube:
        raise HTTPException(status_code=503, detail="YOUTUBE_API_KEY가 설정되지 않았습니다.")

    try:
        video_title, channel_name, view_count, published_at, comment_count_str, comments = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_youtube_comments, video_id
        )
    except HTTPException:
        raise
    except HttpError as e:
        if "commentsDisabled" in str(e):
            raise HTTPException(status_code=403, detail="댓글이 비활성화된 영상입니다.")
        raise HTTPException(status_code=500, detail=f"YouTube API 오류: {str(e)}")

    if not comments:
        raise HTTPException(status_code=404, detail="수집된 댓글이 없습니다.")

    return StreamingResponse(
        _stream_analysis(comments, video_title, comment_count_str, channel_name, view_count, published_at),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/analyze/comments/stream")
async def stream_comments_direct(comments: List[CommentItem]):
    """댓글 직접 입력 SSE 스트리밍 버전."""
    if not comments:
        raise HTTPException(status_code=400, detail="댓글이 없습니다.")

    return StreamingResponse(
        _stream_analysis(comments, "직접 입력", str(len(comments))),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
