import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import AsyncOpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple
from dataclasses import dataclass
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
    authorName: Optional[str] = None
    authorId: Optional[str] = None

class SentimentResult(BaseModel):
    text: str
    likes: Optional[int] = 0
    authorName: Optional[str] = None
    authorId: Optional[str] = None
    sentiment: str
    sentimentScore: float
    botScore: int
    isBot: bool
    botReasons: List[str]

class YoutubeResult(BaseModel):
    videoTitle: str
    channelName: Optional[str] = None
    viewCount: Optional[int] = None
    publishedAt: Optional[str] = None
    videoCommentCount: str
    total: int
    positive: int
    negative: int
    neutral: int
    positivePct: float
    negativePct: float
    neutralPct: float
    botCount: int
    botPct: float
    positiveSummary: str
    negativeSummary: str
    neutralSummary: str
    specialNotes: str
    comments: List[SentimentResult]

# ===== 감정분류 + 봇탐지 통합 프롬프트 =====
ANALYZE_SYSTEM_PROMPT = """당신은 한국어 뉴스·정치·경제·예능 댓글의 감정을 분류하고 봇 여부를 판별하는 전문가입니다.

━━━ 감정 분류 기준 ━━━

[라벨 정의]
- 부정: 분노, 실망, 비판, 혐오, 불만, 냉소, 불신, 우려, 반대, 조롱, 위협 인식 (단, 욕설이 강조어로만 쓰인 경우 제외)
- 긍정: 지지, 칭찬, 공감, 감사, 기대, 응원, 놀라움(긍정적), 희망, 웃음/유머
- 중립: 감정 없이 사실만 전달하거나 순수하게 판단을 보류

[욕설 강조어 판별 — 중요]
한국어 인터넷 슬랭에서 "ㅅㅂ", "존나", "ㅈㄴ", "개" 등은 욕설이 아닌 강조어로 쓰일 수 있습니다.
아래 패턴은 욕설이 포함되어도 긍정으로 분류하세요:
- "ㅋ"가 2개 이상 포함 + "웃기다/웃겨/재밌다/대박/미쳤다" 등 긍정 단어
- 예: "존나 웃기네 ㅅㅂ ㅋㅋㅋㅋㅋ" → 긍정 (웃음 반응)
- 예: "ㅅㅂ ㅋㅋㅋㅋ 개웃겨" → 긍정
- 예: "ㅈㄴ 대박이다 ㅋㅋㅋ" → 긍정
반면 욕설 + 비판 대상 + 분노 맥락이면 부정으로 분류하세요:
- 예: "ㅅㅂ 저 놈들 때문에 나라 망한다" → 부정

[ㅋ + 긍정 단어 규칙의 예외 — 중요]
뉴스·범죄·재판·비리·선고 맥락에서는 "ㅋ + 웃기다/웃겨/폭망/연기" 등이 분노·황당함의 비꼬기 표현입니다. 이 경우 부정으로 분류하세요.

판별 기준:
- 댓글 주제가 범죄 피의자/피고인, 선고 결과, 비리·부패, 판사·검사 비판이면 → 부정
- 댓글 주제가 엔터테인먼트·스포츠·일상 콘텐츠이면 → 긍정 유지

예시:
- "ㅋㅋㅋㅋ존나웃기네ㅋㅋ" (선고 4년 뉴스 댓글) → 부정 (판결 황당함에 대한 분노)
- "연기를 ㅋㅋㅋㅋㅋ..." (피고인 병약 연기 조롱) → 부정
- "성공 웃었겠지 ㅋㅋ" (경미한 선고에 대한 비꼬기) → 부정
- "그놈의 마스크는 ㅋㅋㅋ" (피의자 마스크 조롱) → 부정
- "탬버린 쳐 흔들때 좋았을꺼야 ㅋㅋ 지금은 폭망 ㅋㅋㅋ" (피고인 조롱) → 부정
- "메모지 : 이정도면 만족하시죠 ㅋㅋ" (판결 비꼬기) → 부정
- "양평고속도로가 훨씬 큼... 추가기소는 막을수 없어...ㅋㅋ" (수사 결과 비꼬기) → 부정
- "존나 웃기네 ㅅㅂ ㅋㅋㅋㅋㅋ" (유튜버/예능 콘텐츠 댓글) → 긍정 유지

[반드시 긍정으로 분류할 인터넷 슬랭 — 중요]
아래 표현들은 겉보기에 욕설·비하처럼 보여도 칭찬·감탄·응원의 의미입니다:
1. "ㅈ된다 / ㅈ되네" — 단독 사용 또는 긍정 맥락(재능, 실력 칭찬)이면 "엄청나다/대단하다"의 감탄
   - 예: "근데 진심 재능 ㅈ된다" → 긍정
   - 예: "저 사람 진짜 ㅈ되네 ㅋㅋ" → 긍정
   - 주의: "나 ㅈ됐다" 처럼 자신의 불행을 표현하면 부정
2. "ㄱㅇㅇ" — "개이득/좋다/훌륭하다"의 줄임말로 긍정 감탄
   - 예: "(누구누구) ㄱㅇㅇ" → 긍정
   - 예: "(누구누구) ㄱㅇㅇ ㄱㅇㅇ" → 긍정
3. "그래 ~해라 / 그냥 ~해라" — 응원·지지 뉘앙스
   - 예: "그래 발로만 해라 (누구누구)" → 긍정 (계속 해줘 = 응원)
   - 주의: 비판 대상 + 비꼬는 맥락이면 부정
4. "지리다 / 지리네" — "대단하다/최고다"의 감탄
   - 예: "어떤 스킨 개맛있네" (게임 맥락의 "맛있다") → 긍정
   - 예: "지리노 이분" → 긍정
5. "빡치다 / 개빡치다" + 칭찬 맥락 — 부럽거나 놀랍다는 감탄
   - 예: "개빡치게 왜 잘함?ㅋㅋㅋㅋ" → 긍정

[반드시 부정으로 분류할 패턴]
1. 인터넷 슬랭 충격/경악 — 부정적 맥락에서 사용 시
   - "ㄷㄷ", "ㄷㄷㄷ", "덜덜" + (세금낭비/비리/재난 등)
   - 예: "140조 ㄷㄷㄷ 그것도 증액" → 부정
   - 단, "ㄷㄷ" 단독 (다른 맥락 없이 이 표현만 있는 경우) → 중립
2. 신조어·합성어 비판
   - 인물+히틀러/나치, 정당+비하어 합성
   - 예: "또람프틀러", "틀딱", "꼰대정치" → 부정
3. 암묵적 부정 — 기도/소원 형식이지만 비판 의도
   - "하루속히 끌어내려주소서", "제발 사라져줬으면" → 부정
4. 불신·의심: "안 믿는다", "말이 되냐", "황당하다", 음모론 암시
5. 위기·전쟁·재난을 부정 시각으로 서술: "전쟁 재개", "민간인 살상", "나라 망한다"
6. 기관·언론 직접 비판: "기레기", "적폐", "편파보도", "어용 언론"
7. 강한 요구/주장 — 비판적 맥락: "탄핵해야", "심판해야", "막아야 한다"
8. 풍자·비꼬기 — 긍정 단어를 썼지만 실제 비판·조롱 의도
   - 범죄·사건·비리 맥락에서 "~클라스", "~다운", "~답다" + ㅋ/ㅋㅋ
     예: "이대남 클라스 ㅋㅋㅋ" (범죄 관련) → 부정
   - 별표(*) 또는 따옴표로 감싼 과장된 칭찬·노래 형식으로 조롱
     예: "*이대남 신나는 노래~ 불러보자*" → 부정
   - "역시", "대단하다", "훌륭하다" + 부정 사건 맥락 조합
     예: "역시 대단한 나라다 ㅋㅋ" (비리/사고 뉴스에서) → 부정
9. 한탄·충격 감탄사 — 부정적 사건에 대한 탄식
   - "맙소사", "세상에나", "아이고", "에휴", "허참", "어이없다", "기가 막혀"
     예: "맙소사" (범죄 뉴스 댓글) → 부정
   - "말세다", "세상이 어떻게 되려고", "이게 뭔 세상이야" → 부정
10. 간접적 부정 반응 — 불편·두려움·체념을 사실처럼 표현
    - "이젠 ~도 챙겨야겠네 + ㅠ/ㅜ/😢 등 부정 이모지"
      예: "이젠 휴지 꼭 가지고 다녀야겠네~ㅠ" → 부정
    - "이러다 ~되겠다", "앞으로 ~못하겠네", "~해야 할 판이네" (체념·우려)
      예: "이제 화장실도 못 가겠네" → 부정
    - 주의: 이모지·ㅠ 없이 순수 사실 서술이면 중립 가능

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
댓글 감정 분류 결과와 대표 댓글을 받아 아래 형식으로 정확히 출력하세요.

[출력 형식 — 엄격히 준수]
[긍정]
(긍정 댓글들의 주요 의견을 3~4문장으로 요약. 긍정 댓글이 없으면 "긍정 댓글 없음")
[부정]
(부정 댓글들의 주요 의견을 3~4문장으로 요약. 부정 댓글이 없으면 "부정 댓글 없음")
[중립]
(중립 댓글들의 주요 의견을 3~4문장으로 요약. 중립 댓글이 없으면 "중립 댓글 없음")
[특이사항]
(봇 의심 댓글, 선동 패턴, 여론 양극화 등 눈에 띄는 패턴 — 없으면 "없음")

- 각 섹션 헤더([긍정], [부정], [중립], [특이사항])는 반드시 포함하세요.
- 헤더 외 부가 설명은 금지합니다."""


def _parse_summary(text: str) -> dict:
    """GPT 감정별 요약 텍스트를 파싱해서 dict로 반환."""
    sections = {"긍정": "", "부정": "", "중립": "", "특이사항": ""}
    current = None
    lines_buf: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped in ("[긍정]", "[부정]", "[중립]", "[특이사항]"):
            if current is not None:
                sections[current] = " ".join(lines_buf).strip()
            current = stripped[1:-1]
            lines_buf = []
        elif current is not None and stripped:
            lines_buf.append(stripped)

    if current is not None:
        sections[current] = " ".join(lines_buf).strip()

    return {
        "positive_summary": sections["긍정"] or "(긍정 댓글 없음)",
        "negative_summary": sections["부정"] or "(부정 댓글 없음)",
        "neutral_summary":  sections["중립"] or "(중립 댓글 없음)",
        "special_notes":    sections["특이사항"] or "없음",
    }


async def summarize_comments_async(
    texts: List[str],
    analysis: List[Tuple],
    positive: int,
    negative: int,
    neutral: int,
    bot_count: int,
) -> dict:
    total = len(analysis)
    pos, neg, neu = positive, negative, neutral

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
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_summary(raw)
    except Exception as e:
        print(f"[GPT 요약 오류] {e}")
        return {
            "positive_summary": "(요약 실패)",
            "negative_summary": "(요약 실패)",
            "neutral_summary":  "(요약 실패)",
            "special_notes":    "(요약 실패)",
        }


@dataclass
class AnalysisOutput:
    results: List[SentimentResult]
    positive: int
    negative: int
    neutral: int
    bot_count: int
    total: int
    positive_summary: str
    negative_summary: str
    neutral_summary: str
    special_notes: str


# ===== 핵심 분석 로직 (비동기) =====
async def _analyze_comments(comments: List[CommentItem]) -> AnalysisOutput:
    texts = [c.text for c in comments]

    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
    except Exception:
        sim_matrix = None

    analysis = await analyze_all_parallel(texts, sim_matrix)

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
            authorName=comment.authorName,
            authorId=comment.authorId,
            sentiment=label,
            sentimentScore=score,
            botScore=bot_score,
            isBot=is_bot,
            botReasons=reasons,
        ))

    summary = await summarize_comments_async(texts, analysis, positive, negative, neutral, bot_count)

    return AnalysisOutput(
        results=results,
        positive=positive,
        negative=negative,
        neutral=neutral,
        bot_count=bot_count,
        total=len(results),
        positive_summary=summary["positive_summary"],
        negative_summary=summary["negative_summary"],
        neutral_summary=summary["neutral_summary"],
        special_notes=summary["special_notes"],
    )


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
                authorName=snippet.get("authorDisplayName"),
                authorId=author_channel.get("value") if author_channel else None,
            ))

    return video_title, channel_name, view_count, published_at, comment_count_str, comments


# ===== API 엔드포인트 =====

@app.get("/health")
def health():
    return {"status": "ok", "engine": "gpt-4o-mini"}


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
        out = await _analyze_comments(comments)
        return YoutubeResult(
            videoTitle=video_title,
            channelName=channel_name,
            viewCount=view_count,
            publishedAt=published_at,
            videoCommentCount=comment_count_str,
            total=out.total,
            positive=out.positive,
            negative=out.negative,
            neutral=out.neutral,
            positivePct=round(out.positive / out.total * 100, 1),
            negativePct=round(out.negative / out.total * 100, 1),
            neutralPct=round(out.neutral   / out.total * 100, 1),
            botCount=out.bot_count,
            botPct=round(out.bot_count / out.total * 100, 1),
            positiveSummary=out.positive_summary,
            negativeSummary=out.negative_summary,
            neutralSummary=out.neutral_summary,
            specialNotes=out.special_notes,
            comments=out.results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


