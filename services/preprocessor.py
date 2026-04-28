import re
from bs4 import BeautifulSoup

# 기본 정제 패턴
_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL = re.compile(r"https?://\S+")
_REPORTER_SUFFIX = re.compile(r"[가-힣]{2,4}\s*기자")
_COPYRIGHT = re.compile(r"(무단\s*전재|재배포\s*금지|저작권|Copyright|ⓒ|©).{0,50}")
_MULTI_SPACE = re.compile(r"\s+")
_SPECIAL = re.compile(r"[^가-힣a-zA-Z0-9\s.,!?%()·\-\"']")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[가-힣A-Z\"])")

# 출처 표기 추출 패턴 — 제거하지 않고 별도 수집
_SOURCE = re.compile(r"(사진|자료|제공|출처|이미지)\s*=\s*[^\n,./]{1,30}")

# 문장 레벨 광고성 키워드 — 해당 키워드가 포함된 문장 전체를 제거
_AD_KEYWORDS = re.compile(
    r"구독\s*(하기|해주세요|바랍니다|신청|버튼|하면)"
    r"|알림\s*(설정|신청)"
    r"|좋아요\s*(눌러|클릭|부탁)"
    r"|팔로우\s*(하기|해주세요|바랍니다)"
    r"|앱\s*(다운로드|설치|스토어)"
    r"|구글\s*플레이|앱\s*스토어|플레이\s*스토어"
    r"|\[광고\]|\[PR\]|\[협찬\]|\[AD\]"
    r"|이\s*기사는\s*.{0,15}(과|와)\s*(함께|협력|제휴)"
    r"|뉴스레터\s*(구독|신청)"
    r"|이메일로\s*(받아|구독|신청)"
    r"|관련\s*기사\s*(▶|►|→|>|바로가기)"
    r"|이\s*기사와\s*함께\s*읽으"
    r"|선착순|무료\s*체험\s*(신청|하기)"
    r"|지금\s*바로\s*(신청|구매|주문|다운)"
    r"|구매\s*(하기|링크|바로가기)"
    r"|쇼핑\s*(하기|바로가기)"
    r"|유튜브\s*(구독|채널|영상\s*보기)"
    r"|인스타그램|페이스북|트위터|카카오톡\s*(채널|친구추가)"
)


def clean_html(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "img", "figure", "figcaption"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def clean_text(text: str) -> str:
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _REPORTER_SUFFIX.sub(" ", text)
    text = _COPYRIGHT.sub(" ", text)
    text = _SPECIAL.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def extract_sources(text: str) -> list[str]:
    """출처 표기(사진=, 제공= 등)를 원문에서 추출"""
    return [m.group().strip() for m in _SOURCE.finditer(text)]


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if len(s.strip()) > 10]


def filter_ad_sentences(sentences: list[str]) -> list[str]:
    """광고성 키워드가 포함된 문장을 제외"""
    return [s for s in sentences if not _AD_KEYWORDS.search(s)]


def preprocess(raw: str, is_html: bool = False) -> dict:
    """
    입력 텍스트를 정제하고 문장 분리합니다.
    반환:
      cleaned   - 특수문자·저작권 등이 제거된 본문 전체
      sentences - 광고성 문장이 제거된 본문 문장 리스트
      sources   - 기사 내 출처 표기 (사진=, 제공= 등) 리스트
    """
    text = clean_html(raw) if is_html else raw
    sources = extract_sources(text)        # 정제 전 원문에서 출처 추출
    cleaned = clean_text(text)
    all_sentences = split_sentences(cleaned)
    sentences = filter_ad_sentences(all_sentences)
    return {"cleaned": cleaned, "sentences": sentences, "sources": sources}
