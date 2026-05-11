import re
from bs4 import BeautifulSoup

# 한국어 뉴스 기사 정제를 위한 패턴들
# email 양식, url 양식, 
_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL = re.compile(r"https?://\S+")
_REPORTER_SUFFIX = re.compile(r"[가-힣]{2,4}\s*기자")
_AD_PHRASES = re.compile(r"(무단\s*전재|재배포\s*금지|저작권|Copyright|ⓒ|©).{0,50}")
_MULTI_SPACE = re.compile(r"\s+")
_SPECIAL = re.compile(r"[^\uAC00-\uD7A3a-zA-Z0-9\s.,!?%()·\-\"']")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[가-힣A-Z\"])")


def clean_html(raw: str) -> str:
    """HTML 태그 제거 + BeautifulSoup 기반 텍스트 추출"""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "img", "figure", "figcaption"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def clean_text(text: str) -> str:
    """이메일, URL, 기자명, 광고 문구, 특수문자 제거"""
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _REPORTER_SUFFIX.sub(" ", text)
    text = _AD_PHRASES.sub(" ", text)
    text = _SPECIAL.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if len(s.strip()) > 10]


def preprocess(raw: str, is_html: bool = False) -> dict:
    """
    입력 텍스트를 정제하고 문장 분리합니다.
    반환: {"cleaned": str, "sentences": list[str]}
    """
    text = clean_html(raw) if is_html else raw
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    return {"cleaned": cleaned, "sentences": sentences}
