# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고하는 가이드입니다.

**모든 응답은 반드시 한국어로 작성하세요.**

## 환경 설정

스크립트 실행 전 가상환경을 활성화해야 합니다:
```powershell
# PowerShell
& venv\Scripts\Activate.ps1
```

`.env` 파일에 필요한 환경 변수:
- `YOUTUBE_API_KEY` — YouTube Data API v3 키
- `HUGGINGFACE_TOKEN` — HuggingFace 쓰기 토큰 (모델 업로드용)
- `OPENAI_API_KEY` — (아직 미사용)

## 주요 실행 명령어

```bash
# 메인 유튜브 댓글 분석 파이프라인 실행
python src/test_real_comments.py

# 3클래스 감정 분석 모델 학습 (GPU 필요, 약 10~15분 소요)
python src/train/train_sentiment_3class.py

# 학습된 모델을 HuggingFace Hub에 업로드
python src/train/upload_model.py

# 업로드된 HuggingFace 모델 테스트
python src/test_uploaded_model.py

# KR-FinBert 베이스라인 모델 테스트
python src/test_kote.py
```

## 아키텍처 개요

이 프로젝트는 한국어 뉴스 댓글 분석 시스템("뉴스 렌즈")으로, 두 가지 핵심 기능을 제공합니다.

### 1. 감정 분석 (3클래스)
- **베이스 모델**: 한국어 뉴스/게임/예능 댓글에 파인튜닝된 `klue/roberta-base`
- **라벨**: 0=부정, 1=긍정, 2=중립
- **학습 데이터**: NSMC 데이터셋 (6,000개 샘플) + 수작업 한국어 댓글 (뉴스/게임/예능) × 3배 증강
- **학습 하이퍼파라미터**: lr=2e-5, warmup_ratio=0.1, weight_decay=0.01, epochs=7
- **데이터 파일**: `src/train/training_data.py` — 댓글 데이터만 별도 관리 (뉴스/게임/예능 카테고리)
- **로컬 모델 경로**: `models/sentiment_3class/best/` (RoBERTa, 32k 어휘, hidden size 768)
- **HuggingFace**: `thd011124/news-comment-sentiment`

### 2. 봇/스팸 탐지 (하이브리드)
`src/test_real_comments.py::detect_bot()`에 구현:
- **AI 모델 점수** (50%): `openai-community/roberta-base-openai-detector`로 각 댓글 점수 산출
- **규칙 기반 점수** (50%): 4가지 신호 — TF-IDF 코사인 유사도(>0.8) 기반 중복 댓글, 구어체/이모지 부재, 선동 키워드(반드시, 촉구, 규탄 등), 문장 길이 균일성
- 최종 점수 ≥50이면 봇으로 판정

### 메인 파이프라인 (`src/test_real_comments.py`)
1. 유튜브 영상 ID 또는 URL을 입력받음
2. YouTube Data API v3로 상위 댓글 최대 50개 수집
3. 전체 댓글에 대한 TF-IDF 유사도 행렬 생성
4. 각 댓글에 감정 분석 + 봇 탐지 실행
5. 집계 통계 및 댓글별 결과를 stdout에 출력

### 디렉토리 구조
- `src/train/` — 모델 학습 및 HuggingFace 업로드 스크립트
- `src/collect/`, `src/inference/`, `src/preprocess/` — 구조만 잡혀 있고 현재 비어 있음
- `models/sentiment_3class/` — 학습 체크포인트 + `best/` (배포된 최종 모델)
- `data/raw/`, `data/processed/`, `data/labeled/` — 데이터 디렉토리 (raw는 git 제외)
- `tests/`, `notebooks/` — 구조만 잡혀 있고 현재 비어 있음

### 라벨 규칙
모든 스크립트에서 `{0: "부정", 1: "긍정", 2: "중립"}`을 사용합니다. 로컬 `best/` 모델의 config는 아직 `LABEL_0/1/2`로 되어 있으며, `upload_model.py`에서 HuggingFace 업로드 전에 한국어 라벨로 설정합니다.
