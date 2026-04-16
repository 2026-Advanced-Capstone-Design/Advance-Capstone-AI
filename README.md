# Advance-Capstone-AI
2026 심화 캡스톤 디자인 AI 부분

-크게 기사 부분 AI 랑 여론 분석 AI 2개 모델로 구분

-기사부분 주차별 계획
## 📅 주차별 개발 계획
 
#### 1~3주차: 기획 및 설계
- [x] 프로젝트 주제 선정 및 기능 구상
- [x] 유사 서비스 분석 (AllSides, Ground News)
- [x] 시스템 아키텍처 설계 (4-Layer)
- [x] 기술 스택 선정 및 근거 정리
 
#### 4주차 (3/26 ~ 4/1): 구조 발표 + AI 실험
- [x] Week 4 프로젝트 구조 발표 (12장 PPT)
- [x] AI 모델 비교 실험 (5개 모델, 샘플 68건 — 코드 검증)
- [x] DB 스키마 설계 (10개 테이블)
- [x] 요구사항 명세서 작성
- [x] GitHub PR 템플릿 · 컨벤션 문서 작성
 
---
 
### 🔄 진행 중
 
#### 5주차 (4/2 ~ 4/8): 백엔드 기반 세팅 + AI 본격 실험
 
**백엔드**
- [ ] Docker Compose로 MySQL + Redis 로컬 환경 구축
- [ ] application.yml 설정 (local/prod 분리)
- [ ] JPA 엔티티 10개 생성 + 테이블 자동 생성 확인
- [v] Spring Boot 정상 실행 확인
- [ ] Swagger 설정 + API stub 작성
- [ v] GitHub 레포 생성 + develop 브랜치 + 팀원 공유
 
**AI (Colab)**
- [v] NSMC 10,000건으로 5개 모델 본격 비교 실험
- [v] 실험 결과 정리 + 최종 모델 확정
 
> **🎯 마일스톤 M1:** Spring Boot 실행 + Swagger + GitHub 공유 완료
 
---
 
### 📋 예정
 
#### 6주차 (4/9 ~ 4/15): API 뼈대 + 라벨링 시작
 
**백엔드**
- [ ] 사용자 인증 구현(???) (회원가입 / 로그인 / JWT)
- [ ] 뉴스 분석 요청 API (POST /api/v1/articles/analyze)
- [ ] 분석 결과 조회 API (GET /api/v1/articles/{id}/result)
- [ ] 피드백 API (POST /api/v1/feedbacks)
- [ ] GlobalExceptionHandler + 통일 응답 포맷
 
**AI**
- [ ] 라벨링 데이터 수집 시작 (BigKinds 뉴스 기사) -> 이번주 바로 시작
- [ ] LLM 보조 라벨링 (GPT-4o-mini 초벌 → 사람 검수)
- [ ] 목표: 1,000건 이상
 
> **🎯 마일스톤 M2:** 프론트팀 개발 시작 가능 (Swagger 문서 공유)
 
---
 
#### 7주차 (4/16 ~ 4/22): 크롤링 + 전처리 + 파인튜닝
 
**백엔드**
- [ ] NewsCrawlerService 구현 (Jsoup, 언론사별 파서)
- [ ] OcrService 구현 (Tess4J + Clova OCR fallback)
- [ ] PreprocessService 구현 (Komoran + 정규식 + 문장 분리)
- [ ] input_type별 분기 처리 (TEXT / IMAGE / URL)
 
**AI (Colab)**
- [ ] 라벨링 데이터 2,000건+ 확보
- [ ] klue/roberta-base 파인튜닝 (감정 편향 태스크)
- [ ] 사실/의견 분류 태스크 파인튜닝
 
---
 
#### 8주차 (4/23 ~ 4/29): AI 서빙 + Spring 연동
 
**AI**
- [ ] 감정 분석 태스크 파인튜닝 완료 (3개 태스크 모두)
- [ ] Flask 워커 서버 구현 (/bias, /fact-opinion, /emotion)
- [ ] Docker 이미지 패키징
 
**백엔드**
- [ ] AiWorkerClient 구현 (RestTemplate → Flask 워커 호출)
- [ ] AnalysisService 구현 (CompletableFuture 병렬 호출)
- [ ] Docker Compose에 ai-worker 서비스 추가
- [ ] Spring Boot ↔ Flask 워커 연동 테스트
 
> **🎯 마일스톤 M3:** 텍스트 → AI 분석 → 결과 반환 파이프라인 동작
 
---
 
#### 9주차 (4/30 ~ 5/6): 점수 집계 + 외부 API 연동
 
- [ ] BiasScoreCalculator 구현 (4대 지표 가중 합산 → trust_score)
- [ ] 문장별 분석 결과 저장 (sentence_analyses, 하이라이팅 판정)
- [ ] LLM 연동 (GPT-4o-mini 요약 + 스펙트럼)
- [ ] AI 생성 판별 연동 (Binoculars + GPTZero)
- [ ] 팩트체크 연동 (공공데이터포털 + Google Fact Check API)
 
---
 
#### 10주차 (5/7 ~ 5/13): 캐싱 + 성능 최적화 + 중간 통합
 
- [ ] Redis 캐싱 구현 (URL 해시 기반, TTL 30분)
- [ ] Clova OCR 무료 한도 카운터 (Redis)
- [ ] API Rate Limiting
- [ ] 분석 상태 폴링 API (GET /api/v1/articles/{id}/status)
- [ ] N+1 쿼리 최적화 (fetch join)
- [ ] 프론트 ↔ 백엔드 첫 연동 테스트
 
> **🎯 마일스톤 M4:** 전체 기능 완성 (점수 + 캐싱 + 외부 API)
 
---
 
#### 11주차 (5/14 ~ 5/20): 배포 + CI/CD
 
- [ ] AWS EC2 세팅 (t3.medium, Swap 2GB)
- [ ] Docker Compose 프로덕션 설정
- [ ] Nginx 리버스 프록시 + HTTPS
- [ ] GitHub Actions CI/CD (develop → 빌드+테스트, main → 배포)
- [ ] 환경변수 관리 (.env)
 
> **🎯 마일스톤 M5:** EC2에서 전체 서비스 동작 + 자동 배포
 
---
 
#### 12주차 (5/21 ~ 5/27): 통합 테스트 + QA
 
- [ ] End-to-End 테스트 (입력 → 분석 → 결과 시각화)
- [ ] 크롬 확장 프로그램 연동 테스트
- [ ] 에지 케이스 테스트 (빈 텍스트, 잘못된 URL, OCR 실패 등)
- [ ] 부하 테스트 (JMeter, 동시 10건)
- [ ] 버그 수정 + 성능 개선
 
> **🎯 마일스톤 M6:** QA 완료, 버그 없는 상태
 
---
 
#### 13주차 (5/28 ~ 6/3): 마무리 + 문서화
 
- [ ] 최종 버그 수정 + 코드 리팩토링
- [ ] README 최종 정리 (설치 가이드, 실행 방법)
- [ ] API 문서 최종 정리 (Swagger)
- [ ] 최종 보고서 작성
- [ ] 실험 결과 정리 (모델 비교, 성능 테스트)
 
---
 
#### 14주차 (6/4 ~ 6/8): 발표 준비
 
- [ ] 최종 발표 PPT 제작 (15~20장)
- [ ] 발표 스크립트 작성
- [ ] 데모 시나리오 준비 (실시간 뉴스 분석 시연)
- [ ] 발표 리허설 2~3회
- [ ] 최종 보고서 제출
 
---
 
#### 15주차 (6/9 ~ 6/10): 최종 발표 🎓
 
- [ ] **6/10 최종 발표**
- [ ] 질의응답 대비 예상 질문 정리
- [ ] 시연 환경 최종 점검
 
---
 
## 🏆 마일스톤 요약
 
| 마일스톤 | 기한 | 체크포인트 | 상태 |
|----------|------|-----------|------|
| **M1** | 5주차 (4/8) | Spring Boot 실행 + Swagger + GitHub 공유 | 🔄 진행 중 |
| **M2** | 6주차 (4/15) | API + 인증 완성 → 프론트 개발 시작 가능 | ⬜ 예정 |
| **M3** | 8주차 (4/29) | 텍스트 → AI 분석 → 결과 반환 동작 | ⬜ 예정 |
| **M4** | 10주차 (5/13) | 전체 기능 완성 (점수 + 캐싱 + 외부 API) | ⬜ 예정 |
| **M5** | 11주차 (5/20) | EC2 배포 + CI/CD 자동화 | ⬜ 예정 |
| **M6** | 12주차 (5/27) | QA 완료 | ⬜ 예정 |
| **M7** | 15주차 (6/10) | **최종 발표** | ⬜ 예정 |
 
---
 
## 🚀 로컬 실행 방법
 
### 사전 요구사항
- Java 17+
- Docker & Docker Compose
- Node.js 18+ (프론트엔드)
 
### 실행
 
```bash
# 1. 레포 클론
git clone https://github.com/your-org/newslens-backend.git
cd newslens-backend
 
# 2. MySQL + Redis 실행
docker-compose up -d
 
# 3. Spring Boot 실행
./gradlew bootRun
 
# 4. Swagger 확인
# http://localhost:8080/swagger-ui.html
```
 
---
 
## 📁 프로젝트 구조
 
```
src/main/java/com/example/Advance_Capstone/
├── global/                    # 공통 설정
│   ├── config/                # Security, Redis, Swagger, CORS
│   ├── exception/             # 전역 예외 처리
│   ├── response/              # 통일 응답 포맷
│   └── auth/                  # JWT 토큰
├── domain/                    # 도메인별 패키지
│   ├── user/                  # 사용자 (인증)
│   ├── article/               # 기사 (입력)
│   ├── analysis/              # 분석 결과
│   ├── youtube/               # 유튜브 댓글
│   └── feedback/              # 피드백
└── infra/                     # 외부 연동
    ├── ai/                    # AI 워커 클라이언트
    ├── crawler/               # Jsoup 크롤링
    ├── ocr/                   # Tess4J + Clova OCR
    └── external/              # GPT, GPTZero, 팩트체크
```
 
---
 
## 📐 컨벤션
 
### 브랜치 전략
```
main ← 배포용 (PR 머지만)
└── develop ← 개발 통합
    ├── feature/기능명
    ├── fix/버그명
    └── refactor/대상명
```
 
### 커밋 메시지
```
feat: 뉴스 URL 크롤링 서비스 구현
fix: JWT 토큰 만료 시 500 에러 수정
refactor: AnalysisService 메서드 분리
docs: API 엔드포인트 Swagger 추가
test: ArticleService 단위 테스트 추가
chore: Redis 의존성 추가
```
 
### PR 규칙
- `develop` 브랜치로만 PR 생성
- 최소 1명 리뷰 승인 후 머지
- PR 템플릿 양식 준수
- 로컬 빌드 성공 확인 (`./gradlew build`)
 
---
