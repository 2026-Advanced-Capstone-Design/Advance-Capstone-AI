FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 소스 복사 (API 서버만)
COPY src/api_server ./src/api_server

# 포트
EXPOSE 8000

# 실행
CMD ["uvicorn", "src.api_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
