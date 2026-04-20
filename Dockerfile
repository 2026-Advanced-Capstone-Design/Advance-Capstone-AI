FROM python:3.11-slim

WORKDIR /app

# Java가 필요한 KoNLPy를 위해 JDK 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU 전용 PyTorch 먼저 설치 (CUDA 패키지 수백 MB 방지)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
