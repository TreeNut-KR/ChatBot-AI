# 베이스 이미지 설정
FROM python:3.12.1

# wait-for-it.sh 복사 및 실행 권한 부여
COPY wait-for-it.sh /usr/local/bin/wait-for-it
RUN sed -i 's/\r$//' /usr/local/bin/wait-for-it && chmod +x /usr/local/bin/wait-for-it

# Step 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# 필요한 패키지를 먼저 설치
COPY requirements.txt .
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install -r requirements.txt

# Step 2: Final stage
FROM python:3.11-slim

WORKDIR /app

# 빌드 단계에서 생성한 venv를 복사
COPY --from=builder /app/venv /app/venv
COPY . /app

# 환경 변수 설정
ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:$PATH"

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]


# 어플리케이션 실행 (MySQL 준비 대기)
CMD ["/usr/local/bin/wait-for-it", "mysql:3306", "--", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
