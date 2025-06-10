FROM chatbotai-base:latest

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone
ENV TZ=Asia/Seoul

# 새로운 구조에 맞춰 애플리케이션 코드 복사
COPY src/api/character/ /app/src/api/character/
COPY src/core/character/ /app/src/core/character/
COPY src/domain/character/ /app/src/domain/character/
COPY src/domain/shared/ /app/src/domain/shared/
COPY src/llm/ /app/src/llm/
COPY src/server-character /app/src/server-character
COPY prompt/ /app/prompt/

ENV PYTHONPATH="/opt/python-libs/lib/python3.11/site-packages:${PYTHONPATH}:/app:/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# server-character 디렉토리로 이동
WORKDIR /app/src/server-character

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003", "--http", "h11", "--loop", "asyncio"]
