FROM chatbotai-base:latest

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone
ENV TZ=Asia/Seoul

# 애플리케이션 코드만 복사
COPY src/server-office /app/src/server-office
COPY prompt/ /app/prompt/

ENV PYTHONPATH="/opt/python-libs/lib/python3.11/site-packages:${PYTHONPATH}:/app:/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# server-office 디렉토리로 이동
WORKDIR /app/src/server-office

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002", "--http", "h11", "--loop", "asyncio"]
