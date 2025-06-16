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
COPY src/core/__init__.py /app/src/core/__init__.py
COPY src/domain/character/ /app/src/domain/character/
COPY src/domain/shared/ /app/src/domain/shared/
COPY src/domain/__init__.py /app/src/domain/__init__.py
COPY src/llm/ /app/src/llm/
COPY src/api/__init__.py /app/src/api/__init__.py
COPY src/server/character /app/src/server/character
COPY prompt/ /app/prompt/

ENV PYTHONPATH="/opt/python-libs/lib/python3.11/site-packages:/app:/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# src 디렉토리에서 실행하도록 변경
WORKDIR /app/src

# 애플리케이션 파일을 직접 실행하도록 변경
CMD ["python", "server/character/server.py"]