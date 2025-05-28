FROM chatbotai-base:latest

WORKDIR /app

# 애플리케이션 코드만 복사
COPY src/server-character /app/src/server-character
COPY prompt/ /app/prompt/

# Python 경로 설정 - RTX 3060 강제 할당
ENV PYTHONPATH="/opt/python-libs/lib/python3.11/site-packages:${PYTHONPATH}:/app:/app/src"

# server-character 디렉토리로 이동
WORKDIR /app/src/server-character

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
