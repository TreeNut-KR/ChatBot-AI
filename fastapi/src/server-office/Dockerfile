FROM chatbotai-base:latest

WORKDIR /app

# 애플리케이션 코드만 복사
COPY src/server-office /app/src/server-office
COPY prompt/ /app/prompt/

# 환경변수 설정 - RTX 2080 강제 할당
ENV PYTHONPATH="/opt/python-libs/lib/python3.11/site-packages:${PYTHONPATH}:/app:/app/src"

# server-office 디렉토리로 이동
WORKDIR /app/src/server-office

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
