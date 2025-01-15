# 베이스 이미지 설정
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Python 설치
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev curl

# pip 업그레이드
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

# spacy 설치 및 모델 다운로드
RUN python3.11 -m pip install spacy
RUN python3.11 -m spacy download en_core_web_sm
RUN python3.11 -m spacy download ko_core_news_sm

# FastAPI 서버 코드 복사
COPY src /app/src

# 환경 변수 파일 복사
COPY src/.env /app/src/.env

# bot.yaml 파일 복사
COPY src/bot.yaml /app/src/bot.yaml

# PYTHONPATH 설정
ENV PYTHONPATH=/app/src

# 포트 설정
EXPOSE 8001

# FastAPI 서버 실행
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8001"]