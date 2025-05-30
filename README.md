# 🤖 ChatBot-AI Project

> AI 기반 챗봇 API 프로젝트입니다.  
> FastAPI 기반의 Office/Character API 서버와 Llama 기반 AI 모델을 Docker로 통합 운영합니다.

---

## 🏗️ 전체 아키텍처

- **office**: 업무용 챗봇 API (FastAPI, 8002)
- **character**: 캐릭터 챗봇 API (FastAPI, 8003)
- **nginx**: API Gateway (8001, reverse proxy, 커스텀 404 지원)
- **python-libs-init**: 공통 Python 라이브러리 볼륨 초기화

---

## 📂 주요 폴더 구조

```
ChatBot-AI/
├── fastapi/
│   ├── ai_model/           # AI 모델 파일 (볼륨 마운트)
│   ├── logs/               # 로그 파일 (공유 볼륨)
│   ├── prompt/             # 프롬프트 설정
│   ├── src/
│   │   ├── server-office/  # Office API 서버 코드
│   │   └── server-character/ # Character API 서버 코드
│   ├── .env                # 환경 변수
│   └── bot.yaml            # 봇 설정
├── nginx/
│   ├── nginx.conf          # nginx 리버스 프록시 설정
│   └── 404.html            # 커스텀 404 페이지
├── docker-compose.yml
└── README.md
```

---

## 🚀 빠른 시작 (Docker 기반)

### 1. **필수 요구사항**
- Docker, docker-compose
- NVIDIA GPU 및 드라이버 (CUDA 12.1 이상)
- (선택) 호스트 시간대가 Asia/Seoul로 설정되어 있으면 nginx 로그도 한국 시간으로 기록됨

### 2. **AI 모델 파일 준비**
- `fastapi/ai_model/MLP-KTLim/`, `fastapi/ai_model/QuantFactory/` 등  
  필요한 모델 파일을 Hugging Face 등에서 다운로드 후 해당 폴더에 위치시킵니다.
- `.dockerignore`에 의해 모델 파일은 이미지에 포함되지 않고,  
  반드시 **볼륨 마운트**로만 사용됩니다.

### 3. **환경 변수 파일 준비**
- `fastapi/src/.env` 파일에 필요한 환경 변수(OPENAI_API_KEY 등) 입력

### 4. **커스텀 404 페이지 준비**
- `nginx/404.html` 파일을 원하는 디자인으로 작성

### 5. **컨테이너 빌드 및 실행**
```bash
docker compose up --build
```

---

## 🌐 API Gateway (nginx) 구조

- **8001 포트**에서 모든 API를 통합 제공
- `/office/` → office 서버(8002)로 프록시
- `/character/` → character 서버(8003)로 프록시
- 존재하지 않는 경로는 `/404.html` 커스텀 페이지 반환

---

## 📝 주요 nginx 설정

```nginx
server {
    listen 8001;

    location ^~ /office/ {
        proxy_pass http://office_backend/;
        # ...헤더 설정 생략...
    }
    location ^~ /character/ {
        proxy_pass http://character_backend/;
        # ...헤더 설정 생략...
    }
    error_page 404 /404.html;
    location = /404.html {
        root /etc/nginx/html;
        internal;
    }
    location / {
        return 404;
    }
}
```

---

## 📦 도커 볼륨/마운트 구조

- **공통 라이브러리**: `python-libs` 볼륨 (컨테이너간 공유)
- **모델 파일**: 호스트의 `fastapi/ai_model/` → 컨테이너 내부 `/app/fastapi/ai_model/`
- **로그**: 호스트의 `fastapi/logs/` → 컨테이너 내부 `/app/logs/`
- **nginx 404.html**: 호스트의 `nginx/404.html` → 컨테이너 `/etc/nginx/html/404.html`

---

## 🛠️ 개발/운영 팁

- FastAPI 서버의 docs/redoc/openapi 경로는  
  각각 `/office/docs`, `/character/docs` 등으로 prefix를 다르게 설정해야  
  nginx 프록시 환경에서 충돌이 없습니다.
- 라우터 등록 시 prefix는 빈 문자열로 두고,  
  nginx에서 prefix를 붙여주는 구조가 권장됩니다.
- 모델 파일은 반드시 완전히 다운로드되어야 하며,  
  파일 크기/해시가 공식 배포본과 일치해야 합니다.

---

## 🔑 라이선스

- **AI 모델**: Meta AI 라이선스

---

## 📌 참고

- [AI 모델 정보](./fastapi/ai_model/README.md)
- [데이터셋 정보](./fastapi/datasets/README.md)
- [도메인/SSL 설정](./fastapi/certificates/DNS_README.md)
- [pem 파일 생성](./fastapi/certificates/PEM_README.md)

---