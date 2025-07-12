# ChatBot-AI 프로젝트

[![alt text](https://lh3.googleusercontent.com/d/1H62LOQ8yeql3HQ5OZT4fIzdydTdMhbiw)](https://treenut.ddns.net)

<div align="center">
  <a href="https://github.com/TreeNut-KR/ChatBot-AI">
    <img src="https://github-readme-stats.vercel.app/api/pin/?username=TreeNut-KR&repo=ChatBot-AI&theme=dark&show_owner=true" alt="ChatBot Repository"/>
  </a>
</div>

<br>

---
# 프로젝트 구성원

| 구성원 | 업무 | 사용 기술 |  
|--------|--------|------------|  
| 서정훈 (CutTheWire) | 프로젝트 매니저, 백엔드 | FastAPI, Llama CPP CUDA |  

## 🏗️ 전체 아키텍처

- **office**: 업무용 챗봇 API (FastAPI, 8002)
- **character**: 캐릭터 챗봇 API (FastAPI, 8003)
- **nginx**: API Gateway (8001, reverse proxy, 커스텀 404 지원)
- **python-libs-init**: 공통 Python 라이브러리 볼륨 초기화

## 📋 시스템 아키텍처 다이어그램
![System-Architecture-Diagram-ChatBot](https://cutwire.myddns.me/images/System-Architecture-Diagram-ChatBot.webp)

## 📋 패키지 다이어그램 
![Package-Diagram-ChatBot(AI)](https://cutwire.myddns.me/images/Package-Diagram-ChatBot(AI).webp)

## 🌐 API Gateway (nginx) 구조

- **8001 포트**에서 모든 API를 통합 제공
- `/office/` → office 서버(8002)로 프록시
- `/character/` → character 서버(8003)로 프록시
- 존재하지 않는 경로는 `/404.html` 커스텀 페이지 반환


## 📊 요청 성능
- **v1.7.4** 버전 기준
- **측정 일자**: 2025-07-12 (토) 15:08:57 GMT+0900 (한국 표준시)

<div align="left">
    <a href="/visualization/chatbot-ai">
        <img src="https://img.shields.io/badge/성능차트-상세보기-green?style=for-the-badge&logo=chartdotjs" alt="성능차트 상세보기"/>
    </a>
</div>


## 📅 개발 로드맵 및 버전 릴리즈 일정

### 간트 차트 (ChatBot AI 버전 릴리즈)
![Gantt-Chart-ChatBot(AI)](https://cutwire.myddns.me/images/Gantt-Chart-ChatBot(AI).webp)

### 주요 마일스톤

| 버전 | 기간 | 주요 성과 | 아키텍처 변화 |
|------|------|-----------|---------------|
| **v1.0.x** | 2024.09-2024.10 | 단일 Llama 모델, 스트리밍 지원 | Transformers 기반 스트리밍 |
| **v1.1.x** | 2024.10-2025.01 | 듀얼 GPU 구성, Bllossom 모델 추가 | Llama + Bllossom 멀티모델 |
| **v1.2.x** | 2025.01-2025.02 | Lumimaid GGUF 전환 | 성능 최적화 (GGUF) |
| **v1.3.x** | 2025.02 | DuckDuckGo 검색 API 연동 | 외부 검색 통합 |
| **v1.4.x** | 2025.02-2025.03 | SSL/TLS 보안, 인증서 관리 | HTTPS 프로덕션 환경 |
| **v1.5.x** | 2025.03-2025.04 | 라우터 분리, OpenAI 모델 추가 | 하이브리드 API 아키텍처 |
| **v1.6.x** | 2025.04-2025.05 | MVC 구조, GitHub Actions | 체계적인 개발 파이프라인 |
| **v1.7.x** | 2025.05-2025.06 | Docker 컨테이너화, nginx 게이트웨이 | 마이크로서비스 완성 |

### 개발 통계

- **총 개발 기간**: 9개월 (2024.09 ~ 2025.06)
- **메이저 버전**: 8개 (v1.0.x ~ v1.7.x)
- **릴리즈 횟수**: 20회
- **주요 기술 전환**: 4회 (단일→듀얼→GGUF→마이크로서비스)

### 📄 v1.0.x
<div align="left">
    <a href="https://cutwire.myddns.me/portfolio/reference/chatbot-ai/version(1.0.x).md">
        <img src="https://img.shields.io/badge/명세-상세보기-blue?style=for-the-badge&logo=markdown" alt="명세 상세보기"/>
    </a>
</div>

- `First Commit Days` : 2024-10-19 (토) 23:02:45 GMT+0900 (한국 표준시)
- `Last Commit Days` : 2024-12-16 (월) 18:22:23 GMT+0900 (한국 표준시)

### 📄 v1.1.x
<div align="left">
    <a href="https://cutwire.myddns.me/portfolio/reference/chatbot-ai/version(1.1.x).md">
        <img src="https://img.shields.io/badge/명세-상세보기-blue?style=for-the-badge&logo=markdown" alt="명세 상세보기"/>
    </a>
</div>

- `First Commit Days` : 2025-01-15 (수) 15:40:49 GMT+0900 (한국 표준시)

### 📄 v1.2.x
<div align="left">
    <a href="https://cutwire.myddns.me/portfolio/reference/chatbot-ai/version(1.2.x).md">
        <img src="https://img.shields.io/badge/명세-상세보기-blue?style=for-the-badge&logo=markdown" alt="명세 상세보기"/>
    </a>
</div>

- `First Commit Days` : 2025-02-18 (화) 10:42:34 GMT+0900 (한국 표준시)

### 📄 v1.3.x
<div align="left">
    <a href="https://cutwire.myddns.me/portfolio/reference/chatbot-ai/version(1.3.x).md">
        <img src="https://img.shields.io/badge/명세-상세보기-blue?style=for-the-badge&logo=markdown" alt="명세 상세보기"/>
    </a>
</div>

- `First Commit Days` : 2025-02-18 (화) 11:26:36 GMT+0900 (한국 표준시)
- `Last Commit Days` : 2024-03-15 (토) 15:44:49 GMT+0900 (한국 표준시)

### 📄 v1.4.x
<div align="left">
    <a href="https://cutwire.myddns.me/portfolio/reference/chatbot-ai/version(1.4.x).md">
        <img src="https://img.shields.io/badge/명세-상세보기-blue?style=for-the-badge&logo=markdown" alt="명세 상세보기"/>
    </a>
</div>

- `First Commit Days` : 2024-03-15 (토) 15:47:20 GMT+0900 (한국 표준시)
- `Last Commit Days` : 2024-03-16 (일) 18:24:02 GMT+0900 (한국 표준시)

### 📄 v1.5.x
<div align="left">
    <a href="https://cutwire.myddns.me/portfolio/reference/chatbot-ai/version(1.5.x).md">
        <img src="https://img.shields.io/badge/명세-상세보기-blue?style=for-the-badge&logo=markdown" alt="명세 상세보기"/>
    </a>
</div>

- `First Commit Days` : 2024-03-21 (금) 15:41:35 GMT+0900 (한국 표준시)
- `Last Commit Days` : 2024-05-03 (토) 18:56:29 GMT+0900 (한국 표준시)

### 📄 v1.6.x
<div align="left">
    <a href="https://cutwire.myddns.me/portfolio/reference/chatbot-ai/version(1.6.x).md">
        <img src="https://img.shields.io/badge/명세-상세보기-blue?style=for-the-badge&logo=markdown" alt="명세 상세보기"/>
    </a>
</div>

- `First Commit Days` : 2024-05-10 (토) 04:43:23 GMT+0900 (한국 표준시)
- `Last Commit Days` : 2024-05-16 (금) 01:30:44 GMT+0900 (한국 표준시)

### 📄 v1.7.x
<div align="left">
    <a href="https://cutwire.myddns.me/portfolio/reference/chatbot-ai/version(1.7.x).md">
        <img src="https://img.shields.io/badge/명세-상세보기-blue?style=for-the-badge&logo=markdown" alt="명세 상세보기"/>
    </a>
</div>

- `First Commit Days` : 2024-05-30 (금) 19:19:05 GMT+0900 (한국 표준시)
- `Last Commit Days` : 2024-06-16 (월) 16:36:43 GMT+0900 (한국 표준시)