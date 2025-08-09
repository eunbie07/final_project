# 🚀 Final Project - Environment Setup Guide

이 프로젝트를 새로운 환경에서 실행하기 위한 완전한 설정 가이드입니다.

## 📋 목차
- [필수 설치 프로그램](#필수-설치-프로그램)
- [프로젝트 클론](#프로젝트-클론)
- [환경변수 설정](#환경변수-설정)
- [서비스별 설정](#서비스별-설정)
- [실행 확인](#실행-확인)
- [문제 해결](#문제-해결)

## 🛠️ 필수 설치 프로그램

### 1. 기본 도구
```bash
# Git
https://git-scm.com/

# Node.js (v18 이상 권장)
https://nodejs.org/

# Python (3.9 이상)
https://python.org/

# uv (Python 패키지 매니저)
pip install uv
```

### 2. 데이터베이스 (room-measure용)
```bash
# PostgreSQL
https://postgresql.org/

# 또는 Docker 사용
docker pull postgres:15
```

## 📥 프로젝트 클론

```bash
git clone <repository-url>
cd final_project
```

## 🔧 환경변수 설정

### 자동 설정 스크립트 실행
```bash
# Windows
setup-env.bat

# macOS/Linux
chmod +x setup-env.sh
./setup-env.sh
```

### 수동 설정 (자동 스크립트 실패 시)

#### 1. imggen-realistic-uv7000-triple 설정

**📁 파일 경로:** `imggen-realistic-uv7000-triple/backend/.env`

```bash
# API Keys (발급 필요)
STABILITY_API_KEY=sk-your-stability-key-here
REPLICATE_API_TOKEN=r8_your-replicate-token-here

# Google Cloud (선택사항)
GCP_SERVICE_ACCOUNT_JSON_PATH=./service-account.json
```

**🔑 API 키 발급 방법:**
- **Stability AI**: https://platform.stability.ai/account/keys
- **Replicate**: https://replicate.com/account/api-tokens

**📄 Google Cloud 서비스 계정 (Vertex AI 사용 시):**
1. Google Cloud Console → IAM & Admin → Service Accounts
2. Create Service Account → Vertex AI User 권한 부여
3. Create Key → JSON 다운로드
4. `imggen-realistic-uv7000-triple/backend/service-account.json`로 저장

#### 2. room-measure 설정

**📁 파일 경로:** `room-measure/.env`

```bash
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=room_measure
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://postgres:your_secure_password_here@localhost:5432/room_measure

# JWT Authentication
JWT_SECRET=your-super-secret-jwt-key-minimum-32-characters-long

# Environment
NODE_ENV=development
```

**🔒 보안 권장사항:**
- `POSTGRES_PASSWORD`: 최소 12자 이상의 강력한 비밀번호
- `JWT_SECRET`: 최소 32자 이상의 랜덤 문자열

## 🏗️ 서비스별 설정

### 1. 이미지 생성 서비스 (imggen-realistic-uv7000-triple)

```bash
cd imggen-realistic-uv7000-triple/backend

# 의존성 설치
uv pip install -e .

# 서버 실행 (포트: 7000)
uv run uvicorn main:app --host 0.0.0.0 --port 7000 --reload
```

**테스트:** http://localhost:7000/docs

### 2. 룸 측정 서비스 (room-measure)

#### Backend Cloud (포트: 3000)
```bash
cd room-measure/eunbi/backend-cloud

# 의존성 설치
uv pip install -e .

# 데이터베이스 초기화
psql -U postgres -d room_measure -f schema.sql

# 서버 실행
uv run uvicorn main:app --host 0.0.0.0 --port 3000 --reload
```

#### Backend Local (포트: 3010)
```bash
cd room-measure/eunbi/backend-local

# 의존성 설치
uv pip install -e .

# 서버 실행
uv run uvicorn main:app --host 0.0.0.0 --port 3010 --reload
```

#### Frontend (포트: 4000)
```bash
cd room-measure/eunbi/frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

#### Main Frontend (포트: 4010)
```bash
cd room-measure/frontend-main

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

### 3. 이미지 생성 Frontend (포트: 8080)

```bash
cd imggen-realistic-uv7000-triple/frontend

# 개발 서버 실행 (간단한 HTML 파일)
python -m http.server 8080
# 또는
npx serve . -p 8080
```

## ✅ 실행 확인

### 포트 확인
```bash
# Windows
netstat -ano | findstr :3000
netstat -ano | findstr :3010
netstat -ano | findstr :4000
netstat -ano | findstr :4010
netstat -ano | findstr :7000
netstat -ano | findstr :8080

# macOS/Linux
lsof -i :3000,:3010,:4000,:4010,:7000,:8080
```

### 서비스 접속 테스트
- **이미지 생성 API**: http://localhost:7000/docs
- **이미지 생성 Frontend**: http://localhost:8080
- **룸 측정 Backend Cloud**: http://localhost:3000/docs
- **룸 측정 Backend Local**: http://localhost:3010/docs
- **룸 측정 Frontend**: http://localhost:4000
- **메인 Frontend**: http://localhost:4010

## 🐳 Docker 실행 (권장)

### 이미지 생성 서비스
```bash
cd imggen-realistic-uv7000-triple

# 환경변수 파일 확인 후 실행
docker compose --env-file backend/.env up --build -d
```

### 룸 측정 서비스
```bash
cd room-measure

# Docker Compose 실행
docker compose up --build -d
```

## 🔧 문제 해결

### 1. 포트 충돌 시
```bash
# 포트 사용 중인 프로세스 종료 (Windows)
netstat -ano | findstr :<PORT>
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:<PORT> | xargs kill -9
```

### 2. 데이터베이스 연결 오류
```bash
# PostgreSQL 상태 확인
pg_ctl status

# 데이터베이스 재시작
sudo systemctl restart postgresql
```

### 3. API 키 오류
- API 키 유효성 확인
- 요금제 및 크레딧 잔액 확인
- `.env` 파일 경로 및 내용 확인

### 4. Python 의존성 오류
```bash
# uv 재설치
pip install --upgrade uv

# 가상환경 재생성
uv venv --python 3.9
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 5. Node.js 의존성 오류
```bash
# node_modules 재설치
rm -rf node_modules package-lock.json
npm install

# 또는 yarn 사용
rm -rf node_modules yarn.lock
yarn install
```

## 📞 지원

환경 설정 중 문제가 발생하면:
1. [문제 해결](#문제-해결) 섹션 확인
2. 각 서비스별 README 파일 참조
3. 로그 파일 확인 및 에러 메시지 분석

---

**⚠️ 주의사항:**
- 모든 `.env` 파일과 API 키는 절대 Git에 커밋하지 마세요
- 프로덕션 환경에서는 더 강력한 시크릿 키를 사용하세요
- 정기적으로 API 키를 갱신하세요