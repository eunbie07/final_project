# 이집맞집 - 방측정 서비스

은비의 방측정/가구배치 서비스입니다. MSA(Micro Service Architecture) 방식으로 구성되어 있습니다.

## 📁 프로젝트 구조

```
IjipMatjip/
├── eunbi/                       # 은비 전용 서비스
│   ├── frontend/               # 방측정 전용 앱
│   ├── backend-cloud/          # 클라우드 데이터 관리 (MongoDB)
│   └── backend-local/          # 로컬 AI 이미지 처리
├── frontend-main/              # 팀 통합 홈페이지
├── integration-files/          # 팀 통합용 템플릿
├── frontend-backup/                   # 기존 백업
└── INTEGRATION_GUIDE.md        # 팀 통합 가이드
```

## 🚀 실행 방법

### 1. 백엔드 서버들 (필수)

**클라우드 API 서버 (터미널 1)**
```bash
cd eunbi/backend-cloud
uv run uvicorn main:app --host 0.0.0.0 --port 3000 --reload --log-level debug
```

**로컬 AI 처리 서버 (터미널 2)**
```bash
cd eunbi/backend-local
uv run uvicorn main:app --host 0.0.0.0 --port 3010 --reload --log-level debug
```

### 2. 프론트엔드 (택1 또는 동시)

**방측정 전용 앱 (터미널 3)**
```bash
cd eunbi/frontend
npm run dev -- --host 0.0.0.0 --port 4000
```

**팀 통합 앱 (터미널 4)**
```bash
cd frontend-main
npm run dev -- --host 0.0.0.0 --port 4010
```

## 🌐 접속 주소

| 서비스 | 주소 | 설명 |
|--------|------|------|
| **방측정 전용** | `http://localhost:4000` | 은비 개발용 (순수 방측정만) |
| **팀 통합** | `http://localhost:4010` | 이집맞집 통합 플랫폼 |
| 클라우드 API | `http://localhost:3000` | 데이터 저장/조회 |
| 로컬 AI API | `http://localhost:3010` | 이미지 AI 처리 |

## 💻 개발 환경

### 권장 개발 방식
- **평상시**: 방측정 전용 앱(`localhost:4000`)에서 개발
- **통합 테스트**: 팀 통합 앱(`localhost:4010`)에서 확인

### 서비스별 특징

**방측정 전용 앱 (`eunbi/frontend`)**
- 브랜드명: "방측정"
- 라우팅: `/`, `/login`, `/signup`
- 기능: 순수 방측정/가구배치만

**팀 통합 앱 (`frontend-main`)**
- 브랜드명: "이집맞집"  
- 라우팅: `/`, `/room-planner`, `/ai-design`, `/find-house`, `/login`, `/signup`
- 기능: 홈페이지 + 모든 팀 서비스 연결

## 🛠️ 기술 스택

### 백엔드
- **Python**: FastAPI, uvicorn
- **AI**: YOLO, MiDaS, OpenCV
- **데이터베이스**: MongoDB
- **패키지 관리**: uv

### 프론트엔드
- **React**: Vite, React Router
- **3D**: Three.js, @react-three/fiber
- **스타일링**: Tailwind CSS
- **패키지 관리**: npm

## 📋 주요 기능

### 방측정 서비스
- 📷 이미지 업로드 및 분석
- 🏠 방 크기 자동 측정
- 🪟 창문 자동 감지
- 🪑 3D 가구 배치 시뮬레이션
- 💾 레이아웃 저장/불러오기
- 👤 사용자 인증 (회원가입/로그인)

### API 엔드포인트
- `POST /detect-windows` - 창문 감지
- `POST /auto-detect-room` - 방 자동 인식  
- `POST /estimate-room-size` - 방 크기 측정
- `GET /room-layouts` - 레이아웃 조회
- `POST /save-room-layout` - 레이아웃 저장

## 🤝 팀 통합

이 서비스는 IjipMatjip 프로젝트의 일부입니다.

- **은비**: 방측정/가구배치 서비스 ✅
- **민아**: AI 인테리어 디자인 서비스 
- **단비**: 집찾기/추천 서비스

각 서비스는 독립적으로 개발/배포되며, `frontend-main`에서 통합됩니다.

## 📚 추가 문서

- [팀 통합 가이드](./INTEGRATION_GUIDE.md)
- [통합 API 명세](./integration-files/teamApi.js)
- [환경변수 설정](./integration-files/.env.example)