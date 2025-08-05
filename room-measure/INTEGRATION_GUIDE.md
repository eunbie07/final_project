# IjipMatjip 팀 통합 가이드

이 문서는 3팀(은비, 민아, 단비)의 서비스를 하나의 플랫폼으로 통합하는 가이드입니다.

## 📁 새 레포 폴더 구조 (브랜치별 완전 분리)

**🔥 브랜치별 완전 분리 전략으로 변경되었습니다!**

### main 브랜치 (통합 홈페이지)
```
IjipMatjip/
├── frontend-main/           # 🏠 통합 홈페이지 (포트: 4010)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   │   ├── HomePage.jsx
│   │   │   ├── AIDesignPage.jsx      # 플레이스홀더
│   │   │   ├── FindHousePage.jsx     # 플레이스홀더
│   │   │   └── RoomPlannerPage.jsx   # 플레이스홀더
│   │   └── utils/
│   ├── public/
│   └── package.json
├── integration-files/       # 🔗 팀 API 통합 유틸리티
│   ├── teamApi.js          # 모든 팀 API 통합 함수
│   └── .env.example        # 환경변수 템플릿
└── README.md               # 프로젝트 메인 문서
```

### dev-eunbi 브랜치 (은비 서비스)
```
IjipMatjip/
├── eunbi/                  # 은비 서비스 전체
│   ├── frontend/           # React + Vite (포트: 4000)
│   │   ├── src/
│   │   │   ├── components/3D/
│   │   │   ├── pages/RoomPlannerPage.jsx
│   │   │   └── utils/
│   │   └── package.json
│   ├── backend-cloud/      # 클라우드 데이터 관리 (포트: 3000)
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── backend-local/      # 로컬 AI 이미지 처리 (포트: 3010)
│   │   ├── main.py
│   │   ├── ai_room_detection.py
│   │   └── requirements.txt
│   └── k8s/               # Kubernetes 배포 설정
└── README.md              # 은비 서비스 전용 문서
```

### dev-minah 브랜치 (민아 서비스)
```
IjipMatjip/
├── minah/                 # 민아 서비스
│   ├── frontend/          # React (포트: 4001)
│   │   └── .gitkeep      # 개발 시작용 플레이스홀더
│   └── backend/           # FastAPI (포트: 8001)
│       └── .gitkeep      # 개발 시작용 플레이스홀더
└── README.md             # 민아 서비스 전용 문서
```

### dev-danbi 브랜치 (단비 서비스)
```
IjipMatjip/
├── danbi/                 # 단비 서비스
│   ├── frontend/          # React (포트: 4002)
│   │   └── .gitkeep      # 개발 시작용 플레이스홀더
│   └── backend/           # FastAPI (포트: 8002)
│       └── .gitkeep      # 개발 시작용 플레이스홀더
└── README.md             # 단비 서비스 전용 문서
```

### integration 브랜치 (통합 테스트용)
```
IjipMatjip/
├── eunbi/                 # 은비 서비스 전체
├── minah/                 # 민아 서비스 전체
├── danbi/                 # 단비 서비스 전체
├── frontend-main/         # 통합 홈페이지
├── integration-files/     # 공유 유틸리티
├── docker-compose.yml     # 통합 테스트용
└── README.md             # 통합 테스트 가이드
```

## 🌿 브랜치 전략 (완전 분리)

**새로운 브랜치별 완전 분리 전략:**

- **`main`** - 통합 홈페이지만 (frontend-main + integration-files)
- **`dev-eunbi`** - 은비 서비스만 (eunbi/ 폴더 + README)
- **`dev-minah`** - 민아 서비스만 (minah/ 폴더 + README)
- **`dev-danbi`** - 단비 서비스만 (danbi/ 폴더 + README)
- **`integration`** - 모든 서비스 통합 (모든 폴더)

## 🔌 현재 API 구조 (은비 서비스)

### 포트 분리
- **포트 3000**: 클라우드 데이터 저장/조회 (사용자 인증, 레이아웃 저장)
- **포트 3010**: 로컬 AI 이미지 처리 (방측정, 창문감지, 깊이맵)
- **포트 4000**: 은비 개발용 프론트엔드
- **포트 4010**: 통합 홈페이지

### 환경변수 설정 (.env)
```env
# 팀별 API 서버 URL
VITE_EUNBI_CLOUD_API=http://localhost:3000
VITE_EUNBI_LOCAL_API=http://localhost:3010
VITE_MINAH_API=http://localhost:8001  
VITE_DANBI_API=http://localhost:8002

# 데이터베이스 설정
POSTGRES_HOST=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=user_auth
MONGO_HOST=13.55.21.100

# JWT 설정
JWT_SECRET=your-secret-key-here
```

## 🚀 통합 개발 워크플로우

### 1. 현재 개발 단계
- **은비**: `eunbi/frontend`에서 방측정 서비스 개발 중
- **민아/단비**: 각자 폴더 생성 후 개발 시작
- **통합**: `frontend-main`은 플레이스홀더 상태

### 2. 통합 절차
1. **각자 개발**: `dev-이름` 브랜치에서 개발
2. **API 명세 확정**: 각 팀의 엔드포인트 정리
3. **통합 테스트**: `integration` 브랜치에서 API 연동
4. **최종 배포**: `main` 브랜치로 머지

## 📋 API 명세 (통합용)

### 은비 - 방측정/가구배치 (현재 구현됨)

**클라우드 API (포트 3000):**
- `POST /signup` - 회원가입
- `POST /login` - 로그인  
- `GET /me` - 사용자 정보 조회
- `POST /save-room-layout` - 방 레이아웃 저장 (로그인 사용자)
- `POST /save-room-layout-guest` - 방 레이아웃 저장 (게스트)
- `GET /room-layouts` - 사용자별 레이아웃 조회
- `POST /convert-furniture-coordinates` - 가구 좌표 변환

**로컬 AI API (포트 3010):**
- `POST /undistort` - 이미지 왜곡 보정
- `POST /depth-map` - 깊이 맵 생성
- `GET /depth-map-image` - 깊이 맵 이미지 조회
- `POST /auto-detect-room` - AI 자동 방 감지
- `POST /estimate-room-size` - 방 크기 측정
- `POST /detect-windows` - 창문 감지
- `POST /depth-distance` - 깊이 거리 계산

### 민아 - AI 인테리어 (예정)
- `POST /api/ai-design/generate` - AI 디자인 생성
- `POST /api/interior/recommend-styles` - 스타일 추천
- `POST /api/ai-design/color-palette` - 색상 팔레트

### 단비 - 집찾기/추천 (예정)
- `POST /api/recommend/houses` - 집 추천
- `GET /api/find-house/details/:id` - 집 상세정보
- `GET /api/find-house/search` - 집 검색
- `POST /api/recommend/infrastructure` - 인프라 정보

## 🔗 통합 방법

### 1. API 통합 (teamApi.js 사용)
```javascript
import { teamApi } from '../integration-files/teamApi.js';

// 은비 서비스 호출
const roomLayouts = await teamApi.eunbi.getRoomLayouts();

// 민아 서비스 호출
const aiDesign = await teamApi.minah.generateDesign(designInput);

// 단비 서비스 호출
const houses = await teamApi.danbi.getHouseRecommendations(preferences);
```

### 2. 라우팅 통합 (frontend-main에서)
```jsx
// App.jsx
import { Suspense, lazy } from 'react';

// 각 팀 서비스를 lazy loading으로
const RoomPlannerPage = lazy(() => import('./pages/room-measure/RoomPlannerPage'));
const AIDesignPage = lazy(() => import('./pages/ai-design/AIDesignPage'));
const FindHousePage = lazy(() => import('./pages/find-house/FindHousePage'));

// 라우팅 설정
<Routes>
  <Route path="/" element={<HomePage />} />
  <Route path="/room-planner" element={
    <Suspense fallback={<Loading />}>
      <RoomPlannerPage />
    </Suspense>
  } />
  <Route path="/ai-design" element={
    <Suspense fallback={<Loading />}>
      <AIDesignPage />
    </Suspense>
  } />
  <Route path="/find-house" element={
    <Suspense fallback={<Loading />}>
      <FindHousePage />
    </Suspense>
  } />
</Routes>
```

### 3. 컴포넌트 통합
```bash
# 은비 컴포넌트 통합 시
cp -r eunbi/frontend/src/components/* frontend-main/src/components/room-measure/
cp -r eunbi/frontend/src/utils/* frontend-main/src/utils/room-measure/

# 민아 컴포넌트 통합 시 (예정)
cp -r minah/frontend/src/components/* frontend-main/src/components/ai-design/

# 단비 컴포넌트 통합 시 (예정)  
cp -r danbi/frontend/src/components/* frontend-main/src/components/find-house/
```

## 🛠️ 로컬 통합 테스트 환경

### 서버 실행 순서
```bash
# 1. 은비 백엔드 서버들
cd eunbi/backend-cloud && uv run uvicorn main:app --port 3000
cd eunbi/backend-local && uv run uvicorn main:app --port 3010

# 2. 민아 백엔드 (예정)
cd minah/backend && python -m uvicorn main:app --port 8001

# 3. 단비 백엔드 (예정)
cd danbi/backend && python -m uvicorn main:app --port 8002

# 4. 통합 프론트엔드
cd frontend-main && npm run dev -- --port 4010
```

### 접속 주소
- **통합 홈페이지**: http://localhost:4010
- **은비 개발용**: http://localhost:4000  
- **클라우드 API**: http://localhost:3000
- **로컬 AI API**: http://localhost:3010

## 📝 통합 TODO

- [x] 은비 서비스 MSA 분리 완료
- [x] frontend-main 정리 완료 (플레이스홀더 상태)
- [x] 브랜치 전략 수립
- [ ] 민아/단비 폴더 구조 생성
- [ ] API 명세 세부 조율 (3팀 회의)
- [ ] teamApi.js 실제 엔드포인트 매핑
- [ ] 통합 테스트 환경 구축
- [ ] Docker Compose 설정 (배포용)
- [ ] CI/CD 파이프라인 구축

## 💡 통합 시 주의사항

1. **포트 충돌 방지**: 각 팀은 서로 다른 포트 사용
2. **CORS 설정**: 모든 백엔드에서 프론트엔드 도메인 허용
3. **인증 토큰 공유**: JWT 토큰을 모든 서비스에서 공통 사용
4. **환경변수 관리**: .env 파일로 API URL 관리
5. **번들 최적화**: Lazy loading으로 초기 로딩 시간 단축

## 🤝 팀 협업 가이드

- **코드 리뷰**: PR 시 다른 팀원 1명 이상 리뷰
- **API 변경**: 다른 팀에 영향 있는 변경 시 사전 공지  
- **테스트**: 통합 전 각자 서비스 개별 테스트 완료
- **문서화**: API 변경 시 이 문서도 함께 업데이트