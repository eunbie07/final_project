# IjipMatjip 팀 통합 가이드

## 📁 폴더 구조

```
room-measure/
├── eunbi/                    # 은비 개발 공간
│   ├── frontend/            # 개발/테스트용 프론트엔드
│   ├── backend/             # 메인 백엔드 (FastAPI)
│   ├── backend-local/       # 로컬 이미지 처리
│   └── backend-cloud/       # 클라우드 데이터 관리
├── integration-files/       # 팀 통합시 사용할 파일들
│   ├── teamApi.js          # 팀 API 통합 유틸리티
│   └── .env.example        # 환경변수 템플릿
├── frontend/                # 기존 프론트엔드 (백업)
└── INTEGRATION_GUIDE.md
```

## 🌿 브랜치 전략

- `feature/eunbi` - 은비 개발 브랜치
- `feature/minah` - 민아 개발 브랜치  
- `feature/danbi` - 단비 개발 브랜치
- `frontend-main` - 통합 프론트엔드 브랜치
- `dev` - 팀 통합 테스트 브랜치
- `main` - 최종 배포 브랜치

## 🔌 API 통합 방식

### 환경변수 설정 (.env)
```
# 팀별 API 서버 URL
VITE_EUNBI_API_URL=http://localhost:8000
VITE_MINAH_API_URL=http://localhost:8001  
VITE_DANBI_API_URL=http://localhost:8002
```

### API 사용 예시
```javascript
import { teamApi } from './utils/teamApi.js';

// 은비 서비스 호출
const roomLayouts = await teamApi.eunbi.getRoomLayouts();

// 민아 서비스 호출
const aiDesign = await teamApi.minah.generateDesign(designInput);

// 단비 서비스 호출
const houses = await teamApi.danbi.getHouseRecommendations(preferences);
```

## 🚀 개발 워크플로우

1. **개발**: 각자 `feature/이름` 브랜치에서 개발
2. **테스트**: 완성된 기능을 `dev` 브랜치에 통합
3. **통합**: `frontend-main`에서 API 연동 테스트
4. **배포**: 검증 완료 후 `main` 브랜치로 머지

## 📋 팀별 API 명세

### 은비 - 방측정/가구배치
- `POST /api/room/save-layout` - 방 레이아웃 저장
- `GET /api/room/layouts` - 방 레이아웃 조회
- `POST /api/room/detect-windows` - 창문 감지
- `POST /api/furniture/place` - 가구 배치

### 민아 - AI 인테리어 (예상)
- `POST /api/ai-design/generate` - AI 디자인 생성
- `POST /api/interior/recommend-styles` - 스타일 추천
- `POST /api/ai-design/color-palette` - 색상 팔레트

### 단비 - 집찾기/추천 (예상)
- `POST /api/recommend/houses` - 집 추천
- `GET /api/find-house/details/:id` - 집 상세정보
- `GET /api/find-house/search` - 집 검색
- `POST /api/recommend/infrastructure` - 인프라 정보

## 🛠️ 로컬 개발 환경

```bash
# 은비 서비스 실행
cd eunbi/backend && uvicorn main:app --port 8000
cd eunbi/backend-local && uvicorn main:app --port 3010

# 통합 프론트엔드 실행
cd frontend-main && npm run dev

# 각자 개발용 프론트엔드 실행
cd eunbi/frontend && npm run dev
```

## 📝 TODO

- [ ] 민아/단비 폴더 구조 생성
- [ ] API 명세 세부 조율
- [ ] 통합 테스트 환경 구축
- [ ] Docker Compose 설정
- [ ] CI/CD 파이프라인 구축