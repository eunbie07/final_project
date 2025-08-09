# 🔧 Google Cloud Vertex AI 설정 가이드

실제 Vertex AI를 사용하려면 다음 단계를 따라 Google Cloud를 설정하세요.

## 📋 필수 설정 단계

### 1. Google Cloud 프로젝트 생성

1. [Google Cloud Console](https://console.cloud.google.com/)에 접속
2. 새 프로젝트 생성 또는 기존 프로젝트 선택
3. 프로젝트 ID를 기록 (예: `my-vertex-ai-project`)

### 2. Vertex AI API 활성화

1. Google Cloud Console에서 "API 및 서비스" → "라이브러리"로 이동
2. "Vertex AI API" 검색 후 활성화
3. "AI Platform Training & Prediction API"도 활성화

### 3. 서비스 계정 생성

1. "IAM 및 관리" → "서비스 계정"으로 이동
2. "서비스 계정 만들기" 클릭
3. 이름: `vertex-ai-service`
4. 설명: `Vertex AI 이미지 생성용 서비스 계정`

### 4. 권한 설정

서비스 계정에 다음 역할 추가:

- `Vertex AI User`
- `Vertex AI Service Agent`
- `Storage Object Viewer` (이미지 저장용)

### 5. 서비스 계정 키 생성

1. 생성한 서비스 계정 클릭
2. "키" 탭 → "키 추가" → "새 키 만들기"
3. JSON 형식 선택
4. 다운로드된 키 파일을 `backend/` 폴더에 저장
5. 파일명을 `service-account-key.json`으로 변경

### 6. 환경 변수 설정

`backend/` 폴더에 `.env` 파일 생성:

```env
# Google Cloud 설정
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Google Cloud 인증
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json

# 서버 포트
PORT=9001
```

## 🚀 설정 완료 후 실행

```bash
cd 3d-image/backend
npm start
```

성공적으로 설정되면 터미널에 다음 메시지가 나타납니다:

```
✅ Vertex AI 클라이언트 초기화 성공
✅ Vertex AI 연결됨: your-project-id (us-central1)
```

## ⚠️ 주의사항

1. **비용**: Vertex AI 사용 시 Google Cloud 요금이 발생합니다
2. **할당량**: API 호출 제한이 있을 수 있습니다
3. **보안**: 서비스 계정 키 파일을 Git에 커밋하지 마세요
4. **지역**: `us-central1`이 권장됩니다

## 🔍 문제 해결

### 인증 오류

- 서비스 계정 키 파일 경로 확인
- 프로젝트 ID 확인
- API 활성화 상태 확인

### 권한 오류

- 서비스 계정에 필요한 역할 추가
- 프로젝트에서 Vertex AI API 활성화

### 네트워크 오류

- 방화벽 설정 확인
- 인터넷 연결 확인

## 📞 지원

문제가 발생하면 Google Cloud 문서를 참조하거나 이슈를 생성해 주세요.


