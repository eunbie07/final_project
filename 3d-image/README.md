# 🎨 Vertex AI 인테리어 디자인 웹페이지

사용자가 업로드한 이미지를 Google Cloud Vertex AI를 사용하여 가구 스타일 변경 및 실사화 변환을 수행하는 웹 애플리케이션입니다.

## ✨ 주요 기능

- **이미지 업로드**: 드래그 앤 드롭 또는 클릭으로 이미지 업로드
- **스타일 선택**: 6가지 인테리어 스타일 (모던, 스칸디나비안, 인더스트리얼, 럭셔리, 미니멀, 보헤미안)
- **AI 처리**: Vertex AI를 사용한 가구 스타일 변경 (위치 유지)
- **실사화**: 고품질 포토리얼리스틱 이미지 변환
- **갤러리**: 생성된 이미지들의 비교 및 다운로드

## 🚀 설치 및 실행

### 1. 프론트엔드 실행

```bash
cd 3d-image
npm install
npm start
```

프론트엔드는 `http://localhost:9000`에서 실행됩니다.

### 2. 백엔드 실행

```bash
cd 3d-image/backend
npm install
npm start
```

백엔드는 `http://localhost:9001`에서 실행됩니다.

## 🛠️ 기술 스택

### 프론트엔드

- **React 18**: 사용자 인터페이스
- **CSS3**: 현대적인 디자인과 애니메이션
- **Axios**: HTTP 요청 처리

### 백엔드

- **Node.js**: 서버 런타임
- **Express**: 웹 프레임워크
- **Multer**: 파일 업로드 처리
- **CORS**: 크로스 오리진 요청 허용

### AI 서비스

- **Google Cloud Vertex AI**: 이미지 생성 및 변환
- **Imagen 모델**: 고품질 이미지 생성

## 📁 프로젝트 구조

```
3d-image/
├── src/
│   ├── components/
│   │   ├── ImageUploader.js      # 이미지 업로드 컴포넌트
│   │   ├── StyleSelector.js      # 스타일 선택 컴포넌트
│   │   ├── ProcessingStatus.js   # 처리 상태 컴포넌트
│   │   ├── Gallery.js           # 갤러리 컴포넌트
│   │   └── *.css               # 각 컴포넌트 스타일
│   ├── App.js                   # 메인 앱 컴포넌트
│   └── App.css                  # 메인 스타일
├── backend/
│   ├── server.js                # Express 서버
│   └── package.json             # 백엔드 의존성
└── package.json                 # 프론트엔드 의존성
```

## 🔧 API 엔드포인트

### POST `/api/change-furniture-style`

가구 스타일 변경 API

- **Body**: `multipart/form-data`
  - `image`: 업로드할 이미지 파일
  - `style`: 선택한 스타일 (modern, scandinavian, industrial, luxury, minimalist, bohemian)

### POST `/api/photorealistic`

실사화 변환 API

- **Body**: `multipart/form-data`
  - `image`: 업로드할 이미지 파일

## 🎯 사용 방법

1. **이미지 업로드**: 드래그 앤 드롭 또는 클릭하여 이미지 업로드
2. **스타일 선택**: 원하는 인테리어 스타일 선택
3. **AI 처리**: "Vertex AI로 처리 시작" 버튼 클릭
4. **결과 확인**: 갤러리에서 원본, 스타일 변경, 실사화 이미지 비교
5. **다운로드**: 원하는 이미지 다운로드

## 🔐 환경 설정

### Google Cloud 설정

실제 Vertex AI 사용을 위해서는 다음 설정이 필요합니다:

1. **Google Cloud 프로젝트 생성**
2. **Vertex AI API 활성화**
3. **서비스 계정 키 생성**
4. **환경 변수 설정**:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
   PROJECT_ID=your-project-id
   LOCATION=us-central1
   ```

## 🎨 디자인 특징

- **현대적인 UI**: 그라데이션 배경과 글래스모피즘 효과
- **반응형 디자인**: 모바일 및 데스크톱 최적화
- **애니메이션**: 부드러운 전환 효과와 로딩 애니메이션
- **사용자 친화적**: 직관적인 인터페이스와 명확한 피드백

## 🔄 처리 과정

1. **이미지 분석**: 업로드된 이미지의 가구 위치와 레이아웃 분석
2. **스타일 변경**: 선택한 스타일로 가구 디자인 변경 (위치 유지)
3. **실사화**: 고품질 포토리얼리스틱 이미지로 변환
4. **결과 생성**: 최종 인테리어 이미지 생성

## 📱 지원 브라우저

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요.

---

**Vertex AI 인테리어 디자인** - AI로 당신의 공간을 변환하세요! 🏠✨
