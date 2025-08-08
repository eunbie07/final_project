# 3D to Real Image Converter

2단계 AI 변환으로 3D 스크린샷을 실사화하는 프로젝트

## 목적
- 문제: AI가 가구 배치를 자꾸 바꿔버림
- 해결: 2단계로 나누어 배치 유지하면서 스타일 변경

## 처리 과정

### 1단계: 스타일 변경 (배치 유지)
- 입력: 3D 스크린샷
- 출력: 같은 배치에서 스타일만 변경된 이미지
- 사용 모델: DALL-E or Mock processing

### 2단계: 실사화
- 입력: 스타일 변경된 이미지  
- 출력: 실제 사진 같은 품질의 이미지
- 사용 모델: 실사화 전용 모델

## 실행 방법 (uv 사용)

### 1. 프로젝트 설정
```bash
cd C:\Users\kibwa07\Documents\GitHub\final_project\3d-real-image
```

### 2. 종속성 설치
```bash
uv sync
```

### 3. 환경변수 설정
`.env` 파일에서 OpenAI API 키 설정 (이미 설정됨)

### 4. 서버 실행
```bash
uv run app.py
```

또는

```bash
uv run python app.py
```

### 5. 웹 접속
브라우저에서 `http://localhost:5000` 접속

## 사용법
1. 3D 레이아웃 스크린샷 업로드 (드래그&드롭 또는 클릭)
2. 원하는 스타일 선택 (modern, scandinavian, industrial, cozy, luxury)
3. 변환 방식 선택:
   - **1단계만**: 스타일 변경만 실행
   - **2단계만**: 실사화만 실행 (1단계 완료 후)
   - **전체 변환**: 1단계 + 2단계 한번에 실행
4. 결과 이미지 확인 및 다운로드

## 폴더 구조
```
3d-real-image/
├── uploads/          # 업로드된 3D 스크린샷
├── stage1_output/    # 1단계 결과 (스타일 변경)
├── stage2_output/    # 2단계 결과 (실사화)
├── templates/        # HTML 템플릿
├── app.py           # Flask 웹 서버
├── stage1_processor.py  # 1단계 처리기
├── stage2_processor.py  # 2단계 처리기
├── pyproject.toml   # uv 프로젝트 설정
├── .env            # 환경변수
└── README.md       # 이 파일
```

## 기능
- ✅ 웹 기반 인터페이스
- ✅ 드래그&드롭 파일 업로드
- ✅ 5가지 인테리어 스타일 지원
- ✅ 2단계 분리 처리
- ✅ 실시간 상태 표시
- ✅ 결과 이미지 다운로드
- ✅ Mock 모드 (API 키 없어도 테스트 가능)

## API 엔드포인트
- `POST /upload`: 이미지 업로드
- `POST /process/stage1`: 1단계 처리 (스타일 변경)
- `POST /process/stage2`: 2단계 처리 (실사화)
- `POST /process/full`: 전체 처리 (1단계 + 2단계)
- `GET /download/<path>`: 결과 파일 다운로드

## 개발 모드
```bash
# 개발 종속성 설치
uv sync --group dev

# 코드 포맷팅
uv run black .

# 린팅
uv run flake8
```