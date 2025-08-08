# 🎨 AI Interior Design System

MongoDB와 연동된 AI 인테리어 디자인 자동 생성 시스템

## 📁 폴더 구조

```
ai-interior/
├── 📁 archive/          # 아카이브된 파일들
│   ├── colab_*.py      # Colab 테스트 파일들
│   └── colab_*.ipynb   # Jupyter 노트북들
├── 📁 docs/            # 문서 모음
├── 📁 generated_images/ # 생성된 이미지들
├── 📁 generated_masks/  # 생성된 마스크들
├── 📁 image/           # 샘플 이미지들
├── 📁 scripts/         # 예제 스크립트들
├── 📁 verification_reports/ # 검증 보고서들
└── 📁 image_analysis_results/ # 이미지 분석 결과들
```

## 🚀 주요 파일들

### Core API
- `api_server.py` - 메인 API 서버
- `main.py` - 시스템 진입점
- `config.py` - 설정 관리

### AI 생성기
- `dalle_generator.py` - DALL-E 생성기
- `vertex_ai_generator.py` - Vertex AI 생성기
- `stable_diffusion_generator.py` - Stable Diffusion 생성기
- `integrated_generator.py` - 통합 생성기

### MongoDB 연동
- `mongodb_integration.py` - MongoDB 연동 로직
- `mongodb_furniture_extractor.py` - 가구 데이터 추출기
- `roombox_integration.py` - RoomBox 연동

### 유틸리티
- `image_verification.py` - 이미지 검증
- `layout_mask_generator.py` - 레이아웃 마스크 생성
- `performance_monitor.py` - 성능 모니터링
- `cache_manager.py` - 캐시 관리

## 🎯 사용법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
python api_server.py
```

### 3. Colab 사용 (GPU 제한 시)
- `archive/colab_final_notebook.ipynb` 사용
- CPU 모드 지원
- 단계별 실행 가능

## 📊 성능 지표

- **좌표 정확도**: 99.7%
- **이미지 생성 성공률**: 95%+
- **API 응답 시간**: < 30초

## 🔧 기술 스택

- **AI 모델**: DALL-E, Vertex AI, Stable Diffusion
- **데이터베이스**: MongoDB
- **백엔드**: Python Flask
- **이미지 처리**: OpenCV, PIL
- **배포**: Colab, Local Server