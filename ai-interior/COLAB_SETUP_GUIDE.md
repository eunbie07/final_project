# Colab Inpainting 시스템 설정 가이드

## 🎯 목표
MongoDB 좌표를 정확히 변환하여 95%+ 정확도로 가구를 배치하는 Colab Inpainting 시스템 구축

## 📋 현재 상황
- ✅ **좌표 변환 시스템**: MongoDB cm → 픽셀 마스크 완료
- ✅ **API 연동 구조**: FastAPI 서버에 Colab 엔드포인트 추가 완료  
- ✅ **통합 테스트**: 로컬 시스템 검증 완료
- ⚠️ **Colab 연결**: 환경변수 설정 필요

## 🚀 다음 단계

### 1. Google Colab에서 Notebook 실행

1. `colab_inpainting_workflow.ipynb` 파일을 Google Colab에 업로드
2. GPU 런타임으로 변경 (Runtime → Change runtime type → GPU)
3. 셀을 순서대로 실행:
   - 환경 설정 및 라이브러리 설치
   - ComfyUI 다운로드 및 모델 설치
   - 좌표 변환 시스템 테스트
   - Flask API 서버 시작
   - Ngrok 터널 생성

### 2. Ngrok URL 설정

Colab에서 생성된 Ngrok URL을 환경변수로 설정:

```bash
# Windows
set COLAB_API_URL=https://your-generated-url.ngrok.io

# Linux/Mac  
export COLAB_API_URL=https://your-generated-url.ngrok.io
```

### 3. 로컬 API 서버 재시작

```bash
cd ai-interior
uv run api_server.py
```

### 4. 통합 테스트 재실행

```bash
uv run python test_colab_integration.py
```

## 🔧 핵심 구현 내용

### MongoDB 좌표 → 픽셀 마스크 변환

```python
# 실제 프로젝트 데이터
room_data = {
    'dimensions': {
        'width_cm': 387,   # 실제 방 폭
        'depth_cm': 465,   # 실제 방 깊이
    },
    'furniture_3d': [{
        'name': 'bed',
        'position': [203.67, 0, 238.00]  # 실제 침대 위치
    }]
}

# 변환 결과
# 침대 위치: (203.67, 238.0)cm → (269, 262)px
# 상대 위치: 52.6% X, 51.2% Z (방 중심 기준)
```

### API 엔드포인트

```http
POST http://localhost:8000/generate-interior-colab
Content-Type: application/json

{
  "room_data": { ... },
  "style": "scandinavian",
  "generate_image": true
}
```

### 예상 응답

```json
{
  "success": true,
  "image_url": "http://localhost:8000/images/colab_inpaint_scandinavian_20250807.png",
  "accuracy_score": 0.95,
  "accuracy_percentage": "95.0%",
  "position_analysis": {
    "accuracy_status": "SUCCESS",
    "furniture_positions": [{
      "name": "bed",
      "original_cm": [203.67, 238.0],
      "converted_px": [269, 262]
    }]
  }
}
```

## 📁 생성된 파일

- `colab_inpainting_workflow.ipynb` - Colab 워크플로우 노트북
- `colab_integration.py` - 로컬 시스템 연동 클라이언트
- `test_colab_integration.py` - 통합 테스트 스크립트
- `api_server.py` - Colab 엔드포인트 추가된 서버

## 🎯 성공 기준

- ✅ MongoDB 좌표 정확 변환 (269, 262)px
- ✅ API 서버 Colab 엔드포인트 연동
- ⚠️ 95%+ 위치 정확도 달성 (Colab 실행 후 검증)

## 🔍 트러블슈팅

### Colab 연결 실패
- Ngrok URL이 올바른지 확인
- Colab 서버가 실행 중인지 확인  
- 환경변수 설정 확인

### 좌표 변환 오류
- 방 크기 데이터 확인 (387x465cm)
- MongoDB 가구 위치 데이터 확인
- 픽셀 변환 비율 검토 (X=1.323, Y=1.101)

### 모델 다운로드 실패
- Colab GPU 메모리 확인
- HuggingFace 토큰 설정
- 네트워크 연결 상태 확인

## 📈 다음 최적화 방향

1. **ControlNet 파인튜닝**: 가구별 위치 제어 정확도 향상
2. **마스크 정밀도**: 픽셀 단위 마스크 생성 개선
3. **배치 처리**: 여러 가구 동시 처리 최적화
4. **스타일 일관성**: 인테리어 스타일별 프롬프트 최적화