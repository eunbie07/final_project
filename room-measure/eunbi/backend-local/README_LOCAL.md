# Room Measure Local Backend

로컬 개발용 이미지/AI 처리 백엔드입니다. 무거운 딥러닝 모델을 사용합니다.

## 기능

- 이미지 왜곡 보정
- 깊이 맵 생성 (MiDaS)
- AI 방 감지 (RoomNet 시뮬레이션)
- 창문 감지 (YOLO)
- 방 크기 측정

## 실행 방법

### 1. 의존성 설치
```bash
cd backend-local
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
python main.py
```

서버는 `http://localhost:3010`에서 실행됩니다.

## API 엔드포인트

- `GET /health` - 헬스체크
- `POST /undistort` - 이미지 왜곡 보정
- `POST /depth-map` - 깊이 맵 생성
- `GET /depth-map-image` - 깊이 맵 이미지 조회
- `GET /get-depth-at-point` - 특정 좌표 깊이 값
- `GET /depth-meta` - 깊이 맵 메타데이터
- `POST /auto-detect-room` - AI 방 감지
- `POST /estimate-room-size` - 방 크기 측정  
- `POST /detect-windows` - 창문 감지
- `POST /depth-distance` - 깊이 거리 계산

## 리소스 사용량

- **메모리**: ~2-4GB (모델 로딩 시)
- **CPU**: 높음 (이미지 처리 시)
- **디스크**: ~1.8GB
- **GPU**: 선택사항 (CUDA 가능)

## 주요 파일

- `main.py` - 메인 API 서버
- `depth_processing.py` - 깊이 맵 처리
- `room_measurement.py` - 방 크기 측정
- `window_detection.py` - 창문 감지
- `models.py` - 데이터 모델