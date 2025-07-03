# layout-simulator-service

사진에서 가구를 인식하고, 2D 평면도용 좌표로 변환하며 부모님 기준 조언도 포함하는 FastAPI 서비스입니다.

## 실행 방법

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## API 사용

- POST /layout  
  - form-data에 이미지 업로드
  - 반환: 가구 위치 + 부모님 동선 관련 조언
