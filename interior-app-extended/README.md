# Interior App (확장판: Stability SDXL / DALL·E / 2단계)

포트: 프론트 7000, 백엔드 7001

## 환경변수
backend/.env
```
OPENAI_API_KEY=sk-...
STABILITY_API_KEY=stability-...
STABILITY_API_HOST=https://api.stability.ai
STABILITY_SDXL_PATH=/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image
PORT=7001
```

## 실행
```bash
# 백엔드
cd backend
npm i
cp .env.example .env
npm run dev   # http://localhost:7001

# 프론트엔드
cd ../frontend
npm i
npm run dev   # http://localhost:7000
```

## 사용법
1) 이미지 업로드
2) 모델 선택: DALL·E만, Stability만, 2단계 중 하나
3) 프리셋으로 마스크 지정(스테빌리티 단독은 마스크 필요 없음)
4) 가구 추가 금지 체크 유지 권장
5) 필요 시 SDXL strength 0.3~0.4로 구조 보존 강화
6) 생성하면 결과가 우측 결과/히스토리에 쌓임
