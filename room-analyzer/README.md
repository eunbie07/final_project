
# 🏠 Room Analyzer Project

사진 한 장으로 방 크기를 추정하고, 감지된 가구를 포함한 **2D 평면도 도면을 생성**하는 React + FastAPI 기반 AI 웹앱입니다.

---

## 📦 기능 요약

| 기능 | 설명 |
|------|------|
| ✅ 방 사진 업로드 | React에서 사진 업로드 및 층고 입력 |
| ✅ AI 기반 분석 | FastAPI + YOLOv11 + MiDaS로 방 구조, 가구 감지 |
| ✅ 평면도 생성 | SVG 기반 도면 + PNG 다운로드 기능 |
| ✅ Fallback 처리 | 침대 감지 실패 시 MiDaS + 층고 기준 분석으로 대체 |

---

## 🧠 기술 스택

- **Frontend**: React + Vite
- **Backend**: FastAPI
- **AI 모델**: MiDaS (깊이 추정), YOLOv11 (객체 감지)
- **도면 시각화**: SVG + html-to-image (PNG 저장)

---

## 🚀 실행 방법

### 1. 백엔드 (FastAPI)

```bash
cd /home/node1/final_project/room-analyzer/backend
python -m venv .venv
source .venv/bin/activate
pip install uv               # 최초 1회만
uv                           # pyproject.toml 기반 의존성 설치
uvicorn main:app --reload --host 0.0.0.0 --port 3000              # FastAPI 실행
```

> YOLOv11 모델(`yolo11n`)은 처음 실행 시 자동 다운로드됩니다.

### 2. 프론트엔드 (React)

```bash
cd frontend
npm install
npm run dev
```

---

## 🖼 사용 예시

1. 사진 업로드 + 층고 입력  
2. `/analyze-room` API 호출  
3. YOLO로 가구 감지 + MiDaS로 깊이 추정  
4. SVG 도면 생성 후 PNG 다운로드 가능

---

## 📁 주요 폴더 구조

```
room-analyzer-project/
├── backend/
│   ├── main.py                # FastAPI 진입점
│   ├── analyzer.py            # 분석 로직 (YOLO + MiDaS)
│   ├── utils/                 # 분석 유틸 함수들
│   └── static/                # 결과 이미지 저장
├── frontend/
│   ├── src/components/
│   │   ├── UploadForm.jsx     # 분석 요청
│   │   └── RoomVisualizer.jsx # SVG 도면 + 다운로드
```

---

## 🧰 사용 모델 출처

- [MiDaS (Intel-ISL)](https://github.com/isl-org/MiDaS)
- [YOLOv11 (Ultralytics)](https://github.com/ultralytics/ultralytics)

---

## ✨ 향후 확장 가능 기능

- 드래그 가능한 가구 배치
- 회전 및 정렬 기능
- 부동산 사진 왜곡 분석 (광각/밝기)
- 방 구조 자동 감지(RoomFormer 등)

---

## 🧑‍💻 만든이

이 프로젝트는 2025년 7월 기준 AI 기반 실내 공간 분석 기술을 학습하고 적용하기 위해 제작되었습니다.
