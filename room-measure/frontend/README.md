# 방 크기 추정기 (Room Size Estimator)

이 프로젝트는 부동산 방 사진을 기반으로 사용자가 직접 벽의 기준점을 클릭하여 방의 **가로(x)**, **세로(y)** 길이를 자동으로 추정하는 웹 애플리케이션입니다.광각 렌즈로 촬영된 사진에 대해서도 OpenCV 왜곡 보정을 적용하여 비교적 정확한 거리 추정을 제공합니다.

---

## ✅ 주요 기능

### 1. 사진 업로드 및 왜곡 보정

- 사용자가 사진을 업로드하면 FastAPI 서버에서 OpenCV `cv2.undistort()` 함수로 광각 왜곡을 1차 보정합니다.

### 2. 4포인트 클릭 방식 (xz / yz 평면 기반)

사용자는 아래 순서로 4개의 포인트를 클릭합니다:

1. 벽 하단 (기준점)
2. 벽 상단 (천장)
3. 왼쪽 바닥 방향 (xz 평면)
4. 오른쪽 바닥 방향 (yz 평면)

이 순서를 통해 다음을 계산합니다:

- z축 기준 픽셀 거리 → 230cm로 환산
- x, y 거리 → 각각 `xz`, `yz` 평면에서 독립적으로 환산

### 3. 거리 계산 결과

- 가로 길이 (x): cm 단위
- 세로 길이 (y): cm 단위
- 1픽셀당 cm 값

---

## 🧱 기술 스택

- **Frontend**: React + Vite
- **Backend**: FastAPI + OpenCV
- **Styling**: Tailwind CSS (선택)

---

## 🚀 실행 방법

### 1. 백엔드 실행 (FastAPI)

    cd backend
    uvicorn main:app --host 0.0.0.0 --port 3000

### 2. 프론트엔드 실행 (React)

    cd frontend
    npm install
    npm run dev -- --host 0.0.0.0 --port 4000

---

## 📦 주요 디렉토리 구조

    room-measure/
    ├── backend/
    │   └── main.py               # FastAPI 서버 + OpenCV 처리
    ├── frontend/
    │   ├── index.html
    │   ├── vite.config.js
    │   └── src/
    │       ├── App.jsx           # 전체 앱 구조
    │       ├── main.jsx          # React DOM mount
    │       └── components/
    │           ├── ImageUploader.jsx
    │           └── ImageClickArea.jsx

---

## 🎯 향후 확장 가능성

- RoomNet, HorizonNet 등으로 자동 평면 인식 기능 추가
- 방 구조 2D 시각화 기능
- 방 크기 결과 다운로드 (PDF, JSON 등)

---

## ✨ 만든 이유

광각 사진만 있는 부동산 매물에서도,정확한 방 크기를 유추하고 시뮬레이션하거나 가구 배치를 예측하기 위한 기초 데이터 확보가 목적입니다.
