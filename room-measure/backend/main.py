# room-measure/backend/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import shutil
import os
from typing import List
from math import sqrt
from depth_api import router as depth_router

app = FastAPI()

app.include_router(depth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 모델: 좌표 입력용
# ---------------------------
class Point(BaseModel):
    x: float
    y: float

class RoomPoints(BaseModel):
    points: List[Point]  # [1, 2, 3, 4]

# ---------------------------
# 기능: 픽셀 거리 계산
# ---------------------------
def pixel_distance(p1: Point, p2: Point):
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# ---------------------------
# API: 광각 보정
# ---------------------------
@app.post("/undistort")
async def undistort_image(file: UploadFile = File(...)):
    # 저장 경로
    input_path = f"temp_{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 이미지 로드
    img = cv2.imread(input_path)
    h, w = img.shape[:2]

    # 카메라 내부 파라미터 (예시값)
    K = np.array([[900, 0, w / 2], [0, 900, h / 2], [0, 0, 1]])
    dist = np.array([-0.35, 0.15, 0.0, 0.0, 0.0])

    # 보정
    undistorted = cv2.undistort(img, K, dist)
    output_path = f"undistorted_{file.filename}"
    cv2.imwrite(output_path, undistorted)

    os.remove(input_path)
    return {"result": f"saved as {output_path}"}

# ---------------------------
# API: 방 크기 계산
# ---------------------------
@app.post("/estimate-room-size")
def estimate_room_size(req: RoomPoints):
    p = req.points
    if len(p) != 4:
        return {"error": "좌표는 4개여야 합니다."}

    # xz 평면 기준
    z1 = pixel_distance(p[0], p[1])  # 1-2 (z축)
    x = pixel_distance(p[0], p[2])   # 1-3 (x축)
    x_cm = (230 / z1) * x

    # yz 평면 기준
    y = pixel_distance(p[0], p[3])   # 1-4 (y축)
    y_cm = (230 / z1) * y

    return {
        "x_cm": round(x_cm, 1),
        "y_cm": round(y_cm, 1),
        "cm_per_pixel": round(230 / z1, 4),
    }
