# backend/depth_distance.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from math import sqrt

router = APIRouter()

DEPTH_MAP_PATH = "depth_map.npy"

# 클릭 좌표 구조
class PixelPoint(BaseModel):
    x: int
    y: int

class DepthDistanceRequest(BaseModel):
    point1: PixelPoint
    point2: PixelPoint

@router.post("/depth-distance")
def compute_3d_distance(req: DepthDistanceRequest):
    if not os.path.exists(DEPTH_MAP_PATH):
        raise HTTPException(status_code=404, detail="Depth map not found")

    depth_map = np.load(DEPTH_MAP_PATH)
    h, w = depth_map.shape
    x1, y1 = req.point1.x, req.point1.y
    x2, y2 = req.point2.x, req.point2.y

    for x, y in [(x1, y1), (x2, y2)]:
        if not (0 <= x < w and 0 <= y < h):
            raise HTTPException(status_code=400, detail=f"Point ({x},{y}) out of bounds")

    d1 = float(depth_map[y1, x1])
    d2 = float(depth_map[y2, x2])

    if np.isnan(d1) or np.isnan(d2) or d1 <= 0 or d2 <= 0:
        raise HTTPException(status_code=400, detail="Invalid depth value")

    dist_pixel = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (d2 - d1)**2)
    distance_cm = dist_pixel * 1  # 단위 변환 필요시 조정

    return {
        "3d_distance_cm": round(distance_cm, 2),
        "depth1": round(d1, 3),
        "depth2": round(d2, 3),
        "pixel_distance": round(dist_pixel, 3)
    }
