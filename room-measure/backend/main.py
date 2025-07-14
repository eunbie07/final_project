# room-measure/backend/main.py

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import torch
import cv2
import numpy as np
import shutil
import os
from typing import List
from math import sqrt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_MAP_PATH = os.path.join(BASE_DIR, "depth_map.npy")
DEPTH_IMAGE_PATH = os.path.join(BASE_DIR, "depth_map_output.png")
DEPTH_META_PATH = os.path.join(BASE_DIR, "depth_meta.txt")

# ---------------------------
# 모델 정의
# ---------------------------
class Point3D(BaseModel):
    x: float
    y: float
    z: float

class RoomPoints(BaseModel):
    points: List[Point3D]

class PixelPoint(BaseModel):
    x: int
    y: int

class DepthDistanceRequest(BaseModel):
    point1: PixelPoint
    point2: PixelPoint

# ---------------------------
# 유틸리티 함수
# ---------------------------
def distance_3d(p1: Point3D, p2: Point3D):
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# ---------------------------
# 헬스체크 엔드포인트
# ---------------------------
@app.get("/")
def root():
    return {"message": "서버가 정상 작동 중입니다!", "status": "healthy"}

@app.get("/health")
def health_check():
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append(f"{list(route.methods)} {route.path}")
        elif hasattr(route, 'path'):
            routes.append(f"[GET] {route.path}")
    
    return {
        "status": "healthy", 
        "routes": routes,
        "files": {
            "depth_map": os.path.exists(DEPTH_MAP_PATH),
            "depth_image": os.path.exists(DEPTH_IMAGE_PATH),
            "depth_meta": os.path.exists(DEPTH_META_PATH)
        }
    }

# ---------------------------
# 이미지 처리 엔드포인트
# ---------------------------
@app.post("/undistort")
async def undistort_image(file: UploadFile = File(...)):
    print(f"📁 undistort 요청 받음: {file.filename}")
    input_path = f"temp_{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(input_path)
    h, w = img.shape[:2]
    K = np.array([[900, 0, w / 2], [0, 900, h / 2], [0, 0, 1]])
    dist = np.array([-0.35, 0.15, 0.0, 0.0, 0.0])
    undistorted = cv2.undistort(img, K, dist)

    output_path = f"undistorted_{file.filename}"
    cv2.imwrite(output_path, undistorted)
    os.remove(input_path)

    return {"result": f"saved as {output_path}"}

@app.post("/depth-map")
async def get_depth_map(file: UploadFile = File(...)):
    try:
        print("🔄 Depth map 생성 시작...")
        
        # 모델 로딩
        model_type = "MiDaS_small"
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # 이미지 디코딩
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 추론
        input_tensor = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = model(input_tensor)

        depth = prediction.squeeze().cpu().numpy()
        h, w = depth.shape

        # 메타데이터 저장
        with open(DEPTH_META_PATH, "w") as f:
            f.write(f"{w},{h}")

        # depth map 저장
        np.save(DEPTH_MAP_PATH, depth)

        # 시각화 이미지 생성 및 저장
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        cv2.imwrite(DEPTH_IMAGE_PATH, depth_vis)

        print(f"✅ Depth map 생성 완료!")
        print(f"  - 크기: {w} x {h}")
        print(f"  - 파일: {DEPTH_IMAGE_PATH}")
        print(f"  - 파일 존재: {os.path.exists(DEPTH_IMAGE_PATH)}")

        return JSONResponse(content={
            "depth_image_url": DEPTH_IMAGE_PATH,
            "depth_width": w,
            "depth_height": h,
            "message": "depth map 생성 완료"
        })

    except Exception as e:
        print(f"❌ Depth map 생성 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/depth-map-image")
def get_depth_image():
    print(f"🔍 Depth 이미지 요청")
    print(f"  - 파일 경로: {DEPTH_IMAGE_PATH}")
    print(f"  - 파일 존재: {os.path.exists(DEPTH_IMAGE_PATH)}")
    
    if not os.path.exists(DEPTH_IMAGE_PATH):
        print("❌ Depth 이미지 파일이 존재하지 않음")
        return JSONResponse(status_code=404, content={"error": "depth image 파일이 없습니다"})
    
    print("✅ Depth 이미지 반환")
    return FileResponse(DEPTH_IMAGE_PATH, media_type="image/png")

@app.get("/get-depth-at-point")
async def get_depth_at_point(x: int = Query(...), y: int = Query(...)):
    print(f"🔍 깊이 값 요청: ({x}, {y})")
    
    if not os.path.exists(DEPTH_MAP_PATH):
        return JSONResponse(
            status_code=400,
            content={"error": "Depth map not found. 먼저 /depth-map API를 호출해 주세요."}
        )

    depth_map = np.load(DEPTH_MAP_PATH)
    h, w = depth_map.shape

    if x < 0 or y < 0 or x >= w or y >= h:
        return JSONResponse(
            status_code=400,
            content={"error": f"좌표 ({x},{y})가 이미지 범위를 벗어났습니다. 허용 범위: 0~{w-1}, 0~{h-1}"}
        )

    depth_value = float(depth_map[y, x])
    print(f"  - 깊이 값: {depth_value:.3f}")

    if np.isnan(depth_value) or depth_value <= 0:
        return JSONResponse(
            status_code=400,
            content={"error": f"잘못된 깊이 값입니다: {depth_value}"}
        )

    return {"depth": depth_value}

@app.get("/depth-meta")
def get_depth_map_meta():
    if not os.path.exists(DEPTH_META_PATH):
        return JSONResponse(status_code=404, content={"error": "meta 파일 없음"})

    with open(DEPTH_META_PATH, "r") as f:
        w, h = f.read().strip().split(",")
        return {"width": int(w), "height": int(h)}

# ---------------------------
# 거리 계산 엔드포인트
# ---------------------------
@app.post("/depth-distance")
def compute_3d_distance(req: DepthDistanceRequest):
    if not os.path.exists(DEPTH_MAP_PATH):
        return JSONResponse(status_code=404, content={"error": "Depth map not found"})

    depth_map = np.load(DEPTH_MAP_PATH)
    h, w = depth_map.shape
    x1, y1 = req.point1.x, req.point1.y
    x2, y2 = req.point2.x, req.point2.y

    for x, y in [(x1, y1), (x2, y2)]:
        if not (0 <= x < w and 0 <= y < h):
            return JSONResponse(status_code=400, content={"error": f"Point ({x},{y}) out of bounds"})

    d1 = float(depth_map[y1, x1])
    d2 = float(depth_map[y2, x2])

    if np.isnan(d1) or np.isnan(d2) or d1 <= 0 or d2 <= 0:
        return JSONResponse(status_code=400, content={"error": "Invalid depth value"})

    dist_pixel = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (d2 - d1)**2)
    distance_cm = dist_pixel * 1

    return {
        "3d_distance_cm": round(distance_cm, 2),
        "depth1": round(d1, 3),
        "depth2": round(d2, 3),
        "pixel_distance": round(dist_pixel, 3)
    }

# ---------------------------
# 방 크기 추정 엔드포인트
# ---------------------------
@app.post("/estimate-room-size")
def estimate_room_size(req: RoomPoints):
    print("💡 방 크기 추정 요청:", req.points)

    if len(req.points) != 4:
        return {"error": "좌표는 정확히 4개여야 합니다."}

    p = req.points
    
    # 올바른 방법: 2D 픽셀 거리 기반 계산
    # 1. 수직 거리 (층고) - 점1(바닥)과 점2(천장)
    vertical_pixel_distance = sqrt(
        (p[1].x - p[0].x)**2 + (p[1].y - p[0].y)**2
    )
    
    print(f"🔍 수직 픽셀 거리: {vertical_pixel_distance:.2f}")
    print(f"🔍 점1 (바닥): ({p[0].x}, {p[0].y})")
    print(f"🔍 점2 (천장): ({p[1].x}, {p[1].y})")
    print(f"🔍 점3 (왼쪽): ({p[2].x}, {p[2].y})")
    print(f"🔍 점4 (오른쪽): ({p[3].x}, {p[3].y})")
    
    if vertical_pixel_distance == 0:
        return {"error": "수직 거리가 0입니다. 다른 점을 선택해 주세요."}
    
    # 2. 픽셀당 실제 거리 비율 (230cm 층고 기준)
    cm_per_pixel = 230.0 / vertical_pixel_distance
    
    print(f"🔍 픽셀당 실제 거리: {cm_per_pixel:.4f} cm/pixel")
    
    # 3. 가로 거리 - 점1(기준)과 점4(오른쪽 바닥)
    horizontal_pixel_distance = sqrt(
        (p[3].x - p[0].x)**2 + (p[3].y - p[0].y)**2
    )
    
    # 4. 세로 거리 - 점1(기준)과 점3(왼쪽 바닥)  
    depth_pixel_distance = sqrt(
        (p[2].x - p[0].x)**2 + (p[2].y - p[0].y)**2
    )
    
    print(f"🔍 가로 픽셀 거리: {horizontal_pixel_distance:.2f}")
    print(f"🔍 세로 픽셀 거리: {depth_pixel_distance:.2f}")
    
    # 5. 실제 거리로 변환
    width_cm = horizontal_pixel_distance * cm_per_pixel
    depth_cm = depth_pixel_distance * cm_per_pixel
    
    print(f"📏 계산된 방 크기: 가로 {width_cm:.1f}cm × 세로 {depth_cm:.1f}cm")

    return {
        "width_cm": round(width_cm, 1),
        "depth_cm": round(depth_cm, 1),
        "cm_per_pixel": round(cm_per_pixel, 4),
        "vertical_pixels": round(vertical_pixel_distance, 2),
        "horizontal_pixels": round(horizontal_pixel_distance, 2),
        "depth_pixels": round(depth_pixel_distance, 2)
    }

# ---------------------------
# 서버 시작 이벤트
# ---------------------------
@app.on_event("startup")
async def startup_event():
    print("🚀 서버 시작됨")
    print("📁 현재 작업 디렉토리:", os.getcwd())
    print("📝 등록된 엔드포인트:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = list(route.methods)
            if 'HEAD' in methods:
                methods.remove('HEAD')
            if 'OPTIONS' in methods:
                methods.remove('OPTIONS')
            if methods:
                print(f"  - {methods} {route.path}")

if __name__ == "__main__":
    import uvicorn
    print("🔥 서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")