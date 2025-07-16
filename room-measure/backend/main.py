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
from typing import List, Optional
from math import sqrt, isnan, isinf
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    target_height: Optional[float] = 2.3  # m 단위, 기본값 2.3m

class PixelPoint(BaseModel):
    x: int
    y: int

class DepthDistanceRequest(BaseModel):
    point1: PixelPoint
    point2: PixelPoint

# ---------------------------
# 유틸리티 함수
# ---------------------------
def distance_2d(p1: Point3D, p2: Point3D) -> float:
    """2D 픽셀 거리 계산"""
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def validate_points(points: List[Point3D]) -> tuple[bool, str]:
    """좌표 유효성 검사"""
    if len(points) != 4:
        return False, "좌표는 정확히 4개여야 합니다"
    
    # NaN, 무한대 체크
    for i, point in enumerate(points):
        if any(isnan(val) or isinf(val) for val in [point.x, point.y, point.z]):
            return False, f"점 {i+1}에 유효하지 않은 값이 있습니다"
    
    # 기본 기하학적 검사
    vertical_dist = distance_2d(points[0], points[1])
    if vertical_dist < 10:  # 최소 10픽셀
        return False, "수직 거리가 너무 짧습니다"
    
    horizontal_dist = distance_2d(points[0], points[3])
    if horizontal_dist < 10:
        return False, "가로 거리가 너무 짧습니다"
    
    return True, "OK"

def calculate_confidence(points: List[Point3D]) -> float:
    """측정 신뢰도 계산 (0.0 ~ 1.0)"""
    confidence = 0.0
    
    # 1. 기하학적 일관성 (0.4점)
    try:
        vertical_dist = distance_2d(points[0], points[1])
        horizontal_dist = distance_2d(points[0], points[3])
        depth_dist = distance_2d(points[0], points[2])
        
        # 비율이 합리적인지 확인
        if 0.3 <= vertical_dist/horizontal_dist <= 3.0:
            confidence += 0.2
        if 0.3 <= vertical_dist/depth_dist <= 3.0:
            confidence += 0.2
    except:
        pass
    
    # 2. 깊이 값 합리성 (0.3점)
    depth_values = [p.z for p in points]
    depth_range = max(depth_values) - min(depth_values)
    
    # 깊이 변화가 있지만 극단적이지 않은 경우
    if 10 < depth_range < 500:  # MiDaS 출력 범위 고려
        confidence += 0.3
    elif depth_range <= 10:  # 거의 변화 없음
        confidence += 0.1
    
    # 3. 점들의 분포 (0.3점)
    # 점들이 너무 가깝지 않은지 확인
    min_distance = min([
        distance_2d(points[i], points[j]) 
        for i in range(4) for j in range(i+1, 4)
    ])
    
    if min_distance > 20:
        confidence += 0.3
    elif min_distance > 10:
        confidence += 0.2
    elif min_distance > 5:
        confidence += 0.1
    
    return min(confidence, 1.0)

def improved_room_measurement(points: List[Point3D], target_height: float) -> dict:
    """개선된 방 크기 측정 - MiDaS를 상대적 깊이로만 활용"""
    
    logger.info(f"🔍 개선된 측정 시작:")
    for i, p in enumerate(points):
        logger.info(f"   점{i+1}: ({p.x:.1f}, {p.y:.1f}, depth={p.z:.2f})")
    
    # 1. 기본 2D 픽셀 거리 계산
    vertical_pixels = distance_2d(points[0], points[1])    # 층고 기준선
    horizontal_pixels = distance_2d(points[0], points[3])  # 가로
    depth_pixels = distance_2d(points[0], points[2])       # 세로
    
    logger.info(f"📏 2D 픽셀 거리:")
    logger.info(f"   수직: {vertical_pixels:.1f}px")
    logger.info(f"   가로: {horizontal_pixels:.1f}px")
    logger.info(f"   세로: {depth_pixels:.1f}px")
    
    # 2. MiDaS 깊이를 이용한 원근 보정 계산
    depth_values = [p.z for p in points]
    depth_variance = np.std(depth_values)
    depth_range = max(depth_values) - min(depth_values)
    
    # 원근 효과 보정 팩터 (매우 보수적으로 적용)
    perspective_factor_horizontal = 1.0
    perspective_factor_depth = 1.0
    
    if depth_range > 20:  # 충분한 깊이 차이가 있는 경우만
        # 가로 방향 원근 보정 (점0과 점3의 깊이 차이 활용)
        horizontal_depth_diff = abs(points[3].z - points[0].z)
        if horizontal_depth_diff > 10:
            perspective_factor_horizontal = 1.0 + (horizontal_depth_diff * 0.0005)
        
        # 세로 방향 원근 보정 (점0과 점2의 깊이 차이 활용)
        depth_depth_diff = abs(points[2].z - points[0].z)
        if depth_depth_diff > 10:
            perspective_factor_depth = 1.0 + (depth_depth_diff * 0.0005)
    
    logger.info(f"🔧 원근 보정:")
    logger.info(f"   깊이 범위: {depth_range:.2f}")
    logger.info(f"   가로 보정 팩터: {perspective_factor_horizontal:.4f}")
    logger.info(f"   세로 보정 팩터: {perspective_factor_depth:.4f}")
    
    # 3. 스케일 계산 (층고 기준)
    target_height_cm = target_height * 100  # m를 cm로 변환
    scale_factor = target_height_cm / vertical_pixels if vertical_pixels > 0 else 1.0
    
    logger.info(f"📐 스케일 계산:")
    logger.info(f"   목표 층고: {target_height_cm}cm")
    logger.info(f"   스케일 팩터: {scale_factor:.4f} cm/pixel")
    
    # 4. 최종 크기 계산
    final_width_cm = horizontal_pixels * scale_factor * perspective_factor_horizontal
    final_depth_cm = depth_pixels * scale_factor * perspective_factor_depth
    
    # 5. 합리적 범위 체크 및 제한
    final_width_cm = max(100, min(final_width_cm, 2000))  # 1m ~ 20m
    final_depth_cm = max(100, min(final_depth_cm, 2000))  # 1m ~ 20m
    
    logger.info(f"📊 최종 결과:")
    logger.info(f"   가로: {final_width_cm:.1f}cm")
    logger.info(f"   세로: {final_depth_cm:.1f}cm")
    
    # 6. 신뢰도 계산
    confidence = calculate_confidence(points)
    
    return {
        "width_cm": round(final_width_cm, 1),
        "depth_cm": round(final_depth_cm, 1),
        "height_cm": target_height_cm,
        "method": "improved_midas_relative",
        "confidence": round(confidence, 3),
        "scale_factor": round(scale_factor, 4),
        "perspective_correction": {
            "horizontal_factor": round(perspective_factor_horizontal, 4),
            "depth_factor": round(perspective_factor_depth, 4),
            "depth_range": round(depth_range, 2)
        },
        "pixel_distances": {
            "vertical": round(vertical_pixels, 1),
            "horizontal": round(horizontal_pixels, 1),
            "depth": round(depth_pixels, 1)
        },
        "measurement_quality": {
            "confidence_score": round(confidence, 3),
            "reliability": "높음" if confidence > 0.8 else "보통" if confidence > 0.6 else "낮음"
        }
    }

# ---------------------------
# 헬스체크 엔드포인트
# ---------------------------
@app.get("/")
def root():
    try:
        return {"message": "서버가 정상 작동 중입니다!", "status": "healthy"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health_check():
    try:
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
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# 이미지 처리 엔드포인트
# ---------------------------
@app.post("/undistort")
async def undistort_image(file: UploadFile = File(...)):
    try:
        logger.info(f"📁 undistort 요청 받음: {file.filename}")
        input_path = f"temp_{file.filename}"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("이미지를 읽을 수 없습니다")
            
        h, w = img.shape[:2]
        K = np.array([[900, 0, w / 2], [0, 900, h / 2], [0, 0, 1]])
        dist = np.array([-0.35, 0.15, 0.0, 0.0, 0.0])
        undistorted = cv2.undistort(img, K, dist)

        output_path = f"undistorted_{file.filename}"
        cv2.imwrite(output_path, undistorted)
        os.remove(input_path)

        return {"result": f"saved as {output_path}"}
    except Exception as e:
        logger.error(f"Undistort 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/depth-map")
async def get_depth_map(file: UploadFile = File(...)):
    try:
        logger.info("🔄 Depth map 생성 시작...")
        
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
        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다")
            
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

        logger.info(f"✅ Depth map 생성 완료!")
        logger.info(f"  - 크기: {w} x {h}")

        return JSONResponse(content={
            "depth_image_url": DEPTH_IMAGE_PATH,
            "depth_width": w,
            "depth_height": h,
            "message": "depth map 생성 완료"
        })

    except Exception as e:
        logger.error(f"❌ Depth map 생성 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/depth-map-image")
def get_depth_image():
    try:
        logger.info(f"🔍 Depth 이미지 요청")
        
        if not os.path.exists(DEPTH_IMAGE_PATH):
            logger.error("❌ Depth 이미지 파일이 존재하지 않음")
            return JSONResponse(status_code=404, content={"error": "depth image 파일이 없습니다"})
        
        logger.info("✅ Depth 이미지 반환")
        return FileResponse(DEPTH_IMAGE_PATH, media_type="image/png")
    except Exception as e:
        logger.error(f"Depth 이미지 반환 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get-depth-at-point")
async def get_depth_at_point(x: int = Query(...), y: int = Query(...)):
    try:
        logger.info(f"🔍 깊이 값 요청: ({x}, {y})")
        
        if not os.path.exists(DEPTH_MAP_PATH):
            return JSONResponse(
                status_code=404,
                content={"error": "Depth map not found. 먼저 이미지를 업로드하고 depth-map을 생성해주세요."}
            )

        depth_map = np.load(DEPTH_MAP_PATH)
        h, w = depth_map.shape
        
        logger.info(f"📊 Depth map 정보: {w} × {h}")

        # 좌표 범위 체크 및 안전한 클램핑
        if x < 0 or y < 0 or x >= w or y >= h:
            logger.warning(f"⚠️ 좌표 범위 초과: ({x}, {y}), 허용 범위: 0~{w-1}, 0~{h-1}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"좌표가 범위를 벗어났습니다.",
                    "requested": {"x": x, "y": y},
                    "valid_range": {"x": [0, w-1], "y": [0, h-1]},
                    "depth_map_size": {"width": w, "height": h}
                }
            )

        # 안전한 좌표로 클램핑 (혹시 모를 경우를 대비)
        safe_x = max(0, min(x, w-1))
        safe_y = max(0, min(y, h-1))
        
        if safe_x != x or safe_y != y:
            logger.info(f"🔧 좌표 보정: ({x}, {y}) → ({safe_x}, {safe_y})")

        depth_value = float(depth_map[safe_y, safe_x])
        logger.info(f"  ✅ 깊이 값: {depth_value:.6f}")

        if np.isnan(depth_value) or np.isinf(depth_value):
            logger.warning(f"⚠️ 유효하지 않은 깊이 값: {depth_value}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "해당 위치의 깊이 정보가 유효하지 않습니다.",
                    "depth_value": str(depth_value),
                    "suggestion": "다른 지점을 클릭해보세요."
                }
            )

        return {
            "depth": depth_value,
            "coordinates": {"x": safe_x, "y": safe_y},
            "original_request": {"x": x, "y": y},
            "depth_map_size": {"width": w, "height": h}
        }
        
    except Exception as e:
        logger.error(f"❌ 깊이 값 조회 실패: {str(e)}")
        return JSONResponse(status_code=500, content={
            "error": "서버 내부 오류가 발생했습니다.",
            "details": str(e)
        })

@app.get("/depth-meta")
def get_depth_map_meta():
    try:
        if not os.path.exists(DEPTH_META_PATH):
            return JSONResponse(status_code=404, content={"error": "meta 파일 없음"})

        with open(DEPTH_META_PATH, "r") as f:
            w, h = f.read().strip().split(",")
            return {"width": int(w), "height": int(h)}
    except Exception as e:
        logger.error(f"Meta 정보 조회 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# 거리 계산 엔드포인트
# ---------------------------
@app.post("/depth-distance")
def compute_3d_distance(req: DepthDistanceRequest):
    try:
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

        if np.isnan(d1) or np.isnan(d2):
            return JSONResponse(status_code=400, content={"error": "Invalid depth value"})

        dist_pixel = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (d2 - d1)**2)
        distance_cm = dist_pixel * 1

        return {
            "3d_distance_cm": round(distance_cm, 2),
            "depth1": round(d1, 3),
            "depth2": round(d2, 3),
            "pixel_distance": round(dist_pixel, 3)
        }
    except Exception as e:
        logger.error(f"거리 계산 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# 방 크기 추정 엔드포인트 (개선됨)
# ---------------------------
@app.post("/estimate-room-size")
def estimate_room_size(req: RoomPoints):
    try:
        logger.info("🏠 개선된 방 크기 추정 요청")
        
        # 1. 입력 검증
        is_valid, error_msg = validate_points(req.points)
        if not is_valid:
            logger.warning(f"❌ 입력 검증 실패: {error_msg}")
            return JSONResponse(status_code=400, content={"error": error_msg})
        
        # 2. 개선된 측정 수행
        result = improved_room_measurement(req.points, req.target_height)
        
        # 3. 최종 검증
        confidence = result["confidence"]
        if confidence < 0.3:
            logger.warning(f"⚠️ 낮은 신뢰도: {confidence:.3f}")
            result["warning"] = "측정 신뢰도가 낮습니다. 다른 각도에서 시도해보세요."
        
        logger.info(f"✅ 측정 완료: 가로 {result['width_cm']}cm × 세로 {result['depth_cm']}cm (신뢰도: {confidence:.1%})")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 방 크기 추정 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# 서버 시작 이벤트
# ---------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 개선된 서버 시작됨")
    logger.info("📁 현재 작업 디렉토리: " + os.getcwd())
    logger.info("📝 등록된 엔드포인트:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = list(route.methods)
            if 'HEAD' in methods:
                methods.remove('HEAD')
            if 'OPTIONS' in methods:
                methods.remove('OPTIONS')
            if methods:
                logger.info(f"  - {methods} {route.path}")

if __name__ == "__main__":
    import uvicorn
    logger.info("🔥 개선된 서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")