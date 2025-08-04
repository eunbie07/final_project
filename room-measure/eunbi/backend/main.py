# room-measure/backend/main.py (리팩토링 후)

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import shutil
import os
import logging
import requests
from PIL import Image
import io
import json

# 분리된 모듈들 import
from models import (
    Point3D, RoomPoints, PixelPoint, DepthDistanceRequest, RoomNetRequest,
    WindowInfo, RoomAnalysis, FurniturePosition2D, FurniturePosition3D,
    FurnitureCoordinateConversionRequest, FurnitureCoordinateConversionResponse
)
from window_detection import detect_windows_in_image
from room_measurement import (
    detect_room_simple_and_stable, detect_room_with_advanced_cv,
    simulate_roomnet_detection, improved_room_measurement
)
from depth_processing import (
    generate_depth_map, get_depth_image_path, get_depth_at_point,
    get_depth_map_meta, compute_3d_distance, check_depth_files_exist
)
from mongodb_service import mongodb_service, RoomLayoutData

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

# ---------------------------
# 헬스체크 엔드포인트
# ---------------------------
@app.get("/")
def root():
    return {
        "message": "Room Measure Backend API", 
        "version": "2.0.0-refactored",
        "modules": ["window_detection", "room_measurement", "depth_processing", "mongodb_service"]
    }

@app.get("/health")
def health_check():
    """서버 상태 확인"""
    try:
        # MongoDB 연결 상태 확인
        mongodb_connected = mongodb_service.is_connected()
        
        # 깊이 관련 파일 확인
        depth_files = check_depth_files_exist()
        
        return {
            "status": "healthy",
            "timestamp": "2024-01-20T10:30:00Z",
            "services": {
                "mongodb": "connected" if mongodb_connected else "disconnected",
                "depth_processing": "available",
                "window_detection": "available", 
                "room_measurement": "available"
            },
            "files": depth_files
        }
    except Exception as e:
        logger.error(f"Health check 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

# ---------------------------
# 이미지 처리 엔드포인트
# ---------------------------
@app.post("/undistort")
async def undistort_image(file: UploadFile = File(...)):
    """이미지 왜곡 보정"""
    try:
        logger.info("이미지 왜곡 보정 시작...")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다")
        
        # 간단한 왜곡 보정 (실제로는 카메라 캘리브레이션 필요)
        h, w = img.shape[:2]
        
        # 기본 카메라 매트릭스 (예시)
        camera_matrix = np.array([[w*0.8, 0, w/2], [0, h*0.8, h/2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
        
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        
        # 결과 저장
        output_path = os.path.join(BASE_DIR, "undistorted_image.jpg")
        cv2.imwrite(output_path, undistorted)
        
        logger.info("이미지 왜곡 보정 완료")
        return FileResponse(output_path, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"이미지 왜곡 보정 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/depth-map")
async def get_depth_map(file: UploadFile = File(...)):
    """깊이 맵 생성"""
    result = await generate_depth_map(file)
    if result["success"]:
        return JSONResponse(content=result)
    else:
        return JSONResponse(status_code=500, content=result)

@app.get("/depth-map-image")
def get_depth_image():
    """깊이 이미지 반환"""
    try:
        logger.info(f"Depth 이미지 요청")
        
        depth_image_path = get_depth_image_path()
        if not depth_image_path:
            logger.error("Depth 이미지 파일이 존재하지 않음")
            return JSONResponse(status_code=404, content={"error": "depth image 파일이 없습니다"})
        
        logger.info("Depth 이미지 반환")
        return FileResponse(depth_image_path, media_type="image/png")
    except Exception as e:
        logger.error(f"Depth 이미지 반환 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get-depth-at-point")
async def get_depth_at_point_endpoint(x: int = Query(...), y: int = Query(...)):
    """특정 좌표의 깊이 값 조회"""
    result = get_depth_at_point(x, y)
    if result["success"]:
        return result
    else:
        return JSONResponse(
            status_code=400 if "범위" in result["error"] else 500,
            content=result
        )

@app.get("/depth-meta")
def get_depth_map_meta_endpoint():
    """깊이 맵 메타데이터 조회"""
    result = get_depth_map_meta()
    if result["success"]:
        return {"width": result["width"], "height": result["height"]}
    else:
        return JSONResponse(status_code=404 if "meta 파일" in result["error"] else 500, content=result)

# ---------------------------
# 거리 계산 엔드포인트
# ---------------------------
@app.post("/depth-distance")
def compute_3d_distance_endpoint(req: DepthDistanceRequest):
    """두 점 사이의 3D 거리 계산"""
    result = compute_3d_distance(req)
    if result["success"]:
        return {
            "3d_distance_cm": result["3d_distance_cm"],
            "depth1": result["depth1"],
            "depth2": result["depth2"],
            "pixel_distance": result["pixel_distance"]
        }
    else:
        return JSONResponse(status_code=400 if "out of bounds" in result["error"] else 500, content=result)

# ---------------------------
# RoomNet 자동 감지 엔드포인트
# ---------------------------
@app.post("/auto-detect-room")
async def auto_detect_room(file: UploadFile = File(...), confidence_threshold: float = 0.7):
    """방 자동 감지"""
    try:
        logger.info("RoomNet 자동 방 감지 시작...")
        
        # 임시 파일로 저장
        temp_filename = f"temp_roomnet_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # 안정적인 방 감지 알고리즘 사용
            detection_result = detect_room_simple_and_stable(temp_filename, confidence_threshold)
            
            if detection_result["success"]:
                logger.info(f"✅ 방 감지 성공 - 신뢰도: {detection_result['confidence']:.1%}")
                return JSONResponse(content=detection_result)
            else:
                logger.warning("❌ 방 감지 실패")
                return JSONResponse(status_code=422, content=detection_result)
                
        finally:
            # 임시 파일 정리
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                logger.info(f"임시 파일 삭제: {temp_filename}")
        
    except Exception as e:
        logger.error(f"자동 방 감지 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# 방 크기 추정 엔드포인트
# ---------------------------
@app.post("/estimate-room-size")
def estimate_room_size(req: RoomPoints):
    """방 크기 추정"""
    try:
        logger.info(f"🔥 방 크기 추정 API 호출됨!")
        logger.info(f"📊 요청 데이터: 포인트 {len(req.points)}개, 목표높이 {req.target_height}m")
        
        # 받은 포인트들 상세 로깅
        for i, point in enumerate(req.points):
            logger.info(f"  포인트 {i+1}: x={point.x}, y={point.y}, z={point.z}")
        
        result = improved_room_measurement(req.points, req.target_height)
        
        if result["success"]:
            # 프론트엔드 호환성을 위한 추가 형식
            response_data = {
                **result,
                # 기존 형식 유지 (하위 호환성)
                "width_cm": result["dimensions"]["width_cm"],
                "depth_cm": result["dimensions"]["depth_cm"], 
                "height_cm": result["dimensions"]["height_cm"],
                "area_sqm": result["calculated_values"]["area_sqm"],
                "volume_cum": result["calculated_values"]["volume_cum"],
                # 3D 시스템용 추가 정보
                "roomInfo": result.get("room_info", {}),
                "measurementPoints": result.get("measurement_points", {})
            }
            
            logger.info(f"✅ 방 크기 측정 완료: {result['dimensions']['width_cm']}×{result['dimensions']['depth_cm']}×{result['dimensions']['height_cm']}cm")
            
            # CORS 및 Content-Type 헤더 명시적 설정
            return JSONResponse(
                content=response_data,
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )
        else:
            logger.error(f"❌ 방 크기 측정 실패: {result.get('error', '알 수 없는 오류')}")
            return JSONResponse(status_code=400, content=result)
            
    except Exception as e:
        logger.error(f"방 크기 추정 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/test-room-size")
def test_room_size():
    """방 크기 측정 테스트 엔드포인트"""
    try:
        # 테스트용 더미 데이터
        test_response = {
            "success": True,
            "dimensions": {
                "width_cm": 400.0,
                "depth_cm": 350.0,
                "height_cm": 230.0,
                "width_m": 4.0,
                "depth_m": 3.5,
                "height_m": 2.3
            },
            "calculated_values": {
                "area_sqm": 14.0,
                "volume_cum": 32.2,
                "pixels_per_meter": 28.0
            },
            "confidence": 0.85,
            "method": "test",
            # 프론트엔드 호환성
            "width_cm": 400.0,
            "depth_cm": 350.0,
            "height_cm": 230.0,
            "area_sqm": 14.0,
            "volume_cum": 32.2,
            "roomInfo": {
                "width": 400.0,
                "height": 230.0,
                "depth": 350.0
            }
        }
        
        return JSONResponse(
            content=test_response,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/debug-estimate-room-size")
def debug_estimate_room_size(data: dict):
    """디버깅용 방 크기 추정 엔드포인트"""
    logger.info(f"🔥 디버그 API 호출됨!")
    logger.info(f"📊 받은 데이터: {data}")
    
    return JSONResponse(content={
        "message": "디버그 API 호출 성공",
        "received_data": data,
        "timestamp": "2024-01-20T10:30:00Z"
    })

# ---------------------------
# 창문 감지 엔드포인트
# ---------------------------
@app.post("/detect-windows")
async def detect_windows(file: UploadFile = File(...)):
    """창문 감지"""
    try:
        logger.info("창문 감지 시작...")
        
        # 이미지 디코딩
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_array is None:
            raise ValueError("이미지를 디코딩할 수 없습니다")
        
        # 창문 감지 수행
        windows = detect_windows_in_image(image_array)
        
        logger.info(f"창문 감지 완료: {len(windows)}개")
        
        return {
            "success": True,
            "windows": [window.model_dump() for window in windows],
            "total_windows": len(windows),
            "room_analysis": {
                "room_dimensions": {"analyzed": True},
                "windows": [window.model_dump() for window in windows]
            }
        }
        
    except Exception as e:
        logger.error(f"창문 감지 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# 가구 좌표 변환 API
# ---------------------------
@app.post("/convert-furniture-coordinates", response_model=FurnitureCoordinateConversionResponse)
def convert_furniture_coordinates(req: FurnitureCoordinateConversionRequest):
    """가구 좌표 변환 (3D → 2D)"""
    try:
        logger.info(f"가구 좌표 변환: {req.furniture_id}")
        
        # 3D 좌표를 2D 좌표로 변환
        # 간단한 직교 투영 사용 (실제로는 더 복잡한 변환 필요)
        position_2d = FurniturePosition2D(
            x=req.position_3d.x,  # X는 그대로 유지
            z=req.position_3d.z   # Z를 2D의 Y로 사용
        )
        
        # 방 크기에 따른 스케일링
        room_width, room_height, room_depth = req.room_size
        
        # 정규화 후 2D 공간으로 매핑
        normalized_x = position_2d.x / room_width
        normalized_z = position_2d.z / room_depth
        
        # 2D 공간 크기 (예: 400x400)
        canvas_size = 400
        final_x = normalized_x * canvas_size
        final_z = normalized_z * canvas_size
        
        result = FurnitureCoordinateConversionResponse(
            furniture_id=req.furniture_id,
            position_2d=FurniturePosition2D(x=final_x, z=final_z)
        )
        
        logger.info(f"변환 완료: 3D({req.position_3d.x},{req.position_3d.z}) → 2D({final_x:.1f},{final_z:.1f})")
        
        return result
        
    except Exception as e:
        logger.error(f"가구 좌표 변환 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# MongoDB 저장 API
# ---------------------------
@app.post("/save-room-layout")
async def save_room_layout(layout_data: RoomLayoutData):
    """방 레이아웃 데이터 저장"""
    try:
        result = await mongodb_service.save_room_layout(layout_data)
        return result
    except Exception as e:
        logger.error(f"레이아웃 저장 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/room-layouts")
async def get_room_layouts(limit: int = 10, skip: int = 0):
    """저장된 방 레이아웃 목록 조회"""
    try:
        result = await mongodb_service.get_room_layouts(limit, skip)
        return result
    except Exception as e:
        logger.error(f"레이아웃 조회 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/room-layout/{layout_id}")
async def get_room_layout_by_id(layout_id: str):
    """특정 방 레이아웃 조회"""
    try:
        result = await mongodb_service.get_room_layout_by_id(layout_id)
        return result
    except Exception as e:
        logger.error(f"레이아웃 조회 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------------------------
# 서버 시작 이벤트
# ---------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Room Measure Backend API 시작 (리팩토링 완료)")
    logger.info("📦 로드된 모듈: window_detection, room_measurement, depth_processing, mongodb_service")
    
    # MongoDB 연결 확인
    if mongodb_service.is_connected():
        logger.info("✅ MongoDB 연결 성공")
    else:
        logger.warning("⚠️  MongoDB 연결 실패")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)