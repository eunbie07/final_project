# 로컬용 이미지/AI 처리 백엔드 (포트 3010)
# 무거운 이미지 처리 및 AI 분석 담당

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
    WindowInfo, RoomAnalysis
)
from window_detection import detect_windows_in_image
from room_measurement import (
    detect_room_simple_and_stable, detect_vanishing_points_and_room_corners,
    simulate_roomnet_detection, improved_room_measurement
)
from ai_room_detection import detect_room_with_ai
from depth_processing import (
    generate_depth_map, get_depth_image_path, get_depth_at_point,
    get_depth_map_meta, compute_3d_distance, check_depth_files_exist
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Room Measure Local API", version="1.0.0")

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
# 이미지 처리 및 깊이 맵 관련 API
# ---------------------------

@app.get("/")
async def root():
    return {"message": "Room Measure Local API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "local"}

@app.post("/undistort")
async def undistort_image(file: UploadFile = File(...)):
    """이미지 왜곡 보정"""
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400, 
                content={"error": "이미지 파일만 업로드 가능합니다"}
            )

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "이미지를 읽을 수 없습니다"}
            )

        # 이미지 왜곡 보정 처리
        h, w = img.shape[:2]
        
        # 카메라 캘리브레이션 매개변수 (일반적인 값들)
        camera_matrix = np.array([[w, 0, w/2],
                                 [0, w, h/2],
                                 [0, 0, 1]], dtype=np.float32)
        
        dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
        
        # 왜곡 보정
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        
        # 보정된 이미지 저장
        output_path = os.path.join(BASE_DIR, "undistorted_image.jpg")
        cv2.imwrite(output_path, undistorted)
        
        logger.info(f"이미지 왜곡 보정 완료: {output_path}")
        
        # 보정된 이미지를 바로 반환
        return FileResponse(
            path=output_path,
            media_type="image/jpeg",
            filename="undistorted_image.jpg"
        )
        
    except Exception as e:
        logger.error(f"이미지 왜곡 보정 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"이미지 처리 중 오류가 발생했습니다: {str(e)}"}
        )

@app.post("/depth-map")
async def create_depth_map():
    """깊이 맵 생성"""
    try:
        undistorted_path = os.path.join(BASE_DIR, "undistorted_image.jpg")
        
        if not os.path.exists(undistorted_path):
            return JSONResponse(
                status_code=404,
                content={"error": "왜곡 보정된 이미지를 찾을 수 없습니다. 먼저 이미지를 업로드하고 왜곡 보정을 수행해주세요."}
            )
        
        success = await generate_depth_map(undistorted_path)
        
        if success:
            logger.info("깊이 맵 생성 완료")
            return {"success": True, "message": "깊이 맵이 생성되었습니다"}
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "깊이 맵 생성에 실패했습니다"}
            )
            
    except Exception as e:
        logger.error(f"깊이 맵 생성 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"깊이 맵 생성 중 오류가 발생했습니다: {str(e)}"}
        )

@app.get("/depth-map-image")
async def get_depth_map_image():
    """깊이 맵 이미지 조회"""
    try:
        depth_image_path = get_depth_image_path()
        
        if depth_image_path and os.path.exists(depth_image_path):
            return FileResponse(
                path=depth_image_path,
                media_type="image/png",
                filename="depth_map_output.png"
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "깊이 맵 이미지를 찾을 수 없습니다. 먼저 깊이 맵을 생성해주세요."}
            )
            
    except Exception as e:
        logger.error(f"깊이 맵 이미지 조회 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"깊이 맵 조회 중 오류가 발생했습니다: {str(e)}"}
        )

@app.get("/get-depth-at-point")
async def get_depth_at_point_endpoint(x: int = Query(...), y: int = Query(...)):
    """특정 좌표의 깊이 값 조회"""
    try:
        result = get_depth_at_point(x, y)
        
        # get_depth_at_point는 딕셔너리를 반환합니다
        if isinstance(result, dict) and result.get("success"):
            return {"x": x, "y": y, "depth": result["depth"]}
        else:
            error_msg = result.get("error", "해당 좌표의 깊이 값을 조회할 수 없습니다") if isinstance(result, dict) else "깊이 값 조회 실패"
            return JSONResponse(
                status_code=400,
                content={"error": error_msg}
            )
            
    except Exception as e:
        logger.error(f"깊이 값 조회 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"깊이 값 조회 중 오류가 발생했습니다: {str(e)}"}
        )

@app.get("/depth-meta")
async def get_depth_meta():
    """깊이 맵 메타 정보 조회"""
    try:
        meta_info = get_depth_map_meta()
        
        if meta_info:
            return meta_info
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "깊이 맵 메타 정보를 찾을 수 없습니다"}
            )
            
    except Exception as e:
        logger.error(f"깊이 맵 메타 정보 조회 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"메타 정보 조회 중 오류가 발생했습니다: {str(e)}"}
        )

@app.post("/depth-distance")
async def calculate_depth_distance(request: DepthDistanceRequest):
    """깊이 맵을 이용한 3D 거리 계산"""
    try:
        distance = compute_3d_distance(request.point1, request.point2)
        
        if distance is not None:
            return {
                "point1": request.point1.dict(),
                "point2": request.point2.dict(),
                "distance": float(distance),
                "unit": "cm"
            }
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "거리 계산에 실패했습니다"}
            )
            
    except Exception as e:
        logger.error(f"거리 계산 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"거리 계산 중 오류가 발생했습니다: {str(e)}"}
        )

@app.post("/auto-detect-room")
async def auto_detect_room(file: UploadFile = File(...), confidence_threshold: float = Query(0.7)):
    """AI를 이용한 자동 방 경계 감지"""
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "이미지 파일만 업로드 가능합니다"}
            )

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "이미지를 읽을 수 없습니다"}
            )

        # 임시 이미지 파일 저장
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            cv2.imwrite(temp_file.name, img)
            temp_image_path = temp_file.name
        
        try:
            # 새로운 AI 기반 방 감지 수행 (더 낮은 임계값으로)
            logger.info("🤖 개선된 AI 기반 방 모서리 감지 시작...")
            # 개선된 AI는 더 낮은 임계값을 사용 (원래 요청된 임계값의 70%)
            adjusted_threshold = max(confidence_threshold * 0.7, 0.4)
            result = detect_room_with_ai(temp_image_path, adjusted_threshold)
            
            if result and result.get("success"):
                logger.info(f"✅ AI 감지 완료: {len(result.get('detected_points', []))}개 포인트, 방법: {result.get('method')}")
                return result
            else:
                # AI 감지 실패시 기존 방법으로 폴백
                logger.info("⚠️ AI 감지 실패, 기존 방법으로 폴백...")
                fallback_result = simulate_roomnet_detection(temp_image_path, confidence_threshold)
                
                if fallback_result and fallback_result.get("success"):
                    return fallback_result
                else:
                    return JSONResponse(
                        status_code=422,
                        content={
                            "success": False,
                            "error": "이미지에서 방 경계를 자동으로 감지할 수 없습니다"
                        }
                    )
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            
    except Exception as e:
        logger.error(f"자동 감지 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"자동 감지 중 오류가 발생했습니다: {str(e)}"
            }
        )

@app.post("/estimate-room-size")
async def estimate_room_size(room_points: RoomPoints):
    """방 크기 측정"""
    try:
        logger.info(f"방 크기 측정 요청: {len(room_points.points)}개 포인트")
        logger.info(f"목표 높이: {room_points.target_height}m")
        
        # 개선된 방 측정 알고리즘 사용 (target_height 전달)
        result = improved_room_measurement(room_points.points, room_points.target_height)
        
        if result and result.get('success'):
            logger.info(f"방 크기 측정 완료: {result}")
            return result
        else:
            return JSONResponse(
                status_code=422,
                content={"error": "방 크기를 측정할 수 없습니다. 포인트를 다시 확인해주세요."}
            )
            
    except Exception as e:
        logger.error(f"방 크기 측정 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"방 크기 측정 중 오류가 발생했습니다: {str(e)}"}
        )

@app.post("/detect-windows")
async def detect_windows(file: UploadFile = File(...)):
    """창문 감지"""
    try:
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "이미지 파일만 업로드 가능합니다"}
            )

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "이미지를 읽을 수 없습니다"}
            )

        # 창문 감지 수행
        windows = detect_windows_in_image(img)
        
        logger.info(f"창문 감지 완료: {len(windows)}개 창문")
        
        return {
            "success": True,
            "windows": [window.dict() for window in windows],
            "total_windows": len(windows)
        }
        
    except Exception as e:
        logger.error(f"창문 감지 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"창문 감지 중 오류가 발생했습니다: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3010)