# room-measure/backend/depth_processing.py

import torch
import cv2
import numpy as np
import os
import logging
from math import sqrt
from fastapi import UploadFile
from models import DepthDistanceRequest

logger = logging.getLogger(__name__)

# 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_MAP_PATH = os.path.join(BASE_DIR, "depth_map.npy")
DEPTH_IMAGE_PATH = os.path.join(BASE_DIR, "depth_map_output.png")
DEPTH_META_PATH = os.path.join(BASE_DIR, "depth_meta.txt")

async def generate_depth_map(file) -> dict:
    """이미지로부터 깊이 맵 생성"""
    try:
        logger.info("Depth map 생성 시작...")
        
        # 모델 로딩
        model_type = "MiDaS_small"
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # 이미지 디코딩 (파일 경로 또는 UploadFile 처리)
        if isinstance(file, str):
            # 파일 경로인 경우
            img = cv2.imread(file, cv2.IMREAD_COLOR)
        else:
            # UploadFile인 경우
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

        logger.info(f"Depth map 생성 완료!")
        logger.info(f"  - 크기: {w} x {h}")

        return {
            "success": True,
            "depth_image_url": DEPTH_IMAGE_PATH,
            "depth_width": w,
            "depth_height": h,
            "message": "depth map 생성 완료"
        }

    except Exception as e:
        logger.error(f"Depth map 생성 실패: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def get_depth_image_path() -> str:
    """깊이 이미지 파일 경로 반환"""
    return DEPTH_IMAGE_PATH if os.path.exists(DEPTH_IMAGE_PATH) else None

def get_depth_at_point(x: int, y: int) -> dict:
    """특정 좌표의 깊이 값 조회"""
    try:
        logger.info(f"깊이 값 요청: ({x}, {y})")
        
        if not os.path.exists(DEPTH_MAP_PATH):
            return {
                "success": False,
                "error": "Depth map not found. 먼저 이미지를 업로드하고 depth-map을 생성해주세요."
            }

        depth_map = np.load(DEPTH_MAP_PATH)
        h, w = depth_map.shape
        
        logger.info(f"Depth map 정보: {w} × {h}")

        # 좌표 범위 체크 및 안전한 클램핑
        if x < 0 or y < 0 or x >= w or y >= h:
            logger.warning(f"좌표 범위 초과: ({x}, {y}), 허용 범위: 0~{w-1}, 0~{h-1}")
            return {
                "success": False,
                "error": f"좌표가 범위를 벗어났습니다.",
                "requested": {"x": x, "y": y},
                "valid_range": {"x": [0, w-1], "y": [0, h-1]},
                "depth_map_size": {"width": w, "height": h}
            }

        # 안전한 좌표로 클램핑 (혹시 모를 경우를 대비)
        safe_x = max(0, min(x, w-1))
        safe_y = max(0, min(y, h-1))
        
        if safe_x != x or safe_y != y:
            logger.info(f"🔧 좌표 보정: ({x}, {y}) → ({safe_x}, {safe_y})")

        depth_value = float(depth_map[safe_y, safe_x])
        logger.info(f"  깊이 값: {depth_value:.6f}")

        if np.isnan(depth_value) or np.isinf(depth_value):
            logger.warning(f"유효하지 않은 깊이 값: {depth_value}")
            return {
                "success": False,
                "error": "해당 위치의 깊이 정보가 유효하지 않습니다.",
                "depth_value": str(depth_value),
                "suggestion": "다른 지점을 클릭해보세요."
            }

        return {
            "success": True,
            "depth": depth_value,
            "coordinates": {"x": safe_x, "y": safe_y},
            "original_request": {"x": x, "y": y},
            "depth_map_size": {"width": w, "height": h}
        }
        
    except Exception as e:
        logger.error(f"깊이 값 조회 실패: {str(e)}")
        return {
            "success": False,
            "error": "서버 내부 오류가 발생했습니다.",
            "details": str(e)
        }

def get_depth_map_meta() -> dict:
    """깊이 맵 메타데이터 조회"""
    try:
        if not os.path.exists(DEPTH_META_PATH):
            return {
                "success": False,
                "error": "meta 파일 없음"
            }

        with open(DEPTH_META_PATH, "r") as f:
            w, h = f.read().strip().split(",")
            return {
                "success": True,
                "width": int(w), 
                "height": int(h)
            }
    except Exception as e:
        logger.error(f"Meta 정보 조회 실패: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def compute_3d_distance(req: DepthDistanceRequest) -> dict:
    """두 점 사이의 3D 거리 계산"""
    try:
        if not os.path.exists(DEPTH_MAP_PATH):
            return {
                "success": False,
                "error": "Depth map not found"
            }

        depth_map = np.load(DEPTH_MAP_PATH)
        h, w = depth_map.shape
        x1, y1 = req.point1.x, req.point1.y
        x2, y2 = req.point2.x, req.point2.y

        for x, y in [(x1, y1), (x2, y2)]:
            if not (0 <= x < w and 0 <= y < h):
                return {
                    "success": False,
                    "error": f"Point ({x},{y}) out of bounds"
                }

        d1 = float(depth_map[y1, x1])
        d2 = float(depth_map[y2, x2])

        if np.isnan(d1) or np.isnan(d2):
            return {
                "success": False,
                "error": "Invalid depth value"
            }

        dist_pixel = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (d2 - d1)**2)
        distance_cm = dist_pixel * 1

        return {
            "success": True,
            "3d_distance_cm": round(distance_cm, 2),
            "depth1": round(d1, 3),
            "depth2": round(d2, 3),
            "pixel_distance": round(dist_pixel, 3)
        }
    except Exception as e:
        logger.error(f"거리 계산 실패: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def check_depth_files_exist() -> dict:
    """깊이 관련 파일들의 존재 여부 확인"""
    return {
        "depth_map": os.path.exists(DEPTH_MAP_PATH),
        "depth_image": os.path.exists(DEPTH_IMAGE_PATH),
        "depth_meta": os.path.exists(DEPTH_META_PATH)
    }