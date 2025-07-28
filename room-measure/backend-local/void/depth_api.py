# room-measure/backend/depth_api.py

import torch
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
import os

router = APIRouter()

# 절대 경로 사용하여 파일 저장 위치 명확화
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_MAP_PATH = os.path.join(BASE_DIR, "depth_map.npy")
DEPTH_IMAGE_PATH = os.path.join(BASE_DIR, "depth_map_output.png")
DEPTH_META_PATH = os.path.join(BASE_DIR, "depth_meta.txt")

@router.post("/depth-map")
async def get_depth_map(file: UploadFile = File(...)):
    try:
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

        print(f"✅ Depth map 생성 완료: {DEPTH_IMAGE_PATH}")
        print(f"✅ 파일 존재 여부: {os.path.exists(DEPTH_IMAGE_PATH)}")

        return JSONResponse(content={
            "depth_image_url": DEPTH_IMAGE_PATH,
            "depth_width": w,
            "depth_height": h
        })

    except Exception as e:
        print(f"❌ Depth map 생성 실패: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/get-depth-at-point")
async def get_depth_at_point(x: int = Query(...), y: int = Query(...)):
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
    print(f"[INFO] 좌표 ({x},{y}) → depth: {depth_value:.3f}")

    if np.isnan(depth_value) or depth_value <= 0:
        return JSONResponse(
            status_code=400,
            content={"error": f"잘못된 깊이 값입니다: {depth_value}"}
        )

    return {"depth": depth_value}

@router.get("/depth-meta")
def get_depth_map_meta():
    if not os.path.exists(DEPTH_META_PATH):
        return JSONResponse(status_code=404, content={"error": "meta 파일 없음"})

    with open(DEPTH_META_PATH, "r") as f:
        w, h = f.read().strip().split(",")
        return {"width": int(w), "height": int(h)}

@router.get("/depth-map-image")
def get_depth_image():
    print(f"🔍 Depth 이미지 요청 - 파일 경로: {DEPTH_IMAGE_PATH}")
    print(f"🔍 파일 존재 여부: {os.path.exists(DEPTH_IMAGE_PATH)}")
    
    if not os.path.exists(DEPTH_IMAGE_PATH):
        return JSONResponse(status_code=404, content={"error": "depth image 없음"})
    
    return FileResponse(DEPTH_IMAGE_PATH, media_type="image/png")