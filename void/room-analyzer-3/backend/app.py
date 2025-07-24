# backend/app.py 파일 수정

import cv2
import numpy as np
from PIL import Image
import io
import torch
import json
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .processing import estimate_room_dimensions

# 이 변수들은 전역으로 선언하고, load_midas_model에서 초기화됩니다.
midas = None
transform = None
device = None

def load_midas_model():
    global midas, transform, device
    if midas is None:
        model_type = "MiDaS_small" # 사용할 MiDaS 모델 타입 (DPT_Hybrid, DPT_Large도 가능)
        try:
            device = torch.device("cpu") # GPU를 사용하려면 "cuda"로 변경
            midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            midas.to(device)
            midas.eval() # 모델을 평가 모드로 설정

            # --- 핵심 수정 부분 시작 ---
            # MiDaS 모델의 특정 변환을 가져오기 위한 허브 로드
            # 그러나 이 transform 자체를 직접 사용하지 않고, 필요한 부분을 조합합니다.
            midas_transforms_module = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

            # torchvision.transforms를 임포트
            from torchvision import transforms

            # MiDaS 모델이 요구하는 표준화 값
            # 이 값들은 MiDaS 모델의 공식 허브 코드에서 가져옵니다.
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            # PIL Image를 Torch Tensor로 변환하고, 정규화하는 custom transform을 정의합니다.
            # MiDaS 모델은 0-1 범위의 float 텐서를 기대합니다.
            if model_type == "MiDaS_small":
                transform = transforms.Compose([
                    transforms.Resize(256), # MiDaS_small에 맞는 크기로 리사이즈 (필요시)
                    transforms.CenterCrop(256), # MiDaS_small에 맞는 크기로 크롭 (필요시)
                    transforms.ToTensor(), # PIL Image를 [0, 1] 범위의 FloatTensor로 변환
                    transforms.Normalize(mean=mean, std=std) # MiDaS에 맞는 정규화
                ])
            else: # DPT_Hybrid, DPT_Large 등 다른 모델의 경우
                transform = transforms.Compose([
                    transforms.Resize(384), # 다른 MiDaS 모델에 맞는 크기로 리사이즈
                    transforms.CenterCrop(384), # 다른 MiDaS 모델에 맞는 크기로 크롭
                    transforms.ToTensor(), # PIL Image를 [0, 1] 범위의 FloatTensor로 변환
                    transforms.Normalize(mean=mean, std=std) # MiDaS에 맞는 정규화
                ])
            # --- 핵심 수정 부분 끝 ---

            print(f"MiDaS model loaded successfully on {device}!")
        except Exception as e:
            print(f"Failed to load MiDaS model: {e}")
            traceback.print_exc() # 오류 스택 트레이스 출력
            midas = None

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    load_midas_model()

# /initial-image-analysis 엔드포인트는 이전과 동일합니다.
@app.post("/initial-image-analysis")
async def initial_image_analysis(file: UploadFile = File(...)):
    if midas is None or transform is None: # transform도 확인
        raise HTTPException(status_code=500, detail="MiDaS model or transform not loaded. Please check backend logs.")

    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 여기서 정의된 transform을 사용합니다.
        input_batch = transform(image_pil).unsqueeze(0).to(device) # unsqueeze(0)로 배치 차원 추가

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_pil.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()

        return JSONResponse({
            "message": "Image analyzed. Ready for point selection.",
            "image_width_px": image_pil.width,
            "image_height_px": image_pil.height,
            "min_depth_value": float(depth_map.min()),
            "max_depth_value": float(depth_map.max()),
            "estimated_width_m": "N/A",
            "estimated_height_m": "N/A",
            "notes": "Upload successful. Select points on the image for measurement."
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Initial image analysis failed: {str(e)}")


# /measure-room-with-points 엔드포인트도 이전과 동일합니다.
@app.post("/measure-room-with-points")
async def measure_room_with_points(
    file: UploadFile = File(...),
    points: str = Form(...)
):
    if midas is None or transform is None: # transform도 확인
        raise HTTPException(status_code=500, detail="MiDaS model or transform not loaded. Please check backend logs.")

    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv2 = np.array(image_pil)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

        # 여기서 정의된 transform을 사용합니다.
        input_batch = transform(image_pil).unsqueeze(0).to(device) # unsqueeze(0)로 배치 차원 추가
        
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_pil.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()

        selected_points_list = json.loads(points)

        measurement_results = estimate_room_dimensions(img_cv2, depth_map, selected_points_list)

        return JSONResponse({
            "message": "Image and points processed successfully for measurement.",
            "image_width_px": img_cv2.shape[1],
            "image_height_px": img_cv2.shape[0],
            "estimated_width_m": measurement_results['estimated_width_m'],
            "estimated_height_m": measurement_results['estimated_height_m'],
            "notes": measurement_results['notes']
        })

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for points.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Measurement failed: {str(e)}")