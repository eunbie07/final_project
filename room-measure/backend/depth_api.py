# depth_api.py (일부 발췌)
import torch
import cv2
from fastapi import APIRouter, UploadFile, File
import numpy as np
from fastapi.responses import FileResponse

router = APIRouter()

@router.post("/depth-map")
async def get_depth_map(file: UploadFile = File(...)):
    # MiDaS 모델 불러오기
    model_type = "MiDaS_small"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 파일 저장
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 전처리
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))  # 입력 해상도 제한
    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        prediction = model(img)
        depth = prediction.squeeze().cpu().numpy()

    # 이미지 저장
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    output_path = "depth_map_output.png"
    cv2.imwrite(output_path, depth_colored)

    return FileResponse(output_path, media_type="image/png")
