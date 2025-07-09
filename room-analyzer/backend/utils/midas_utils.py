import torch
import cv2
from PIL import Image
import numpy as np

def load_midas_model():
    model_type = "MiDaS_small"  # 또는 "DPT_Hybrid"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transforms.dpt_transform
        expects_pil = True
    else:
        transform = transforms.small_transform
        expects_pil = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, transform, device, expects_pil

def predict_depth(model, transform, device, img_path, expects_pil=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if expects_pil:
        img_input = Image.fromarray(img)
    else:
        img_input = img  # numpy 형태 그대로 사용

    # 수정된 transform 적용
    input_tensor = transform(img_input).to(device)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)  # ← ✅ 들여쓰기 되어야 함

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output

def normalize_depth(depth_map):
    return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
