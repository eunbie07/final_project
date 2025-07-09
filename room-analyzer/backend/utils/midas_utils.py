
import torch
import cv2
from torchvision.transforms import Compose

def load_midas_model():
    model_type = "DPT_Large"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, transform, device

def predict_depth(model, transform, device, img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    return prediction.cpu().numpy()

def normalize_depth(depth_map):
    return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
