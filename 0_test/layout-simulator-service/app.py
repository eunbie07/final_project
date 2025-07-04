from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REAL_BED_WIDTH_CM = 200  # 실제 침대 너비 기준

# YOLO 모델 로드
model = YOLO('yolov8n.pt')

def detect_objects(image_path):
    results = model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                'label': label,
                'bbox': [x1, y1, x2, y2]
            })
    return detections

def normalize_boxes(detections, image_width, image_height):
    normalized = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        normalized.append({
            'label': d['label'],
            'x': x1 / image_width,
            'y': y1 / image_height,
            'w': (x2 - x1) / image_width,
            'h': (y2 - y1) / image_height
        })
    return normalized

def check_parent_friendly(normalized_boxes):
    warnings = []
    for item in normalized_boxes:
        if item['w'] < 0.2 and item['label'] in ['table', 'sofa']:
            warnings.append(f"{item['label']} 주변 간격이 좁아 이동이 불편할 수 있어요.")
    return warnings

@app.post("/layout")
async def analyze_layout(file: UploadFile = File(...)):
    image = Image.open(file.file)
    width, height = image.size
    image_path = f"tmp_{file.filename}"
    image.save(image_path)

    detections = detect_objects(image_path)
    normalized = normalize_boxes(detections, width, height)
    warnings = check_parent_friendly(normalized)

    bed = next((obj for obj in normalized if obj['label'] == 'bed'), None)
    if not bed:
        return {
            "layout": normalized,
            "layout_cm": [],
            "parent_warnings": warnings,
            "error": "침대를 찾을 수 없어 실제 크기 환산 불가"
        }

    bed_width_ratio = bed['w']
    cm_per_ratio = REAL_BED_WIDTH_CM / bed_width_ratio

    layout_cm = []
    for obj in normalized:
        layout_cm.append({
            "label": obj['label'],
            "x": obj['x'] * cm_per_ratio,
            "y": obj['y'] * cm_per_ratio,
            "w": obj['w'] * cm_per_ratio,
            "h": obj['h'] * cm_per_ratio
        })

    return {
        "layout": normalized,
        "layout_cm": layout_cm,
        "parent_warnings": warnings,
        "image_width": width,
        "image_height": height
    }
