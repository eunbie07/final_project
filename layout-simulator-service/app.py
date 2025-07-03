from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from yolo_detector import detect_objects
from layout_generator import normalize_boxes
from parent_filter import check_parent_friendly
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REAL_BED_WIDTH_CM = 200  # 실제 침대 너비 기준

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
