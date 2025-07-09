
from ultralytics import YOLO

REAL_BED_WIDTH_CM = 200
model = YOLO("yolo11n")

def detect_furniture(image_path):
    results = model(image_path)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append({
                "label": label,
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1
            })
    return boxes

def get_scale_from_bed(boxes):
    for box in boxes:
        if box["label"] == "bed":
            return REAL_BED_WIDTH_CM / box["w"]
    return None
