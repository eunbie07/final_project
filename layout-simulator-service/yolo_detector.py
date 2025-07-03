from ultralytics import YOLO

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
