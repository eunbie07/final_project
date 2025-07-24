from utils.midas_utils import load_midas_model, predict_depth, normalize_depth
from utils.measure_utils import estimate_room_size, draw_floor_plan_with_furniture
from utils.yolo_utils import detect_furniture, get_scale_from_bed

# 전역 모델 로드 (한 번만 수행)
model, transform, device, expects_pil = load_midas_model()

def analyze_room(img_path, ceiling_cm):
    # 1. 깊이 추정
    depth_map = predict_depth(model, transform, device, img_path, expects_pil)
    norm = normalize_depth(depth_map)

    # 2. 가구 감지 (YOLO)
    boxes = detect_furniture(img_path)

    # 3. 침대 기준 스케일 추정
    scale = get_scale_from_bed(boxes)

    if scale:
        # 침대 정보 추출
        beds = [b for b in boxes if b["label"] == "bed"]
        if beds:
            bed = max(beds, key=lambda b: b["w"] * b["h"])
            w_cm = round(bed["w"] * scale)
            h_cm = round(bed["h"] * scale)
            bbox = (bed["x"], bed["y"], bed["w"], bed["h"])
            draw_floor_plan_with_furniture(img_path, bbox, w_cm, h_cm, boxes, "static/floor_plan.png")
            return bbox, w_cm, h_cm, boxes

    # 4. 침대 없음 → 깊이 기반 추정
    w_cm, h_cm, bbox = estimate_room_size(norm, ceiling_cm)
    if bbox:
        draw_floor_plan_with_furniture(img_path, bbox, w_cm, h_cm, boxes, "static/floor_plan.png")
        return bbox, w_cm, h_cm, boxes

    # 실패
    return None, None, None, []
