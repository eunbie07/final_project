
from utils.midas_utils import load_midas_model, predict_depth, normalize_depth
from utils.measure_utils import estimate_room_size, draw_floor_plan_with_furniture
from utils.yolo_utils import detect_furniture, get_scale_from_bed

model, transform, device = load_midas_model()

def analyze_room(img_path, ceiling_cm):
    depth_map = predict_depth(model, transform, device, img_path)
    norm = normalize_depth(depth_map)
    boxes = detect_furniture(img_path)
    scale = get_scale_from_bed(boxes)

    if scale:
        bed = max([b for b in boxes if b["label"] == "bed"], key=lambda b: b["w"] * b["h"])
        w_cm = round(bed["w"] * scale)
        h_cm = round(bed["h"] * scale)
        bbox = (bed["x"], bed["y"], bed["w"], bed["h"])
        draw_floor_plan_with_furniture(img_path, bbox, w_cm, h_cm, boxes, "static/floor_plan.png")
        return bbox, w_cm, h_cm, boxes

    w_cm, h_cm, bbox = estimate_room_size(norm, ceiling_cm)
    if bbox:
        draw_floor_plan_with_furniture(img_path, bbox, w_cm, h_cm, boxes, "static/floor_plan.png")
        return bbox, w_cm, h_cm, boxes
    return None, None, None, []
