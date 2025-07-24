# utils/measure_utils.py
import numpy as np
import cv2

def estimate_room_size(depth_map, ceiling_cm=230):
    h, w = depth_map.shape
    flat = depth_map.flatten()
    threshold = np.percentile(flat, 5)
    mask = (depth_map <= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    largest = max(contours, key=cv2.contourArea)
    x, y, w_pix, h_pix = cv2.boundingRect(largest)
    scale_per_px = ceiling_cm / h
    w_cm = round(w_pix * scale_per_px)
    h_cm = round(h_pix * scale_per_px)
    return w_cm, h_cm, (x, y, w_pix, h_pix)

def draw_floor_plan_with_furniture(img_path, bbox, w_cm, h_cm, objects, output_path):
    x, y, w_pix, h_pix = bbox
    img = cv2.imread(img_path)
    cropped = img[y:y+h_pix, x:x+w_pix].copy()
    scale_x = 300 / w_pix
    scale_y = 300 / h_pix
    resized = cv2.resize(cropped, (300, 300))

    for obj in objects:
        ox, oy, ow, oh = obj["x"], obj["y"], obj["w"], obj["h"]
        if not (x <= ox <= x + w_pix and y <= oy <= y + h_pix): continue
        rx1 = int((ox - x) * scale_x)
        ry1 = int((oy - y) * scale_y)
        rx2 = int(rx1 + ow * scale_x)
        ry2 = int(ry1 + oh * scale_y)
        cv2.rectangle(resized, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.putText(resized, obj["label"], (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(resized, f"{w_cm}cm x {h_cm}cm", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imwrite(output_path, resized)
