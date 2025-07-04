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
