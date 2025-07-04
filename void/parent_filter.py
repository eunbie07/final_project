def check_parent_friendly(normalized_boxes):
    warnings = []
    for item in normalized_boxes:
        if item['w'] < 0.2 and item['label'] in ['table', 'sofa']:
            warnings.append(f"{item['label']} 주변 간격이 좁아 이동이 불편할 수 있어요.")
    return warnings
