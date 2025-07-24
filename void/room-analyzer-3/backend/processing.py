import numpy as np
import cv2

# 가상 기준 길이 (예: 사람이 서 있을 때의 평균 키 1.7m)
# 이 값은 추정된 깊이 맵의 스케일을 실제 미터 단위로 변환하는 데 사용됩니다.
# 사용자가 아는 실제 길이를 입력받는다면 더 정확해집니다.
REFERENCE_OBJECT_HEIGHT_METERS = 1.7

def estimate_room_dimensions(img_cv2, depth_map, selected_points):
    """
    이미지, 깊이 맵, 선택된 점들을 기반으로 방의 크기와 2D 평면도 데이터를 추정합니다.
    선택된 점들은 최소 2개(길이 측정) 또는 4개(평면도 추정)가 필요합니다.
    """
    # 초기 반환 값 설정
    result = {
        "estimated_width_m": "N/A",
        "estimated_height_m": "N/A",
        "floor_plan_vertices": [], # 2D 평면도 꼭짓점 추가
        "notes": ""
    }

    if not selected_points:
        result["notes"] = "No points selected for measurement."
        return result

    # 점의 픽셀 좌표와 해당 깊이 값 추출
    points_with_depth = []
    for p in selected_points:
        x, y = p['x'], p['y']
        # 이미지 범위를 벗어나는 점에 대한 예외 처리
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth = depth_map[y, x]
            points_with_depth.append({'x': x, 'y': y, 'depth': depth})
        else:
            print(f"Warning: Point ({x}, {y}) is out of image bounds.")

    if len(points_with_depth) < 2:
        result["notes"] = "Not enough valid points selected for measurement (need at least 2)."
        return result

    # ------------------ 길이 추정 (기존 로직 보강) ------------------
    # 가장 가까운 두 점 사이의 픽셀 거리 계산
    p1 = points_with_depth[0]
    p2 = points_with_depth[1]
    
    pixel_dist_p1_p2 = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
    avg_depth_p1_p2 = (p1['depth'] + p2['depth']) / 2.0
    
    # MiDaS 깊이 값 스케일링을 위한 임의의 상수 (조정 필요)
    # 이 값은 MiDaS 깊이 1단위가 실제 몇 미터에 해당하는지 대략적으로 연결합니다.
    # 이 상수는 환경 및 MiDaS 모델 출력에 따라 튜닝이 필요합니다.
    pixel_to_meter_scale_factor = 0.003 # 1픽셀당 미터 (깊이에 따라 가변적)

    # 길이 계산 (매우 단순화된 추정)
    # 실제 방 측정을 위해선 더 복잡한 기하학적 계산과 카메라 캘리브레이션이 필요합니다.
    # 여기서는 픽셀 거리에 평균 깊이와 임의의 스케일 팩터를 곱하여 대략적인 미터 길이를 추정합니다.
    estimated_width_m = round(pixel_dist_p1_p2 * pixel_to_meter_scale_factor * avg_depth_p1_p2, 2)
    result["estimated_width_m"] = estimated_width_m
    
    if len(points_with_depth) >= 3:
        p2 = points_with_depth[1]
        p3 = points_with_depth[2]
        pixel_dist_p2_p3 = np.sqrt((p2['x'] - p3['x'])**2 + (p2['y'] - p3['y'])**2)
        avg_depth_p2_p3 = (p2['depth'] + p3['depth']) / 2.0
        estimated_height_m = round(pixel_dist_p2_p3 * pixel_to_meter_scale_factor * avg_depth_p2_p3, 2)
        result["estimated_height_m"] = estimated_height_m
    
    result["notes"] = f"Homography applied using {len(points_with_depth)} points. Meter scale is a 'very rough approximation' based on estimated depth and a heuristic scaling factor. Needs ceiling height and camera parameters for accuracy."

    # ------------------ 2D 평면도 꼭짓점 추정 ------------------
    if len(points_with_depth) >= 4:
        # 사용자가 선택한 4개의 점을 바닥의 직사각형 코너로 가정하고, 2D 평면도 꼭짓점을 생성
        # 이 로직은 `estimated_width_m`와 `estimated_height_m`가 유효할 때만 실행됩니다.
        if isinstance(result["estimated_width_m"], (int, float)) and \
           isinstance(result["estimated_height_m"], (int, float)):
            
            # 2D 평면도의 꼭짓점 (상대 좌표, 미터 단위)
            # 좌상단 (0,0)을 기준으로 직사각형을 그립니다.
            result["floor_plan_vertices"] = [
                {"x": 0, "y": 0},
                {"x": result["estimated_width_m"], "y": 0},
                {"x": result["estimated_width_m"], "y": result["estimated_height_m"]},
                {"x": 0, "y": result["estimated_height_m"]}
            ]
            print(f"DEBUG: Generated floor_plan_vertices: {result['floor_plan_vertices']}")
        else:
            result["floor_plan_vertices"] = []
            print("DEBUG: estimated_width_m or estimated_height_m is N/A, floor_plan_vertices empty.")
    else:
        result["floor_plan_vertices"] = []
        result["notes"] += " Not enough points (need 4) for 2D floor plan estimation."
        print(f"DEBUG: Not enough points ({len(points_with_depth)}), floor_plan_vertices empty.")

    return result