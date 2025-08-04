# 세 선이 만나는 모서리 점 감지 (바닥 + 두 벽)
# 방의 모서리는 일반적으로 3개의 면(바닥, 벽1, 벽2)이 만나는 지점

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

def detect_three_line_intersections(img, w, h):
    """세 선이 만나는 모서리 점 감지"""
    
    logger.info("🔍 세 선 교차점 감지 시작...")
    
    try:
        # 이미지 전처리
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 에지 감지 강화
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 다중 에지 감지
        edges1 = cv2.Canny(enhanced, 30, 90)
        edges2 = cv2.Canny(enhanced, 50, 150) 
        edges3 = cv2.Canny(enhanced, 80, 200)
        edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # 직선 감지 (더 세밀하게)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=int(min(w, h) * 0.08),  # 더 짧은 선도 감지
            maxLineGap=15
        )
        
        if lines is None or len(lines) < 3:
            logger.warning(f"충분한 직선을 감지하지 못함: {len(lines) if lines is not None else 0}개")
            return []
        
        logger.info(f"📏 {len(lines)}개 직선 감지됨")
        
        # 직선 분류 (수직선, 수평선, 대각선)
        vertical_lines, horizontal_lines, diagonal_lines = classify_lines_detailed(lines)
        
        logger.info(f"분류: 수직 {len(vertical_lines)}개, 수평 {len(horizontal_lines)}개, 대각 {len(diagonal_lines)}개")
        
        # 세 선이 만나는 교차점 찾기
        three_line_intersections = find_three_line_intersections(
            vertical_lines, horizontal_lines, diagonal_lines, w, h
        )
        
        logger.info(f"🎯 세 선 교차점 후보: {len(three_line_intersections)}개")
        
        return three_line_intersections
        
    except Exception as e:
        logger.error(f"세 선 교차점 감지 오류: {str(e)}")
        return []

def classify_lines_detailed(lines):
    """직선을 더 세밀하게 분류"""
    
    vertical_lines = []
    horizontal_lines = []
    diagonal_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 직선 길이 계산
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 너무 짧은 직선 제외
        if length < 20:
            continue
        
        # 각도 계산
        if x2 - x1 == 0:
            angle = 90
        else:
            angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
        
        # 각도로 분류
        if 80 <= angle <= 100:  # 수직선 (80~100도)
            vertical_lines.append((line[0], length, angle))
        elif angle <= 15 or angle >= 165:  # 수평선 (0~15도, 165~180도)
            horizontal_lines.append((line[0], length, angle))
        elif 25 <= angle <= 65:  # 대각선 (원근법에 의한 사선)
            diagonal_lines.append((line[0], length, angle))
    
    return vertical_lines, horizontal_lines, diagonal_lines

def find_three_line_intersections(vertical_lines, horizontal_lines, diagonal_lines, w, h):
    """세 종류의 직선이 만나는 교차점 찾기"""
    
    intersections = []
    
    # 바닥-벽 모서리 찾기: 수직선 + 수평선 + 대각선이 만나는 점
    for v_line, v_length, v_angle in vertical_lines:
        for h_line, h_length, h_angle in horizontal_lines:
            for d_line, d_length, d_angle in diagonal_lines:
                
                # 세 직선의 교차점들 계산
                intersection1 = calculate_line_intersection_safe(v_line, h_line)
                intersection2 = calculate_line_intersection_safe(v_line, d_line)
                intersection3 = calculate_line_intersection_safe(h_line, d_line)
                
                # 세 교차점이 모두 유효하고 서로 가까운지 확인
                if intersection1 and intersection2 and intersection3:
                    center_point = calculate_center_point([intersection1, intersection2, intersection3])
                    
                    if center_point and is_valid_three_line_intersection(
                        center_point, [intersection1, intersection2, intersection3], w, h
                    ):
                        # 모서리 강도 계산
                        strength = calculate_corner_strength(
                            center_point, v_length, h_length, d_length, w, h
                        )
                        
                        intersections.append({
                            "x": int(center_point[0]),
                            "y": int(center_point[1]),
                            "strength": strength,
                            "type": "three_line_intersection",
                            "lines": {
                                "vertical": v_line,
                                "horizontal": h_line,
                                "diagonal": d_line
                            }
                        })
    
    # 수직선 + 수평선이 만나는 주요 교차점도 추가 (2선 교차)
    for v_line, v_length, v_angle in vertical_lines:
        for h_line, h_length, h_angle in horizontal_lines:
            intersection = calculate_line_intersection_safe(v_line, h_line)
            
            if intersection and is_valid_room_corner_intersection(intersection, w, h):
                strength = calculate_corner_strength_2line(intersection, v_length, h_length, w, h)
                
                intersections.append({
                    "x": int(intersection[0]),
                    "y": int(intersection[1]),
                    "strength": strength,
                    "type": "two_line_intersection",
                    "lines": {
                        "vertical": v_line,
                        "horizontal": h_line
                    }
                })
    
    # 강도 순으로 정렬
    intersections.sort(key=lambda x: x["strength"], reverse=True)
    
    # 중복 제거
    unique_intersections = remove_duplicate_intersections(intersections, min(w, h) * 0.05)
    
    return unique_intersections

def calculate_line_intersection_safe(line1, line2):
    """안전한 직선 교차점 계산"""
    
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    if abs(denom) < 1e-10:
        return None  # 평행선
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    
    # 교차점이 두 선분 내부에 있는지 확인 (약간 여유를 둠)
    if -0.1 <= t <= 1.1 and -0.1 <= u <= 1.1:
        intersection_x = x1 + t*(x2-x1)
        intersection_y = y1 + t*(y2-y1)
        return (intersection_x, intersection_y)
    
    return None

def calculate_center_point(points):
    """여러 점의 중심점 계산"""
    
    if not points:
        return None
    
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    
    return (x_sum / len(points), y_sum / len(points))

def is_valid_three_line_intersection(center_point, intersections, w, h):
    """세 선 교차점이 유효한지 확인"""
    
    x, y = center_point
    
    # 이미지 경계 내부인지 확인
    margin = min(w, h) * 0.02
    if not (margin <= x <= w - margin and margin <= y <= h - margin):
        return False
    
    # 세 교차점이 서로 충분히 가까운지 확인
    max_distance = min(w, h) * 0.03  # 3% 이내
    
    for intersection in intersections:
        distance = np.sqrt((center_point[0] - intersection[0])**2 + 
                          (center_point[1] - intersection[1])**2)
        if distance > max_distance:
            return False
    
    return True

def is_valid_room_corner_intersection(point, w, h):
    """방 모서리로 유효한 교차점인지 확인"""
    
    x, y = point
    
    # 경계 확인
    margin = min(w, h) * 0.05
    if not (margin <= x <= w - margin and margin <= y <= h - margin):
        return False
    
    # 방 모서리는 일반적으로 가장자리나 특정 위치에 있음
    edge_threshold = min(w, h) * 0.2
    
    # 가장자리 근처이거나 바닥 근처에 있어야 함
    is_near_edge = (x < edge_threshold or x > w - edge_threshold or 
                   y < edge_threshold or y > h - edge_threshold)
    is_near_floor = y > h * 0.6  # 바닥 근처
    
    return is_near_edge or is_near_floor

def calculate_corner_strength(point, v_length, h_length, d_length, w, h):
    """세 선 모서리의 강도 계산"""
    
    x, y = point
    
    # 1. 직선 길이 기반 강도 (긴 직선일수록 신뢰도 높음)
    length_score = min((v_length + h_length + d_length) / (min(w, h) * 2), 1.0)
    
    # 2. 위치 기반 강도 (방 모서리 위치일수록 높음)
    position_score = 0.0
    
    # 바닥 모서리 (방의 주요 모서리)
    if y > h * 0.7:  # 하단 30%
        if x < w * 0.3 or x > w * 0.7:  # 좌우 모서리
            position_score = 1.0
        else:  # 중앙 바닥
            position_score = 0.8
    # 중간 높이
    elif h * 0.3 < y < h * 0.7:
        if x < w * 0.2 or x > w * 0.8:  # 벽 모서리
            position_score = 0.6
        else:
            position_score = 0.3
    # 상단
    else:
        if x < w * 0.3 or x > w * 0.7:  # 천장 모서리
            position_score = 0.7
        else:
            position_score = 0.4
    
    # 3. 각도 다양성 (세 방향이 다양할수록 좋음)
    angle_diversity_score = 0.8  # 세 선이므로 기본적으로 높은 점수
    
    # 최종 강도
    strength = length_score * 0.4 + position_score * 0.4 + angle_diversity_score * 0.2
    
    return strength

def calculate_corner_strength_2line(point, v_length, h_length, w, h):
    """두 선 모서리의 강도 계산"""
    
    x, y = point
    
    # 직선 길이 기반
    length_score = min((v_length + h_length) / (min(w, h) * 1.5), 1.0)
    
    # 위치 기반 (2선 교차는 약간 낮은 점수)
    position_score = 0.0
    if y > h * 0.7 and (x < w * 0.3 or x > w * 0.7):
        position_score = 0.8
    elif y > h * 0.6:
        position_score = 0.6
    elif x < w * 0.2 or x > w * 0.8:
        position_score = 0.5
    else:
        position_score = 0.3
    
    strength = length_score * 0.5 + position_score * 0.5
    
    return strength * 0.8  # 2선이므로 3선보다 낮은 가중치

def remove_duplicate_intersections(intersections, min_distance):
    """중복 교차점 제거"""
    
    unique_intersections = []
    
    for intersection in intersections:
        is_duplicate = False
        
        for existing in unique_intersections:
            distance = np.sqrt((intersection["x"] - existing["x"])**2 + 
                             (intersection["y"] - existing["y"])**2)
            if distance < min_distance:
                # 더 강한 교차점으로 교체
                if intersection["strength"] > existing["strength"]:
                    unique_intersections.remove(existing)
                    unique_intersections.append(intersection)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_intersections.append(intersection)
    
    return unique_intersections

def find_best_floor_corner_intersection(intersections, w, h):
    """바닥 모서리로 가장 적합한 교차점 찾기 (1번 포인트용)"""
    
    if not intersections:
        return None
    
    logger.info(f"🔍 1번 포인트 찾기: 총 {len(intersections)}개 교차점 분석")
    
    # 중앙 영역 정의 (좀 더 넓게)
    center_x = w * 0.5
    center_zone_width = w * 0.3  # 중앙 30% 영역
    center_left = center_x - center_zone_width
    center_right = center_x + center_zone_width
    
    # 바닥 영역 정의
    floor_zone = h * 0.7  # 하단 30%
    
    # 1단계: 중앙 바닥 영역의 교차점들만 필터링
    center_floor_candidates = [
        intersection for intersection in intersections
        if (center_left <= intersection["x"] <= center_right and  # 중앙 영역
            intersection["y"] >= floor_zone)  # 바닥 영역
    ]
    
    logger.info(f"📍 중앙 바닥 영역 후보: {len(center_floor_candidates)}개")
    
    if center_floor_candidates:
        # 중앙 바닥 영역에서 가장 좋은 후보 선택
        best_candidate = None
        best_score = 0
        
        for candidate in center_floor_candidates:
            # 점수 계산: 중앙 근접도 + 바닥 근접도 + 강도
            center_distance = abs(candidate["x"] - center_x) / (w * 0.5)
            center_score = (1 - center_distance) * 0.5  # 중앙 근접도
            
            bottom_score = (candidate["y"] - floor_zone) / (h - floor_zone) * 0.3  # 바닥 근접도
            
            strength_score = candidate["strength"] * 0.2  # 원래 강도
            
            total_score = center_score + bottom_score + strength_score
            
            if candidate["type"] == "three_line_intersection":
                total_score += 0.1  # 3선 교차 보너스
            
            logger.info(f"  후보 ({candidate['x']}, {candidate['y']}): 점수 {total_score:.3f}")
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate
        
        if best_candidate:
            logger.info(f"✅ 중앙 바닥 최적 후보 선택: ({best_candidate['x']}, {best_candidate['y']}) 점수: {best_score:.3f}")
            return best_candidate
    
    # 2단계: 중앙 바닥 영역에 후보가 없으면 범위 확장
    logger.info("⚠️ 중앙 바닥 후보 없음, 범위 확장...")
    
    # 바닥 영역에서 중앙에 가장 가까운 점
    floor_candidates = [
        intersection for intersection in intersections
        if intersection["y"] >= h * 0.6  # 하단 40%로 확장
    ]
    
    if floor_candidates:
        best_floor = min(floor_candidates, key=lambda c: abs(c["x"] - center_x))
        logger.info(f"📍 확장된 바닥 영역 최적 후보: ({best_floor['x']}, {best_floor['y']})")
        return best_floor
    
    # 3단계: 최종 폴백 - 전체에서 중앙+아래쪽 우선
    logger.info("⚠️ 바닥 후보 없음, 전체 영역에서 선택...")
    
    best_fallback = min(intersections, 
                       key=lambda c: abs(c["x"] - center_x) * 0.7 + (h - c["y"]) * 0.3)
    
    logger.info(f"📍 폴백 후보: ({best_fallback['x']}, {best_fallback['y']})")
    return best_fallback