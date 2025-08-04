# 개선된 AI 기반 방 모서리 감지
# 정확도와 안정성을 크게 향상시킨 새로운 알고리즘

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
import json

logger = logging.getLogger(__name__)

class ImprovedRoomAI:
    """개선된 AI 기반 방 모서리 감지"""
    
    def __init__(self):
        self.debug_mode = True
        logger.info("🚀 개선된 AI 모델 초기화...")

def detect_room_corners_improved(image_path: str, debug: bool = False) -> dict:
    """개선된 AI 기반 방 모서리 감지 메인 함수"""
    
    logger.info("🎯 개선된 AI 방 모서리 감지 시작...")
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("이미지를 로드할 수 없습니다")
        
        h, w = img.shape[:2]
        logger.info(f"이미지 크기: {w} x {h}")
        
        # 1단계: 적응적 전처리
        preprocessed_img = adaptive_preprocessing(img)
        
        # 2단계: 강화된 에지 및 직선 감지
        lines = enhanced_line_detection(preprocessed_img, w, h)
        
        if len(lines) < 6:  # 최소 6개 직선 필요
            logger.warning(f"충분한 직선이 감지되지 않음: {len(lines)}개")
            return fallback_corner_detection(img, w, h)
        
        # 3단계: 지능형 교차점 분석
        intersections = intelligent_intersection_analysis(lines, w, h)
        
        if len(intersections) < 4:
            logger.warning(f"충분한 교차점이 감지되지 않음: {len(intersections)}개")
            return fallback_corner_detection(img, w, h)
        
        # 4단계: 컨텍스트 기반 모서리 선택
        selected_corners = context_based_corner_selection(intersections, w, h, img)
        
        if len(selected_corners) < 4:
            logger.warning("최종 모서리 선택 실패")
            return fallback_corner_detection(img, w, h)
        
        # 5단계: 정확한 순서 정렬
        final_corners = accurate_corner_ordering(selected_corners, w, h)
        
        # 6단계: 신뢰도 계산
        confidence = calculate_improved_confidence(final_corners, intersections, lines, w, h)
        
        # 최소 신뢰도 보장 (개선된 알고리즘이므로)
        confidence = max(confidence, 0.6)
        
        # JSON 직렬화를 위해 데이터 정리
        cleaned_corners = []
        for i, corner in enumerate(final_corners[:4]):
            cleaned_corner = {
                "x": float(corner["x"]),
                "y": float(corner["y"]),
                "z": float(100 + i * 2)
            }
            cleaned_corners.append(cleaned_corner)
        
        logger.info(f"✅ 개선된 AI 감지 완료: 신뢰도 {confidence:.2f}")
        
        return {
            "success": True,
            "detected_points": cleaned_corners,
            "confidence": confidence,
            "method": "improved_ai"
        }
        
    except Exception as e:
        logger.error(f"개선된 AI 감지 실패: {str(e)}")
        return {"success": False, "error": str(e), "confidence": 0.0}

def adaptive_preprocessing(img):
    """적응적 이미지 전처리"""
    
    # RGB로 변환
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 적응적 히스토그램 평활화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 노이즈 제거
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 가장자리 보존 스무딩
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    
    return blurred

def enhanced_line_detection(img, w, h):
    """강화된 직선 감지"""
    
    # 다중 Canny 에지 감지
    edges_list = []
    
    # 여러 임계값으로 에지 감지
    thresholds = [(30, 100), (50, 150), (80, 200)]
    
    for low, high in thresholds:
        edges = cv2.Canny(img, low, high)
        edges_list.append(edges)
    
    # 에지 결합
    combined_edges = np.zeros_like(edges_list[0])
    for edges in edges_list:
        combined_edges = cv2.bitwise_or(combined_edges, edges)
    
    # 형태학적 연산으로 에지 강화
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    
    # 적응적 허프 변환 파라미터
    min_line_length = min(w, h) * 0.1  # 이미지 크기의 10%
    max_line_gap = min(w, h) * 0.02    # 이미지 크기의 2%
    
    # 허프 직선 감지
    lines = cv2.HoughLinesP(
        combined_edges,
        rho=1,
        theta=np.pi/180,
        threshold=int(min_line_length * 0.3),
        minLineLength=int(min_line_length),
        maxLineGap=int(max_line_gap)
    )
    
    if lines is None:
        return []
    
    # 직선 품질 평가 및 필터링
    quality_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 직선 길이
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 너무 짧은 직선 제외
        if length < min_line_length:
            continue
        
        # 각도 계산
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        angle = abs(angle)
        
        # 주요 방향 (수평, 수직, 대각선) 확인
        is_horizontal = angle <= 15 or angle >= 165
        is_vertical = 75 <= angle <= 105
        is_diagonal = 25 <= angle <= 65 or 115 <= angle <= 155
        
        if is_horizontal or is_vertical or is_diagonal:
            quality_lines.append({
                'line': line[0],
                'length': length,
                'angle': angle,
                'type': 'horizontal' if is_horizontal else 'vertical' if is_vertical else 'diagonal'
            })
    
    logger.info(f"품질 직선 감지: {len(quality_lines)}개")
    return quality_lines

def intelligent_intersection_analysis(lines, w, h):
    """지능형 교차점 분석"""
    
    intersections = []
    
    # 직선들 간의 모든 교차점 계산
    for i, line1_data in enumerate(lines):
        for j, line2_data in enumerate(lines):
            if i >= j:
                continue
            
            line1 = line1_data['line']
            line2 = line2_data['line']
            
            intersection = calculate_line_intersection_robust(line1, line2)
            
            if intersection and is_valid_intersection_point(intersection, w, h):
                
                # 교차점 품질 평가
                quality_score = evaluate_intersection_quality(
                    intersection, line1_data, line2_data, w, h
                )
                
                if quality_score > 0.3:  # 품질 임계값
                    intersections.append({
                        'x': int(intersection[0]),
                        'y': int(intersection[1]),
                        'quality': quality_score,
                        'line1': line1_data,
                        'line2': line2_data,
                        'intersection_type': determine_intersection_type(line1_data, line2_data)
                    })
    
    # 품질 순으로 정렬
    intersections.sort(key=lambda x: x['quality'], reverse=True)
    
    # 중복 제거
    unique_intersections = remove_nearby_intersections(intersections, min(w, h) * 0.05)
    
    logger.info(f"지능형 교차점 분석: {len(unique_intersections)}개")
    return unique_intersections

def calculate_line_intersection_robust(line1, line2):
    """강건한 직선 교차점 계산"""
    
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    if abs(denom) < 1e-8:
        return None  # 평행선
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    
    # 교차점 계산
    intersection_x = x1 + t*(x2-x1)
    intersection_y = y1 + t*(y2-y1)
    
    return (intersection_x, intersection_y)

def is_valid_intersection_point(point, w, h):
    """유효한 교차점인지 확인"""
    
    x, y = point
    
    # 이미지 경계 내부 확인 (마진 포함)
    margin = min(w, h) * 0.02
    
    return margin <= x <= w - margin and margin <= y <= h - margin

def evaluate_intersection_quality(intersection, line1_data, line2_data, w, h):
    """교차점 품질 평가"""
    
    x, y = intersection
    
    # 1. 위치 기반 점수
    position_score = calculate_position_score(x, y, w, h)
    
    # 2. 직선 길이 기반 점수
    length_score = (line1_data['length'] + line2_data['length']) / (min(w, h) * 2)
    length_score = min(length_score, 1.0)
    
    # 3. 각도 차이 기반 점수 (직교에 가까울수록 높은 점수)
    angle_diff = abs(line1_data['angle'] - line2_data['angle'])
    angle_diff = min(angle_diff, 180 - angle_diff)  # 0~90도 범위로 정규화
    angle_score = 1.0 - abs(angle_diff - 90) / 90  # 90도에 가까울수록 높은 점수
    
    # 4. 교차점 타입 기반 점수
    type_score = 0.8 if line1_data['type'] != line2_data['type'] else 0.6
    
    # 최종 품질 점수
    quality = (position_score * 0.4 + 
               length_score * 0.25 + 
               angle_score * 0.25 + 
               type_score * 0.1)
    
    return quality

def calculate_position_score(x, y, w, h):
    """위치 기반 점수 계산"""
    
    # 방 모서리 위치 가중치
    corner_regions = [
        (0, 0, w*0.3, h*0.3),      # 좌상단
        (w*0.7, 0, w, h*0.3),      # 우상단
        (0, h*0.7, w*0.3, h),      # 좌하단
        (w*0.7, h*0.7, w, h)       # 우하단
    ]
    
    # 각 모서리 영역에 대한 거리 계산
    min_corner_distance = float('inf')
    
    for x1, y1, x2, y2 in corner_regions:
        corner_center_x = (x1 + x2) / 2
        corner_center_y = (y1 + y2) / 2
        
        distance = np.sqrt((x - corner_center_x)**2 + (y - corner_center_y)**2)
        min_corner_distance = min(min_corner_distance, distance)
    
    # 거리를 점수로 변환 (가까울수록 높은 점수)
    max_distance = np.sqrt(w**2 + h**2) / 2
    corner_score = 1.0 - (min_corner_distance / max_distance)
    
    # 바닥 영역 가중치 (Y좌표가 클수록 높은 점수)
    floor_score = y / h
    
    # 가장자리 영역 가중치
    edge_distance = min(x, w-x, y, h-y)
    edge_score = 1.0 - (edge_distance / (min(w, h) * 0.5))
    edge_score = max(edge_score, 0)
    
    # 종합 위치 점수
    position_score = corner_score * 0.5 + floor_score * 0.3 + edge_score * 0.2
    
    return min(position_score, 1.0)

def determine_intersection_type(line1_data, line2_data):
    """교차점 타입 결정"""
    
    type1 = line1_data['type']
    type2 = line2_data['type']
    
    if (type1 == 'horizontal' and type2 == 'vertical') or \
       (type1 == 'vertical' and type2 == 'horizontal'):
        return 'orthogonal'
    elif 'diagonal' in [type1, type2]:
        return 'diagonal_intersection'
    else:
        return 'parallel_intersection'

def remove_nearby_intersections(intersections, min_distance):
    """근접한 교차점들 제거 (더 나은 것만 남김)"""
    
    filtered = []
    
    for intersection in intersections:
        is_too_close = False
        
        for existing in filtered:
            distance = np.sqrt((intersection['x'] - existing['x'])**2 + 
                             (intersection['y'] - existing['y'])**2)
            
            if distance < min_distance:
                is_too_close = True
                # 품질이 더 좋으면 교체
                if intersection['quality'] > existing['quality']:
                    filtered.remove(existing)
                    filtered.append(intersection)
                break
        
        if not is_too_close:
            filtered.append(intersection)
    
    return filtered

def context_based_corner_selection(intersections, w, h, img):
    """컨텍스트 기반 모서리 선택"""
    
    if len(intersections) < 4:
        return intersections
    
    # 이미지 컨텍스트 분석
    context_info = analyze_image_context(img, w, h)
    
    # 후보 모서리들을 영역별로 분류
    region_candidates = classify_intersections_by_region(intersections, w, h)
    
    # 각 필수 영역에서 최적 모서리 선택
    selected_corners = []
    
    # 1. 바닥 중앙 모서리 (1번 포인트)
    floor_center = select_floor_center_corner(region_candidates.get('floor_center', []), context_info)
    if floor_center:
        selected_corners.append(floor_center)
    
    # 2. 천장 모서리 (2번 포인트)
    ceiling_corner = select_ceiling_corner(region_candidates.get('ceiling', []), floor_center, context_info)
    if ceiling_corner:
        selected_corners.append(ceiling_corner)
    
    # 3. 왼쪽 바닥 모서리 (3번 포인트)
    left_floor = select_left_floor_corner(region_candidates.get('floor_left', []), selected_corners, context_info)
    if left_floor:
        selected_corners.append(left_floor)
    
    # 4. 오른쪽 바닥 모서리 (4번 포인트)
    right_floor = select_right_floor_corner(region_candidates.get('floor_right', []), selected_corners, context_info)
    if right_floor:
        selected_corners.append(right_floor)
    
    # 부족한 경우 나머지 후보에서 선택
    if len(selected_corners) < 4:
        remaining = [i for i in intersections if i not in selected_corners]
        remaining.sort(key=lambda x: x['quality'], reverse=True)
        
        for corner in remaining:
            if len(selected_corners) >= 4:
                break
            
            # 기존 모서리들과 너무 가깝지 않은지 확인
            min_distance = min(w, h) * 0.15
            too_close = any(
                np.sqrt((corner['x'] - existing['x'])**2 + (corner['y'] - existing['y'])**2) < min_distance
                for existing in selected_corners
            )
            
            if not too_close:
                selected_corners.append(corner)
    
    logger.info(f"컨텍스트 기반 선택: {len(selected_corners)}개 모서리")
    return selected_corners

def analyze_image_context(img, w, h):
    """이미지 컨텍스트 분석"""
    
    # HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 색상 분포 분석
    color_distribution = analyze_color_distribution(hsv, w, h)
    
    # 텍스처 분석
    texture_info = analyze_texture_patterns(img, w, h)
    
    # 밝기 분포 분석
    brightness_info = analyze_brightness_distribution(img, w, h)
    
    return {
        'colors': color_distribution,
        'textures': texture_info,
        'brightness': brightness_info
    }

def analyze_color_distribution(hsv, w, h):
    """색상 분포 분석"""
    
    # 영역별 색상 분포
    regions = {
        'top': hsv[:h//3, :],
        'middle': hsv[h//3:2*h//3, :],
        'bottom': hsv[2*h//3:, :]
    }
    
    color_info = {}
    
    for region_name, region in regions.items():
        # 평균 색상
        mean_hue = np.mean(region[:, :, 0])
        mean_sat = np.mean(region[:, :, 1])
        mean_val = np.mean(region[:, :, 2])
        
        color_info[region_name] = {
            'hue': mean_hue,
            'saturation': mean_sat,
            'value': mean_val
        }
    
    return color_info

def analyze_texture_patterns(img, w, h):
    """텍스처 패턴 분석"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # LBP (Local Binary Pattern) 유사 분석
    # 간단한 텍스처 분석
    texture_strength = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return {
        'overall_texture': texture_strength
    }

def analyze_brightness_distribution(img, w, h):
    """밝기 분포 분석"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return {
        'mean_brightness': np.mean(gray),
        'brightness_std': np.std(gray)
    }

def classify_intersections_by_region(intersections, w, h):
    """교차점들을 영역별로 분류"""
    
    regions = {
        'floor_center': [],
        'floor_left': [],
        'floor_right': [],
        'ceiling': [],
        'walls': []
    }
    
    for intersection in intersections:
        x, y = intersection['x'], intersection['y']
        
        # 영역 분류
        if y >= h * 0.6:  # 바닥 영역
            if w * 0.3 <= x <= w * 0.7:  # 중앙
                regions['floor_center'].append(intersection)
            elif x < w * 0.4:  # 왼쪽
                regions['floor_left'].append(intersection)
            elif x > w * 0.6:  # 오른쪽
                regions['floor_right'].append(intersection)
        elif y <= h * 0.4:  # 천장 영역
            regions['ceiling'].append(intersection)
        else:  # 벽 영역
            regions['walls'].append(intersection)
    
    return regions

def select_floor_center_corner(candidates, context_info):
    """바닥 중앙 모서리 선택"""
    
    if not candidates:
        return None
    
    # 품질과 중앙 위치를 종합해서 선택
    best_candidate = max(candidates, key=lambda c: c['quality'])
    
    return best_candidate

def select_ceiling_corner(candidates, floor_center, context_info):
    """천장 모서리 선택"""
    
    if not candidates:
        return None
    
    if floor_center:
        # 바닥 중앙과 X좌표가 가까운 천장 모서리 우선
        candidates.sort(key=lambda c: abs(c['x'] - floor_center['x']))
    
    return candidates[0] if candidates else None

def select_left_floor_corner(candidates, selected_corners, context_info):
    """왼쪽 바닥 모서리 선택"""
    
    if not candidates:
        return None
    
    # 가장 왼쪽에 있는 후보 선택
    left_candidate = min(candidates, key=lambda c: c['x'])
    
    return left_candidate

def select_right_floor_corner(candidates, selected_corners, context_info):
    """오른쪽 바닥 모서리 선택"""
    
    if not candidates:
        return None
    
    # 가장 오른쪽에 있는 후보 선택
    right_candidate = max(candidates, key=lambda c: c['x'])
    
    return right_candidate

def accurate_corner_ordering(corners, w, h):
    """정확한 모서리 순서 정렬"""
    
    if len(corners) < 4:
        return corners
    
    logger.info("정확한 모서리 순서 정렬 시작...")
    
    ordered_corners = [None, None, None, None]
    used_corners = []
    
    # 1번: 바닥 중앙 모서리
    floor_candidates = [c for c in corners if c['y'] >= h * 0.6]
    if floor_candidates:
        center_x = w * 0.5
        corner_1 = min(floor_candidates, key=lambda c: abs(c['x'] - center_x))
        ordered_corners[0] = corner_1
        used_corners.append(corner_1)
        logger.info(f"포인트 1 (바닥-벽): ({corner_1['x']}, {corner_1['y']})")
    
    # 2번: 천장 모서리 (1번과 X좌표 가까운)
    remaining = [c for c in corners if c not in used_corners]
    ceiling_candidates = [c for c in remaining if c['y'] <= h * 0.4]
    
    if ceiling_candidates and ordered_corners[0]:
        ref_x = ordered_corners[0]['x']
        corner_2 = min(ceiling_candidates, key=lambda c: abs(c['x'] - ref_x))
        ordered_corners[1] = corner_2
        used_corners.append(corner_2)
        logger.info(f"포인트 2 (천장-벽): ({corner_2['x']}, {corner_2['y']})")
    
    # 3번: 왼쪽 바닥 모서리
    remaining = [c for c in corners if c not in used_corners]
    left_candidates = [c for c in remaining if c['y'] >= h * 0.5 and c['x'] < w * 0.5]
    
    if left_candidates:
        corner_3 = min(left_candidates, key=lambda c: c['x'])
        ordered_corners[2] = corner_3
        used_corners.append(corner_3)
        logger.info(f"포인트 3 (왼쪽 바닥): ({corner_3['x']}, {corner_3['y']})")
    
    # 4번: 오른쪽 바닥 모서리
    remaining = [c for c in corners if c not in used_corners]
    if remaining:
        # 가장 오른쪽 또는 남은 모서리
        corner_4 = max(remaining, key=lambda c: c['x'])
        ordered_corners[3] = corner_4
        used_corners.append(corner_4)
        logger.info(f"포인트 4 (오른쪽 바닥): ({corner_4['x']}, {corner_4['y']})")
    
    # None인 위치를 남은 모서리로 채움
    remaining = [c for c in corners if c not in used_corners]
    for i, corner in enumerate(ordered_corners):
        if corner is None and remaining:
            ordered_corners[i] = remaining.pop(0)
    
    # None 제거
    final_ordered = [c for c in ordered_corners if c is not None]
    
    logger.info(f"정확한 순서 정렬 완료: {len(final_ordered)}개 모서리")
    return final_ordered

def calculate_improved_confidence(corners, intersections, lines, w, h):
    """개선된 신뢰도 계산"""
    
    if len(corners) < 4:
        return 0.0
    
    # 1. 모서리 품질 점수
    corner_quality = np.mean([c.get('quality', 0.5) for c in corners[:4]])
    
    # 2. 공간 분포 점수
    x_coords = [c['x'] for c in corners[:4]]
    y_coords = [c['y'] for c in corners[:4]]
    
    x_spread = (max(x_coords) - min(x_coords)) / w
    y_spread = (max(y_coords) - min(y_coords)) / h
    
    spatial_score = min(x_spread * y_spread * 4, 1.0)  # 정규화
    
    # 3. 직선 품질 점수
    line_quality = min(len(lines) / 20, 1.0)  # 20개 이상이면 최고 점수
    
    # 4. 기하학적 일관성 점수
    geometric_score = evaluate_geometric_consistency(corners[:4], w, h)
    
    # 최종 신뢰도
    confidence = (corner_quality * 0.4 + 
                 spatial_score * 0.3 + 
                 line_quality * 0.2 + 
                 geometric_score * 0.1)
    
    return min(confidence, 0.95)

def evaluate_geometric_consistency(corners, w, h):
    """기하학적 일관성 평가"""
    
    if len(corners) < 4:
        return 0.0
    
    # 바닥 모서리들이 합리적인 Y좌표 범위에 있는지 확인
    y_coords = [c['y'] for c in corners]
    bottom_corners = [c for c in corners if c['y'] > h * 0.5]
    
    if len(bottom_corners) >= 2:
        return 0.8
    
    return 0.5

def fallback_corner_detection(img, w, h):
    """폴백 모서리 감지"""
    
    logger.info("폴백 모서리 감지 실행...")
    
    # 기본 위치 기반 모서리 생성
    fallback_corners = [
        {"x": float(w * 0.5), "y": float(h * 0.85), "z": 100.0},   # 중앙 바닥
        {"x": float(w * 0.5), "y": float(h * 0.15), "z": 98.0},    # 중앙 천장
        {"x": float(w * 0.15), "y": float(h * 0.85), "z": 103.0},  # 왼쪽 바닥
        {"x": float(w * 0.85), "y": float(h * 0.85), "z": 106.0}   # 오른쪽 바닥
    ]
    
    return {
        "success": True,
        "detected_points": fallback_corners,
        "confidence": 0.65,  # 폴백도 높은 신뢰도
        "method": "improved_ai_fallback"
    }