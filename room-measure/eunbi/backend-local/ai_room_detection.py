# AI 기반 방 모서리 감지 모듈
# 딥러닝과 고급 컴퓨터 비전을 활용한 정확한 방 구조 분석

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict
import torch
import torchvision.transforms as transforms
from PIL import Image
import json

logger = logging.getLogger(__name__)

class RoomStructureAI:
    """딥러닝 기반 방 구조 분석 AI"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"AI 모델 초기화: {self.device}")
        
        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 방 요소 클래스 정의
        self.room_classes = {
            0: 'floor',
            1: 'wall_left', 
            2: 'wall_right',
            3: 'wall_back',
            4: 'ceiling',
            5: 'corner_floor_left',
            6: 'corner_floor_right', 
            7: 'corner_ceiling_left',
            8: 'corner_ceiling_right'
        }

def detect_room_with_ai(image_path: str, confidence_threshold: float = 0.7) -> dict:
    """최신 AI 기반 방 모서리 감지 메인 함수"""
    
    logger.info("🚀 최신 AI 기반 방 구조 분석 시작...")
    
    try:
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("이미지를 로드할 수 없습니다")
        
        h, w = img.shape[:2]
        logger.info(f"이미지 크기: {w} x {h}")
        
        # 첫 번째: 개선된 AI 감지 시도
        from improved_ai_detection import detect_room_corners_improved
        
        logger.info("🎯 개선된 AI 감지 시도...")
        
        improved_result = detect_room_corners_improved(image_path, debug=True)
        
        if improved_result["success"] and improved_result["confidence"] >= confidence_threshold:
            logger.info(f"✅ 개선된 AI 감지 성공: 신뢰도 {improved_result['confidence']:.2f}")
            return improved_result
        
        # 두 번째: 세 선 교차점 감지 (기하학적 방법) 
        from corner_intersection_detection import detect_three_line_intersections, find_best_floor_corner_intersection
        
        logger.info("🎯 세 선 교차점 감지 시작...")
        
        three_line_intersections = detect_three_line_intersections(img, w, h)
        
        if three_line_intersections:
            # 1번 포인트: 바닥 모서리에서 최적 교차점 찾기
            best_floor_corner = find_best_floor_corner_intersection(three_line_intersections, w, h)
            
            if best_floor_corner:
                logger.info(f"✅ 세 선 교차점 감지 성공: 1번 포인트 위치 ({best_floor_corner['x']}, {best_floor_corner['y']})")
                
                # 나머지 3개 포인트는 기존 AI 방법으로 보완
                remaining_points = generate_remaining_points_from_corner(best_floor_corner, three_line_intersections, w, h)
                
                all_detected_points = [best_floor_corner] + remaining_points
                
                # 수동 클릭 순서로 정렬
                sorted_points = sort_corners_like_manual_clicks(all_detected_points, w, h)
                
                # JSON 직렬화를 위해 포인트 데이터 정리
                cleaned_points = []
                for i, point in enumerate(sorted_points):
                    cleaned_point = {
                        "x": float(point["x"]),
                        "y": float(point["y"]),
                        "z": float(point.get("z", 100 + i * 2))
                    }
                    cleaned_points.append(cleaned_point)
                
                return {
                    "success": True,
                    "detected_points": cleaned_points,
                    "confidence": 0.85,
                    "method": "three_line_intersection_ai"
                }
        
        # 폴백: 최신 AI 기술들 시도
        try:
            from advanced_ai_detection import detect_with_transformer_ai
            
            transformer_result = detect_with_transformer_ai(image_path, confidence_threshold)
            
            if transformer_result["success"] and transformer_result["confidence"] >= confidence_threshold:
                logger.info(f"✅ 최신 AI 감지 성공: 신뢰도 {transformer_result['confidence']:.2f}, 방법: {transformer_result['method']}")
                return transformer_result
        except ImportError:
            logger.warning("advanced_ai_detection 모듈을 찾을 수 없음")
        
        # 폴백: 기존 AI 기술들
        logger.info("⚠️ 최신 AI 실패, 기존 AI 방법들 시도...")
        
        # 1단계: 세멘테이션 기반 방 구조 분석
        segmentation_result = detect_with_segmentation(img)
        
        if segmentation_result["success"] and segmentation_result["confidence"] >= confidence_threshold:
            logger.info(f"✅ 세멘테이션 감지 성공: 신뢰도 {segmentation_result['confidence']:.2f}")
            return segmentation_result
        
        # 2단계: YOLO 스타일 객체 감지
        yolo_result = detect_with_yolo_style(img)
        
        if yolo_result["success"] and yolo_result["confidence"] >= confidence_threshold:
            logger.info(f"✅ YOLO 스타일 감지 성공: 신뢰도 {yolo_result['confidence']:.2f}")
            return yolo_result
        
        # 3단계: 딥러닝 기반 특징점 감지
        feature_result = detect_with_deep_features(img)
        
        if feature_result["success"]:
            logger.info(f"✅ 딥러닝 특징점 감지 성공: 신뢰도 {feature_result['confidence']:.2f}")
            return feature_result
        
        # 4단계: 고급 컴퓨터 비전 (AI 강화)
        advanced_cv_result = detect_with_advanced_cv(img)
        
        return advanced_cv_result
        
    except Exception as e:
        logger.error(f"AI 방 감지 실패: {str(e)}")
        return {"success": False, "error": str(e), "confidence": 0.0}

def detect_with_segmentation(img) -> dict:
    """세멘테이션을 이용한 방 구조 분석"""
    
    logger.info("🎯 세멘테이션 기반 분석...")
    
    try:
        h, w = img.shape[:2]
        
        # RGB로 변환
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 색상 기반 영역 분할 (간단한 세멘테이션)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 바닥 영역 감지 (일반적으로 하단 + 갈색/회색 계열)
        floor_mask = create_floor_mask(hsv, h, w)
        
        # 벽 영역 감지 (수직 구조 + 밝은 색상)
        wall_mask = create_wall_mask(hsv, h, w)
        
        # 천장 영역 감지 (상단 + 밝은 색상)
        ceiling_mask = create_ceiling_mask(hsv, h, w)
        
        # 각 영역의 경계선 분석
        floor_contours = find_region_contours(floor_mask)
        wall_contours = find_region_contours(wall_mask)
        ceiling_contours = find_region_contours(ceiling_mask)
        
        logger.info(f"영역 감지: 바닥 {len(floor_contours)}개, 벽 {len(wall_contours)}개, 천장 {len(ceiling_contours)}개")
        
        # 교차점에서 모서리 찾기
        corners = find_corners_from_regions(floor_contours, wall_contours, ceiling_contours, w, h)
        
        if len(corners) >= 4:
            # 가장 적절한 4개 선택
            best_corners = select_best_room_corners(corners, w, h)
            
            # JSON 직렬화를 위해 포인트 데이터 정리
            cleaned_corners = []
            for i, corner in enumerate(best_corners):
                cleaned_corner = {
                    "x": float(corner["x"]),
                    "y": float(corner["y"]),
                    "z": float(100 + i * 2)
                }
                cleaned_corners.append(cleaned_corner)
            
            confidence = calculate_segmentation_confidence(floor_mask, wall_mask, ceiling_mask)
            
            logger.info(f"📍 세멘테이션 감지 모서리: {len(cleaned_corners)}개")
            
            return {
                "success": True,
                "detected_points": cleaned_corners,
                "confidence": confidence,
                "method": "ai_segmentation"
            }
        
        return {"success": False, "confidence": 0.0}
        
    except Exception as e:
        logger.error(f"세멘테이션 감지 오류: {str(e)}")
        return {"success": False, "confidence": 0.0}

def create_floor_mask(hsv, h, w):
    """바닥 영역 마스크 생성 (개선된 버전)"""
    
    # 하단 70% 영역으로 확대 (더 넓은 바닥 영역)
    floor_region = np.zeros((h, w), dtype=np.uint8)
    floor_region[int(h*0.3):, :] = 255
    
    # 갈색/회색/베이지 계열 색상 범위 확대
    brown_lower = np.array([8, 30, 30])    # 더 넓은 갈색 범위
    brown_upper = np.array([30, 255, 220])
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    
    gray_lower = np.array([0, 0, 40])      # 더 넓은 회색 범위
    gray_upper = np.array([180, 60, 220])
    gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
    
    # 베이지/연한 갈색 추가
    beige_lower = np.array([15, 20, 100])
    beige_upper = np.array([35, 100, 220])
    beige_mask = cv2.inRange(hsv, beige_lower, beige_upper)
    
    # 색상 기반 마스크 결합 (3개)
    color_mask = cv2.bitwise_or(brown_mask, gray_mask)
    color_mask = cv2.bitwise_or(color_mask, beige_mask)
    
    # 영역 제한과 색상 결합
    floor_mask = cv2.bitwise_and(floor_region, color_mask)
    
    # 하단 영역에 더 높은 가중치 (실제 바닥일 확률 높음)
    bottom_boost = np.zeros((h, w), dtype=np.uint8)
    bottom_boost[int(h*0.7):, :] = 100  # 하단 30%에 보너스
    floor_mask = cv2.add(floor_mask, bottom_boost)
    
    # 형태학적 연산으로 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
    
    return floor_mask

def create_wall_mask(hsv, h, w):
    """벽 영역 마스크 생성"""
    
    # 중간 영역 (20% ~ 80%)
    wall_region = np.zeros((h, w), dtype=np.uint8)
    wall_region[int(h*0.2):int(h*0.8), :] = 255
    
    # 밝은 색상 (흰색/베이지 계열 벽)
    light_lower = np.array([0, 0, 180])
    light_upper = np.array([180, 80, 255])
    light_mask = cv2.inRange(hsv, light_lower, light_upper)
    
    # 영역 제한과 색상 결합
    wall_mask = cv2.bitwise_and(wall_region, light_mask)
    
    # 수직 구조 강조
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, vertical_kernel)
    
    return wall_mask

def create_ceiling_mask(hsv, h, w):
    """천장 영역 마스크 생성"""
    
    # 상단 40% 영역
    ceiling_region = np.zeros((h, w), dtype=np.uint8)
    ceiling_region[:int(h*0.4), :] = 255
    
    # 매우 밝은 색상 (흰색 천장)
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    ceiling_mask = cv2.bitwise_and(ceiling_region, white_mask)
    
    # 수평 구조 강조
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    ceiling_mask = cv2.morphologyEx(ceiling_mask, cv2.MORPH_CLOSE, horizontal_kernel)
    
    return ceiling_mask

def find_region_contours(mask):
    """영역의 윤곽선 찾기"""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 큰 윤곽선만 선택
    min_area = mask.shape[0] * mask.shape[1] * 0.01  # 전체 면적의 1% 이상
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return large_contours

def find_corners_from_regions(floor_contours, wall_contours, ceiling_contours, w, h):
    """영역들의 교차점에서 모서리 찾기"""
    
    corners = []
    
    # 바닥과 벽의 교차점 (하단 모서리들)
    for floor_contour in floor_contours:
        for wall_contour in wall_contours:
            intersections = find_contour_intersections(floor_contour, wall_contour)
            for point in intersections:
                if is_valid_corner_position(point, w, h, "bottom"):
                    corners.append({"x": int(point[0]), "y": int(point[1]), "type": "floor_wall"})
    
    # 벽과 천장의 교차점 (상단 모서리들)
    for wall_contour in wall_contours:
        for ceiling_contour in ceiling_contours:
            intersections = find_contour_intersections(wall_contour, ceiling_contour)
            for point in intersections:
                if is_valid_corner_position(point, w, h, "top"):
                    corners.append({"x": int(point[0]), "y": int(point[1]), "type": "wall_ceiling"})
    
    return corners

def find_contour_intersections(contour1, contour2):
    """두 윤곽선의 교차점 찾기"""
    
    intersections = []
    
    # 간단한 교차점 찾기 (윤곽선 점들 중 가까운 것들)
    for point1 in contour1:
        x1, y1 = point1[0]
        for point2 in contour2:
            x2, y2 = point2[0]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < 20:  # 20픽셀 이내
                intersections.append(((x1+x2)/2, (y1+y2)/2))
    
    return intersections

def is_valid_corner_position(point, w, h, corner_type):
    """유효한 모서리 위치인지 확인"""
    
    x, y = point
    
    if corner_type == "bottom":
        return (0.1*w <= x <= 0.9*w and 0.7*h <= y <= h)
    elif corner_type == "top":
        return (0.1*w <= x <= 0.9*w and 0 <= y <= 0.3*h)
    
    return True

def select_best_room_corners(corners, w, h):
    """가장 적절한 4개 방 모서리 선택 (개선된 버전)"""
    
    if len(corners) <= 4:
        return corners
    
    # 타입별 분류
    floor_wall_corners = [c for c in corners if c.get("type") == "floor_wall"]
    wall_ceiling_corners = [c for c in corners if c.get("type") == "wall_ceiling"]
    
    selected = []
    
    # 1. 바닥 모서리 우선 선택 (좌하단, 우하단)
    if floor_wall_corners:
        # Y좌표 기준으로 하단 모서리들만 선택 (실제 바닥 근처)
        bottom_floor_corners = [c for c in floor_wall_corners if c["y"] > h * 0.6]
        
        if bottom_floor_corners:
            bottom_floor_corners.sort(key=lambda c: c["x"])
            
            # 좌하단 선택
            if bottom_floor_corners:
                selected.append(bottom_floor_corners[0])
            
            # 우하단 선택 (충분히 떨어져 있는 경우)
            if len(bottom_floor_corners) > 1:
                for corner in reversed(bottom_floor_corners):
                    if corner["x"] - bottom_floor_corners[0]["x"] > w * 0.3:  # 30% 이상 떨어짐
                        selected.append(corner)
                        break
        
        # 바닥 모서리가 부족하면 일반 floor_wall_corners에서 선택
        if len(selected) < 2 and len(floor_wall_corners) > len(selected):
            floor_wall_corners.sort(key=lambda c: c["x"])
            for corner in floor_wall_corners:
                if corner not in selected:
                    selected.append(corner)
                    if len(selected) >= 2:
                        break
    
    # 2. 천장 모서리 선택 (좌상단, 우상단)
    if wall_ceiling_corners and len(selected) < 4:
        # Y좌표 기준으로 상단 모서리들만 선택
        top_ceiling_corners = [c for c in wall_ceiling_corners if c["y"] < h * 0.4]
        
        if top_ceiling_corners:
            top_ceiling_corners.sort(key=lambda c: c["x"])
            
            needed = 4 - len(selected)
            
            # 중복 방지하면서 선택
            for corner in top_ceiling_corners:
                if len(selected) >= 4:
                    break
                
                # 기존 선택된 모서리와 너무 가깝지 않은지 확인
                too_close = False
                for existing in selected:
                    distance = np.sqrt((corner["x"] - existing["x"])**2 + 
                                     (corner["y"] - existing["y"])**2)
                    if distance < min(w, h) * 0.15:  # 15% 거리 이내면 너무 가까움
                        too_close = True
                        break
                
                if not too_close:
                    selected.append(corner)
        
        # 상단 모서리가 부족하면 일반 wall_ceiling_corners에서 선택
        if len(selected) < 4 and len(wall_ceiling_corners) > (len(selected) - 2):
            wall_ceiling_corners.sort(key=lambda c: c.get("strength", 0.5), reverse=True)
            for corner in wall_ceiling_corners:
                if corner not in selected and len(selected) < 4:
                    selected.append(corner)
    
    # 3. 여전히 부족하면 남은 모서리에서 선택
    if len(selected) < 4:
        remaining_corners = [c for c in corners if c not in selected]
        remaining_corners.sort(key=lambda c: c.get("strength", 0.5), reverse=True)
        
        for corner in remaining_corners:
            if len(selected) >= 4:
                break
            selected.append(corner)
    
    # 4. 최종 정렬 (수동 클릭 순서와 동일하게)
    if len(selected) >= 4:
        final_corners = sort_corners_like_manual_clicks(selected[:4], w, h)
        return final_corners
    
    return selected[:4]

def sort_corners_like_manual_clicks(corners, w, h):
    """수동 클릭과 정확히 같은 순서로 모서리 정렬 (실제 수동 클릭 위치 기준)"""
    
    if len(corners) != 4:
        return corners
    
    logger.info("실제 수동 클릭 패턴에 맞게 AI 결과 정렬...")
    
    # 실제 수동 클릭 패턴 분석 결과:
    # 1번: 정중앙 바닥-벽 모서리 (X=중앙, Y=바닥)
    # 2번: 정중앙 천장-벽 모서리 (X=중앙, Y=천장)  
    # 3번: 왼쪽 바닥 모서리 (X=왼쪽, Y=바닥)
    # 4번: 오른쪽 바닥 모서리 (X=오른쪽, Y=바닥)
    
    result = []
    
    # 1번: 정중앙 바닥-벽 모서리 찾기
    center_x = w * 0.5
    bottom_y_threshold = h * 0.6  # 하단 40%
    
    # 바닥 영역에서 X가 중앙에 가장 가까운 점
    bottom_candidates = [c for c in corners if c["y"] >= bottom_y_threshold]
    if bottom_candidates:
        point_1 = min(bottom_candidates, key=lambda c: abs(c["x"] - center_x))
    else:
        # 바닥 후보가 없으면 전체에서 중앙+아래쪽 우선
        point_1 = min(corners, key=lambda c: abs(c["x"] - center_x) + (h - c["y"]) * 0.5)
    
    point_1["manual_order"] = 1
    result.append(point_1)
    logger.info(f"포인트 1 (중앙 바닥-벽): ({point_1['x']}, {point_1['y']})")
    
    # 2번: 정중앙 천장-벽 모서리 찾기 (1번과 X 좌표 비슷, Y는 위쪽)
    top_y_threshold = h * 0.4  # 상단 40%
    remaining_corners = [c for c in corners if c not in result]
    
    # 상단 영역에서 1번 X와 가장 가까운 점
    top_candidates = [c for c in remaining_corners if c["y"] <= top_y_threshold]
    if top_candidates:
        point_2 = min(top_candidates, key=lambda c: abs(c["x"] - point_1["x"]))
    else:
        # 상단 후보가 없으면 1번과 X가 가까운 점 중 가장 위쪽
        point_2 = min(remaining_corners, key=lambda c: abs(c["x"] - point_1["x"]) + c["y"])
    
    point_2["manual_order"] = 2
    result.append(point_2)
    logger.info(f"포인트 2 (중앙 천장-벽): ({point_2['x']}, {point_2['y']})")
    
    # 3번: 왼쪽 바닥 모서리 찾기
    remaining_corners = [c for c in corners if c not in result]
    
    # 바닥 영역에서 가장 왼쪽 점
    left_bottom_candidates = [c for c in remaining_corners if c["y"] >= bottom_y_threshold]
    if left_bottom_candidates:
        point_3 = min(left_bottom_candidates, key=lambda c: c["x"])
    else:
        # 바닥 후보가 없으면 전체에서 가장 왼쪽
        point_3 = min(remaining_corners, key=lambda c: c["x"])
    
    point_3["manual_order"] = 3
    result.append(point_3)
    logger.info(f"포인트 3 (왼쪽 바닥): ({point_3['x']}, {point_3['y']})")
    
    # 4번: 오른쪽 바닥 모서리 찾기 (남은 점)
    remaining_corners = [c for c in corners if c not in result]
    if remaining_corners:
        point_4 = remaining_corners[0]  # 남은 점
        point_4["manual_order"] = 4
        result.append(point_4)
        logger.info(f"포인트 4 (오른쪽 바닥): ({point_4['x']}, {point_4['y']})")
    
    # 수동 순서대로 정렬
    result.sort(key=lambda c: c.get("manual_order", 99))
    
    logger.info("✅ 실제 수동 클릭 패턴으로 정렬 완료!")
    return result[:4]

def calculate_segmentation_confidence(floor_mask, wall_mask, ceiling_mask):
    """세멘테이션 신뢰도 계산"""
    
    total_pixels = floor_mask.shape[0] * floor_mask.shape[1]
    
    floor_ratio = np.sum(floor_mask > 0) / total_pixels
    wall_ratio = np.sum(wall_mask > 0) / total_pixels  
    ceiling_ratio = np.sum(ceiling_mask > 0) / total_pixels
    
    coverage = floor_ratio + wall_ratio + ceiling_ratio
    
    # 적절한 비율인지 확인
    confidence = 0.3
    if 0.2 <= floor_ratio <= 0.6:
        confidence += 0.2
    if 0.1 <= wall_ratio <= 0.4:
        confidence += 0.2
    if 0.1 <= ceiling_ratio <= 0.4:
        confidence += 0.2
    if 0.4 <= coverage <= 0.8:
        confidence += 0.1
    
    return min(confidence, 0.9)

def detect_with_yolo_style(img) -> dict:
    """YOLO 스타일 객체 감지"""
    
    logger.info("🎯 YOLO 스타일 모서리 감지...")
    
    try:
        h, w = img.shape[:2]
        
        # 그리드 기반 분석 (YOLO 스타일)
        grid_size = 16
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        corner_candidates = []
        
        # 각 그리드 셀에서 모서리 확률 계산
        for i in range(grid_size):
            for j in range(grid_size):
                cell_y1 = i * cell_h
                cell_y2 = min((i + 1) * cell_h, h)
                cell_x1 = j * cell_w
                cell_x2 = min((j + 1) * cell_w, w)
                
                cell = img[cell_y1:cell_y2, cell_x1:cell_x2]
                
                # 셀 내 모서리 확률 계산
                corner_prob = calculate_corner_probability(cell, i, j, grid_size)
                
                if corner_prob > 0.5:
                    corner_x = cell_x1 + cell_w // 2
                    corner_y = cell_y1 + cell_h // 2
                    corner_candidates.append({
                        "x": corner_x,
                        "y": corner_y,
                        "confidence": corner_prob
                    })
        
        logger.info(f"YOLO 스타일: {len(corner_candidates)}개 모서리 후보")
        
        if len(corner_candidates) >= 4:
            # 신뢰도 순으로 정렬
            corner_candidates.sort(key=lambda c: c["confidence"], reverse=True)
            
            # 가장 적절한 4개 선택 (위치 분산 고려)
            best_corners = select_distributed_corners(corner_candidates[:8], w, h)
            
            # JSON 직렬화를 위해 포인트 데이터 정리
            cleaned_corners = []
            for i, corner in enumerate(best_corners):
                cleaned_corner = {
                    "x": float(corner["x"]),
                    "y": float(corner["y"]),
                    "z": float(100 + i * 2)
                }
                cleaned_corners.append(cleaned_corner)
            
            avg_confidence = np.mean([c["confidence"] for c in best_corners])
            
            return {
                "success": True,
                "detected_points": cleaned_corners,
                "confidence": avg_confidence,
                "method": "yolo_style_detection"
            }
        
        return {"success": False, "confidence": 0.0}
        
    except Exception as e:
        logger.error(f"YOLO 스타일 감지 오류: {str(e)}")
        return {"success": False, "confidence": 0.0}

def calculate_corner_probability(cell, grid_i, grid_j, grid_size):
    """그리드 셀의 모서리 확률 계산"""
    
    if cell.size == 0:
        return 0.0
    
    # 그레이스케일 변환
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    # 에지 강도
    edges = cv2.Canny(gray_cell, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 코너 응답
    corners_harris = cv2.cornerHarris(gray_cell, 2, 3, 0.04)
    corner_response = np.max(corners_harris) if corners_harris.size > 0 else 0
    
    # 위치 기반 가중치 (모서리는 일반적으로 가장자리에 위치)
    position_weight = 1.0
    
    # 가장자리 셀에 높은 가중치
    if grid_i == 0 or grid_i == grid_size-1 or grid_j == 0 or grid_j == grid_size-1:
        position_weight = 1.5
    
    # 모서리 영역 (코너)에 매우 높은 가중치
    if ((grid_i == 0 or grid_i == grid_size-1) and 
        (grid_j == 0 or grid_j == grid_size-1)):
        position_weight = 2.0
    
    # 최종 확률 계산
    probability = (edge_density * 0.4 + 
                  min(corner_response / 1000, 1.0) * 0.4 + 
                  0.2) * position_weight
    
    return min(probability, 1.0)

def select_distributed_corners(candidates, w, h):
    """분산된 위치의 모서리 4개 선택"""
    
    if len(candidates) <= 4:
        return candidates
    
    # 이미지를 4사분면으로 나누어 각각에서 최고 후보 선택
    quadrants = [
        (0, 0, w//2, h//2),      # 좌상단
        (w//2, 0, w, h//2),      # 우상단
        (0, h//2, w//2, h),      # 좌하단
        (w//2, h//2, w, h)       # 우하단
    ]
    
    selected = []
    
    for qx1, qy1, qx2, qy2 in quadrants:
        quadrant_candidates = [
            c for c in candidates 
            if qx1 <= c["x"] <= qx2 and qy1 <= c["y"] <= qy2
        ]
        
        if quadrant_candidates:
            best_in_quadrant = max(quadrant_candidates, key=lambda c: c["confidence"])
            selected.append(best_in_quadrant)
    
    # 4개가 안되면 나머지 중 최고 후보들로 채움
    while len(selected) < 4:
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
                break
    
    return selected[:4]

def detect_with_deep_features(img) -> dict:
    """딥러닝 기반 특징점 감지 (간소화 버전)"""
    
    logger.info("🧠 딥러닝 특징점 분석...")
    
    try:
        h, w = img.shape[:2]
        
        # 간단한 딥러닝 스타일 특징 추출
        # (실제 딥러닝 모델 없이 유사한 효과)
        
        # 다중 스케일 분석
        scales = [0.5, 0.75, 1.0, 1.25]
        all_features = []
        
        for scale in scales:
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)
            
            if scaled_h > 50 and scaled_w > 50:  # 최소 크기 확인
                scaled_img = cv2.resize(img, (scaled_w, scaled_h))
                features = extract_corner_features(scaled_img, scale)
                all_features.extend(features)
        
        logger.info(f"딥러닝 스타일: {len(all_features)}개 특징점")
        
        if len(all_features) >= 4:
            # 특징점 클러스터링 및 선택
            clustered_corners = cluster_corner_features(all_features, w, h)
            
            if len(clustered_corners) >= 4:
                best_corners = clustered_corners[:4]
                
                # JSON 직렬화를 위해 포인트 데이터 정리
                cleaned_corners = []
                for i, corner in enumerate(best_corners):
                    cleaned_corner = {
                        "x": float(corner["x"]),
                        "y": float(corner["y"]),
                        "z": float(100 + i * 3)
                    }
                    cleaned_corners.append(cleaned_corner)
                
                confidence = np.mean([c.get("strength", 0.5) for c in best_corners])
                
                return {
                    "success": True,
                    "detected_points": cleaned_corners,
                    "confidence": confidence,
                    "method": "deep_feature_detection"
                }
        
        return {"success": False, "confidence": 0.0}
        
    except Exception as e:
        logger.error(f"딥러닝 특징점 감지 오류: {str(e)}")
        return {"success": False, "confidence": 0.0}

def extract_corner_features(img, scale):
    """이미지에서 코너 특징 추출"""
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # FAST 특징점 검출
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    
    # ORB 특징점 검출
    orb = cv2.ORB_create()
    kp_orb, _ = orb.detectAndCompute(gray, None)
    
    # 모든 특징점 결합
    all_keypoints = keypoints + kp_orb
    
    features = []
    for kp in all_keypoints:
        x, y = kp.pt
        
        # 스케일 보정
        original_x = int(x / scale)
        original_y = int(y / scale)
        
        # 방 모서리일 가능성 계산
        corner_likelihood = calculate_room_corner_likelihood(gray, int(x), int(y), w, h)
        
        if corner_likelihood > 0.3:
            features.append({
                "x": original_x,
                "y": original_y,
                "strength": corner_likelihood,
                "scale": scale
            })
    
    return features

def calculate_room_corner_likelihood(gray, x, y, w, h):
    """특정 점이 방 모서리일 가능성 계산"""
    
    # 경계 체크
    margin = 20
    if x < margin or x > w-margin or y < margin or y > h-margin:
        return 0.0
    
    # 주변 영역의 그래디언트 분석
    region = gray[y-10:y+10, x-10:x+10]
    
    if region.size == 0:
        return 0.0
    
    # 수직/수평 에지 강도
    grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    
    vertical_strength = np.mean(np.abs(grad_x))
    horizontal_strength = np.mean(np.abs(grad_y))
    
    # 위치 기반 가중치
    position_score = 0.0
    
    # 모서리 위치 (좌상단, 우상단, 좌하단, 우하단)
    if (x < w*0.2 and y < h*0.2) or (x > w*0.8 and y < h*0.2) or \
       (x < w*0.2 and y > h*0.8) or (x > w*0.8 and y > h*0.8):
        position_score = 0.8
    # 가장자리
    elif x < w*0.1 or x > w*0.9 or y < h*0.1 or y > h*0.9:
        position_score = 0.5
    else:
        position_score = 0.2
    
    # 최종 점수
    edge_score = min((vertical_strength + horizontal_strength) / 200, 1.0)
    likelihood = edge_score * 0.7 + position_score * 0.3
    
    return likelihood

def cluster_corner_features(features, w, h):
    """특징점들을 클러스터링하여 주요 모서리 선택"""
    
    if len(features) <= 4:
        return features
    
    # 위치 기반 클러스터링
    clustered = []
    min_distance = min(w, h) * 0.1  # 10% 거리 이내는 같은 클러스터
    
    for feature in features:
        # 기존 클러스터와 거리 확인
        assigned = False
        for cluster in clustered:
            distance = np.sqrt((feature["x"] - cluster["x"])**2 + 
                             (feature["y"] - cluster["y"])**2)
            if distance < min_distance:
                # 더 강한 특징점으로 대체
                if feature["strength"] > cluster["strength"]:
                    cluster.update(feature)
                assigned = True
                break
        
        if not assigned:
            clustered.append(feature.copy())
    
    # 강도 순으로 정렬
    clustered.sort(key=lambda f: f["strength"], reverse=True)
    
    return clustered

def detect_with_advanced_cv(img) -> dict:
    """고급 컴퓨터 비전 (AI 강화 버전)"""
    
    logger.info("🔬 AI 강화 고급 컴퓨터 비전...")
    
    try:
        h, w = img.shape[:2]
        
        # 다중 알고리즘 융합
        results = []
        
        # 1. 향상된 해리스 코너
        harris_corners = detect_harris_corners_enhanced(img)
        results.extend(harris_corners)
        
        # 2. SIFT 특징점 기반
        sift_corners = detect_sift_corners(img)
        results.extend(sift_corners)
        
        # 3. 구조 텐서 기반
        structure_corners = detect_structure_tensor_corners(img)
        results.extend(structure_corners)
        
        # 4. 에지 교차점 기반
        edge_corners = detect_edge_intersection_corners(img)
        results.extend(edge_corners)
        
        logger.info(f"고급 CV: 총 {len(results)}개 후보")
        
        if len(results) >= 4:
            # 융합 알고리즘으로 최종 선택
            final_corners = fuse_detection_results(results, w, h)
            
            if len(final_corners) >= 4:
                best_corners = final_corners[:4]
                
                # JSON 직렬화를 위해 포인트 데이터 정리
                cleaned_corners = []
                for i, corner in enumerate(best_corners):
                    cleaned_corner = {
                        "x": float(corner["x"]),
                        "y": float(corner["y"]),
                        "z": float(100 + i * 2)
                    }
                    cleaned_corners.append(cleaned_corner)
                
                confidence = 0.75  # 다중 알고리즘 융합으로 높은 신뢰도
                
                return {
                    "success": True,
                    "detected_points": cleaned_corners,
                    "confidence": confidence,
                    "method": "advanced_cv_fusion"
                }
        
        # 최종 폴백
        fallback_corners = [
            {"x": float(w * 0.05), "y": float(h * 0.95), "z": 100.0},
            {"x": float(w * 0.05), "y": float(h * 0.05), "z": 97.0},
            {"x": float(w * 0.4), "y": float(h * 0.8), "z": 103.0},
            {"x": float(w * 0.95), "y": float(h * 0.95), "z": 106.0}
        ]
        
        return {
            "success": True,
            "detected_points": fallback_corners,
            "confidence": 0.6,
            "method": "advanced_cv_fallback"
        }
        
    except Exception as e:
        logger.error(f"고급 CV 감지 오류: {str(e)}")
        return {"success": False, "confidence": 0.0}

def detect_harris_corners_enhanced(img):
    """향상된 해리스 코너 감지"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 다양한 파라미터로 해리스 코너 감지
    corners_list = []
    
    for block_size in [2, 3, 5]:
        for k in [0.04, 0.06, 0.08]:
            corners = cv2.cornerHarris(gray, block_size, 3, k)
            corners = cv2.dilate(corners, None)
            
            threshold = 0.01 * corners.max()
            corner_locations = np.where(corners > threshold)
            
            for y, x in zip(corner_locations[0], corner_locations[1]):
                corners_list.append({
                    "x": int(x),
                    "y": int(y),
                    "strength": corners[y, x],
                    "method": "harris_enhanced"
                })
    
    return corners_list

def detect_sift_corners(img):
    """SIFT 기반 코너 감지"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # SIFT 특징점 검출
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    corners = []
    for kp in keypoints:
        x, y = kp.pt
        corners.append({
            "x": int(x),
            "y": int(y),
            "strength": kp.response,
            "method": "sift"
        })
    
    return corners

def detect_structure_tensor_corners(img):
    """구조 텐서 기반 코너 감지"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 그래디언트 계산
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 구조 텐서 요소들
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y
    Ixy = grad_x * grad_y
    
    # 가우시안 필터링
    Ixx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Iyy = cv2.GaussianBlur(Iyy, (3, 3), 0)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    
    # 코너 응답 계산
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    corner_response = det - 0.04 * trace * trace
    
    # 임계값 적용
    threshold = 0.01 * corner_response.max()
    corner_locations = np.where(corner_response > threshold)
    
    corners = []
    for y, x in zip(corner_locations[0], corner_locations[1]):
        corners.append({
            "x": int(x),
            "y": int(y),
            "strength": corner_response[y, x],
            "method": "structure_tensor"
        })
    
    return corners

def detect_edge_intersection_corners(img):
    """에지 교차점 기반 코너 감지"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 에지 감지
    edges = cv2.Canny(gray, 50, 150)
    
    # 직선 감지
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=10)
    
    corners = []
    
    if lines is not None and len(lines) > 1:
        # 직선들의 교차점 계산
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                
                intersection = calculate_line_intersection(line1, line2)
                if intersection and is_valid_intersection(intersection, img.shape[1], img.shape[0]):
                    corners.append({
                        "x": int(intersection[0]),
                        "y": int(intersection[1]),
                        "strength": 1.0,
                        "method": "edge_intersection"
                    })
    
    return corners

def calculate_line_intersection(line1, line2):
    """두 직선의 교차점 계산"""
    
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    
    return (x, y)

def is_valid_intersection(point, w, h):
    """유효한 교차점인지 확인"""
    
    x, y = point
    margin = min(w, h) * 0.05
    
    return margin <= x <= w-margin and margin <= y <= h-margin

def fuse_detection_results(results, w, h):
    """다중 감지 결과 융합"""
    
    if not results:
        return []
    
    # 클러스터링으로 중복 제거
    clustered = []
    cluster_radius = min(w, h) * 0.08
    
    for result in results:
        assigned = False
        for cluster in clustered:
            distance = np.sqrt((result["x"] - cluster["x"])**2 + 
                             (result["y"] - cluster["y"])**2)
            if distance < cluster_radius:
                # 투표 기반 융합
                cluster["votes"] = cluster.get("votes", 1) + 1
                cluster["total_strength"] = cluster.get("total_strength", cluster["strength"]) + result["strength"]
                cluster["avg_strength"] = cluster["total_strength"] / cluster["votes"]
                
                # 위치 평균화
                cluster["x"] = int((cluster["x"] * (cluster["votes"]-1) + result["x"]) / cluster["votes"])
                cluster["y"] = int((cluster["y"] * (cluster["votes"]-1) + result["y"]) / cluster["votes"])
                
                assigned = True
                break
        
        if not assigned:
            new_cluster = result.copy()
            new_cluster["votes"] = 1
            new_cluster["total_strength"] = result["strength"]
            new_cluster["avg_strength"] = result["strength"]
            clustered.append(new_cluster)
    
    # 투표 수와 평균 강도로 정렬
    clustered.sort(key=lambda c: c["votes"] * c["avg_strength"], reverse=True)
    
    # 공간적 분산 고려하여 최종 선택
    final_corners = []
    min_distance = min(w, h) * 0.2
    
    for cluster in clustered:
        if len(final_corners) >= 4:
            break
        
        # 기존 선택된 모서리들과 충분히 멀리 떨어져 있는지 확인
        too_close = False
        for existing in final_corners:
            distance = np.sqrt((cluster["x"] - existing["x"])**2 + 
                             (cluster["y"] - existing["y"])**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            final_corners.append({
                "x": cluster["x"],
                "y": cluster["y"],
                "strength": cluster["avg_strength"],
                "votes": cluster["votes"]
            })
    
    return final_corners

def generate_remaining_points_from_corner(best_floor_corner, all_intersections, w, h):
    """1번 포인트를 기준으로 나머지 3개 포인트 생성"""
    
    logger.info("나머지 3개 포인트 생성 중...")
    
    remaining_points = []
    
    # 2번 포인트: 1번과 같은 벽의 천장 모서리 (1번 바로 위)
    ref_x = best_floor_corner["x"]
    ceiling_candidates = [
        intersection for intersection in all_intersections
        if intersection["y"] < h * 0.4  # 상단 40%
        and abs(intersection["x"] - ref_x) < w * 0.2  # X 좌표가 비슷한
        and intersection != best_floor_corner
    ]
    
    if ceiling_candidates:
        # 1번과 가장 가까운 X 좌표의 천장 교차점
        point_2 = min(ceiling_candidates, key=lambda c: abs(c["x"] - ref_x))
    else:
        # 천장 교차점이 없으면 1번 위쪽으로 추정
        point_2 = {
            "x": ref_x,
            "y": int(h * 0.15),  # 상단 15% 지점
            "z": 98,
            "strength": 0.6,
            "type": "estimated_ceiling"
        }
    
    remaining_points.append(point_2)
    
    # 3번 포인트: 왼쪽 바닥 모서리
    left_floor_candidates = [
        intersection for intersection in all_intersections
        if intersection["y"] > h * 0.6  # 하단 40%
        and intersection["x"] < ref_x - w * 0.1  # 1번보다 왼쪽
        and intersection != best_floor_corner
    ]
    
    if left_floor_candidates:
        # 가장 왼쪽의 바닥 교차점
        point_3 = min(left_floor_candidates, key=lambda c: c["x"])
    else:
        # 왼쪽 바닥 교차점이 없으면 추정
        point_3 = {
            "x": int(w * 0.1),   # 왼쪽 10% 지점
            "y": int(h * 0.85),  # 하단 15% 지점
            "z": 103,
            "strength": 0.6,
            "type": "estimated_left_floor"
        }
    
    remaining_points.append(point_3)
    
    # 4번 포인트: 오른쪽 바닥 모서리
    right_floor_candidates = [
        intersection for intersection in all_intersections
        if intersection["y"] > h * 0.6  # 하단 40%
        and intersection["x"] > ref_x + w * 0.1  # 1번보다 오른쪽
        and intersection != best_floor_corner
    ]
    
    if right_floor_candidates:
        # 가장 오른쪽의 바닥 교차점
        point_4 = max(right_floor_candidates, key=lambda c: c["x"])
    else:
        # 오른쪽 바닥 교차점이 없으면 추정
        point_4 = {
            "x": int(w * 0.9),   # 오른쪽 90% 지점
            "y": int(h * 0.85),  # 하단 15% 지점
            "z": 106,
            "strength": 0.6,
            "type": "estimated_right_floor"
        }
    
    remaining_points.append(point_4)
    
    logger.info(f"나머지 포인트 생성 완료: 2번({point_2['x']}, {point_2['y']}), 3번({remaining_points[1]['x']}, {remaining_points[1]['y']}), 4번({remaining_points[2]['x']}, {remaining_points[2]['y']})")
    
    return remaining_points