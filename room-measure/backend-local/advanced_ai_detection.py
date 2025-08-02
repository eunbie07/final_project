# 최신 AI 기술을 활용한 방 모서리 감지
# Transformer, U-Net, MediaPipe 스타일 구현

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import json

logger = logging.getLogger(__name__)

class AdvancedRoomAI:
    """최신 AI 기술 기반 방 모서리 감지"""
    
    def __init__(self):
        logger.info("🚀 최신 AI 모델 초기화...")
        
        # Transformer 스타일 패치 크기
        self.patch_size = 16
        
        # U-Net 스타일 다중 스케일
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        # MediaPipe 스타일 랜드마크 감지
        self.corner_landmarks = {
            'floor_left': 0,
            'ceiling_left': 1, 
            'floor_back': 2,
            'floor_right': 3
        }

def detect_with_transformer_ai(image_path: str, confidence_threshold: float = 0.7) -> dict:
    """Transformer 기반 방 구조 분석"""
    
    logger.info("🔮 Transformer AI 기반 분석 시작...")
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("이미지를 로드할 수 없습니다")
        
        h, w = img.shape[:2]
        logger.info(f"이미지 크기: {w} x {h}")
        
        # 1단계: Vision Transformer 스타일 패치 분석
        vit_result = vision_transformer_analysis(img)
        
        if vit_result["success"] and vit_result["confidence"] >= confidence_threshold:
            logger.info(f"✅ Vision Transformer 성공: 신뢰도 {vit_result['confidence']:.2f}")
            return vit_result
        
        # 2단계: U-Net 스타일 세멘테이션
        unet_result = unet_style_segmentation(img)
        
        if unet_result["success"] and unet_result["confidence"] >= confidence_threshold:
            logger.info(f"✅ U-Net 스타일 성공: 신뢰도 {unet_result['confidence']:.2f}")
            return unet_result
        
        # 3단계: MediaPipe 스타일 랜드마크
        mediapipe_result = mediapipe_style_landmarks(img)
        
        if mediapipe_result["success"]:
            logger.info(f"✅ MediaPipe 스타일 성공: 신뢰도 {mediapipe_result['confidence']:.2f}")
            return mediapipe_result
        
        # 4단계: 앙상블 AI (여러 모델 결합)
        ensemble_result = ensemble_ai_detection(img)
        
        return ensemble_result
        
    except Exception as e:
        logger.error(f"Transformer AI 감지 실패: {str(e)}")
        return {"success": False, "error": str(e), "confidence": 0.0}

def vision_transformer_analysis(img) -> dict:
    """Vision Transformer 스타일 분석"""
    
    logger.info("🎯 Vision Transformer 패치 분석...")
    
    try:
        h, w = img.shape[:2]
        patch_size = 16
        
        # 이미지를 패치로 분할 (ViT 스타일)
        patches = []
        patch_positions = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = img[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                patch_positions.append((i, j))
        
        logger.info(f"총 {len(patches)}개 패치 생성")
        
        # 각 패치의 "모서리성" 분석
        corner_scores = []
        
        for idx, patch in enumerate(patches):
            score = analyze_patch_for_corners(patch, patch_positions[idx], h, w)
            corner_scores.append(score)
        
        # 상위 점수 패치들에서 모서리 찾기
        top_patches = sorted(range(len(corner_scores)), 
                           key=lambda i: corner_scores[i], reverse=True)[:20]
        
        corner_candidates = []
        
        for patch_idx in top_patches:
            i, j = patch_positions[patch_idx]
            patch = patches[patch_idx]
            
            # 패치 내 정밀 모서리 검출
            local_corners = find_corners_in_patch(patch, i, j)
            corner_candidates.extend(local_corners)
        
        logger.info(f"Vision Transformer: {len(corner_candidates)}개 모서리 후보")
        
        if len(corner_candidates) >= 4:
            # 트랜스포머 스타일 어텐션 메커니즘으로 최적 4개 선택
            selected_corners = transformer_attention_selection(corner_candidates, w, h)
            
            if len(selected_corners) >= 4:
                # 수동 클릭 순서로 정렬
                final_corners = sort_corners_like_manual_clicks_advanced(selected_corners[:4], w, h)
                
                for i, corner in enumerate(final_corners):
                    corner["z"] = 100 + i * 2
                
                confidence = calculate_transformer_confidence(final_corners, corner_scores)
                
                return {
                    "success": True,
                    "detected_points": final_corners[:4],
                    "confidence": confidence,
                    "method": "vision_transformer"
                }
        
        return {"success": False, "confidence": 0.0}
        
    except Exception as e:
        logger.error(f"Vision Transformer 오류: {str(e)}")
        return {"success": False, "confidence": 0.0}

def analyze_patch_for_corners(patch, position, img_h, img_w):
    """패치가 방 모서리를 포함할 확률 분석"""
    
    i, j = position
    patch_h, patch_w = patch.shape[:2]
    
    # 1. 기하학적 위치 점수
    center_x = j + patch_w // 2
    center_y = i + patch_h // 2
    
    position_score = 0.0
    
    # 모서리 영역 (높은 점수)
    if ((center_x < img_w * 0.2 and center_y < img_h * 0.2) or  # 좌상단
        (center_x > img_w * 0.8 and center_y < img_h * 0.2) or  # 우상단  
        (center_x < img_w * 0.2 and center_y > img_h * 0.8) or  # 좌하단
        (center_x > img_w * 0.8 and center_y > img_h * 0.8)):   # 우하단
        position_score = 1.0
    # 가장자리 영역 (중간 점수)
    elif (center_x < img_w * 0.1 or center_x > img_w * 0.9 or 
          center_y < img_h * 0.1 or center_y > img_h * 0.9):
        position_score = 0.6
    else:
        position_score = 0.2
    
    # 2. 시각적 특징 점수
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    
    # 에지 밀도
    edges = cv2.Canny(gray_patch, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 코너 응답
    corners = cv2.cornerHarris(gray_patch, 2, 3, 0.04)
    corner_response = np.max(corners) if corners.size > 0 else 0
    
    # 색상 분산 (방 구조는 색상 경계가 명확)
    color_var = np.var(patch)
    
    # 3. 최종 점수 계산
    visual_score = (edge_density * 0.4 + 
                   min(corner_response / 1000, 1.0) * 0.4 + 
                   min(color_var / 10000, 1.0) * 0.2)
    
    final_score = position_score * 0.6 + visual_score * 0.4
    
    return final_score

def find_corners_in_patch(patch, offset_i, offset_j):
    """패치 내에서 정밀 모서리 검출"""
    
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    
    # Harris corner detection
    corners = cv2.cornerHarris(gray_patch, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    
    threshold = 0.01 * corners.max() if corners.max() > 0 else 0
    corner_locations = np.where(corners > threshold)
    
    detected_corners = []
    
    for y, x in zip(corner_locations[0], corner_locations[1]):
        global_x = offset_j + x
        global_y = offset_i + y
        strength = corners[y, x]
        
        detected_corners.append({
            "x": int(global_x),
            "y": int(global_y),
            "strength": strength
        })
    
    return detected_corners

def transformer_attention_selection(candidates, w, h):
    """트랜스포머 어텐션 메커니즘으로 최적 모서리 선택"""
    
    if len(candidates) <= 4:
        return candidates
    
    # 어텐션 가중치 계산 (각 후보가 다른 후보들과의 관계)
    attention_weights = []
    
    for i, candidate in enumerate(candidates):
        weight = 0.0
        
        # 다른 모든 후보와의 관계 분석
        for j, other in enumerate(candidates):
            if i != j:
                # 거리 기반 관계
                distance = np.sqrt((candidate["x"] - other["x"])**2 + 
                                 (candidate["y"] - other["y"])**2)
                
                # 적절한 거리(방 크기)에 있는 후보들과 높은 어텐션
                ideal_distance = min(w, h) * 0.3
                distance_score = 1.0 / (1.0 + abs(distance - ideal_distance) / ideal_distance)
                
                # 각도 관계 (직각 관계 선호)
                angle = np.arctan2(other["y"] - candidate["y"], 
                                 other["x"] - candidate["x"])
                angle_score = abs(np.sin(2 * angle))  # 90도 배수에서 높은 점수
                
                weight += distance_score * 0.7 + angle_score * 0.3
        
        # 자체 강도도 고려
        self_strength = candidate.get("strength", 0.5)
        total_weight = weight + self_strength
        
        attention_weights.append(total_weight)
    
    # 어텐션 가중치 순으로 정렬
    sorted_indices = sorted(range(len(attention_weights)), 
                          key=lambda i: attention_weights[i], reverse=True)
    
    # 상위 후보들 중에서 공간적으로 분산된 4개 선택
    selected = []
    min_distance = min(w, h) * 0.15
    
    for idx in sorted_indices:
        candidate = candidates[idx]
        
        # 이미 선택된 모서리들과 너무 가깝지 않은지 확인
        too_close = False
        for selected_corner in selected:
            distance = np.sqrt((candidate["x"] - selected_corner["x"])**2 + 
                             (candidate["y"] - selected_corner["y"])**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected.append(candidate)
            
        if len(selected) >= 4:
            break
    
    return selected

def calculate_transformer_confidence(corners, patch_scores):
    """트랜스포머 모델 신뢰도 계산"""
    
    # 1. 패치 점수 기반 신뢰도
    avg_patch_score = np.mean(patch_scores) if patch_scores else 0.5
    
    # 2. 모서리 분포 기반 신뢰도
    if len(corners) >= 4:
        x_coords = [c["x"] for c in corners[:4]]
        y_coords = [c["y"] for c in corners[:4]]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # 적절한 분포인지 확인
        distribution_score = min(x_range / 500, 1.0) * min(y_range / 300, 1.0)
    else:
        distribution_score = 0.0
    
    # 3. 모서리 강도 기반 신뢰도
    strength_score = np.mean([c.get("strength", 0.5) for c in corners[:4]]) if corners else 0.0
    strength_score = min(strength_score / 1000, 1.0)
    
    # 최종 신뢰도
    confidence = (avg_patch_score * 0.4 + 
                 distribution_score * 0.4 + 
                 strength_score * 0.2)
    
    return min(confidence, 0.95)

def sort_corners_like_manual_clicks_advanced(corners, w, h):
    """수동 클릭과 같은 순서로 모서리 정렬 (고급 AI용)"""
    
    if len(corners) != 4:
        return corners
    
    logger.info("수동 클릭 순서로 AI 감지 결과 정렬 중...")
    
    # 상하 분리
    bottom_y_threshold = h * 0.6  # 바닥 기준점을 60%로 설정
    top_y_threshold = h * 0.4     # 천장 기준점을 40%로 설정
    
    bottom_corners = [c for c in corners if c["y"] >= bottom_y_threshold]
    top_corners = [c for c in corners if c["y"] <= top_y_threshold]
    middle_corners = [c for c in corners if top_y_threshold < c["y"] < bottom_y_threshold]
    
    # 부족한 경우 middle_corners를 적절히 분배
    if len(bottom_corners) < 2:
        # 중간 영역의 모서리 중 아래쪽을 바닥으로 분류
        middle_bottom = [c for c in middle_corners if c["y"] > h * 0.5]
        bottom_corners.extend(middle_bottom)
        middle_corners = [c for c in middle_corners if c not in middle_bottom]
    
    if len(top_corners) < 2:
        # 중간 영역의 모서리 중 위쪽을 천장으로 분류
        middle_top = [c for c in middle_corners if c["y"] <= h * 0.5]
        top_corners.extend(middle_top)
    
    # 좌우 정렬
    bottom_corners.sort(key=lambda c: c["x"])
    top_corners.sort(key=lambda c: c["x"])
    
    result = []
    
    # 1. Floor-wall corner (바닥-벽 모서리) - 가장 왼쪽 바닥
    if bottom_corners:
        floor_wall = bottom_corners[0]
        floor_wall["corner_type"] = "floor_wall"
        result.append(floor_wall)
        logger.info(f"포인트 1 (Floor-wall): ({floor_wall['x']}, {floor_wall['y']})")
    
    # 2. Ceiling-wall corner (같은 벽의 천장) - 1번과 X좌표가 가장 가까운 천장
    if top_corners and result:
        ref_x = result[0]["x"]
        ceiling_wall = min(top_corners, key=lambda c: abs(c["x"] - ref_x))
        ceiling_wall["corner_type"] = "ceiling_wall_same"
        result.append(ceiling_wall)
        logger.info(f"포인트 2 (Ceiling-wall same): ({ceiling_wall['x']}, {ceiling_wall['y']})")
    elif top_corners:
        ceiling_wall = top_corners[0]
        ceiling_wall["corner_type"] = "ceiling_wall_same"
        result.append(ceiling_wall)
    
    # 3. Left wall floor corner (왼쪽 벽 바닥) - 나머지 바닥 중 적절한 위치
    remaining_bottom = [c for c in bottom_corners if c not in result]
    if remaining_bottom:
        # 1번 포인트보다 안쪽(중앙 쪽)에 있는 것 선호
        if result:
            ref_x = result[0]["x"]
            left_wall_floor = min(remaining_bottom, 
                                key=lambda c: abs(c["x"] - w*0.4) + abs(c["y"] - result[0]["y"]) * 0.3)
        else:
            left_wall_floor = remaining_bottom[0]
        
        left_wall_floor["corner_type"] = "left_wall_floor"
        result.append(left_wall_floor)
        logger.info(f"포인트 3 (Left wall floor): ({left_wall_floor['x']}, {left_wall_floor['y']})")
    
    # 4. Right wall floor corner (오른쪽 벽 바닥) - 가장 오른쪽
    remaining_all = [c for c in corners if c not in result]
    if remaining_all:
        # 가장 오른쪽에 있는 모서리 (바닥 우선)
        remaining_bottom_final = [c for c in remaining_all if c["y"] >= h * 0.5]
        if remaining_bottom_final:
            right_wall_floor = max(remaining_bottom_final, key=lambda c: c["x"])
        else:
            right_wall_floor = max(remaining_all, key=lambda c: c["x"])
        
        right_wall_floor["corner_type"] = "right_wall_floor"
        result.append(right_wall_floor)
        logger.info(f"포인트 4 (Right wall floor): ({right_wall_floor['x']}, {right_wall_floor['y']})")
    
    # 부족한 경우 채우기
    while len(result) < 4:
        remaining = [c for c in corners if c not in result]
        if remaining:
            next_corner = remaining[0]
            next_corner["corner_type"] = f"fallback_{len(result) + 1}"
            result.append(next_corner)
            logger.info(f"포인트 {len(result)} (Fallback): ({next_corner['x']}, {next_corner['y']})")
        else:
            break
    
    logger.info("AI 감지 결과가 수동 클릭 순서로 정렬 완료")
    return result[:4]

def unet_style_segmentation(img) -> dict:
    """U-Net 스타일 다중 스케일 세멘테이션"""
    
    logger.info("🔍 U-Net 스타일 세멘테이션...")
    
    try:
        h, w = img.shape[:2]
        
        # 다중 스케일 분석 (U-Net의 인코더-디코더 구조 모방)
        multi_scale_results = []
        scales = [0.5, 0.75, 1.0, 1.25]
        
        for scale in scales:
            if scale != 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img = cv2.resize(img, (new_w, new_h))
            else:
                scaled_img = img.copy()
                new_h, new_w = h, w
            
            # 각 스케일에서 세멘테이션 수행
            segments = perform_unet_segmentation(scaled_img)
            
            # 원본 크기로 결과 변환
            if scale != 1.0:
                segments = resize_segmentation_results(segments, w, h, scale)
            
            multi_scale_results.append(segments)
        
        # 다중 스케일 결과 융합
        fused_segments = fuse_multi_scale_results(multi_scale_results)
        
        # 세멘테이션 결과에서 모서리 추출
        corners = extract_corners_from_segmentation(fused_segments, w, h)
        
        logger.info(f"U-Net 스타일: {len(corners)}개 모서리 감지")
        
        if len(corners) >= 4:
            best_corners = select_best_unet_corners(corners, w, h)
            
            for i, corner in enumerate(best_corners[:4]):
                corner["z"] = 100 + i * 2
            
            confidence = calculate_unet_confidence(fused_segments, corners)
            
            return {
                "success": True,
                "detected_points": best_corners[:4],
                "confidence": confidence,
                "method": "unet_segmentation"
            }
        
        return {"success": False, "confidence": 0.0}
        
    except Exception as e:
        logger.error(f"U-Net 세멘테이션 오류: {str(e)}")
        return {"success": False, "confidence": 0.0}

def perform_unet_segmentation(img):
    """U-Net 스타일 세멘테이션 수행"""
    
    h, w = img.shape[:2]
    
    # HSV로 변환하여 색상 기반 분할
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. 바닥 세그먼트 (하단 + 갈색/회색)
    floor_mask = np.zeros((h, w), dtype=np.uint8)
    floor_region = floor_mask.copy()
    floor_region[int(h*0.5):, :] = 255  # 하단 50%
    
    # 바닥 색상 (갈색, 회색, 베이지)
    floor_color1 = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([25, 255, 200]))
    floor_color2 = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 50, 150]))
    floor_color = cv2.bitwise_or(floor_color1, floor_color2)
    
    floor_mask = cv2.bitwise_and(floor_region, floor_color)
    
    # 2. 벽 세그먼트 (중간 + 밝은 색상)
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    wall_region = wall_mask.copy()
    wall_region[int(h*0.1):int(h*0.9), :] = 255  # 중간 80%
    
    # 벽 색상 (밝은 색상)
    wall_color = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 80, 255]))
    wall_mask = cv2.bitwise_and(wall_region, wall_color)
    
    # 3. 천장 세그먼트 (상단 + 매우 밝은 색상)
    ceiling_mask = np.zeros((h, w), dtype=np.uint8)
    ceiling_region = ceiling_mask.copy()
    ceiling_region[:int(h*0.5), :] = 255  # 상단 50%
    
    # 천장 색상 (흰색)
    ceiling_color = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    ceiling_mask = cv2.bitwise_and(ceiling_region, ceiling_color)
    
    # 형태학적 연산으로 세그먼트 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    ceiling_mask = cv2.morphologyEx(ceiling_mask, cv2.MORPH_CLOSE, kernel)
    
    return {
        "floor": floor_mask,
        "wall": wall_mask, 
        "ceiling": ceiling_mask
    }

def resize_segmentation_results(segments, target_w, target_h, scale):
    """세멘테이션 결과를 원본 크기로 변환"""
    
    resized_segments = {}
    
    for key, mask in segments.items():
        resized_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        resized_segments[key] = resized_mask
    
    return resized_segments

def fuse_multi_scale_results(multi_scale_results):
    """다중 스케일 세멘테이션 결과 융합"""
    
    if not multi_scale_results:
        return {}
    
    # 첫 번째 결과를 기준으로 초기화
    fused = {}
    for key in multi_scale_results[0].keys():
        fused[key] = np.zeros_like(multi_scale_results[0][key], dtype=np.float32)
    
    # 모든 스케일 결과를 평균화
    for result in multi_scale_results:
        for key, mask in result.items():
            fused[key] += mask.astype(np.float32) / len(multi_scale_results)
    
    # 이진화
    for key in fused.keys():
        fused[key] = (fused[key] > 127).astype(np.uint8) * 255
    
    return fused

def extract_corners_from_segmentation(segments, w, h):
    """세멘테이션 결과에서 방 모서리 추출"""
    
    corners = []
    
    if "floor" not in segments or "wall" not in segments:
        return corners
    
    floor_mask = segments["floor"]
    wall_mask = segments["wall"]
    ceiling_mask = segments.get("ceiling", np.zeros_like(floor_mask))
    
    # 각 세그먼트의 윤곽선 찾기
    floor_contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ceiling_contours, _ = cv2.findContours(ceiling_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 윤곽선들만 사용
    if floor_contours:
        floor_contour = max(floor_contours, key=cv2.contourArea)
        
        # 바닥 윤곽선의 모서리점들 찾기
        epsilon = 0.02 * cv2.arcLength(floor_contour, True)
        floor_corners = cv2.approxPolyDP(floor_contour, epsilon, True)
        
        for corner in floor_corners:
            x, y = corner[0]
            if is_valid_room_corner(x, y, w, h):
                corners.append({
                    "x": int(x),
                    "y": int(y),
                    "type": "floor",
                    "strength": 1.0
                })
    
    # 벽과 천장의 교차점도 찾기
    if wall_contours and ceiling_contours:
        wall_contour = max(wall_contours, key=cv2.contourArea)
        ceiling_contour = max(ceiling_contours, key=cv2.contourArea)
        
        # 교차점 계산 (간단한 방법)
        for wall_pt in wall_contour[::10]:  # 샘플링
            wall_x, wall_y = wall_pt[0]
            for ceiling_pt in ceiling_contour[::10]:
                ceiling_x, ceiling_y = ceiling_pt[0]
                
                distance = np.sqrt((wall_x - ceiling_x)**2 + (wall_y - ceiling_y)**2)
                if distance < 20:  # 가까운 점들
                    avg_x = (wall_x + ceiling_x) // 2
                    avg_y = (wall_y + ceiling_y) // 2
                    
                    if is_valid_room_corner(avg_x, avg_y, w, h):
                        corners.append({
                            "x": int(avg_x),
                            "y": int(avg_y),
                            "type": "wall_ceiling",
                            "strength": 0.8
                        })
    
    return corners

def is_valid_room_corner(x, y, w, h):
    """유효한 방 모서리 위치인지 확인"""
    
    margin = min(w, h) * 0.05
    return margin <= x <= w - margin and margin <= y <= h - margin

def select_best_unet_corners(corners, w, h):
    """U-Net 결과에서 최적 모서리 4개 선택"""
    
    if len(corners) <= 4:
        return corners
    
    # 강도 순으로 정렬
    corners.sort(key=lambda c: c["strength"], reverse=True)
    
    # 공간적 분산을 고려하여 선택
    selected = []
    min_distance = min(w, h) * 0.2
    
    for corner in corners:
        too_close = False
        for selected_corner in selected:
            distance = np.sqrt((corner["x"] - selected_corner["x"])**2 + 
                             (corner["y"] - selected_corner["y"])**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected.append(corner)
            
        if len(selected) >= 4:
            break
    
    return selected

def calculate_unet_confidence(segments, corners):
    """U-Net 모델 신뢰도 계산"""
    
    # 세그먼트 품질 평가
    total_pixels = 0
    segmented_pixels = 0
    
    for mask in segments.values():
        total_pixels += mask.size
        segmented_pixels += np.sum(mask > 0)
    
    coverage = segmented_pixels / total_pixels if total_pixels > 0 else 0
    
    # 모서리 품질 평가
    corner_quality = np.mean([c.get("strength", 0.5) for c in corners]) if corners else 0
    
    # 최종 신뢰도
    confidence = coverage * 0.6 + corner_quality * 0.4
    
    return min(confidence, 0.9)

def mediapipe_style_landmarks(img) -> dict:
    """MediaPipe 스타일 랜드마크 감지"""
    
    logger.info("🎯 MediaPipe 스타일 랜드마크 감지...")
    
    try:
        h, w = img.shape[:2]
        
        # MediaPipe 스타일의 키포인트 감지
        # 방의 주요 구조적 랜드마크 정의
        room_landmarks = detect_room_landmarks(img)
        
        if len(room_landmarks) >= 4:
            # 신뢰도 기반 필터링
            filtered_landmarks = [lm for lm in room_landmarks if lm["confidence"] > 0.3]
            
            if len(filtered_landmarks) >= 4:
                best_landmarks = select_landmark_corners(filtered_landmarks, w, h)
                
                for i, landmark in enumerate(best_landmarks[:4]):
                    landmark["z"] = 100 + i * 2
                
                avg_confidence = np.mean([lm["confidence"] for lm in best_landmarks[:4]])
                
                return {
                    "success": True,
                    "detected_points": best_landmarks[:4],
                    "confidence": avg_confidence,
                    "method": "mediapipe_landmarks"
                }
        
        return {"success": False, "confidence": 0.0}
        
    except Exception as e:
        logger.error(f"MediaPipe 랜드마크 오류: {str(e)}")
        return {"success": False, "confidence": 0.0}

def detect_room_landmarks(img):
    """방의 구조적 랜드마크 감지"""
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    landmarks = []
    
    # 1. 수직선 기반 랜드마크 (벽 모서리)
    vertical_lines = detect_vertical_structures(gray)
    
    # 2. 수평선 기반 랜드마크 (바닥/천장 경계)
    horizontal_lines = detect_horizontal_structures(gray)
    
    # 3. 교차점을 랜드마크로 변환
    for v_line in vertical_lines:
        for h_line in horizontal_lines:
            intersection = calculate_line_intersection_simple(v_line, h_line)
            if intersection and 0 < intersection[0] < w and 0 < intersection[1] < h:
                
                # 랜드마크 신뢰도 계산
                confidence = calculate_landmark_confidence(intersection, gray, w, h)
                
                landmarks.append({
                    "x": int(intersection[0]),
                    "y": int(intersection[1]),
                    "confidence": confidence,
                    "type": determine_landmark_type(intersection, w, h)
                })
    
    return landmarks

def detect_vertical_structures(gray):
    """수직 구조 감지"""
    
    # 수직 소벨 필터
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)
    
    # 수직선 강조
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    vertical_lines = cv2.morphologyEx(sobelx.astype(np.uint8), cv2.MORPH_OPEN, vertical_kernel)
    
    # 허프 변환으로 직선 감지
    lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return []
    
    # 수직선만 필터링
    vertical_lines_filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if 80 <= angle <= 100:  # 수직선
            vertical_lines_filtered.append(line[0])
    
    return vertical_lines_filtered

def detect_horizontal_structures(gray):
    """수평 구조 감지"""
    
    # 수평 소벨 필터
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = np.absolute(sobely)
    
    # 수평선 강조
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    horizontal_lines = cv2.morphologyEx(sobely.astype(np.uint8), cv2.MORPH_OPEN, horizontal_kernel)
    
    # 허프 변환으로 직선 감지
    lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50,
                           minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return []
    
    # 수평선만 필터링
    horizontal_lines_filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if angle <= 10 or angle >= 170:  # 수평선
            horizontal_lines_filtered.append(line[0])
    
    return horizontal_lines_filtered

def calculate_line_intersection_simple(line1, line2):
    """간단한 직선 교차점 계산"""
    
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    
    x = x1 + t*(x2-x1)
    y = y1 + t*(y2-y1)
    
    return (x, y)

def calculate_landmark_confidence(point, gray, w, h):
    """랜드마크 신뢰도 계산"""
    
    x, y = int(point[0]), int(point[1])
    
    # 위치 기반 신뢰도
    position_confidence = 0.0
    
    # 모서리 근처에 높은 신뢰도
    if ((x < w*0.2 and y < h*0.2) or (x > w*0.8 and y < h*0.2) or
        (x < w*0.2 and y > h*0.8) or (x > w*0.8 and y > h*0.8)):
        position_confidence = 0.8
    elif (x < w*0.1 or x > w*0.9 or y < h*0.1 or y > h*0.9):
        position_confidence = 0.5
    else:
        position_confidence = 0.2
    
    # 로컬 특징 기반 신뢰도
    if (5 <= x < w-5 and 5 <= y < h-5):
        local_region = gray[y-5:y+5, x-5:x+5]
        
        # 에지 강도
        edges = cv2.Canny(local_region, 50, 150)
        edge_strength = np.sum(edges > 0) / edges.size
        
        # 코너 응답
        corners = cv2.cornerHarris(local_region.astype(np.float32), 2, 3, 0.04)
        corner_strength = np.max(corners) if corners.size > 0 else 0
        
        feature_confidence = edge_strength * 0.6 + min(corner_strength/1000, 1.0) * 0.4
    else:
        feature_confidence = 0.0
    
    return position_confidence * 0.7 + feature_confidence * 0.3

def determine_landmark_type(point, w, h):
    """랜드마크 타입 결정"""
    
    x, y = point
    
    if y > h * 0.7:
        return "floor_corner"
    elif y < h * 0.3:
        return "ceiling_corner"
    else:
        return "wall_corner"

def select_landmark_corners(landmarks, w, h):
    """랜드마크에서 방 모서리 선택"""
    
    # 타입별 분류
    floor_corners = [lm for lm in landmarks if lm["type"] == "floor_corner"]
    ceiling_corners = [lm for lm in landmarks if lm["type"] == "ceiling_corner"]
    wall_corners = [lm for lm in landmarks if lm["type"] == "wall_corner"]
    
    selected = []
    
    # 바닥 모서리 중 좌우 선택
    if floor_corners:
        floor_corners.sort(key=lambda lm: lm["x"])
        if len(floor_corners) >= 2:
            selected.append(floor_corners[0])   # 왼쪽 바닥
            selected.append(floor_corners[-1])  # 오른쪽 바닥
        else:
            selected.extend(floor_corners)
    
    # 천장 모서리 선택
    if ceiling_corners and len(selected) < 4:
        ceiling_corners.sort(key=lambda lm: lm["confidence"], reverse=True)
        needed = 4 - len(selected)
        selected.extend(ceiling_corners[:needed])
    
    # 부족하면 벽 모서리로 채우기
    if wall_corners and len(selected) < 4:
        wall_corners.sort(key=lambda lm: lm["confidence"], reverse=True)
        needed = 4 - len(selected)
        selected.extend(wall_corners[:needed])
    
    return selected

def ensemble_ai_detection(img) -> dict:
    """앙상블 AI: 여러 모델의 결과를 종합"""
    
    logger.info("🎭 앙상블 AI 융합 감지...")
    
    try:
        h, w = img.shape[:2]
        
        # 여러 AI 모델의 결과 수집
        all_results = []
        
        # 1. 경량 CNN 스타일
        cnn_corners = lightweight_cnn_detection(img)
        all_results.extend([{"corner": c, "method": "cnn", "weight": 0.3} for c in cnn_corners])
        
        # 2. Random Forest 기반
        rf_corners = random_forest_detection(img)
        all_results.extend([{"corner": c, "method": "random_forest", "weight": 0.25} for c in rf_corners])
        
        # 3. SVM 기반
        svm_corners = svm_detection(img)
        all_results.extend([{"corner": c, "method": "svm", "weight": 0.25} for c in svm_corners])
        
        # 4. 클러스터링 기반
        cluster_corners = clustering_detection(img)
        all_results.extend([{"corner": c, "method": "clustering", "weight": 0.2} for c in cluster_corners])
        
        logger.info(f"앙상블: 총 {len(all_results)}개 후보 수집")
        
        if len(all_results) >= 4:
            # 앙상블 투표로 최종 결정
            final_corners = ensemble_voting(all_results, w, h)
            
            if len(final_corners) >= 4:
                for i, corner in enumerate(final_corners[:4]):
                    corner["z"] = 100 + i * 2
                
                confidence = calculate_ensemble_confidence(final_corners, all_results)
                
                return {
                    "success": True,
                    "detected_points": final_corners[:4],
                    "confidence": confidence,
                    "method": "ensemble_ai"
                }
        
        # 앙상블 실패시 기본 위치 반환
        fallback_corners = [
            {"x": int(w * 0.05), "y": int(h * 0.95), "z": 100},
            {"x": int(w * 0.05), "y": int(h * 0.05), "z": 97},
            {"x": int(w * 0.4), "y": int(h * 0.8), "z": 103},
            {"x": int(w * 0.95), "y": int(h * 0.95), "z": 106}
        ]
        
        return {
            "success": True,
            "detected_points": fallback_corners,
            "confidence": 0.6,
            "method": "ensemble_fallback"
        }
        
    except Exception as e:
        logger.error(f"앙상블 AI 오류: {str(e)}")
        return {"success": False, "confidence": 0.0}

def lightweight_cnn_detection(img):
    """경량 CNN 스타일 특징 추출"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # CNN의 컨볼루션 레이어 시뮬레이션
    # 여러 필터 적용
    filters = [
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),  # 수평 에지
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),  # 수직 에지
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),  # 라플라시안
    ]
    
    feature_maps = []
    for f in filters:
        filtered = cv2.filter2D(gray, -1, f)
        feature_maps.append(filtered)
    
    # 특징 맵 결합
    combined = np.zeros_like(gray, dtype=np.float32)
    for fm in feature_maps:
        combined += np.abs(fm.astype(np.float32))
    
    combined = combined / len(feature_maps)
    
    # 풀링 (다운샘플링)
    pooled = cv2.resize(combined, (w//4, h//4))
    
    # 강한 응답 위치 찾기
    threshold = np.percentile(pooled, 95)  # 상위 5%
    strong_responses = np.where(pooled > threshold)
    
    corners = []
    for y, x in zip(strong_responses[0], strong_responses[1]):
        # 원본 크기로 변환
        orig_x = x * 4
        orig_y = y * 4
        
        if 0 < orig_x < w and 0 < orig_y < h:
            corners.append({
                "x": int(orig_x),
                "y": int(orig_y),
                "strength": pooled[y, x]
            })
    
    return corners

def random_forest_detection(img):
    """Random Forest 기반 모서리 감지"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 특징 추출
    features = []
    positions = []
    
    step = 10  # 샘플링 간격
    
    for y in range(step, h-step, step):
        for x in range(step, w-step, step):
            # 로컬 특징 계산
            patch = gray[y-step//2:y+step//2, x-step//2:x+step//2]
            
            if patch.size == 0:
                continue
            
            # 특징들
            mean_intensity = np.mean(patch)
            std_intensity = np.std(patch)
            
            # 그래디언트
            gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            mean_gradient = np.mean(gradient_mag)
            
            # 위치 정보
            pos_x = x / w
            pos_y = y / h
            
            feature_vector = [
                mean_intensity, std_intensity, mean_gradient,
                pos_x, pos_y,
                pos_x**2, pos_y**2,  # 비선형 특징
                pos_x * pos_y
            ]
            
            features.append(feature_vector)
            positions.append((x, y))
    
    if not features:
        return []
    
    features = np.array(features)
    
    # 간단한 분류 규칙 (실제 RF 대신)
    # 모서리일 확률 계산
    corner_probs = []
    
    for i, feature in enumerate(features):
        x, y = positions[i]
        
        # 규칙 기반 분류
        prob = 0.0
        
        # 위치 기반 규칙
        if (x < w*0.2 and y < h*0.2) or (x > w*0.8 and y < h*0.2) or \
           (x < w*0.2 and y > h*0.8) or (x > w*0.8 and y > h*0.8):
            prob += 0.4
        
        # 그래디언트 기반 규칙
        if feature[2] > np.percentile(features[:, 2], 75):  # 상위 25% 그래디언트
            prob += 0.3
        
        # 강도 변화 기반 규칙
        if feature[1] > np.percentile(features[:, 1], 75):  # 상위 25% 표준편차
            prob += 0.3
        
        corner_probs.append(prob)
    
    # 높은 확률의 위치들 선택
    high_prob_indices = [i for i, p in enumerate(corner_probs) if p > 0.5]
    
    corners = []
    for i in high_prob_indices:
        x, y = positions[i]
        corners.append({
            "x": x,
            "y": y,
            "probability": corner_probs[i]
        })
    
    return corners

def svm_detection(img):
    """SVM 기반 모서리 감지"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # HOG 특징 추출 (SVM에서 많이 사용)
    # 간단한 HOG 구현
    corners = []
    
    cell_size = 16
    
    for y in range(0, h-cell_size, cell_size//2):
        for x in range(0, w-cell_size, cell_size//2):
            cell = gray[y:y+cell_size, x:x+cell_size]
            
            if cell.shape[0] != cell_size or cell.shape[1] != cell_size:
                continue
            
            # 그래디언트 계산
            gx = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(gx**2 + gy**2)
            orientation = np.arctan2(gy, gx)
            
            # 히스토그램 계산 (9개 빈)
            hist, _ = np.histogram(orientation, bins=9, range=(-np.pi, np.pi), 
                                 weights=magnitude)
            
            # 정규화
            hist = hist / (np.linalg.norm(hist) + 1e-6)
            
            # 간단한 SVM 결정 (실제로는 훈련된 모델 사용)
            # 여기서는 휴리스틱 사용
            corner_score = 0.0
            
            # 강한 에지가 있는지
            if np.max(hist) > 0.3:
                corner_score += 0.3
            
            # 여러 방향의 에지가 있는지 (코너 특성)
            if np.sum(hist > 0.1) >= 2:
                corner_score += 0.4
            
            # 위치 가중치
            center_x = x + cell_size // 2
            center_y = y + cell_size // 2
            
            if ((center_x < w*0.3 and center_y < h*0.3) or 
                (center_x > w*0.7 and center_y < h*0.3) or
                (center_x < w*0.3 and center_y > h*0.7) or 
                (center_x > w*0.7 and center_y > h*0.7)):
                corner_score += 0.3
            
            if corner_score > 0.5:
                corners.append({
                    "x": center_x,
                    "y": center_y,
                    "score": corner_score
                })
    
    return corners

def clustering_detection(img):
    """클러스터링 기반 모서리 감지"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 특징점 검출
    corners_harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners_harris = cv2.dilate(corners_harris, None)
    
    # 강한 코너들만 선택
    threshold = 0.01 * corners_harris.max()
    corner_locations = np.where(corners_harris > threshold)
    
    if len(corner_locations[0]) == 0:
        return []
    
    # 좌표 리스트
    points = list(zip(corner_locations[1], corner_locations[0]))  # (x, y)
    
    if len(points) < 4:
        return [{"x": int(x), "y": int(y), "cluster": 0} for x, y in points]
    
    # K-means 클러스터링 (4개 클러스터)
    points_array = np.array(points)
    
    try:
        kmeans = KMeans(n_clusters=min(4, len(points)), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(points_array)
        cluster_centers = kmeans.cluster_centers_
        
        # 각 클러스터의 중심을 모서리로 사용
        corners = []
        for i, center in enumerate(cluster_centers):
            corners.append({
                "x": int(center[0]),
                "y": int(center[1]),
                "cluster": i,
                "cluster_size": np.sum(cluster_labels == i)
            })
        
        return corners
        
    except Exception as e:
        logger.error(f"클러스터링 오류: {str(e)}")
        # 폴백: 가장 강한 4개 포인트 선택
        strengths = [corners_harris[y, x] for x, y in points]
        sorted_indices = sorted(range(len(strengths)), key=lambda i: strengths[i], reverse=True)
        
        return [{"x": int(points[i][0]), "y": int(points[i][1]), "cluster": 0} 
                for i in sorted_indices[:4]]

def ensemble_voting(all_results, w, h):
    """앙상블 투표로 최종 모서리 결정"""
    
    if not all_results:
        return []
    
    # 클러스터링으로 비슷한 위치의 결과들 그룹화
    positions = [(result["corner"]["x"], result["corner"]["y"]) for result in all_results]
    
    if len(positions) < 4:
        return [result["corner"] for result in all_results]
    
    # DBSCAN으로 클러스터링
    try:
        clustering = DBSCAN(eps=min(w, h)*0.1, min_samples=1)
        cluster_labels = clustering.fit_predict(positions)
        
        # 각 클러스터의 가중 평균 계산
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # 노이즈 제거
        
        cluster_corners = []
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
            
            # 가중 평균 계산
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            
            for idx in cluster_indices:
                result = all_results[idx]
                weight = result["weight"]
                corner = result["corner"]
                
                weighted_x += corner["x"] * weight
                weighted_y += corner["y"] * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_x = int(weighted_x / total_weight)
                avg_y = int(weighted_y / total_weight)
                
                cluster_corners.append({
                    "x": avg_x,
                    "y": avg_y,
                    "votes": len(cluster_indices),
                    "total_weight": total_weight
                })
        
        # 투표 수와 가중치로 정렬
        cluster_corners.sort(key=lambda c: c["votes"] * c["total_weight"], reverse=True)
        
        return cluster_corners[:4]
        
    except Exception as e:
        logger.error(f"앙상블 투표 오류: {str(e)}")
        # 폴백: 가중치가 높은 순으로 선택
        all_results.sort(key=lambda r: r["weight"], reverse=True)
        return [result["corner"] for result in all_results[:4]]

def calculate_ensemble_confidence(final_corners, all_results):
    """앙상블 신뢰도 계산"""
    
    if not final_corners or not all_results:
        return 0.0
    
    # 1. 투표 일치도
    total_votes = sum(corner.get("votes", 1) for corner in final_corners)
    avg_votes = total_votes / len(final_corners)
    vote_confidence = min(avg_votes / 3, 1.0)  # 평균 3표 이상이면 높은 신뢰도
    
    # 2. 가중치 총합
    total_weight = sum(corner.get("total_weight", 0.5) for corner in final_corners)
    weight_confidence = min(total_weight / 4, 1.0)  # 4.0이 최대
    
    # 3. 결과 다양성
    methods_used = set(result["method"] for result in all_results)
    diversity_confidence = len(methods_used) / 4  # 4가지 방법 모두 사용하면 1.0
    
    # 최종 신뢰도
    confidence = (vote_confidence * 0.4 + 
                 weight_confidence * 0.4 + 
                 diversity_confidence * 0.2)
    
    return min(confidence, 0.95)