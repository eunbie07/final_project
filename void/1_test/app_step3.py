# 파일명: app_step3.py
# 3단계: 절대 크기 추정 시스템 (실제 방 크기를 모를 때)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
import math
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class AbsoluteRoomAnalyzer:
    def __init__(self):
        # 한국 건축 표준 치수 (cm)
        self.standard_dimensions = {
            'door': {
                'width': [70, 80, 90],      # 일반문 너비
                'height': [200, 210, 220],  # 일반문 높이
                'typical': {'width': 80, 'height': 200}
            },
            'window': {
                'width': [90, 120, 150, 180],   # 창문 너비
                'height': [120, 150, 180],      # 창문 높이  
                'typical': {'width': 120, 'height': 150}
            },
            'outlet': {
                'width': [8, 10, 12],       # 콘센트 너비
                'height': [8, 10, 12],      # 콘센트 높이
                'typical': {'width': 10, 'height': 10}
            },
            'ceiling_height': [240, 260, 280],  # 천장 높이
            'room_sizes': {
                # 일반적인 방 크기 (가로 x 세로 cm)
                'small': [(250, 300), (300, 350)],       # 작은 방
                'medium': [(300, 400), (350, 450)],      # 중간 방  
                'large': [(400, 500), (450, 600)]        # 큰 방
            }
        }
    
    def analyze_image(self, image_data, reference_size=None, options=None):
        """절대 크기 추정 분석"""
        if options is None:
            options = {'detect_windows': True, 'detect_doors': True}
            
        try:
            # 이미지 디코딩
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("이미지를 디코딩할 수 없습니다.")
            
            logger.info(f"절대 크기 추정 시작 - 이미지 크기: {image.shape}")
            
            # 1단계: 고급 전처리
            processed_image = self.enhanced_preprocessing(image)
            
            # 2단계: 표준 객체 감지 및 분류
            detected_objects = self.detect_standard_objects(image)
            logger.info(f"감지된 표준 객체: {len(detected_objects)}개")
            
            # 3단계: 다중 스케일 추정
            scale_candidates = self.estimate_multiple_scales(detected_objects, image.shape)
            best_scale = self.select_best_scale(scale_candidates)
            logger.info(f"최적 스케일: {best_scale:.4f} cm/pixel")
            
            # 4단계: 방 경계선 감지  
            room_corners = self.detect_room_boundaries_advanced(processed_image, detected_objects)
            
            # 5단계: 크기 검증 및 보정
            dimensions = self.calculate_verified_dimensions(room_corners, best_scale, scale_candidates)
            
            # 6단계: 신뢰도 및 대안 제시
            analysis_result = self.comprehensive_analysis(dimensions, detected_objects, scale_candidates)
            
            # 7단계: 시각화
            result_image = self.absolute_visualization(
                image, room_corners, detected_objects, analysis_result
            )
            
            return {
                'success': True,
                'dimensions': analysis_result['primary'],
                'alternatives': analysis_result['alternatives'],
                'detected_objects': detected_objects,
                'scale_info': {
                    'best_scale': best_scale,
                    'scale_candidates': scale_candidates,
                    'confidence': analysis_result['confidence']
                },
                'result_image': result_image,
                'analysis_info': {
                    'method': 'absolute_estimation_v3',
                    'objects_used': len([obj for obj in detected_objects if obj.get('used_for_scale')]),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"분석 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def enhanced_preprocessing(self, image):
        """향상된 전처리"""
        # 노이즈 제거
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 그레이스케일
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # 적응적 히스토그램 평활화
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def detect_standard_objects(self, image):
        """표준 치수 객체 감지 및 분류"""
        detected = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 문 감지 (더 정교함)
        doors = self.detect_doors_with_standards(gray, image)
        detected.extend(doors)
        
        # 창문 감지
        windows = self.detect_windows_with_standards(gray, image)
        detected.extend(windows)
        
        # 콘센트 감지
        outlets = self.detect_outlets_with_standards(gray)
        detected.extend(outlets)
        
        # 추가: 타일/패턴 감지
        patterns = self.detect_floor_patterns(gray)
        detected.extend(patterns)
        
        return detected
    
    def detect_doors_with_standards(self, gray, color_image):
        """표준 치수 기반 문 감지"""
        doors = []
        
        # 엣지 강화
        edges = cv2.Canny(gray, 50, 150)
        
        # 수직 라인 강조 (문은 세로가 긴 형태)
        kernel_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
        vertical_edges = cv2.filter2D(edges, -1, kernel_v)
        
        contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 100000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w
                
                # 문의 종횡비 체크 (높이/너비 = 2.0~3.5)
                if 2.0 < aspect_ratio < 3.5:
                    # 추가 검증
                    confidence = self.verify_door_features(color_image[y:y+h, x:x+w], gray[y:y+h, x:x+w])
                    
                    if confidence > 0.3:
                        # 표준 치수와 매칭
                        estimated_size = self.match_door_standards(w, h)
                        
                        doors.append({
                            'type': 'door',
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'aspect_ratio': aspect_ratio,
                            'estimated_real_size': estimated_size,
                            'pixel_size': {'width': w, 'height': h}
                        })
        
        return doors
    
    def verify_door_features(self, color_roi, gray_roi):
        """문 특징 검증"""
        if color_roi.size == 0:
            return 0.0
        
        confidence = 0.0
        
        # 1. 색상 분석 (문은 보통 갈색, 흰색, 회색 계열)
        hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
        
        # 갈색 계열 (나무 문)
        brown_mask = cv2.inRange(hsv, np.array([5, 50, 20]), np.array([25, 255, 200]))
        
        # 회색/흰색 계열 (도장 문)  
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 30, 255]))
        
        door_color_ratio = (cv2.countNonZero(brown_mask) + cv2.countNonZero(gray_mask)) / color_roi.size
        confidence += door_color_ratio * 0.4
        
        # 2. 엣지 패턴 분석 (문은 직선 엣지가 많음)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if 0.05 < edge_density < 0.3:  # 적당한 엣지 밀도
            confidence += 0.3
        
        # 3. 수직성 검증 (문은 세로 구조물)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=10)
        if lines is not None:
            vertical_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
                if 80 < angle < 100:  # 수직선
                    vertical_lines += 1
            
            if vertical_lines >= 2:
                confidence += 0.3
        
        return min(1.0, confidence)
    
    def match_door_standards(self, pixel_width, pixel_height):
        """문 표준 치수 매칭"""
        # 한국 표준 문 치수와 비교
        door_standards = self.standard_dimensions['door']
        
        # 종횡비 기반 매칭
        pixel_aspect = pixel_height / pixel_width
        
        best_match = None
        min_diff = float('inf')
        
        for real_width in door_standards['width']:
            for real_height in door_standards['height']:
                real_aspect = real_height / real_width
                diff = abs(pixel_aspect - real_aspect)
                
                if diff < min_diff:
                    min_diff = diff
                    best_match = {'width': real_width, 'height': real_height}
        
        return best_match or door_standards['typical']
    
    def detect_windows_with_standards(self, gray, color_image):
        """표준 치수 기반 창문 감지"""
        windows = []
        
        # 밝은 영역 감지 (창문 = 자연광)
        _, bright_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        
        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 200000:  # 창문 크기 범위
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # 창문 종횡비 (0.8~3.0)
                if 0.8 < aspect_ratio < 3.0:
                    confidence = self.verify_window_features(color_image[y:y+h, x:x+w], gray[y:y+h, x:x+w])
                    
                    if confidence > 0.4:
                        estimated_size = self.match_window_standards(w, h)
                        
                        windows.append({
                            'type': 'window',
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'aspect_ratio': aspect_ratio,
                            'estimated_real_size': estimated_size,
                            'pixel_size': {'width': w, 'height': h}
                        })
        
        return windows
    
    def verify_window_features(self, color_roi, gray_roi):
        """창문 특징 검증"""
        if color_roi.size == 0:
            return 0.0
        
        confidence = 0.0
        
        # 1. 밝기 분석
        mean_brightness = np.mean(gray_roi)
        if mean_brightness > 120:
            confidence += 0.4
        
        # 2. 프레임 검출 (창문 테두리)
        edges = cv2.Canny(gray_roi, 30, 100)
        
        # 직사각형 프레임 찾기
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=20, maxLineGap=5)
        if lines is not None:
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
                
                if angle < 20 or angle > 160:  # 수평선
                    horizontal_lines += 1
                elif 70 < angle < 110:  # 수직선
                    vertical_lines += 1
            
            if horizontal_lines >= 2 and vertical_lines >= 2:
                confidence += 0.4
        
        # 3. 유리창 반사/투명도 (밝기 편차)
        brightness_std = np.std(gray_roi)
        if 20 < brightness_std < 60:  # 적당한 밝기 편차
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def match_window_standards(self, pixel_width, pixel_height):
        """창문 표준 치수 매칭"""
        window_standards = self.standard_dimensions['window']
        
        pixel_aspect = pixel_width / pixel_height
        
        best_match = None
        min_diff = float('inf')
        
        for real_width in window_standards['width']:
            for real_height in window_standards['height']:
                real_aspect = real_width / real_height
                diff = abs(pixel_aspect - real_aspect)
                
                if diff < min_diff:
                    min_diff = diff
                    best_match = {'width': real_width, 'height': real_height}
        
        return best_match or window_standards['typical']
    
    def detect_outlets_with_standards(self, gray):
        """콘센트 표준 치수 감지"""
        outlets = []
        
        edges = cv2.Canny(gray, 80, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # 콘센트 크기
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 0.6 < aspect_ratio < 1.7:  # 정사각형에 가까운 형태
                    outlets.append({
                        'type': 'outlet',
                        'bbox': (x, y, w, h),
                        'confidence': 0.5,
                        'estimated_real_size': self.standard_dimensions['outlet']['typical'],
                        'pixel_size': {'width': w, 'height': h}
                    })
        
        return outlets
    
    def detect_floor_patterns(self, gray):
        """바닥 패턴 감지 (타일, 마루 등)"""
        patterns = []
        
        # 반복 패턴 검출 (타일 등)
        # 템플릿 매칭이나 FFT 기반 패턴 분석 가능
        # 여기서는 간단한 그리드 패턴 검출
        
        return patterns  # 현재는 빈 배열 반환
    
    def estimate_multiple_scales(self, detected_objects, image_shape):
        """다중 스케일 추정"""
        scale_candidates = []
        
        for obj in detected_objects:
            obj_type = obj['type']
            pixel_size = obj['pixel_size']
            estimated_real = obj['estimated_real_size']
            confidence = obj['confidence']
            
            if obj_type == 'door':
                # 문 높이 기준 스케일
                scale_height = estimated_real['height'] / pixel_size['height']
                scale_candidates.append({
                    'scale': scale_height,
                    'confidence': confidence,
                    'source': f'door_height_{estimated_real["height"]}cm',
                    'object': obj
                })
                
                # 문 너비 기준 스케일
                scale_width = estimated_real['width'] / pixel_size['width']
                scale_candidates.append({
                    'scale': scale_width,
                    'confidence': confidence * 0.8,  # 높이보다 신뢰도 낮음
                    'source': f'door_width_{estimated_real["width"]}cm',
                    'object': obj
                })
                
            elif obj_type == 'window':
                # 창문 너비 기준
                scale_width = estimated_real['width'] / pixel_size['width']
                scale_candidates.append({
                    'scale': scale_width,
                    'confidence': confidence,
                    'source': f'window_width_{estimated_real["width"]}cm',
                    'object': obj
                })
                
                # 창문 높이 기준
                scale_height = estimated_real['height'] / pixel_size['height']
                scale_candidates.append({
                    'scale': scale_height,
                    'confidence': confidence * 0.9,
                    'source': f'window_height_{estimated_real["height"]}cm',
                    'object': obj
                })
                
            elif obj_type == 'outlet':
                # 콘센트 크기 기준
                scale_avg = (estimated_real['width'] + estimated_real['height']) / (pixel_size['width'] + pixel_size['height']) * 2
                scale_candidates.append({
                    'scale': scale_avg,
                    'confidence': confidence * 0.6,  # 작은 객체라 신뢰도 낮음
                    'source': f'outlet_{estimated_real["width"]}cm',
                    'object': obj
                })
        
        # 스케일이 없으면 일반적인 방 크기 추정
        if not scale_candidates:
            h, w = image_shape[:2]
            # 일반적인 방: 3.5m x 4.0m 가정
            estimated_scale = 350 / min(w, h)  # 더 작은 쪽을 3.5m로 가정
            scale_candidates.append({
                'scale': estimated_scale,
                'confidence': 0.3,
                'source': 'default_room_estimation',
                'object': None
            })
        
        return scale_candidates
    
    def select_best_scale(self, scale_candidates):
        """최적 스케일 선택"""
        if not scale_candidates:
            return 0.5  # 기본값
        
        # 신뢰도 가중 평균
        total_weight = sum(candidate['confidence'] for candidate in scale_candidates)
        
        if total_weight == 0:
            return scale_candidates[0]['scale']
        
        weighted_scale = sum(
            candidate['scale'] * candidate['confidence'] 
            for candidate in scale_candidates
        ) / total_weight
        
        return weighted_scale
    
    def detect_room_boundaries_advanced(self, processed_image, detected_objects):
        """고급 방 경계선 감지"""
        # 객체 영역 마스킹
        mask = self.create_advanced_mask(processed_image.shape, detected_objects)
        
        # 마스크 적용된 엣지
        edges = cv2.Canny(processed_image, 50, 150)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 직선 검출
        lines = cv2.HoughLinesP(
            masked_edges, 1, np.pi/180, threshold=50, 
            minLineLength=80, maxLineGap=30
        )
        
        if lines is None:
            return self.generate_default_corners(processed_image.shape)
        
        # 방 모서리 추출
        corners = self.extract_room_corners_v2(lines, processed_image.shape)
        
        return corners
    
    def create_advanced_mask(self, image_shape, detected_objects):
        """고급 마스크 생성"""
        h, w = image_shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        for obj in detected_objects:
            x, y, width, height = obj['bbox']
            # 객체 주변 여백 설정 (객체 타입별 다름)
            if obj['type'] == 'door':
                margin = 15
            elif obj['type'] == 'window':
                margin = 10
            else:
                margin = 5
            
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + width + margin)
            y2 = min(h, y + height + margin)
            
            mask[y1:y2, x1:x2] = 0
        
        return mask
    
    def extract_room_corners_v2(self, lines, image_shape):
        """방 모서리 추출 v2"""
        h, w = image_shape[:2]
        
        # 수평선/수직선 분류
        h_lines, v_lines = [], []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length > 50:
                if abs(angle) < 20 or abs(angle) > 160:
                    h_lines.append(line[0])
                elif 70 < abs(angle) < 110:
                    v_lines.append(line[0])
        
        # 경계선 선택 및 교점 계산
        corners = []
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            # 경계선들 선택
            boundaries = self.select_room_boundaries(h_lines, v_lines, w, h)
            
            # 교점 계산
            if all(b is not None for b in boundaries.values()):
                corners = self.calculate_boundary_intersections(boundaries)
        
        # 백업 방법
        if len(corners) < 4:
            corners = self.generate_default_corners(image_shape)
        
        return corners[:4]
    
    def select_room_boundaries(self, h_lines, v_lines, width, height):
        """방 경계선 선택"""
        boundaries = {}
        
        if h_lines:
            boundaries['top'] = min(h_lines, key=lambda l: (l[1] + l[3]) / 2)
            boundaries['bottom'] = max(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        
        if v_lines:
            boundaries['left'] = min(v_lines, key=lambda l: (l[0] + l[2]) / 2)
            boundaries['right'] = max(v_lines, key=lambda l: (l[0] + l[2]) / 2)
        
        return boundaries
    
    def calculate_boundary_intersections(self, boundaries):
        """경계선 교점 계산"""
        corners = []
        
        intersections = [
            ('top', 'left'),
            ('top', 'right'), 
            ('bottom', 'right'),
            ('bottom', 'left')
        ]
        
        for h_key, v_key in intersections:
            if h_key in boundaries and v_key in boundaries:
                intersection = self.line_intersection(boundaries[h_key], boundaries[v_key])
                if intersection:
                    corners.append(intersection)
        
        return corners
    
    def line_intersection(self, line1, line2):
        """두 직선 교점"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (int(x), int(y))
    
    def generate_default_corners(self, image_shape):
        """기본 모서리점 생성"""
        h, w = image_shape[:2]
        margin = min(w, h) * 0.1
        
        return [
            (int(margin), int(margin)),
            (int(w - margin), int(margin)),
            (int(w - margin), int(h - margin)),
            (int(margin), int(h - margin))
        ]
    
    def calculate_verified_dimensions(self, corners, best_scale, scale_candidates):
        """검증된 치수 계산"""
        if len(corners) < 4:
            return {'width': 0, 'height': 0, 'area': 0, 'perimeter': 0, 'confidence': 0.0}
        
        # 픽셀 거리
        tl, tr, br, bl = corners
        width_pixels = math.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
        height_pixels = math.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
        
        # 실제 크기
        width_cm = width_pixels * best_scale
        height_cm = height_pixels * best_scale
        area_m2 = (width_cm * height_cm) / 10000
        perimeter_m = (width_cm + height_cm) * 2 / 100
        
        # 신뢰도 계산
        confidence = self.calculate_comprehensive_confidence(
            corners, width_cm, height_cm, scale_candidates
        )
        
        return {
            'width': round(width_cm, 1),
            'height': round(height_cm, 1), 
            'area': round(area_m2, 2),
            'perimeter': round(perimeter_m, 2),
            'confidence': confidence
        }
    
    def calculate_comprehensive_confidence(self, corners, width_cm, height_cm, scale_candidates):
        """종합 신뢰도 계산"""
        confidence = 1.0
        
        # 1. 스케일 신뢰도
        if scale_candidates:
            avg_scale_confidence = sum(sc['confidence'] for sc in scale_candidates) / len(scale_candidates)
            confidence *= avg_scale_confidence
        
        # 2. 크기 합리성 (한국 주거 기준)
        if 200 <= width_cm <= 600 and 200 <= height_cm <= 600:
            confidence *= 1.0
        elif 150 <= width_cm <= 800 and 150 <= height_cm <= 800:
            confidence *= 0.8
        else:
            confidence *= 0.4
        
        # 3. 종횡비 합리성
        aspect_ratio = max(width_cm, height_cm) / min(width_cm, height_cm)
        if aspect_ratio <= 2.0:
            confidence *= 1.0
        elif aspect_ratio <= 3.0:
            confidence *= 0.7
        else:
            confidence *= 0.4
        
        # 4. 직사각형 형태
        rectangularity = self.check_rectangularity(corners)
        confidence *= rectangularity
        
        return max(0.1, min(1.0, confidence))
    
    def check_rectangularity(self, corners):
        """직사각형 형태 확인"""
        if len(corners) < 4:
            return 0.0
        
        angles = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = corners[(i + 2) % 4]
            
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = max(-1, min(1, dot_product / (norm1 * norm2)))
                angle = math.degrees(math.acos(cos_angle))
                angles.append(angle)
        
        rectangularity = 1.0
        for angle in angles:
            diff = abs(angle - 90)
            rectangularity *= max(0.2, 1.0 - diff / 45)
        
        return rectangularity
    
    def comprehensive_analysis(self, dimensions, detected_objects, scale_candidates):
        """종합 분석 및 대안 제시"""
        # 주 결과
        primary = dimensions
        
        # 대안 시나리오들
        alternatives = []
        
        # 각 스케일 후보별로 대안 계산
        for i, scale_candidate in enumerate(scale_candidates[:3]):  # 상위 3개만
            alt_scale = scale_candidate['scale']
            alt_width = primary['width'] * (alt_scale / (primary['width'] / 100))  # 간단 비례 계산
            alt_height = primary['height'] * (alt_scale / (primary['height'] / 100))
            
            alternatives.append({
                'scenario': f"시나리오 {i+1}",
                'source': scale_candidate['source'],
                'width': round(alt_width, 1),
                'height': round(alt_height, 1),
                'area': round(alt_width * alt_height / 10000, 2),
                'confidence': scale_candidate['confidence']
            })
        
        # 일반적인 방 크기와 비교
        room_category = self.categorize_room_size(primary['width'], primary['height'])
        
        return {
            'primary': primary,
            'alternatives': alternatives,
            'room_category': room_category,
            'confidence': primary['confidence']
        }
    
    def categorize_room_size(self, width_cm, height_cm):
        """방 크기 분류"""
        area_m2 = width_cm * height_cm / 10000
        
        if area_m2 < 10:
            return "소형 방 (10m² 미만)"
        elif area_m2 < 20:
            return "중형 방 (10-20m²)"
        elif area_m2 < 30:
            return "대형 방 (20-30m²)"
        else:
            return "특대형 방 (30m² 이상)"
    
    def absolute_visualization(self, image, corners, detected_objects, analysis_result):
        """절대 크기 시각화"""
        result = image.copy()
        
        # 1. 감지된 객체들 표시 (스케일 기준 객체 강조)
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            obj_type = obj['type']
            confidence = obj['confidence']
            
            # 객체 타입별 색상
            colors = {
                'door': (255, 0, 0),      # 빨강
                'window': (0, 255, 255),  # 노랑  
                'outlet': (255, 0, 255)   # 마젠타
            }
            
            color = colors.get(obj_type, (128, 128, 128))
            thickness = 4 if obj.get('used_for_scale') else 2
            
            # 객체 박스
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # 라벨 (실제 추정 크기 포함)
            real_size = obj['estimated_real_size']
            if obj_type == 'door':
                label = f"문 {real_size['width']}x{real_size['height']}cm"
            elif obj_type == 'window':
                label = f"창문 {real_size['width']}x{real_size['height']}cm"
            else:
                label = f"{obj_type} {confidence:.2f}"
            
            # 라벨 배경
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result, (x, y - 25), (x + text_size[0] + 10, y), color, -1)
            cv2.putText(result, label, (x + 5, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 2. 방 경계선 및 모서리점
        if len(corners) >= 4:
            pts = np.array(corners, dtype=np.int32)
            
            # 반투명 방 영역
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            result = cv2.addWeighted(result, 0.75, overlay, 0.25, 0)
            
            # 방 경계선
            cv2.polylines(result, [pts], True, (0, 255, 0), 4)
            
            # 모서리점
            for i, corner in enumerate(corners):
                cv2.circle(result, corner, 12, (255, 0, 0), -1)
                cv2.circle(result, corner, 16, (255, 255, 255), 3)
                cv2.putText(result, str(i+1), 
                           (corner[0] + 20, corner[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 3. 상세 정보 패널
        self.draw_comprehensive_info_panel(result, analysis_result)
        
        # 4. 치수 라벨
        if len(corners) >= 4:
            self.draw_absolute_dimension_labels(result, corners, analysis_result['primary'])
        
        # base64 인코딩
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{result_base64}"
    
    def draw_comprehensive_info_panel(self, image, analysis_result):
        """종합 정보 패널"""
        panel_height = 200
        panel_width = image.shape[1]
        
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel.fill(20)
        
        # 제목
        cv2.putText(panel, "Absolute Room Size Analysis", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 주 결과
        primary = analysis_result['primary']
        main_info = [
            f"Primary: {primary['width']}cm x {primary['height']}cm",
            f"Area: {primary['area']}m² | Confidence: {primary['confidence']:.1%}",
            f"Category: {analysis_result['room_category']}"
        ]
        
        for i, line in enumerate(main_info):
            cv2.putText(panel, line, (20, 55 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 대안 시나리오
        if analysis_result['alternatives']:
            cv2.putText(panel, "Alternative Scenarios:", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            for i, alt in enumerate(analysis_result['alternatives'][:2]):  # 상위 2개만
                alt_text = f"{alt['scenario']}: {alt['width']}x{alt['height']}cm ({alt['confidence']:.1%})"
                cv2.putText(panel, alt_text, (20, 145 + i * 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 패널을 이미지에 추가
        result = np.vstack([image, panel])
        image[:] = result[:image.shape[0]]
    
    def draw_absolute_dimension_labels(self, image, corners, dimensions):
        """절대 치수 라벨"""
        # 가로 치수 (상단)
        top_center = (
            (corners[0][0] + corners[1][0]) // 2,
            max(40, corners[0][1] - 50)
        )
        
        width_text = f"{dimensions['width']}cm"
        text_size = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        
        # 배경 박스
        cv2.rectangle(image, 
                     (top_center[0] - text_size[0]//2 - 10, top_center[1] - text_size[1] - 10),
                     (top_center[0] + text_size[0]//2 + 10, top_center[1] + 10),
                     (0, 0, 0), -1)
        cv2.rectangle(image, 
                     (top_center[0] - text_size[0]//2 - 10, top_center[1] - text_size[1] - 10),
                     (top_center[0] + text_size[0]//2 + 10, top_center[1] + 10),
                     (0, 255, 255), 2)
        
        cv2.putText(image, width_text, 
                   (top_center[0] - text_size[0]//2, top_center[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # 세로 치수 (좌측)
        left_center = (
            max(100, corners[0][0] - 120),
            (corners[0][1] + corners[3][1]) // 2
        )
        
        height_text = f"{dimensions['height']}cm"
        text_size = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        
        # 배경 박스
        cv2.rectangle(image, 
                     (left_center[0] - text_size[0]//2 - 10, left_center[1] - text_size[1]//2 - 10),
                     (left_center[0] + text_size[0]//2 + 10, left_center[1] + text_size[1]//2 + 10),
                     (0, 0, 0), -1)
        cv2.rectangle(image, 
                     (left_center[0] - text_size[0]//2 - 10, left_center[1] - text_size[1]//2 - 10),
                     (left_center[0] + text_size[0]//2 + 10, left_center[1] + text_size[1]//2 + 10),
                     (0, 255, 255), 2)
        
        cv2.putText(image, height_text, 
                   (left_center[0] - text_size[0]//2, left_center[1] + text_size[1]//2 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

# Flask 라우트
analyzer = AbsoluteRoomAnalyzer()

@app.route('/')
def index():
    return render_template('room_analyzer.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_room():
    try:
        data = request.get_json()
        
        image_data = data.get('image')
        reference_size = data.get('reference_size')  # 3단계에서는 무시됨
        options = data.get('options', {})
        
        if not image_data:
            return jsonify({'success': False, 'error': '이미지가 없습니다.'})
        
        logger.info("3단계 절대 크기 추정 분석 시작")
        result = analyzer.analyze_image(image_data, reference_size, options)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '3.0.0',
        'features': [
            'absolute_size_estimation', 
            'standard_object_detection', 
            'multi_scale_analysis',
            'confidence_scoring',
            'alternative_scenarios'
        ],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 3단계 절대 크기 추정 시스템을 시작합니다...")
    print("📍 http://localhost:5000 에서 접속하세요")
    print("🎯 3단계 혁신 기능:")
    print("  🏠 한국 건축 표준 기반 분석")
    print("  📏 문/창문 표준 치수 자동 매칭")
    print("  🎯 다중 스케일 후보 생성 및 검증")
    print("  📊 신뢰도 기반 최적 스케일 선택")
    print("  📋 대안 시나리오 제시")
    print("  🏆 방 크기 카테고리 분류")
    print("  ✨ 기준 크기 입력 불필요!")
    app.run(debug=True, host='0.0.0.0', port=5000)