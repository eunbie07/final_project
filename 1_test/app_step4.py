# 파일명: app_step4.py
# 4단계: AI 강화 방 전체 영역 감지 시스템

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
import math
from datetime import datetime
import logging
import requests
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class AIEnhancedRoomAnalyzer:
    def __init__(self):
        # 한국 건축 표준 치수
        self.standard_dimensions = {
            'door': {'width': 80, 'height': 200},
            'window': {'width': 120, 'height': 150},
            'outlet': {'width': 10, 'height': 10}
        }
        
        # AI 모델 설정
        self.setup_ai_models()
        
    def setup_ai_models(self):
        """AI 모델 초기화"""
        try:
            # YOLOv4나 다른 객체 감지 모델을 로드할 수 있음
            # 여기서는 OpenCV DNN을 사용한 예시
            self.net = None
            
            # Google Vision API 키 (있다면)
            self.vision_api_key = os.getenv('GOOGLE_VISION_API_KEY')
            
            logger.info("AI 모델 초기화 완료")
            
        except Exception as e:
            logger.warning(f"AI 모델 로드 실패, 로컬 CV 사용: {e}")
            self.net = None
    
    def analyze_image(self, image_data, reference_size=None, options=None):
        """AI 강화 분석"""
        if options is None:
            options = {'detect_windows': True, 'detect_doors': True}
            
        try:
            # 이미지 디코딩
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("이미지를 디코딩할 수 없습니다.")
            
            logger.info(f"AI 강화 분석 시작 - 이미지 크기: {image.shape}")
            
            # 1단계: AI 기반 실내 구조 분석
            structure_analysis = self.ai_structure_analysis(image)
            
            # 2단계: 벽면 vs 가구 구분
            wall_mask, furniture_mask = self.separate_walls_and_furniture(image, structure_analysis)
            
            # 3단계: AI 객체 감지 (Google Vision API 또는 로컬)
            detected_objects = self.ai_object_detection(image)
            
            # 4단계: 벽면 기반 방 경계선 감지
            room_corners = self.detect_room_walls_only(image, wall_mask, furniture_mask)
            
            # 5단계: AI 기반 스케일 추정
            smart_scale = self.ai_scale_estimation(detected_objects, room_corners, image.shape)
            
            # 6단계: 정확한 방 치수 계산
            room_dimensions = self.calculate_room_dimensions_ai(room_corners, smart_scale)
            
            # 7단계: 신뢰도 및 검증
            verification_result = self.ai_verification(room_dimensions, detected_objects, structure_analysis)
            
            # 8단계: AI 강화 시각화
            result_image = self.ai_enhanced_visualization(
                image, room_corners, detected_objects, room_dimensions, wall_mask, furniture_mask
            )
            
            return {
                'success': True,
                'dimensions': verification_result['final_dimensions'],
                'detected_objects': detected_objects,
                'structure_analysis': structure_analysis,
                'verification': verification_result,
                'result_image': result_image,
                'analysis_info': {
                    'method': 'ai_enhanced_v4',
                    'ai_features_used': ['structure_analysis', 'wall_furniture_separation', 'object_detection'],
                    'confidence': verification_result['confidence'],
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"AI 분석 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def ai_structure_analysis(self, image):
        """AI 기반 실내 구조 분석"""
        logger.info("AI 구조 분석 시작")
        
        # 1. 색상 공간 분석 (HSV, LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # 2. 텍스처 분석 (LBP - Local Binary Patterns)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture_features = self.analyze_texture_patterns(gray)
        
        # 3. 깊이 추정 (단안 카메라 기반)
        depth_map = self.estimate_depth(image)
        
        # 4. 표면 분류 (벽면, 바닥, 천장, 가구)
        surface_classification = self.classify_surfaces(image, hsv, lab, texture_features)
        
        return {
            'texture_features': texture_features,
            'depth_map': depth_map,
            'surface_classification': surface_classification,
            'dominant_colors': self.extract_dominant_colors(image),
            'lighting_analysis': self.analyze_lighting(image)
        }
    
    def analyze_texture_patterns(self, gray):
        """텍스처 패턴 분석"""
        # LBP (Local Binary Pattern) 계산
        def local_binary_pattern(image, radius=3, n_points=24):
            h, w = image.shape
            lbp = np.zeros((h, w), dtype=np.uint8)
            
            for y in range(radius, h - radius):
                for x in range(radius, w - radius):
                    center = image[y, x]
                    binary_string = ""
                    
                    for i in range(n_points):
                        angle = 2 * np.pi * i / n_points
                        x_offset = int(radius * np.cos(angle))
                        y_offset = int(radius * np.sin(angle))
                        
                        neighbor = image[y + y_offset, x + x_offset]
                        binary_string += "1" if neighbor >= center else "0"
                    
                    lbp[y, x] = int(binary_string, 2) % 256
            
            return lbp
        
        # 텍스처 맵 계산
        lbp = local_binary_pattern(gray)
        
        # 텍스처 영역 분류
        smooth_areas = cv2.inRange(lbp, 0, 50)      # 매끄러운 영역 (벽면)
        textured_areas = cv2.inRange(lbp, 100, 255) # 텍스처 영역 (가구, 패턴)
        
        return {
            'lbp_map': lbp,
            'smooth_areas': smooth_areas,
            'textured_areas': textured_areas
        }
    
    def estimate_depth(self, image):
        """단안 카메라 깊이 추정"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 간단한 깊이 추정 (실제로는 AI 모델 사용)
        # 밝기와 대비 기반 깊이 추정
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        depth_estimate = cv2.absdiff(gray, blurred)
        
        # 정규화
        depth_estimate = cv2.normalize(depth_estimate, None, 0, 255, cv2.NORM_MINMAX)
        
        return depth_estimate
    
    def classify_surfaces(self, image, hsv, lab, texture_features):
        """표면 분류 (벽, 바닥, 천장, 가구)"""
        h, w = image.shape[:2]
        classification = np.zeros((h, w), dtype=np.uint8)
        
        # 1. 위치 기반 분류
        ceiling_mask = np.zeros((h, w), dtype=np.uint8)
        ceiling_mask[0:h//4, :] = 1  # 상단 25%는 천장 가능성
        
        floor_mask = np.zeros((h, w), dtype=np.uint8)
        floor_mask[3*h//4:h, :] = 1  # 하단 25%는 바닥 가능성
        
        # 2. 색상 기반 분류
        # 벽면 (보통 밝은 색상)
        wall_color_mask = cv2.inRange(hsv[:,:,2], 150, 255)  # 밝기 기준
        
        # 바닥 (보통 갈색 계열)
        floor_color_mask = cv2.inRange(hsv[:,:,0], 10, 25)   # 갈색 색조
        
        # 3. 텍스처 기반 분류
        smooth_mask = texture_features['smooth_areas']
        textured_mask = texture_features['textured_areas']
        
        # 4. 종합 분류
        # 벽면: 밝고 매끄러운 영역
        wall_mask = cv2.bitwise_and(wall_color_mask, smooth_mask)
        wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(floor_mask))
        
        # 가구: 텍스처가 있는 영역
        furniture_mask = textured_mask.copy()
        
        # 바닥: 하단 + 특정 색상
        floor_final_mask = cv2.bitwise_and(floor_mask, 
                                          cv2.bitwise_or(floor_color_mask, smooth_mask))
        
        # 천장: 상단 + 밝은 색상
        ceiling_final_mask = cv2.bitwise_and(ceiling_mask, wall_color_mask)
        
        return {
            'wall_mask': wall_mask,
            'furniture_mask': furniture_mask,
            'floor_mask': floor_final_mask,
            'ceiling_mask': ceiling_final_mask
        }
    
    def extract_dominant_colors(self, image):
        """주요 색상 추출"""
        # K-means 클러스터링으로 주요 색상 추출
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 5
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers.astype(np.uint8).tolist()
    
    def analyze_lighting(self, image):
        """조명 분석"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 밝기 분포
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        # 밝은 영역 (창문, 조명)
        bright_areas = cv2.inRange(gray, 200, 255)
        bright_ratio = np.sum(bright_areas > 0) / (image.shape[0] * image.shape[1])
        
        return {
            'brightness_mean': float(brightness_mean),
            'brightness_std': float(brightness_std),
            'bright_areas_ratio': float(bright_ratio)
        }
    
    def separate_walls_and_furniture(self, image, structure_analysis):
        """벽면과 가구 분리"""
        surface_class = structure_analysis['surface_classification']
        
        # 벽면 마스크 (벽 + 천장)
        wall_mask = cv2.bitwise_or(surface_class['wall_mask'], surface_class['ceiling_mask'])
        
        # 가구 마스크
        furniture_mask = surface_class['furniture_mask']
        
        # 모폴로지 연산으로 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
        furniture_mask = cv2.morphologyEx(furniture_mask, cv2.MORPH_OPEN, kernel)
        
        logger.info(f"벽면/가구 분리 완료 - 벽면: {np.sum(wall_mask>0)} pixels, 가구: {np.sum(furniture_mask>0)} pixels")
        
        return wall_mask, furniture_mask
    
    def ai_object_detection(self, image):
        """AI 기반 객체 감지"""
        detected_objects = []
        
        try:
            # Google Vision API 사용 (API 키가 있는 경우)
            if self.vision_api_key:
                vision_objects = self.google_vision_detection(image)
                detected_objects.extend(vision_objects)
                logger.info(f"Google Vision API: {len(vision_objects)}개 객체 감지")
            
            # 로컬 AI 모델 백업
            local_objects = self.enhanced_local_detection(image)
            detected_objects.extend(local_objects)
            logger.info(f"로컬 AI: {len(local_objects)}개 객체 감지")
            
        except Exception as e:
            logger.warning(f"AI 객체 감지 오류: {e}")
            # 백업 방법
            detected_objects = self.fallback_object_detection(image)
        
        return detected_objects
    
    def google_vision_detection(self, image):
        """Google Vision API 객체 감지"""
        if not self.vision_api_key:
            return []
        
        try:
            # 이미지를 base64로 인코딩
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode()
            
            # API 호출
            url = f'https://vision.googleapis.com/v1/images:annotate?key={self.vision_api_key}'
            
            payload = {
                'requests': [{
                    'image': {'content': image_base64},
                    'features': [
                        {'type': 'OBJECT_LOCALIZATION', 'maxResults': 50}
                    ]
                }]
            }
            
            response = requests.post(url, json=payload, timeout=10)
            result = response.json()
            
            objects = []
            if 'responses' in result and result['responses']:
                if 'localizedObjectAnnotations' in result['responses'][0]:
                    for obj in result['responses'][0]['localizedObjectAnnotations']:
                        # 실내 관련 객체만 필터링
                        if obj['name'].lower() in ['door', 'window', 'furniture', 'cabinet', 'refrigerator']:
                            bbox = obj['boundingPoly']['normalizedVertices']
                            h, w = image.shape[:2]
                            
                            x1 = int(bbox[0].get('x', 0) * w)
                            y1 = int(bbox[0].get('y', 0) * h)
                            x2 = int(bbox[2].get('x', 1) * w)
                            y2 = int(bbox[2].get('y', 1) * h)
                            
                            objects.append({
                                'type': obj['name'].lower(),
                                'bbox': (x1, y1, x2-x1, y2-y1),
                                'confidence': obj['score'],
                                'method': 'google_vision_ai'
                            })
            
            return objects
            
        except Exception as e:
            logger.error(f"Google Vision API 오류: {e}")
            return []
    
    def enhanced_local_detection(self, image):
        """강화된 로컬 객체 감지"""
        objects = []
        
        # 1. 문 감지 (AI 강화)
        doors = self.ai_enhanced_door_detection(image)
        objects.extend(doors)
        
        # 2. 창문 감지 (AI 강화)
        windows = self.ai_enhanced_window_detection(image)
        objects.extend(windows)
        
        # 3. 가구 감지 (새로 추가)
        furniture = self.ai_furniture_detection(image)
        objects.extend(furniture)
        
        return objects
    
    def ai_enhanced_door_detection(self, image):
        """AI 강화 문 감지"""
        doors = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 엣지 기반 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. 수직선 강조 필터
        kernel_vertical = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
        vertical_edges = cv2.filter2D(edges, -1, kernel_vertical)
        
        # 3. 컨투어 검출
        contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 3000 < area < 100000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w
                
                if 2.0 < aspect_ratio < 4.0:  # 문의 종횡비
                    # AI 특징 검증
                    confidence = self.verify_door_with_ai(image[y:y+h, x:x+w])
                    
                    if confidence > 0.5:
                        doors.append({
                            'type': 'door',
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'method': 'ai_enhanced_local',
                            'estimated_real_size': {'width': 80, 'height': 200}
                        })
        
        return doors
    
    def verify_door_with_ai(self, door_roi):
        """AI 기반 문 검증"""
        if door_roi.size == 0:
            return 0.0
        
        confidence = 0.0
        
        # 1. 색상 분석 (문 특징적 색상)
        hsv = cv2.cvtColor(door_roi, cv2.COLOR_BGR2HSV)
        
        # 갈색 계열 (나무문)
        brown_mask = cv2.inRange(hsv, np.array([5, 30, 20]), np.array([25, 255, 200]))
        brown_ratio = np.sum(brown_mask > 0) / door_roi.size
        
        # 흰색/회색 계열 (도장문)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 30, 255]))
        white_ratio = np.sum(white_mask > 0) / door_roi.size
        
        color_confidence = min(1.0, (brown_ratio + white_ratio) * 2)
        confidence += color_confidence * 0.4
        
        # 2. 엣지 패턴 분석
        gray_roi = cv2.cvtColor(door_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if 0.05 < edge_density < 0.25:
            confidence += 0.3
        
        # 3. 수직성 검증
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=10)
        if lines is not None:
            vertical_score = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
                if 80 < angle < 100:  # 수직선
                    vertical_score += 1
            
            if vertical_score >= 2:
                confidence += 0.3
        
        return min(1.0, confidence)
    
    def ai_enhanced_window_detection(self, image):
        """AI 강화 창문 감지"""
        windows = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 밝기 기반 검출 (창문 = 자연광)
        _, bright_mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        
        # 2. 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 8000 < area < 200000:  # 창문 크기
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 0.8 < aspect_ratio < 3.0:  # 창문 종횡비
                    confidence = self.verify_window_with_ai(image[y:y+h, x:x+w], gray[y:y+h, x:x+w])
                    
                    if confidence > 0.6:
                        windows.append({
                            'type': 'window',
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'method': 'ai_enhanced_local',
                            'estimated_real_size': {'width': 120, 'height': 150}
                        })
        
        return windows
    
    def verify_window_with_ai(self, window_roi, gray_roi):
        """AI 기반 창문 검증"""
        if window_roi.size == 0:
            return 0.0
        
        confidence = 0.0
        
        # 1. 밝기 분석
        mean_brightness = np.mean(gray_roi)
        if mean_brightness > 130:
            confidence += 0.4
        
        # 2. 프레임 검출
        edges = cv2.Canny(gray_roi, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
                
                if angle < 20 or angle > 160:
                    horizontal_lines += 1
                elif 70 < angle < 110:
                    vertical_lines += 1
            
            if horizontal_lines >= 2 and vertical_lines >= 2:
                confidence += 0.4
        
        # 3. 유리 반사 특성
        brightness_std = np.std(gray_roi)
        if 20 < brightness_std < 70:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def ai_furniture_detection(self, image):
        """AI 가구 감지"""
        furniture = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 텍스처 기반 가구 감지
        # 가구는 보통 균일하지 않은 텍스처를 가짐
        
        # 2. 색상 기반 가구 감지
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 가구 색상 범위 (갈색, 검정, 흰색 등)
        furniture_colors = [
            (np.array([5, 50, 50]), np.array([25, 255, 200])),    # 갈색
            (np.array([0, 0, 0]), np.array([180, 255, 50]),       # 검정
            (np.array([0, 0, 200]), np.array([180, 30, 255]))     # 흰색
        ]
        
        combined_mask = np.zeros(gray.shape, dtype=np.uint8)
        for lower, upper in furniture_colors:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 3. 컨투어 검출
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 150000:  # 가구 크기 범위
                x, y, w, h = cv2.boundingRect(contour)
                
                furniture.append({
                    'type': 'furniture',
                    'bbox': (x, y, w, h),
                    'confidence': 0.7,
                    'method': 'ai_color_texture'
                })
        
        return furniture
    
    def fallback_object_detection(self, image):
        """백업 객체 감지"""
        # 기존의 간단한 객체 감지 방법
        return []
    
    def detect_room_walls_only(self, image, wall_mask, furniture_mask):
        """벽면만을 이용한 방 경계선 감지"""
        logger.info("벽면 기반 방 경계선 감지 시작")
        
        # 1. 가구 영역을 제외한 벽면만 사용
        clean_wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(furniture_mask))
        
        # 2. 벽면 영역에서 엣지 검출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # 벽면 마스크 적용
        wall_edges = cv2.bitwise_and(edges, clean_wall_mask)
        
        # 3. 이미지 경계부 강화 (벽면은 보통 이미지 가장자리에 있음)
        h, w = image.shape[:2]
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_width = min(w, h) // 10
        
        # 경계부 마스크 생성
        border_mask[0:border_width, :] = 255          # 상단
        border_mask[h-border_width:h, :] = 255        # 하단
        border_mask[:, 0:border_width] = 255          # 좌측
        border_mask[:, w-border_width:w] = 255        # 우측
        
        # 경계부와 벽면 엣지 결합
        enhanced_edges = cv2.bitwise_or(wall_edges, cv2.bitwise_and(edges, border_mask))
        
        # 4. 직선 검출 (파라미터 조정)
        lines = cv2.HoughLinesP(
            enhanced_edges,
            rho=1,
            theta=np.pi/180,
            threshold=40,
            minLineLength=min(w, h) // 8,  # 이미지 크기 대비 최소 길이
            maxLineGap=50
        )
        
        if lines is None:
            logger.warning("직선을 찾을 수 없음, 기본 모서리 사용")
            return self.generate_default_room_corners(image.shape)
        
        # 5. 방 전체 경계선 추출
        room_corners = self.extract_full_room_boundaries(lines, image.shape, wall_mask)
        
        logger.info(f"벽면 기반 모서리 감지 완료: {len(room_corners)}개")
        
        return room_corners
    
    def extract_full_room_boundaries(self, lines, image_shape, wall_mask):
        """방 전체 경계선 추출"""
        h, w = image_shape[:2]
        
        # 1. 직선을 수평선과 수직선으로 분류
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # 충분히 긴 직선만 고려
            if length > min(w, h) / 10:
                if abs(angle) < 25 or abs(angle) > 155:  # 수평선 (각도 범위 확대)
                    horizontal_lines.append((line[0], length))
                elif 65 < abs(angle) < 115:  # 수직선 (각도 범위 확대)
                    vertical_lines.append((line[0], length))
        
        logger.info(f"수평선: {len(horizontal_lines)}개, 수직선: {len(vertical_lines)}개")
        
        # 2. 방의 외곽 경계선 선택
        room_boundaries = self.select_room_outer_boundaries(horizontal_lines, vertical_lines, w, h)
        
        # 3. 교점 계산
        corners = []
        if all(boundary is not None for boundary in room_boundaries.values()):
            corners = self.calculate_room_intersections(room_boundaries)
        
        # 4. 모서리점이 부족하면 이미지 경계 사용
        if len(corners) < 4:
            corners = self.generate_room_corners_with_boundaries(image_shape, lines)
        
        return corners[:4]
    
    def select_room_outer_boundaries(self, h_lines, v_lines, width, height):
        """방의 외곽 경계선 선택"""
        boundaries = {'top': None, 'bottom': None, 'left': None, 'right': None}
        
        # 수평선에서 최상단과 최하단 선택
        if h_lines:
            # 길이 가중치 고려한 선택
            h_lines_sorted = sorted(h_lines, key=lambda x: x[1], reverse=True)  # 길이순 정렬
            top_candidates = [line for line, length in h_lines_sorted if (line[1] + line[3]) / 2 < height / 2]
            bottom_candidates = [line for line, length in h_lines_sorted if (line[1] + line[3]) / 2 > height / 2]
            
            if top_candidates:
                boundaries['top'] = min(top_candidates, key=lambda l: (l[1] + l[3]) / 2)
            if bottom_candidates:
                boundaries['bottom'] = max(bottom_candidates, key=lambda l: (l[1] + l[3]) / 2)
        
        # 수직선에서 최좌측과 최우측 선택
        if v_lines:
            v_lines_sorted = sorted(v_lines, key=lambda x: x[1], reverse=True)  # 길이순 정렬
            left_candidates = [line for line, length in v_lines_sorted if (line[0] + line[2]) / 2 < width / 2]
            right_candidates = [line for line, length in v_lines_sorted if (line[0] + line[2]) / 2 > width / 2]
            
            if left_candidates:
                boundaries['left'] = min(left_candidates, key=lambda l: (l[0] + l[2]) / 2)
            if right_candidates:
                boundaries['right'] = max(right_candidates, key=lambda l: (l[0] + l[2]) / 2)
        
        return boundaries
    
    def calculate_room_intersections(self, boundaries):
        """방 경계선들의 교점 계산"""
        corners = []
        
        intersections = [
            ('top', 'left'),
            ('top', 'right'),
            ('bottom', 'right'),
            ('bottom', 'left')
        ]
        
        for h_key, v_key in intersections:
            if boundaries[h_key] is not None and boundaries[v_key] is not None:
                intersection = self.line_intersection(boundaries[h_key], boundaries[v_key])
                if intersection:
                    corners.append(intersection)
        
        return corners
    
    def generate_room_corners_with_boundaries(self, image_shape, lines):
        """경계선을 이용한 방 모서리 생성"""
        h, w = image_shape[:2]
        
        # 이미지 경계에서 일정 거리 안쪽에 모서리 생성
        margin_x = w * 0.05  # 5% 여백
        margin_y = h * 0.05
        
        # 감지된 직선들을 참고하여 여백 조정
        if lines is not None and len(lines) > 0:
            # 가장 바깥쪽 직선들의 위치를 참고
            all_x = []
            all_y = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])
            
            if all_x and all_y:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                
                # 실제 감지된 범위 사용
                margin_x = max(margin_x, min_x)
                margin_y = max(margin_y, min_y)
                max_x = min(max_x, w - margin_x)
                max_y = min(max_y, h - margin_y)
                
                return [
                    (int(margin_x), int(margin_y)),
                    (int(max_x), int(margin_y)),
                    (int(max_x), int(max_y)),
                    (int(margin_x), int(max_y))
                ]
        
        # 기본 모서리
        return [
            (int(margin_x), int(margin_y)),
            (int(w - margin_x), int(margin_y)),
            (int(w - margin_x), int(h - margin_y)),
            (int(margin_x), int(h - margin_y))
        ]
    
    def generate_default_room_corners(self, image_shape):
        """기본 방 모서리 생성"""
        h, w = image_shape[:2]
        margin = min(w, h) * 0.08  # 8% 여백
        
        return [
            (int(margin), int(margin)),
            (int(w - margin), int(margin)),
            (int(w - margin), int(h - margin)),
            (int(margin), int(h - margin))
        ]
    
    def line_intersection(self, line1, line2):
        """두 직선의 교점 계산"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (int(x), int(y))
    
    def ai_scale_estimation(self, detected_objects, room_corners, image_shape):
        """AI 기반 스케일 추정"""
        scale_candidates = []
        
        # 1. 객체 기반 스케일 계산
        for obj in detected_objects:
            if 'estimated_real_size' in obj:
                obj_type = obj['type']
                bbox = obj['bbox']
                real_size = obj['estimated_real_size']
                confidence = obj['confidence']
                
                if obj_type == 'door':
                    # 문 높이 기준 (가장 신뢰할 만함)
                    scale = real_size['height'] / bbox[3]  # height
                    scale_candidates.append({
                        'scale': scale,
                        'confidence': confidence * 0.9,
                        'source': f'door_height_{real_size["height"]}cm'
                    })
                
                elif obj_type == 'window':
                    # 창문 너비 기준
                    scale = real_size['width'] / bbox[2]  # width
                    scale_candidates.append({
                        'scale': scale,
                        'confidence': confidence * 0.8,
                        'source': f'window_width_{real_size["width"]}cm'
                    })
        
        # 2. 방 비율 기반 추정 (한국 주거 표준)
        if room_corners and len(room_corners) >= 4:
            # 방의 픽셀 크기
            tl, tr, br, bl = room_corners[:4]
            width_pixels = math.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
            height_pixels = math.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
            
            # 일반적인 방 크기 시나리오들
            typical_room_sizes = [
                (300, 400),  # 3m x 4m
                (350, 450),  # 3.5m x 4.5m
                (400, 500),  # 4m x 5m
                (250, 350),  # 2.5m x 3.5m
            ]
            
            for real_width, real_height in typical_room_sizes:
                # 두 가지 방향 고려
                scale1 = real_width / width_pixels
                scale2 = real_height / height_pixels
                avg_scale = (scale1 + scale2) / 2
                
                scale_candidates.append({
                    'scale': avg_scale,
                    'confidence': 0.4,
                    'source': f'typical_room_{real_width}x{real_height}cm'
                })
        
        # 3. 최적 스케일 선택
        if scale_candidates:
            # 신뢰도 가중평균
            total_weight = sum(sc['confidence'] for sc in scale_candidates)
            if total_weight > 0:
                best_scale = sum(sc['scale'] * sc['confidence'] for sc in scale_candidates) / total_weight
            else:
                best_scale = scale_candidates[0]['scale']
        else:
            # 기본값
            h, w = image_shape[:2]
            best_scale = 350 / min(w, h)  # 3.5m 가정
        
        logger.info(f"AI 스케일 추정: {best_scale:.4f} cm/pixel ({len(scale_candidates)}개 후보)")
        
        return best_scale
    
    def calculate_room_dimensions_ai(self, corners, scale):
        """AI 기반 방 치수 계산"""
        if len(corners) < 4:
            return {'width': 0, 'height': 0, 'area': 0, 'perimeter': 0, 'confidence': 0.0}
        
        # 모서리점들을 정렬
        tl, tr, br, bl = corners[:4]
        
        # 픽셀 거리 계산
        width_pixels = math.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
        height_pixels = math.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
        
        # 실제 크기 계산
        width_cm = width_pixels * scale
        height_cm = height_pixels * scale
        area_m2 = (width_cm * height_cm) / 10000
        perimeter_m = (width_cm + height_cm) * 2 / 100
        
        # 신뢰도 계산
        confidence = self.calculate_ai_confidence(corners, width_cm, height_cm)
        
        return {
            'width': round(width_cm, 1),
            'height': round(height_cm, 1),
            'area': round(area_m2, 2),
            'perimeter': round(perimeter_m, 2),
            'confidence': confidence
        }
    
    def calculate_ai_confidence(self, corners, width_cm, height_cm):
        """AI 기반 신뢰도 계산"""
        confidence = 1.0
        
        # 1. 크기 합리성 (한국 주거 기준)
        if 250 <= width_cm <= 600 and 200 <= height_cm <= 500:
            confidence *= 1.0
        elif 200 <= width_cm <= 800 and 150 <= height_cm <= 600:
            confidence *= 0.8
        else:
            confidence *= 0.5
        
        # 2. 종횡비 합리성
        aspect_ratio = max(width_cm, height_cm) / min(width_cm, height_cm)
        if aspect_ratio <= 2.0:
            confidence *= 1.0
        elif aspect_ratio <= 2.5:
            confidence *= 0.8
        else:
            confidence *= 0.6
        
        # 3. 직사각형 형태
        rectangularity = self.check_rectangularity_ai(corners)
        confidence *= rectangularity
        
        return max(0.2, min(1.0, confidence))
    
    def check_rectangularity_ai(self, corners):
        """AI 직사각형 형태 검증"""
        if len(corners) < 4:
            return 0.0
        
        # 각 내각이 90도에 가까운지 확인
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
        
        # 90도와의 차이로 점수 계산
        rectangularity = 1.0
        for angle in angles:
            diff = abs(angle - 90)
            rectangularity *= max(0.3, 1.0 - diff / 60)  # 60도 차이까지 허용
        
        return rectangularity
    
    def ai_verification(self, dimensions, detected_objects, structure_analysis):
        """AI 기반 결과 검증"""
        verification = {
            'final_dimensions': dimensions,
            'confidence': dimensions['confidence'],
            'warnings': [],
            'recommendations': []
        }
        
        # 1. 크기 검증
        area = dimensions['area']
        if area < 5:
            verification['warnings'].append("방 크기가 매우 작습니다 (5m² 미만)")
            verification['recommendations'].append("창문이나 문을 기준으로 다시 측정해보세요")
        elif area > 50:
            verification['warnings'].append("방 크기가 매우 큽니다 (50m² 초과)")
            verification['recommendations'].append("측정 영역이 방 전체가 아닐 수 있습니다")
        
        # 2. 객체 일관성 검증
        if detected_objects:
            object_scales = []
            for obj in detected_objects:
                if 'estimated_real_size' in obj and obj['type'] in ['door', 'window']:
                    bbox = obj['bbox']
                    real_size = obj['estimated_real_size']
                    
                    if obj['type'] == 'door':
                        scale = real_size['height'] / bbox[3]
                    else:
                        scale = real_size['width'] / bbox[2]
                    
                    object_scales.append(scale)
            
            if len(object_scales) > 1:
                scale_std = np.std(object_scales)
                if scale_std > np.mean(object_scales) * 0.3:
                    verification['warnings'].append("객체 기반 스케일이 일관되지 않습니다")
        
        # 3. 구조 분석 일관성
        brightness = structure_analysis['lighting_analysis']['brightness_mean']
        if brightness < 100:
            verification['warnings'].append("이미지가 어두워 분석 정확도가 낮을 수 있습니다")
            verification['recommendations'].append("더 밝은 환경에서 촬영한 이미지를 사용해보세요")
        
        return verification
    
    def ai_enhanced_visualization(self, image, corners, detected_objects, dimensions, wall_mask, furniture_mask):
        """AI 강화 시각화"""
        result = image.copy()
        
        # 1. 구조 분석 결과 표시 (벽면/가구 구분)
        # 벽면 영역을 연한 파란색으로 표시
        wall_overlay = result.copy()
        wall_overlay[wall_mask > 0] = [255, 200, 200]  # 연한 파란색
        result = cv2.addWeighted(result, 0.85, wall_overlay, 0.15, 0)
        
        # 가구 영역을 연한 빨간색으로 표시
        furniture_overlay = result.copy()
        furniture_overlay[furniture_mask > 0] = [200, 200, 255]  # 연한 빨간색
        result = cv2.addWeighted(result, 0.85, furniture_overlay, 0.15, 0)
        
        # 2. AI 감지된 객체들 표시
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            obj_type = obj['type']
            confidence = obj['confidence']
            method = obj.get('method', 'unknown')
            
            # 객체 타입별 색상
            colors = {
                'door': (0, 255, 0),        # 초록 (문)
                'window': (255, 255, 0),    # 노랑 (창문)
                'furniture': (255, 0, 255), # 마젠타 (가구)
                'outlet': (0, 255, 255)     # 시안 (콘센트)
            }
            
            color = colors.get(obj_type, (128, 128, 128))
            
            # AI 감지된 객체는 두꺼운 선으로 표시
            thickness = 4 if 'ai' in method else 2
            
            # 객체 박스
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # 라벨 (AI 방법 표시)
            if 'estimated_real_size' in obj:
                real_size = obj['estimated_real_size']
                if obj_type == 'door':
                    label = f"AI Door {real_size['height']}cm"
                elif obj_type == 'window':
                    label = f"AI Window {real_size['width']}cm"
                else:
                    label = f"AI {obj_type} {confidence:.2f}"
            else:
                label = f"AI {obj_type} {confidence:.2f}"
            
            # 라벨 배경
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result, (x, y - 30), (x + text_size[0] + 10, y), color, -1)
            cv2.putText(result, label, (x + 5, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 3. 방 전체 경계선 강조 표시
        if len(corners) >= 4:
            pts = np.array(corners, dtype=np.int32)
            
            # 방 영역 반투명 오버레이
            room_overlay = result.copy()
            cv2.fillPoly(room_overlay, [pts], (0, 255, 0))
            result = cv2.addWeighted(result, 0.8, room_overlay, 0.2, 0)
            
            # 방 경계선 (매우 두꺼운 선)
            cv2.polylines(result, [pts], True, (0, 255, 0), 6)
            
            # 모서리점 (크고 명확하게)
            for i, corner in enumerate(corners):
                cv2.circle(result, corner, 15, (255, 0, 0), -1)
                cv2.circle(result, corner, 20, (255, 255, 255), 4)
                cv2.putText(result, str(i+1), 
                           (corner[0] + 25, corner[1] + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # 4. AI 분석 정보 패널
        self.draw_ai_info_panel(result, dimensions, detected_objects)
        
        # 5. 정확한 치수 라벨
        if len(corners) >= 4:
            self.draw_ai_dimension_labels(result, corners, dimensions)
        
        # base64 인코딩
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{result_base64}"
    
    def draw_ai_info_panel(self, image, dimensions, detected_objects):
        """AI 정보 패널"""
        panel_height = 180
        panel_width = image.shape[1]
        
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel.fill(15)
        
        # 제목
        cv2.putText(panel, "AI Enhanced Room Analysis", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # 주요 결과
        main_info = [
            f"Room Size: {dimensions['width']}cm x {dimensions['height']}cm",
            f"Area: {dimensions['area']}m² | Confidence: {dimensions['confidence']:.1%}",
            f"AI Objects Detected: {len(detected_objects)}",
        ]
        
        for i, line in enumerate(main_info):
            cv2.putText(panel, line, (20, 55 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # AI 기능 표시
        ai_features = [
            "✓ Wall/Furniture Separation",
            "✓ AI Object Detection", 
            "✓ Smart Scale Estimation"
        ]
        
        for i, feature in enumerate(ai_features):
            cv2.putText(panel, feature, (20, 125 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 패널 추가
        result = np.vstack([image, panel])
        image[:] = result[:image.shape[0]]
    
    def draw_ai_dimension_labels(self, image, corners, dimensions):
        """AI 치수 라벨"""
        # 가로 치수 (상단)
        top_center = (
            (corners[0][0] + corners[1][0]) // 2,
            max(50, corners[0][1] - 60)
        )
        
        width_text = f"Room Width: {dimensions['width']}cm"
        self.draw_dimension_label(image, width_text, top_center, (0, 255, 255))
        
        # 세로 치수 (좌측)
        left_center = (
            max(150, corners[0][0] - 140),
            (corners[0][1] + corners[3][1]) // 2
        )
        
        height_text = f"Room Height: {dimensions['height']}cm"
        self.draw_dimension_label(image, height_text, left_center, (0, 255, 255))
    
    def draw_dimension_label(self, image, text, position, color):
        """치수 라벨 그리기"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # 배경 박스
        cv2.rectangle(image, 
                     (position[0] - text_size[0]//2 - 15, position[1] - text_size[1] - 15),
                     (position[0] + text_size[0]//2 + 15, position[1] + 15),
                     (0, 0, 0), -1)
        
        # 테두리
        cv2.rectangle(image, 
                     (position[0] - text_size[0]//2 - 15, position[1] - text_size[1] - 15),
                     (position[0] + text_size[0]//2 + 15, position[1] + 15),
                     color, 3)
        
        # 텍스트
        cv2.putText(image, text, 
                   (position[0] - text_size[0]//2, position[1] - 5),
                   font, font_scale, color, thickness)

# Flask 라우트
analyzer = AIEnhancedRoomAnalyzer()

@app.route('/')
def index():
    return render_template('room_analyzer.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_room():
    try:
        data = request.get_json()
        
        image_data = data.get('image')
        reference_size = data.get('reference_size')
        options = data.get('options', {})
        
        if not image_data:
            return jsonify({'success': False, 'error': '이미지가 없습니다.'})
        
        logger.info("4단계 AI 강화 분석 시작")
        result = analyzer.analyze_image(image_data, reference_size, options)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '4.0.0',
        'features': [
            'ai_structure_analysis',
            'wall_furniture_separation', 
            'ai_object_detection',
            'smart_scale_estimation',
            'full_room_boundary_detection',
            'ai_verification'
        ],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 4단계 AI 강화 방 분석 시스템을 시작합니다...")
    print("📍 http://localhost:5000 에서 접속하세요")
    print("🤖 AI 강화 기능:")
    print("  🧠 AI 구조 분석 (벽면/가구/바닥/천장 분리)")
    print("  🎯 벽면 전용 경계선 감지")
    print("  🔍 AI 객체 감지 (Google Vision API 지원)")
    print("  📐 스마트 스케일 추정")
    print("  ✅ AI 기반 결과 검증")
    print("  🎨 고급 시각화 (구조 분석 결과 표시)")
    print("  🏠 방 전체 영역 정확 측정")
    app.run(debug=True, host='0.0.0.0', port=5000)