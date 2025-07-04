# 파일명: app_step2.py
# 2단계: 객체 감지 및 스케일 보정

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

class SmartRoomAnalyzer:
    def __init__(self):
        self.reference_objects = {
            'door': {'width': 80, 'height': 200},
            'window': {'width': 120, 'height': 150},
            'outlet': {'width': 10, 'height': 10}
        }
    
    def analyze_image(self, image_data, reference_size=200, options=None):
        """스마트 분석 함수"""
        if options is None:
            options = {'detect_windows': True, 'detect_doors': True}
            
        try:
            # 이미지 디코딩
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("이미지를 디코딩할 수 없습니다.")
            
            logger.info(f"이미지 크기: {image.shape}, 기준 크기: {reference_size}cm")
            
            # 1단계: 향상된 전처리
            processed_image = self.enhanced_preprocessing(image)
            
            # 2단계: 객체 감지 (문, 창문, 콘센트)
            detected_objects = self.detect_room_objects(image)
            logger.info(f"감지된 객체: {len(detected_objects)}개")
            
            # 3단계: 스마트 스케일 계산
            smart_scale = self.calculate_smart_scale(detected_objects, reference_size, image.shape)
            logger.info(f"스마트 스케일: {smart_scale:.4f} cm/pixel")
            
            # 4단계: 개선된 방 경계선 감지
            room_corners = self.smart_room_detection(processed_image, detected_objects)
            
            # 5단계: 정확한 치수 계산
            dimensions = self.calculate_accurate_dimensions(
                room_corners, smart_scale, detected_objects
            )
            
            # 6단계: 고급 시각화
            result_image = self.smart_visualization(
                image, room_corners, detected_objects, dimensions
            )
            
            return {
                'success': True,
                'dimensions': dimensions,
                'detected_objects': detected_objects,
                'result_image': result_image,
                'analysis_info': {
                    'method': 'smart_cv_v2',
                    'confidence': dimensions.get('confidence', 0.0),
                    'scale_factor': smart_scale,
                    'objects_found': len(detected_objects),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"분석 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def enhanced_preprocessing(self, image):
        """향상된 이미지 전처리"""
        # 1. 노이즈 제거
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. 그레이스케일 변환
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # 3. 적응적 히스토그램 평활화
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 4. 가우시안 블러로 미세 노이즈 제거
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return blurred
    
    def detect_room_objects(self, image):
        """방 내 객체 감지 (문, 창문, 콘센트)"""
        detected_objects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 문 감지
        doors = self.detect_doors_advanced(gray, image)
        detected_objects.extend(doors)
        
        # 2. 창문 감지
        windows = self.detect_windows_advanced(gray, image)
        detected_objects.extend(windows)
        
        # 3. 콘센트/스위치 감지
        outlets = self.detect_outlets_advanced(gray)
        detected_objects.extend(outlets)
        
        return detected_objects
    
    def detect_doors_advanced(self, gray, color_image):
        """고급 문 감지"""
        doors = []
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        
        # 수직선 강조 (문은 보통 세로가 긴 직사각형)
        kernel_vertical = np.array([[-1, 2, -1],
                                   [-1, 2, -1],
                                   [-1, 2, -1]], dtype=np.float32)
        vertical_edges = cv2.filter2D(edges, -1, kernel_vertical)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 3000 < area < 80000:  # 문 크기 범위
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w
                
                # 문의 특징: 높이가 너비의 2-4배
                if 2.0 < aspect_ratio < 4.0:
                    # 추가 검증: 문틀 색상 확인
                    roi = color_image[y:y+h, x:x+w]
                    if self.is_door_like(roi):
                        doors.append({
                            'type': 'door',
                            'bbox': (x, y, w, h),
                            'confidence': 0.8,
                            'aspect_ratio': aspect_ratio
                        })
        
        return doors
    
    def is_door_like(self, roi):
        """문 같은 영역인지 확인"""
        if roi.size == 0:
            return False
        
        # 색상 분석: 문은 보통 갈색, 흰색, 또는 어두운 색
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 갈색 범위 (문틀)
        brown_lower = np.array([10, 50, 20])
        brown_upper = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        # 어두운 영역 (문 틈새)
        dark_mask = cv2.inRange(hsv[:,:,2], 0, 50)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        door_like_pixels = cv2.countNonZero(brown_mask) + cv2.countNonZero(dark_mask)
        
        return (door_like_pixels / total_pixels) > 0.1
    
    def detect_windows_advanced(self, gray, color_image):
        """고급 창문 감지"""
        windows = []
        
        # 밝은 영역 감지 (창문은 자연광으로 밝음)
        _, bright_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 8000 < area < 150000:  # 창문 크기 범위
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # 창문의 특징: 가로가 세로와 비슷하거나 더 긴 직사각형
                if 0.7 < aspect_ratio < 3.0:
                    # 추가 검증: 창문 프레임 확인
                    if self.is_window_like(color_image[y:y+h, x:x+w], gray[y:y+h, x:x+w]):
                        windows.append({
                            'type': 'window',
                            'bbox': (x, y, w, h),
                            'confidence': 0.7,
                            'aspect_ratio': aspect_ratio
                        })
        
        return windows
    
    def is_window_like(self, color_roi, gray_roi):
        """창문 같은 영역인지 확인"""
        if color_roi.size == 0:
            return False
        
        # 밝기 분석
        mean_brightness = np.mean(gray_roi)
        
        # 창문은 보통 밝고, 프레임이 있음
        if mean_brightness > 120:
            # 엣지 밀도 확인 (창문 프레임)
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            return edge_density > 0.05  # 적당한 엣지가 있어야 함
        
        return False
    
    def detect_outlets_advanced(self, gray):
        """고급 콘센트/스위치 감지"""
        outlets = []
        
        # 작은 직사각형 감지
        edges = cv2.Canny(gray, 100, 200)
        
        # 작은 커널로 작은 객체 강조
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 3000:  # 콘센트 크기 범위
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # 콘센트: 정사각형에 가까운 형태
                if 0.5 < aspect_ratio < 2.0:
                    outlets.append({
                        'type': 'outlet',
                        'bbox': (x, y, w, h),
                        'confidence': 0.5,
                        'aspect_ratio': aspect_ratio
                    })
        
        return outlets
    
    def calculate_smart_scale(self, detected_objects, reference_size, image_shape):
        """스마트 스케일 계산"""
        h, w = image_shape[:2]
        
        # 객체 기반 스케일 계산
        for obj in detected_objects:
            obj_type = obj['type']
            bbox = obj['bbox']
            x, y, width, height = bbox
            
            if obj_type == 'door' and obj_type in self.reference_objects:
                # 문의 높이를 기준으로 스케일 계산
                real_height = self.reference_objects['door']['height']
                pixel_height = height
                scale = real_height / pixel_height
                logger.info(f"문 기반 스케일 계산: {scale:.4f} cm/pixel")
                return scale
                
            elif obj_type == 'window' and obj_type in self.reference_objects:
                # 창문의 너비를 기준으로 스케일 계산
                real_width = self.reference_objects['window']['width']
                pixel_width = width
                scale = real_width / pixel_width
                logger.info(f"창문 기반 스케일 계산: {scale:.4f} cm/pixel")
                return scale
        
        # 객체를 찾지 못한 경우 이미지 크기 기반 추정
        # 일반적인 방 크기를 가정 (3m x 4m)
        estimated_room_width = 400  # cm
        scale = estimated_room_width / w
        logger.info(f"이미지 크기 기반 스케일 추정: {scale:.4f} cm/pixel")
        
        return scale
    
    def smart_room_detection(self, processed_image, detected_objects):
        """스마트 방 경계선 감지"""
        # 객체를 제외한 영역에서 벽면 감지
        mask = self.create_object_mask(processed_image.shape, detected_objects)
        
        # 마스크 적용된 엣지 검출
        edges = cv2.Canny(processed_image, 50, 150)
        edges = cv2.bitwise_and(edges, mask)
        
        # 강화된 직선 검출
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=60,
            minLineLength=80,
            maxLineGap=40
        )
        
        if lines is None:
            return self.fallback_corners(processed_image.shape)
        
        # 방 경계선 추출
        room_corners = self.extract_room_boundaries_smart(lines, processed_image.shape)
        
        return room_corners
    
    def create_object_mask(self, image_shape, detected_objects):
        """객체 영역을 제외한 마스크 생성"""
        h, w = image_shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # 감지된 객체 영역을 마스크에서 제외
        for obj in detected_objects:
            x, y, width, height = obj['bbox']
            # 객체 주변에 여백 추가
            margin = 10
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + width + margin)
            y2 = min(h, y + height + margin)
            
            mask[y1:y2, x1:x2] = 0
        
        return mask
    
    def extract_room_boundaries_smart(self, lines, image_shape):
        """스마트 방 경계선 추출"""
        h, w = image_shape[:2]
        
        # 직선을 수평/수직으로 분류
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length > 60:  # 최소 길이 조건
                if abs(angle) < 15 or abs(angle) > 165:
                    horizontal_lines.append(line[0])
                elif 75 < abs(angle) < 105:
                    vertical_lines.append(line[0])
        
        # 경계선 찾기 (가장자리 우선)
        corners = []
        
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # 상하좌우 경계선 선택
            top_line = self.select_boundary_line(horizontal_lines, 'top', h)
            bottom_line = self.select_boundary_line(horizontal_lines, 'bottom', h)
            left_line = self.select_boundary_line(vertical_lines, 'left', w)
            right_line = self.select_boundary_line(vertical_lines, 'right', w)
            
            # 교점 계산
            if all(line is not None for line in [top_line, bottom_line, left_line, right_line]):
                tl = self.line_intersection(top_line, left_line)
                tr = self.line_intersection(top_line, right_line)
                br = self.line_intersection(bottom_line, right_line)
                bl = self.line_intersection(bottom_line, left_line)
                
                valid_corners = [c for c in [tl, tr, br, bl] if c and self.is_valid_corner(c, w, h)]
                
                if len(valid_corners) >= 4:
                    corners = valid_corners[:4]
        
        if len(corners) < 4:
            corners = self.fallback_corners(image_shape)
        
        return corners
    
    def select_boundary_line(self, lines, position, dimension):
        """경계선 선택"""
        if not lines:
            return None
        
        if position == 'top':
            return min(lines, key=lambda l: (l[1] + l[3]) / 2)
        elif position == 'bottom':
            return max(lines, key=lambda l: (l[1] + l[3]) / 2)
        elif position == 'left':
            return min(lines, key=lambda l: (l[0] + l[2]) / 2)
        elif position == 'right':
            return max(lines, key=lambda l: (l[0] + l[2]) / 2)
        
        return lines[0]
    
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
    
    def is_valid_corner(self, corner, width, height):
        """유효한 모서리점인지 확인"""
        x, y = corner
        margin = 0.05  # 5% 여백
        
        return (width * margin <= x <= width * (1 - margin) and 
                height * margin <= y <= height * (1 - margin))
    
    def fallback_corners(self, image_shape):
        """백업 모서리점"""
        h, w = image_shape[:2]
        margin = min(w, h) * 0.1
        
        return [
            (int(margin), int(margin)),
            (int(w - margin), int(margin)),
            (int(w - margin), int(h - margin)),
            (int(margin), int(h - margin))
        ]
    
    def calculate_accurate_dimensions(self, corners, scale_factor, detected_objects):
        """정확한 치수 계산"""
        if len(corners) < 4:
            return {
                'width': 0, 'height': 0, 'area': 0, 'perimeter': 0, 'confidence': 0.0
            }
        
        # 픽셀 거리 계산
        tl, tr, br, bl = corners[:4]
        
        width_pixels = math.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
        height_pixels = math.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
        
        # 실제 크기 계산
        width_cm = width_pixels * scale_factor
        height_cm = height_pixels * scale_factor
        area_m2 = (width_cm * height_cm) / 10000
        perimeter_m = (width_cm + height_cm) * 2 / 100
        
        # 신뢰도 계산
        confidence = self.calculate_confidence_v2(corners, detected_objects, width_cm, height_cm)
        
        logger.info(f"최종 치수: {width_cm:.1f} x {height_cm:.1f}cm (신뢰도: {confidence:.2f})")
        
        return {
            'width': round(width_cm, 1),
            'height': round(height_cm, 1),
            'area': round(area_m2, 2),
            'perimeter': round(perimeter_m, 2),
            'confidence': confidence
        }
    
    def calculate_confidence_v2(self, corners, detected_objects, width_cm, height_cm):
        """향상된 신뢰도 계산"""
        confidence = 1.0
        
        # 1. 객체 감지 신뢰도
        if detected_objects:
            obj_confidence = sum(obj['confidence'] for obj in detected_objects) / len(detected_objects)
            confidence *= (0.5 + obj_confidence * 0.5)
        
        # 2. 크기 합리성 (일반적인 방 크기: 200-600cm)
        if 200 <= width_cm <= 600 and 200 <= height_cm <= 600:
            confidence *= 1.0
        elif 100 <= width_cm <= 800 and 100 <= height_cm <= 800:
            confidence *= 0.8
        else:
            confidence *= 0.4
        
        # 3. 종횡비 합리성
        aspect_ratio = max(width_cm, height_cm) / min(width_cm, height_cm)
        if aspect_ratio <= 2.0:
            confidence *= 1.0
        elif aspect_ratio <= 3.0:
            confidence *= 0.8
        else:
            confidence *= 0.5
        
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
                cos_angle = dot_product / (norm1 * norm2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.degrees(math.acos(cos_angle))
                angles.append(angle)
        
        rectangularity = 1.0
        for angle in angles:
            diff = abs(angle - 90)
            rectangularity *= max(0.2, 1.0 - diff / 45)
        
        return rectangularity
    
    def smart_visualization(self, image, corners, detected_objects, dimensions):
        """스마트 시각화"""
        result = image.copy()
        
        # 1. 감지된 객체 표시
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            obj_type = obj['type']
            confidence = obj['confidence']
            
            colors = {
                'door': (255, 0, 0),     # 빨강
                'window': (0, 255, 255), # 노랑
                'outlet': (255, 0, 255)  # 마젠타
            }
            
            color = colors.get(obj_type, (128, 128, 128))
            
            # 객체 박스
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
            
            # 라벨
            label = f"{obj_type} ({confidence:.2f})"
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 2. 방 경계선 표시
        if len(corners) >= 4:
            pts = np.array(corners, dtype=np.int32)
            
            # 반투명 영역
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            # 경계선
            cv2.polylines(result, [pts], True, (0, 255, 0), 4)
            
            # 모서리점
            for i, corner in enumerate(corners):
                cv2.circle(result, corner, 12, (255, 0, 0), -1)
                cv2.circle(result, corner, 16, (255, 255, 255), 3)
                cv2.putText(result, str(i+1), 
                           (corner[0] + 20, corner[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 3. 정보 패널 추가
        if dimensions:
            self.draw_enhanced_info_panel(result, dimensions, detected_objects)
            
            # 치수 라벨
            if len(corners) >= 4:
                self.draw_dimension_labels_v2(result, corners, dimensions)
        
        # base64 인코딩
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{result_base64}"
    
    def draw_enhanced_info_panel(self, image, dimensions, detected_objects):
        """향상된 정보 패널"""
        panel_height = 160
        panel_width = image.shape[1]
        
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel.fill(25)
        
        # 제목
        cv2.putText(panel, "Smart Analysis Results", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # 측정값
        info_lines = [
            f"Width: {dimensions['width']}cm",
            f"Height: {dimensions['height']}cm",
            f"Area: {dimensions['area']}m²",
            f"Confidence: {dimensions['confidence']:.1%}",
            f"Objects: {len(detected_objects)} detected"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(panel, line, (20, 65 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 패널을 이미지에 추가
        result = np.vstack([image, panel])
        image[:] = result[:image.shape[0]]
    
    def draw_dimension_labels_v2(self, image, corners, dimensions):
        """향상된 치수 라벨 그리기"""
        # 가로 치수 (상단)
        top_center = (
            (corners[0][0] + corners[1][0]) // 2,
            max(30, corners[0][1] - 40)
        )
        
        # 배경 사각형
        text = f"{dimensions['width']}cm"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(image, 
                     (top_center[0] - text_size[0]//2 - 5, top_center[1] - text_size[1] - 5),
                     (top_center[0] + text_size[0]//2 + 5, top_center[1] + 5),
                     (0, 0, 0), -1)
        
        cv2.putText(image, text, 
                   (top_center[0] - text_size[0]//2, top_center[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # 세로 치수 (좌측)
        left_center = (
            max(80, corners[0][0] - 100),
            (corners[0][1] + corners[3][1]) // 2
        )
        
        text = f"{dimensions['height']}cm"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.rectangle(image, 
                     (left_center[0] - text_size[0]//2 - 5, left_center[1] - text_size[1]//2 - 5),
                     (left_center[0] + text_size[0]//2 + 5, left_center[1] + text_size[1]//2 + 5),
                     (0, 0, 0), -1)
        
        cv2.putText(image, text, 
                   (left_center[0] - text_size[0]//2, left_center[1] + text_size[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

# Flask 라우트
analyzer = SmartRoomAnalyzer()

@app.route('/')
def index():
    return render_template('room_analyzer.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_room():
    try:
        data = request.get_json()
        
        image_data = data.get('image')
        reference_size = data.get('reference_size', 200)
        options = data.get('options', {})
        
        if not image_data:
            return jsonify({'success': False, 'error': '이미지가 없습니다.'})
        
        logger.info(f"2단계 분석 시작 - 기준 크기: {reference_size}cm")
        result = analyzer.analyze_image(image_data, reference_size, options)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'features': ['object_detection', 'smart_scaling', 'enhanced_visualization'],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 2단계 스마트 방 분석 서버를 시작합니다...")
    print("📍 http://localhost:5000 에서 접속하세요")
    print("✨ 2단계 새로운 기능:")
    print("  - 🚪 문 자동 감지 (높이 기준 스케일 보정)")
    print("  - 🪟 창문 자동 감지 (밝기 + 프레임 분석)")
    print("  - 🔌 콘센트/스위치 감지")
    print("  - 🧠 스마트 스케일 계산")
    print("  - 📊 향상된 신뢰도 시스템")
    print("  - 🎨 고급 시각화 (객체 라벨링)")
    app.run(debug=True, host='0.0.0.0', port=5000)