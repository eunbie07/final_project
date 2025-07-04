# 파일명: app_step4_fixed.py
# 4단계 수정된 버전 - 문법 오류 해결

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

class FixedRoomAnalyzer:
    def __init__(self):
        self.standard_dimensions = {
            'door': {'width': 80, 'height': 200},
            'window': {'width': 120, 'height': 150},
            'outlet': {'width': 10, 'height': 10}
        }
    
    def analyze_image(self, image_data, reference_size=200, options=None):
        """고정된 분석 함수"""
        if options is None:
            options = {'detect_windows': True, 'detect_doors': True}
            
        try:
            # 이미지 디코딩
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("이미지를 디코딩할 수 없습니다.")
            
            logger.info(f"수정된 분석 시작 - 이미지 크기: {image.shape}")
            
            # 1단계: 향상된 전처리
            processed_image = self.enhanced_preprocessing(image)
            
            # 2단계: 벽면과 가구 분리
            wall_mask, furniture_mask = self.separate_walls_furniture(image)
            
            # 3단계: 객체 감지
            detected_objects = self.detect_room_objects(image)
            
            # 4단계: 방 전체 경계선 감지 (가구 제외)
            room_corners = self.detect_full_room_boundaries(processed_image, wall_mask, furniture_mask)
            
            # 5단계: 스케일 계산
            scale_factor = self.calculate_smart_scale(detected_objects, reference_size, image.shape)
            
            # 6단계: 방 치수 계산
            dimensions = self.calculate_room_dimensions(room_corners, scale_factor)
            
            # 7단계: 시각화
            result_image = self.enhanced_visualization(
                image, room_corners, detected_objects, dimensions, wall_mask, furniture_mask
            )
            
            return {
                'success': True,
                'dimensions': dimensions,
                'detected_objects': detected_objects,
                'result_image': result_image,
                'analysis_info': {
                    'method': 'fixed_room_analyzer_v4',
                    'confidence': dimensions.get('confidence', 0.0),
                    'scale_factor': scale_factor,
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
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # 적응적 히스토그램 평활화
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def separate_walls_furniture(self, image):
        """벽면과 가구 분리"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # 벽면 마스크 (밝고 매끄러운 영역)
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 밝기 기반 벽면 감지
        bright_mask = cv2.inRange(gray, 120, 255)
        
        # 위치 기반 벽면 추정 (가장자리 영역)
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_width = min(w, h) // 8
        border_mask[0:border_width, :] = 255          # 상단
        border_mask[h-border_width:h, :] = 255        # 하단
        border_mask[:, 0:border_width] = 255          # 좌측
        border_mask[:, w-border_width:w] = 255        # 우측
        
        # 벽면 = 밝은 영역 + 가장자리 영역
        wall_mask = cv2.bitwise_or(bright_mask, border_mask)
        
        # 가구 마스크 (어둡고 텍스처가 있는 영역)
        furniture_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 어두운 영역 (가구)
        dark_mask = cv2.inRange(gray, 0, 120)
        
        # 중앙 영역 (가구가 주로 위치)
        center_mask = np.zeros((h, w), dtype=np.uint8)
        center_x_start = w // 4
        center_x_end = 3 * w // 4
        center_y_start = h // 4
        center_y_end = 3 * h // 4
        center_mask[center_y_start:center_y_end, center_x_start:center_x_end] = 255
        
        # 가구 = 어두운 영역 + 중앙 영역
        furniture_mask = cv2.bitwise_and(dark_mask, center_mask)
        
        # 모폴로지 연산으로 정제
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
        furniture_mask = cv2.morphologyEx(furniture_mask, cv2.MORPH_OPEN, kernel)
        
        logger.info("벽면/가구 분리 완료")
        
        return wall_mask, furniture_mask
    
    def detect_room_objects(self, image):
        """방 객체 감지"""
        detected_objects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 창문 감지 (밝은 영역)
        windows = self.detect_windows(gray, image)
        detected_objects.extend(windows)
        
        # 콘센트 감지
        outlets = self.detect_outlets(gray)
        detected_objects.extend(outlets)
        
        return detected_objects
    
    def detect_windows(self, gray, color_image):
        """창문 감지"""
        windows = []
        
        # 밝은 영역 감지
        _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 8000 < area < 150000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 0.8 < aspect_ratio < 3.0:
                    windows.append({
                        'type': 'window',
                        'bbox': (x, y, w, h),
                        'confidence': 0.7,
                        'estimated_real_size': {'width': 120, 'height': 150}
                    })
        
        return windows
    
    def detect_outlets(self, gray):
        """콘센트 감지"""
        outlets = []
        
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 3000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 0.6 < aspect_ratio < 1.7:
                    outlets.append({
                        'type': 'outlet',
                        'bbox': (x, y, w, h),
                        'confidence': 0.5,
                        'estimated_real_size': {'width': 10, 'height': 10}
                    })
        
        return outlets
    
    def detect_full_room_boundaries(self, processed_image, wall_mask, furniture_mask):
        """방 전체 경계선 감지 (가구 제외)"""
        logger.info("방 전체 경계선 감지 시작")
        
        # 가구 영역 제외한 마스크
        clean_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(furniture_mask))
        
        # 엣지 검출
        edges = cv2.Canny(processed_image, 50, 150)
        
        # 마스크 적용
        masked_edges = cv2.bitwise_and(edges, clean_mask)
        
        # 이미지 경계부 강화
        h, w = processed_image.shape[:2]
        border_edges = np.zeros((h, w), dtype=np.uint8)
        border_width = min(w, h) // 15
        
        # 경계부에 인위적인 엣지 추가
        border_edges[border_width:border_width+5, :] = 255        # 상단 라인
        border_edges[h-border_width-5:h-border_width, :] = 255   # 하단 라인
        border_edges[:, border_width:border_width+5] = 255        # 좌측 라인
        border_edges[:, w-border_width-5:w-border_width] = 255   # 우측 라인
        
        # 엣지 결합
        combined_edges = cv2.bitwise_or(masked_edges, border_edges)
        
        # 직선 검출
        lines = cv2.HoughLinesP(
            combined_edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=min(w, h) // 6,
            maxLineGap=50
        )
        
        if lines is None:
            logger.warning("직선을 찾을 수 없음, 기본 경계 사용")
            return self.generate_default_corners(processed_image.shape)
        
        # 방 모서리 추출
        corners = self.extract_room_corners(lines, processed_image.shape)
        
        logger.info(f"방 경계선 감지 완료: {len(corners)}개 모서리")
        return corners
    
    def extract_room_corners(self, lines, image_shape):
        """방 모서리 추출"""
        h, w = image_shape[:2]
        
        # 수평선과 수직선 분류
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            if length > min(w, h) / 8:  # 충분히 긴 직선만
                if abs(angle) < 25 or abs(angle) > 155:  # 수평선
                    horizontal_lines.append(line[0])
                elif 65 < abs(angle) < 115:  # 수직선
                    vertical_lines.append(line[0])
        
        # 외곽 경계선 선택
        corners = []
        
        if len(horizontal_lines) >= 1 and len(vertical_lines) >= 1:
            # 최외곽 라인들 선택
            if horizontal_lines:
                top_line = min(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
                bottom_line = max(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
            else:
                top_line = [0, h//10, w, h//10]
                bottom_line = [0, 9*h//10, w, 9*h//10]
            
            if vertical_lines:
                left_line = min(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
                right_line = max(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
            else:
                left_line = [w//10, 0, w//10, h]
                right_line = [9*w//10, 0, 9*w//10, h]
            
            # 교점 계산
            tl = self.line_intersection(top_line, left_line)
            tr = self.line_intersection(top_line, right_line)
            br = self.line_intersection(bottom_line, right_line)
            bl = self.line_intersection(bottom_line, left_line)
            
            # 유효한 교점들만 추가
            for corner in [tl, tr, br, bl]:
                if corner and self.is_valid_corner(corner, w, h):
                    corners.append(corner)
        
        # 모서리가 부족하면 기본값 사용
        if len(corners) < 4:
            corners = self.generate_default_corners(image_shape)
        
        return corners[:4]
    
    def line_intersection(self, line1, line2):
        """두 직선의 교점"""
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
        margin = 0.05
        return (width * margin <= x <= width * (1 - margin) and 
                height * margin <= y <= height * (1 - margin))
    
    def generate_default_corners(self, image_shape):
        """기본 모서리점 생성"""
        h, w = image_shape[:2]
        margin = min(w, h) * 0.05  # 5% 여백
        
        return [
            (int(margin), int(margin)),
            (int(w - margin), int(margin)),
            (int(w - margin), int(h - margin)),
            (int(margin), int(h - margin))
        ]
    
    def calculate_smart_scale(self, detected_objects, reference_size, image_shape):
        """스마트 스케일 계산"""
        # 객체 기반 스케일
        for obj in detected_objects:
            if 'estimated_real_size' in obj:
                obj_type = obj['type']
                bbox = obj['bbox']
                real_size = obj['estimated_real_size']
                
                if obj_type == 'window':
                    scale = real_size['width'] / bbox[2]
                    logger.info(f"창문 기반 스케일: {scale:.4f}")
                    return scale
                elif obj_type == 'outlet':
                    scale = real_size['width'] / bbox[2]
                    logger.info(f"콘센트 기반 스케일: {scale:.4f}")
                    return scale
        
        # 기본 스케일 (방 크기 추정)
        h, w = image_shape[:2]
        estimated_scale = 400 / min(w, h)  # 4m 가정
        logger.info(f"기본 스케일: {estimated_scale:.4f}")
        
        return estimated_scale
    
    def calculate_room_dimensions(self, corners, scale_factor):
        """방 치수 계산"""
        if len(corners) < 4:
            return {'width': 0, 'height': 0, 'area': 0, 'perimeter': 0, 'confidence': 0.0}
        
        tl, tr, br, bl = corners[:4]
        
        # 픽셀 거리
        width_pixels = math.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
        height_pixels = math.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
        
        # 실제 크기
        width_cm = width_pixels * scale_factor
        height_cm = height_pixels * scale_factor
        area_m2 = (width_cm * height_cm) / 10000
        perimeter_m = (width_cm + height_cm) * 2 / 100
        
        # 신뢰도
        confidence = self.calculate_confidence(corners, width_cm, height_cm)
        
        return {
            'width': round(width_cm, 1),
            'height': round(height_cm, 1),
            'area': round(area_m2, 2),
            'perimeter': round(perimeter_m, 2),
            'confidence': confidence
        }
    
    def calculate_confidence(self, corners, width_cm, height_cm):
        """신뢰도 계산"""
        confidence = 1.0
        
        # 크기 합리성
        if 250 <= width_cm <= 600 and 200 <= height_cm <= 500:
            confidence *= 1.0
        elif 150 <= width_cm <= 800 and 150 <= height_cm <= 600:
            confidence *= 0.7
        else:
            confidence *= 0.4
        
        # 종횡비
        aspect_ratio = max(width_cm, height_cm) / min(width_cm, height_cm)
        if aspect_ratio <= 2.0:
            confidence *= 1.0
        else:
            confidence *= 0.6
        
        return max(0.2, min(1.0, confidence))
    
    def enhanced_visualization(self, image, corners, detected_objects, dimensions, wall_mask, furniture_mask):
        """향상된 시각화"""
        result = image.copy()
        
        # 1. 벽면/가구 영역 표시
        wall_overlay = result.copy()
        wall_overlay[wall_mask > 0] = [255, 200, 200]
        result = cv2.addWeighted(result, 0.9, wall_overlay, 0.1, 0)
        
        furniture_overlay = result.copy()
        furniture_overlay[furniture_mask > 0] = [200, 200, 255]
        result = cv2.addWeighted(result, 0.9, furniture_overlay, 0.1, 0)
        
        # 2. 감지된 객체
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            obj_type = obj['type']
            confidence = obj['confidence']
            
            colors = {
                'window': (0, 255, 255),
                'outlet': (255, 0, 255)
            }
            
            color = colors.get(obj_type, (128, 128, 128))
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
            
            label = f"{obj_type} {confidence:.2f}"
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 3. 방 경계선
        if len(corners) >= 4:
            pts = np.array(corners, dtype=np.int32)
            
            # 방 영역 표시
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            # 경계선
            cv2.polylines(result, [pts], True, (0, 255, 0), 5)
            
            # 모서리점
            for i, corner in enumerate(corners):
                cv2.circle(result, corner, 12, (255, 0, 0), -1)
                cv2.circle(result, corner, 16, (255, 255, 255), 3)
                cv2.putText(result, str(i+1), 
                           (corner[0] + 20, corner[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 4. 정보 패널
        self.draw_info_panel(result, dimensions)
        
        # 5. 치수 라벨
        if len(corners) >= 4:
            self.draw_dimension_labels(result, corners, dimensions)
        
        # base64 인코딩
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{result_base64}"
    
    def draw_info_panel(self, image, dimensions):
        """정보 패널"""
        panel_height = 120
        panel_width = image.shape[1]
        
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel.fill(25)
        
        cv2.putText(panel, "Fixed Room Analysis Results", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        info_lines = [
            f"Room: {dimensions['width']}cm x {dimensions['height']}cm",
            f"Area: {dimensions['area']}m² | Confidence: {dimensions['confidence']:.1%}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(panel, line, (20, 65 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        result = np.vstack([image, panel])
        image[:] = result[:image.shape[0]]
    
    def draw_dimension_labels(self, image, corners, dimensions):
        """치수 라벨"""
        # 가로 치수
        top_center = (
            (corners[0][0] + corners[1][0]) // 2,
            max(40, corners[0][1] - 50)
        )
        
        width_text = f"{dimensions['width']}cm"
        text_size = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        
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
        
        # 세로 치수
        left_center = (
            max(100, corners[0][0] - 120),
            (corners[0][1] + corners[3][1]) // 2
        )
        
        height_text = f"{dimensions['height']}cm"
        text_size = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        
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
analyzer = FixedRoomAnalyzer()

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
        
        logger.info(f"수정된 4단계 분석 시작 - 기준: {reference_size}cm")
        result = analyzer.analyze_image(image_data, reference_size, options)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '4.1.0',
        'features': ['wall_furniture_separation', 'full_room_detection', 'enhanced_visualization'],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 4단계 수정된 방 분석 시스템을 시작합니다...")
    print("📍 http://localhost:5000 에서 접속하세요")
    print("🔧 수정된 기능:")
    print("  - 문법 오류 해결")
    print("  - 벽면/가구 분리 개선")
    print("  - 방 전체 경계선 감지")
    print("  - 더 안정적인 분석")
    app.run(debug=True, host='0.0.0.0', port=5000)