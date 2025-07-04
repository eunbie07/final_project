# 파일명: app_step1.py
# 1단계: 기본 원근법 보정 및 엣지 검출 개선

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

class ImprovedRoomAnalyzer:
    def __init__(self):
        self.reference_objects = {
            'door': {'width': 80, 'height': 200},
            'window': {'width': 120, 'height': 150}
        }
    
    def analyze_image(self, image_data, reference_size=200, options=None):
        """개선된 분석 함수"""
        if options is None:
            options = {'detect_windows': True, 'detect_doors': True}
            
        try:
            # 이미지 디코딩
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("이미지를 디코딩할 수 없습니다.")
            
            logger.info(f"이미지 크기: {image.shape}")
            
            # 1단계: 개선된 이미지 전처리
            processed_image = self.enhanced_preprocessing(image)
            
            # 2단계: 고급 엣지 검출
            edges = self.advanced_edge_detection(processed_image)
            
            # 3단계: 개선된 직선 검출
            lines = self.improved_line_detection(edges)
            
            # 4단계: 방 경계선 감지
            room_corners = self.detect_room_boundaries(lines, image.shape)
            
            # 5단계: 직사각형으로 정규화
            normalized_corners = self.normalize_to_rectangle(room_corners, image.shape)
            
            # 6단계: 치수 계산
            dimensions = self.calculate_dimensions_with_confidence(
                normalized_corners, reference_size
            )
            
            # 7단계: 시각화
            result_image = self.enhanced_visualization(
                image, normalized_corners, dimensions
            )
            
            return {
                'success': True,
                'dimensions': dimensions,
                'features': [],
                'result_image': result_image,
                'analysis_info': {
                    'method': 'improved_cv',
                    'confidence': dimensions.get('confidence', 0.0),
                    'corners_detected': len(normalized_corners),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"분석 오류: {e}")
            return {'success': False, 'error': str(e)}
    
    def enhanced_preprocessing(self, image):
        """향상된 이미지 전처리"""
        # 1. 노이즈 제거 (bilateral filter)
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. 그레이스케일 변환
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # 3. 적응적 히스토그램 평활화 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        logger.info("향상된 전처리 완료")
        return enhanced
    
    def advanced_edge_detection(self, image):
        """고급 엣지 검출"""
        # 1. 가우시안 블러
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 2. 적응적 Canny 엣지 검출
        sigma = 0.33
        median = np.median(blurred)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        
        edges = cv2.Canny(blurred, lower, upper)
        
        # 3. 모폴로지 연산으로 엣지 강화
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        logger.info(f"엣지 검출 완료 - 임계값: {lower}-{upper}")
        return edges
    
    def improved_line_detection(self, edges):
        """개선된 직선 검출"""
        # 확률적 Hough 변환
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=100,
            maxLineGap=30
        )
        
        if lines is None:
            logger.warning("직선을 찾을 수 없습니다")
            return []
        
        # 직선 필터링 (길이 기준)
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length > 80:  # 최소 길이 조건
                filtered_lines.append(line[0])
        
        logger.info(f"직선 검출 완료 - {len(filtered_lines)}개 직선")
        return filtered_lines
    
    def detect_room_boundaries(self, lines, image_shape):
        """방 경계선 감지"""
        if not lines:
            return self.fallback_corners(image_shape)
        
        h, w = image_shape[:2]
        
        # 수평선과 수직선 분류
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # 수평선 (-20도 ~ 20도, 160도 ~ 200도)
            if abs(angle) < 20 or abs(angle) > 160:
                horizontal_lines.append(line)
            # 수직선 (70도 ~ 110도, -110도 ~ -70도)
            elif 70 < abs(angle) < 110:
                vertical_lines.append(line)
        
        logger.info(f"수평선: {len(horizontal_lines)}개, 수직선: {len(vertical_lines)}개")
        
        # 경계선 찾기
        corners = []
        
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # 가장 외곽의 직선들 선택
            top_line = min(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
            bottom_line = max(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
            left_line = min(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
            right_line = max(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
            
            # 교점 계산
            tl = self.line_intersection(top_line, left_line)
            tr = self.line_intersection(top_line, right_line)
            br = self.line_intersection(bottom_line, right_line)
            bl = self.line_intersection(bottom_line, left_line)
            
            # 유효한 교점들만 추가
            for corner in [tl, tr, br, bl]:
                if corner and self.is_valid_corner(corner, w, h):
                    corners.append(corner)
        
        if len(corners) < 4:
            logger.warning("충분한 모서리점을 찾지 못함, 백업 방법 사용")
            return self.fallback_corners(image_shape)
        
        return corners
    
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
        margin = 0.1  # 10% 여백
        
        return (width * margin <= x <= width * (1 - margin) and 
                height * margin <= y <= height * (1 - margin))
    
    def fallback_corners(self, image_shape):
        """백업 모서리점 생성"""
        h, w = image_shape[:2]
        margin = min(w, h) * 0.15
        
        return [
            (int(margin), int(margin)),
            (int(w - margin), int(margin)),
            (int(w - margin), int(h - margin)),
            (int(margin), int(h - margin))
        ]
    
    def normalize_to_rectangle(self, corners, image_shape):
        """모서리점들을 직사각형으로 정규화"""
        if len(corners) < 4:
            return self.fallback_corners(image_shape)
        
        # 중심점 계산
        center_x = sum(c[0] for c in corners) / len(corners)
        center_y = sum(c[1] for c in corners) / len(corners)
        
        # 각 사분면별로 가장 가까운 점 찾기
        top_left = min(corners, key=lambda c: (c[0] - center_x)**2 + (c[1] - center_y)**2 
                      if c[0] < center_x and c[1] < center_y else float('inf'))
        top_right = min(corners, key=lambda c: (c[0] - center_x)**2 + (c[1] - center_y)**2 
                       if c[0] > center_x and c[1] < center_y else float('inf'))
        bottom_right = min(corners, key=lambda c: (c[0] - center_x)**2 + (c[1] - center_y)**2 
                          if c[0] > center_x and c[1] > center_y else float('inf'))
        bottom_left = min(corners, key=lambda c: (c[0] - center_x)**2 + (c[1] - center_y)**2 
                         if c[0] < center_x and c[1] > center_y else float('inf'))
        
        normalized = [top_left, top_right, bottom_right, bottom_left]
        
        # 유효성 검사
        if any(c == corners[0] for c in normalized[1:]):  # 중복 점 확인
            return self.fallback_corners(image_shape)
        
        logger.info("모서리점 정규화 완료")
        return normalized
    
    def calculate_dimensions_with_confidence(self, corners, reference_size):
        """신뢰도를 포함한 치수 계산"""
        if len(corners) < 4:
            return {
                'width': 0, 'height': 0, 'area': 0, 'perimeter': 0, 'confidence': 0.0
            }
        
        # 픽셀 거리 계산
        top_left, top_right, bottom_right, bottom_left = corners
        
        width_pixels = math.sqrt(
            (top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2
        )
        height_pixels = math.sqrt(
            (bottom_left[0] - top_left[0])**2 + (bottom_left[1] - top_left[1])**2
        )
        
        # 스케일 계산 (기준 크기 기반)
        scale_factor = reference_size / max(width_pixels, height_pixels) * 2.5
        
        # 실제 크기 계산
        width_cm = width_pixels * scale_factor
        height_cm = height_pixels * scale_factor
        area_m2 = (width_cm * height_cm) / 10000
        perimeter_m = (width_cm + height_cm) * 2 / 100
        
        # 신뢰도 계산
        confidence = self.calculate_confidence(corners, width_pixels, height_pixels)
        
        logger.info(f"치수 계산 완료 - {width_cm:.1f}x{height_cm:.1f}cm (신뢰도: {confidence:.2f})")
        
        return {
            'width': round(width_cm, 1),
            'height': round(height_cm, 1),
            'area': round(area_m2, 2),
            'perimeter': round(perimeter_m, 2),
            'confidence': confidence
        }
    
    def calculate_confidence(self, corners, width_pixels, height_pixels):
        """신뢰도 계산"""
        confidence = 1.0
        
        # 1. 직사각형 형태 확인
        rectangularity = self.check_rectangularity(corners)
        confidence *= rectangularity
        
        # 2. 크기 합리성 확인
        if width_pixels < 100 or height_pixels < 100:
            confidence *= 0.5
        
        # 3. 종횡비 확인
        aspect_ratio = max(width_pixels, height_pixels) / min(width_pixels, height_pixels)
        if aspect_ratio > 4:
            confidence *= 0.6
        
        return max(0.1, min(1.0, confidence))
    
    def check_rectangularity(self, corners):
        """직사각형 형태 확인"""
        if len(corners) < 4:
            return 0.0
        
        # 인접한 변들이 수직인지 확인
        angles = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = corners[(i + 2) % 4]
            
            # 벡터 계산
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # 내적으로 각도 계산
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = dot_product / (norm1 * norm2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = math.degrees(math.acos(cos_angle))
                angles.append(angle)
        
        # 90도와의 차이로 직사각형 정도 계산
        rectangularity = 1.0
        for angle in angles:
            diff = abs(angle - 90)
            rectangularity *= max(0.1, 1.0 - diff / 45)
        
        return rectangularity
    
    def enhanced_visualization(self, image, corners, dimensions):
        """향상된 시각화"""
        result = image.copy()
        
        if len(corners) >= 4:
            # 방 영역 하이라이트
            pts = np.array(corners, dtype=np.int32)
            
            # 반투명 오버레이
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            # 경계선 그리기
            cv2.polylines(result, [pts], True, (0, 255, 0), 4)
            
            # 모서리점 표시
            for i, corner in enumerate(corners):
                cv2.circle(result, corner, 10, (255, 0, 0), -1)
                cv2.circle(result, corner, 15, (255, 255, 255), 3)
                
                # 모서리 번호 표시
                cv2.putText(result, str(i+1), 
                           (corner[0] + 20, corner[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 치수 정보 패널
        if dimensions:
            self.draw_info_panel(result, dimensions)
            
            # 이미지 위에 주요 치수 표시
            if len(corners) >= 4:
                self.draw_dimension_labels(result, corners, dimensions)
        
        # base64 인코딩
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{result_base64}"
    
    def draw_info_panel(self, image, dimensions):
        """정보 패널 그리기"""
        panel_height = 140
        panel_width = image.shape[1]
        
        # 패널 배경
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel.fill(30)  # 어두운 배경
        
        # 제목
        cv2.putText(panel, "측정 결과", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 측정값
        info_lines = [
            f"가로: {dimensions['width']}cm",
            f"세로: {dimensions['height']}cm",
            f"면적: {dimensions['area']}m²",
            f"신뢰도: {dimensions['confidence']:.1%}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(panel, line, (20, 65 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 패널을 이미지에 추가
        result = np.vstack([image, panel])
        image[:] = result[:image.shape[0]]
    
    def draw_dimension_labels(self, image, corners, dimensions):
        """치수 라벨 그리기"""
        # 가로 치수 (상단 중앙)
        top_center = (
            (corners[0][0] + corners[1][0]) // 2,
            max(20, corners[0][1] - 30)
        )
        
        cv2.putText(image, f"{dimensions['width']}cm", 
                   top_center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(image, f"{dimensions['width']}cm", 
                   top_center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
        
        # 세로 치수 (좌측 중앙)
        left_center = (
            max(10, corners[0][0] - 120),
            (corners[0][1] + corners[3][1]) // 2
        )
        
        cv2.putText(image, f"{dimensions['height']}cm", 
                   left_center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(image, f"{dimensions['height']}cm", 
                   left_center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

# Flask 라우트
analyzer = ImprovedRoomAnalyzer()

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
        
        logger.info(f"분석 시작 - 기준 크기: {reference_size}cm")
        result = analyzer.analyze_image(image_data, reference_size, options)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '1.1.0',
        'features': ['enhanced_preprocessing', 'advanced_edge_detection', 'improved_line_detection'],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 1단계 개선된 방 사진 분석 서버를 시작합니다...")
    print("📍 http://localhost:5000 에서 접속하세요")
    print("✨ 새로운 기능:")
    print("  - 향상된 이미지 전처리 (CLAHE)")
    print("  - 적응적 Canny 엣지 검출")
    print("  - 개선된 직선 검출 및 필터링")
    print("  - 신뢰도 계산")
    print("  - 향상된 시각화")
    app.run(debug=True, host='0.0.0.0', port=5000)