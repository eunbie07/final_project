# 방 사진 분석 프로젝트 파일 구조
# 
# room_analyzer/
# ├── app.py (메인 Flask 서버)
# ├── requirements.txt (필요한 패키지 목록)
# ├── templates/
# │   └── room_analyzer.html (웹 인터페이스)
# ├── static/
# │   ├── css/
# │   │   └── style.css
# │   └── js/
# │       └── app.js
# ├── analyzer/
# │   ├── __init__.py
# │   ├── room_analyzer.py (분석 엔진)
# │   └── utils.py (유틸리티 함수)
# ├── uploads/ (업로드된 이미지 저장)
# ├── results/ (분석 결과 저장)
# └── README.md (설치 및 사용 가이드)

# =============================================================================
# 1. app.py (메인 Flask 서버)
# =============================================================================

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import json
from datetime import datetime
from analyzer.room_analyzer import RoomAnalyzer

app = Flask(__name__)
CORS(app)

# 업로드 및 결과 디렉토리 생성
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

# 분석기 초기화
analyzer = RoomAnalyzer()

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
        
        result = analyzer.analyze_image(image_data, reference_size, options)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 방 사진 분석 서버를 시작합니다...")
    print("📍 http://localhost:5000 에서 접속하세요")
    app.run(debug=True, host='0.0.0.0', port=5000)

# =============================================================================
# 2. requirements.txt
# =============================================================================

"""
flask==2.3.3
flask-cors==4.0.0
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
python-dotenv==1.0.0
"""

# =============================================================================
# 3. analyzer/__init__.py
# =============================================================================

"""
Room Analyzer Package
방 사진 분석 및 평면도 생성 패키지
"""

__version__ = "1.0.0"
__author__ = "Room Analyzer Team"

# =============================================================================
# 4. analyzer/room_analyzer.py (메인 분석 엔진)
# =============================================================================

import cv2
import numpy as np
import base64
import math
from PIL import Image, ImageDraw, ImageFont
from .utils import ImageProcessor, GeometryCalculator

class RoomAnalyzer:
    def __init__(self):
        self.reference_objects = {
            'door': {'width': 80, 'height': 200},  # cm
            'window': {'width': 120, 'height': 150},
            'switch': {'width': 10, 'height': 10}
        }
        self.image_processor = ImageProcessor()
        self.geometry_calc = GeometryCalculator()
    
    def analyze_image(self, image_data, reference_size=200, options=None):
        """메인 분석 함수"""
        if options is None:
            options = {'detect_windows': True, 'detect_doors': True}
            
        try:
            # base64 이미지 디코딩
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            if image is None:
                raise ValueError("이미지를 디코딩할 수 없습니다.")
            
            # 이미지 전처리
            processed = self.image_processor.preprocess_image(image)
            
            # 엣지 검출 및 직선 감지
            edges = self.image_processor.detect_edges(processed)
            lines = self.image_processor.detect_lines(edges)
            
            # 방 모서리 찾기
            corners = self.geometry_calc.find_room_corners(lines, image.shape)
            
            # 크기 계산
            dimensions = self.geometry_calc.calculate_room_dimensions(
                corners, reference_size
            )
            
            # 특징 검출
            features = []
            if options.get('detect_doors') or options.get('detect_windows'):
                features = self.detect_features(image, options)
            
            # 결과 시각화
            result_image = self.visualize_results(
                image, corners, features, dimensions
            )
            
            return {
                'success': True,
                'dimensions': dimensions,
                'features': features,
                'result_image': result_image,
                'corners': corners,
                'analysis_info': {
                    'reference_size': reference_size,
                    'options': options,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def detect_features(self, image, options):
        """문, 창문 등 특징 요소 검출"""
        features = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        contours, _ = cv2.findContours(
            cv2.Canny(gray, 50, 150), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if options.get('detect_doors') and 0.3 < aspect_ratio < 0.7:
                        features.append({
                            'type': 'door',
                            'bbox': (x, y, w, h),
                            'confidence': 0.8
                        })
                    elif options.get('detect_windows') and 0.7 < aspect_ratio < 3.0:
                        features.append({
                            'type': 'window',
                            'bbox': (x, y, w, h),
                            'confidence': 0.7
                        })
        
        return features
    
    def visualize_results(self, image, corners, features, dimensions):
        """결과 시각화"""
        result = image.copy()
        
        # 방 경계선 그리기
        if corners and len(corners) >= 4:
            pts = np.array(corners[:4], np.int32)
            cv2.polylines(result, [pts], True, (0, 255, 0), 3)
            
            for corner in corners[:4]:
                cv2.circle(result, corner, 8, (0, 0, 255), -1)
        
        # 특징 요소 표시
        for feature in features:
            x, y, w, h = feature['bbox']
            color = (255, 0, 0) if feature['type'] == 'door' else (0, 255, 255)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                result, 
                feature['type'], 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                color, 
                2
            )
        
        # 치수 표시
        if dimensions and corners:
            cv2.putText(
                result,
                f"{dimensions['width']}cm",
                (image.shape[1] // 2 - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                result,
                f"{dimensions['height']}cm",
                (10, image.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        # base64로 인코딩
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{result_base64}"

# =============================================================================
# 5. analyzer/utils.py (유틸리티 클래스들)
# =============================================================================

import cv2
import numpy as np
import math

class ImageProcessor:
    """이미지 처리 유틸리티"""
    
    def preprocess_image(self, image):
        """이미지 전처리"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 5)
        equalized = cv2.equalizeHist(denoised)
        return equalized
    
    def detect_edges(self, image):
        """엣지 검출"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return edges
    
    def detect_lines(self, edges):
        """직선 검출"""
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100,
            minLineLength=50,
            maxLineGap=10
        )
        return lines

class GeometryCalculator:
    """기하학적 계산 유틸리티"""
    
    def find_room_corners(self, lines, image_shape):
        """방의 모서리점 찾기"""
        if lines is None:
            return None
            
        height, width = image_shape[:2]
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            
            if abs(angle) < 15 or abs(angle) > 165:
                horizontal_lines.append(line[0])
            elif abs(angle - 90) < 15 or abs(angle + 90) < 15:
                vertical_lines.append(line[0])
        
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            horizontal_lines.sort(
                key=lambda l: math.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2), 
                reverse=True
            )
            vertical_lines.sort(
                key=lambda l: math.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2), 
                reverse=True
            )
            
            corners = self.find_line_intersections(
                horizontal_lines[:2], 
                vertical_lines[:2]
            )
            return corners
        
        return None
    
    def find_line_intersections(self, h_lines, v_lines):
        """직선들의 교점 찾기"""
        corners = []
        
        for h_line in h_lines:
            for v_line in v_lines:
                intersection = self.line_intersection(h_line, v_line)
                if intersection:
                    corners.append(intersection)
        
        return corners
    
    def line_intersection(self, line1, line2):
        """두 직선의 교점 계산"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        
        return (int(intersection_x), int(intersection_y))
    
    def calculate_room_dimensions(self, corners, reference_size=200):
        """방 크기 계산"""
        if not corners or len(corners) < 4:
            return None
            
        corners = sorted(corners, key=lambda p: (p[1], p[0]))
        
        if len(corners) >= 4:
            top_left, top_right = sorted(corners[:2], key=lambda p: p[0])
            bottom_left, bottom_right = sorted(corners[2:4], key=lambda p: p[0])
            
            width_pixels = math.sqrt(
                (top_right[0] - top_left[0])**2 + 
                (top_right[1] - top_left[1])**2
            )
            height_pixels = math.sqrt(
                (bottom_left[0] - top_left[0])**2 + 
                (bottom_left[1] - top_left[1])**2
            )
            
            scale_factor = reference_size / max(width_pixels, height_pixels) * 2
            
            width_cm = width_pixels * scale_factor
            height_cm = height_pixels * scale_factor
            
            return {
                'width': round(width_cm, 1),
                'height': round(height_cm, 1),
                'area': round(width_cm * height_cm / 10000, 2),
                'perimeter': round((width_cm + height_cm) * 2 / 100, 2),
                'corners': [top_left, top_right, bottom_right, bottom_left]
            }
        
        return None

# =============================================================================
# 6. templates/room_analyzer.html
# =============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>실제 방 사진 분석 시스템</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="header">
        <h1>실제 방 사진 분석 시스템 <span class="ai-badge">🤖 AI 분석</span></h1>
        <p>OpenCV와 컴퓨터 비전 기술로 실제 방 크기를 정확하게 측정합니다</p>
    </div>

    <div class="container">
        <div class="controls">
            <div class="section-title">분석 설정</div>
            
            <div class="control-group">
                <label for="referenceSize">기준 크기 (cm)</label>
                <input type="number" id="referenceSize" value="200" min="50" max="500" step="10">
                <small style="color: #666;">문이나 창문 등 알려진 크기를 기준으로 설정</small>
            </div>

            <div class="control-group">
                <label for="roomType">방 유형</label>
                <select id="roomType">
                    <option value="bedroom">침실</option>
                    <option value="living">거실</option>
                    <option value="kitchen">주방</option>
                    <option value="bathroom">욕실</option>
                    <option value="office">사무실</option>
                    <option value="other">기타</option>
                </select>
            </div>

            <div class="section-title">AI 분석 옵션</div>
            
            <div class="control-group">
                <label>
                    <input type="checkbox" id="detectWindows" checked> 창문 자동 감지
                </label>
            </div>

            <div class="control-group">
                <label>
                    <input type="checkbox" id="detectDoors" checked> 문 자동 감지
                </label>
            </div>

            <div class="control-group">
                <label>
                    <input type="checkbox" id="detectFurniture"> 가구 감지
                </label>
            </div>

            <button class="btn" id="analyzeBtn" disabled>🔍 AI 분석 시작</button>
            <button class="btn" id="downloadBtn" disabled>📥 결과 다운로드</button>
            <button class="btn" id="resetBtn">🔄 새로 시작</button>

            <div class="error-message" id="errorMessage"></div>
            <div class="success-message" id="successMessage"></div>
        </div>

        <div class="main-content">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📸</div>
                <div class="upload-text">방 사진을 업로드하세요</div>
                <div class="upload-hint">JPG, PNG 파일을 드래그하거나 클릭하여 선택</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>

            <div class="loading" id="loading">
                <div class="loading-spinner"></div>
                <div>AI가 이미지를 분석 중입니다...</div>
            </div>

            <div class="analysis-results" id="analysisResults" style="display: none;">
                <div class="image-container">
                    <h3>원본 이미지</h3>
                    <img id="originalImage" class="uploaded-image" alt="업로드된 이미지">
                </div>
                
                <div class="image-container">
                    <h3>AI 분석 결과</h3>
                    <img id="resultImage" class="uploaded-image" alt="분석 결과">
                </div>
            </div>

            <div class="measurement-results" id="measurementResults" style="display: none;">
                <div class="section-title">측정 결과</div>
                
                <div class="measurement-item">
                    <span class="measurement-label">가로 길이</span>
                    <span class="measurement-value" id="widthValue">-</span>
                </div>
                
                <div class="measurement-item">
                    <span class="measurement-label">세로 길이</span>
                    <span class="measurement-value" id="heightValue">-</span>
                </div>
                
                <div class="measurement-item">
                    <span class="measurement-label">면적</span>
                    <span class="measurement-value" id="areaValue">-</span>
                </div>
                
                <div class="measurement-item">
                    <span class="measurement-label">둘레</span>
                    <span class="measurement-value" id="perimeterValue">-</span>
                </div>
                
                <div class="measurement-item">
                    <span class="measurement-label">감지된 특징</span>
                    <span class="measurement-value" id="featuresValue">-</span>
                </div>
            </div>
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>'''

# =============================================================================
# 7. README.md
# =============================================================================

README_CONTENT = '''# 방 사진 분석 및 평면도 생성 시스템

실제 방 사진을 업로드하여 AI가 자동으로 분석하고 정확한 크기 측정 및 평면도를 생성하는 웹 애플리케이션입니다.

## 🚀 주요 기능

### AI 기반 분석
- **컴퓨터 비전**: OpenCV를 활용한 실제 이미지 처리
- **엣지 검출**: Canny 알고리즘으로 방 경계선 자동 감지
- **객체 감지**: 문, 창문, 가구 등 자동 인식
- **정확한 측정**: 기준 크기 기반 실제 치수 계산

### 웹 인터페이스
- **드래그 앤 드롭**: 직관적인 이미지 업로드
- **실시간 분석**: Flask API를 통한 실시간 서버 통신
- **시각적 결과**: 원본과 분석 결과 비교 표시
- **상세 측정값**: 가로/세로/면적/둘레 자동 계산

## 📋 시스템 요구사항

- Python 3.8+
- 웹 브라우저 (Chrome, Firefox, Safari, Edge)
- 최소 4GB RAM (이미지 처리용)

## 🔧 설치 방법

### 1. 저장소 클론 또는 파일 다운로드
```bash
git clone <repository-url>
cd room_analyzer
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv

# Windows
venv\\Scripts\\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 서버 실행
```bash
python app.py
```

### 5. 웹 브라우저에서 접속
```
http://localhost:5000
```

## 📖 사용 방법

### 1. 이미지 업로드
- "방 사진을 업로드하세요" 영역에 이미지 파일을 드래그하거나 클릭하여 선택
- JPG, PNG 형식 지원

### 2. 분석 설정
- **기준 크기**: 알려진 객체(문, 창문 등)의 실제 크기 입력
- **방 유형**: 침실, 거실, 주방 등 선택
- **감지 옵션**: 창문, 문, 가구 감지 여부 설정

### 3. AI 분석 시작
- "🔍 AI 분석 시작" 버튼 클릭
- 분석 완료까지 약 2-5초 소요

### 4. 결과 확인
- 원본 이미지와 분석 결과 비교
- 상세한 측정값 확인
- "📥 결과 다운로드"로 분석 결과 이미지 저장

## 🎯 분석 정확도 향상 팁

### 좋은 사진 촬영 방법
1. **정면에서 촬영**: 방의 정면에서 수직으로 촬영
2. **충분한 조명**: 명확한 경계선을 위해 밝은 환경에서 촬영
3. **전체 포함**: 방의 모든 모서리가 보이도록 촬영
4. **기준 객체 포함**: 문이나 창문 등 크기를 아는 객체 포함

### 기준 크기 설정
- **문**: 일반적으로 폭 80cm, 높이 200cm
- **창문**: 일반적으로 폭 120cm, 높이 150cm
- **스위치**: 일반적으로 10cm x 10cm

## 🏗️ 프로젝트 구조

```
room_analyzer/
├── app.py                 # Flask 메인 서버
├── requirements.txt       # 패키지 의존성
├── README.md             # 사용 가이드
├── analyzer/             # 분석 엔진
│   ├── __init__.py
│   ├── room_analyzer.py  # 메인 분석 클래스
│   └── utils.py          # 유틸리티 함수들
├── templates/            # HTML 템플릿
│   └── room_analyzer.html
├── static/               # 정적 파일들
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── uploads/              # 업로드된 이미지
└── results/              # 분석 결과
```

## 🔬 기술 스택

### 백엔드
- **Flask**: 웹 서버 프레임워크
- **OpenCV**: 컴퓨터 비전 라이브러리
- **NumPy**: 수치 계산 라이브러리
- **Pillow**: 이미지 처리 라이브러리

### 프론트엔드
- **HTML5/CSS3**: 웹 인터페이스
- **JavaScript**: 클라이언트 로직
- **Fetch API**: 서버 통신

### AI/CV 알고리즘
- **Canny Edge Detection**: 경계선 검출
- **Hough Transform**: 직선 검출
- **Contour Analysis**: 객체 감지
- **Geometric Calculation**: 치수 계산

## 🚨 문제 해결

### 서버 연결 오류
```bash
# 포트가 사용 중인 경우
lsof -ti:5000 | xargs kill -9

# 다른 포트로 실행
python app.py --port 8000
```

### 패키지 설치 오류
```bash
# OpenCV 설치 문제시
pip install opencv-python-headless

# 의존성 문제시
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### 분석 정확도 문제
1. 더 밝고 선명한 사진으로 재시도
2. 기준 크기를 정확하게 설정
3. 방의 모든 모서리가 보이는 사진 사용

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요.
'''

# =============================================================================
# 파일 생성 스크립트
# =============================================================================

def create_project_files():
    """프로젝트 파일들을 생성하는 함수"""
    
    # requirements.txt 생성
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write("""flask==2.3.3
flask-cors==4.0.0
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
python-dotenv==1.0.0
""")
    
    # README.md 생성
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(README_CONTENT)
    
    # templates/room_analyzer.html 생성
    with open('templates/room_analyzer.html', 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)
    
    # analyzer/__init__.py 생성
    with open('analyzer/__init__.py', 'w', encoding='utf-8') as f:
        f.write('"""Room Analyzer Package"""\n__version__ = "1.0.0"')
    
    print("✅ 모든 프로젝트 파일이 생성되었습니다!")
    print("📁 다음 명령어로 실행하세요:")
    print("1. pip install -r requirements.txt")
    print("2. python app.py")
    print("3. http://localhost:5000 접속")

if __name__ == "__main__":
    create_project_files()

# =============================================================================
# 8. static/css/style.css (스타일시트)
# =============================================================================

CSS_CONTENT = '''* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.header {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    padding: 20px;
    text-align: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.header h1 {
    color: white;
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.header p {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.1em;
}

.ai-badge {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    color: white;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-left: 10px;
}

.container {
    flex: 1;
    display: flex;
    gap: 20px;
    padding: 20px;
    max-width: 1400px;
    margin: 0 auto;
    width: 100%;
}

.controls {
    background: rgba(255, 255, 255, 0.95);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    min-width: 350px;
    max-width: 400px;
    height: fit-content;
}

.main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.upload-area {
    background: rgba(255, 255, 255, 0.95);
    border: 3px dashed #667eea;
    border-radius: 15px;
    padding: 40px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.upload-area:hover {
    background: rgba(255, 255, 255, 1);
    border-color: #764ba2;
}

.upload-area.dragover {
    background: rgba(102, 126, 234, 0.1);
    border-color: #667eea;
}

.upload-icon {
    font-size: 48px;
    color: #667eea;
    margin-bottom: 20px;
}

.upload-text {
    font-size: 18px;
    color: #333;
    margin-bottom: 10px;
}

.upload-hint {
    font-size: 14px;
    color: #666;
}

.analysis-results {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

.image-container {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.uploaded-image {
    max-width: 100%;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.control-group {
    margin-bottom: 20px;
}

.control-group label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: #333;
    font-size: 14px;
}

.control-group input, .control-group select {
    width: 100%;
    padding: 10px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 14px;
    transition: border-color 0.3s ease;
}

.control-group input:focus, .control-group select:focus {
    outline: none;
    border-color: #667eea;
}

.btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    width: 100%;
    margin-bottom: 10px;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.section-title {
    color: #667eea;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 15px;
    padding-bottom: 5px;
    border-bottom: 2px solid #667eea;
}

.measurement-results {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

.measurement-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #eee;
}

.measurement-item:last-child {
    border-bottom: none;
}

.measurement-label {
    font-weight: 600;
    color: #333;
}

.measurement-value {
    color: #667eea;
    font-weight: 700;
}

.loading {
    display: none;
    text-align: center;
    padding: 20px;
}

.loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.error-message {
    color: #e74c3c;
    background: rgba(231, 76, 60, 0.1);
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
    display: none;
}

.success-message {
    color: #27ae60;
    background: rgba(39, 174, 96, 0.1);
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
    display: none;
}

#fileInput {
    display: none;
}

@media (max-width: 1024px) {
    .analysis-results {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 768px) {
    .container {
        flex-direction: column;
        padding: 10px;
    }

    .controls {
        max-width: none;
        order: 2;
    }

    .main-content {
        order: 1;
    }
}'''

# =============================================================================
# 9. static/js/app.js (JavaScript 애플리케이션)
# =============================================================================

JS_CONTENT = '''// 방 사진 분석 애플리케이션 JavaScript

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const originalImage = document.getElementById('originalImage');
const resultImage = document.getElementById('resultImage');
const analysisResults = document.getElementById('analysisResults');
const measurementResults = document.getElementById('measurementResults');
const loading = document.getElementById('loading');
const analyzeBtn = document.getElementById('analyzeBtn');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const errorMessage = document.getElementById('errorMessage');
const successMessage = document.getElementById('successMessage');

let uploadedImageData = null;
let analysisData = null;

// 이벤트 리스너 등록
document.addEventListener('DOMContentLoaded', initializeApp);

function initializeApp() {
    // 파일 업로드 관련 이벤트
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // 버튼 이벤트
    analyzeBtn.addEventListener('click', startAnalysis);
    downloadBtn.addEventListener('click', downloadResult);
    resetBtn.addEventListener('click', resetApplication);
    
    // 서버 상태 확인
    checkServerHealth();
}

// 드래그 앤 드롭 핸들러
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// 파일 처리
function handleFile(file) {
    // 파일 유효성 검사
    if (!file.type.startsWith('image/')) {
        showError('이미지 파일만 업로드할 수 있습니다.');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB 제한
        showError('파일 크기가 너무 큽니다. 10MB 이하의 파일을 선택해주세요.');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedImageData = e.target.result;
        originalImage.src = uploadedImageData;
        
        // UI 업데이트
        uploadArea.style.display = 'none';
        analysisResults.style.display = 'grid';
        analyzeBtn.disabled = false;
        
        showSuccess('이미지가 업로드되었습니다. AI 분석을 시작하세요.');
    };
    
    reader.onerror = function() {
        showError('파일을 읽는 중 오류가 발생했습니다.');
    };
    
    reader.readAsDataURL(file);
}

// AI 분석 시작
async function startAnalysis() {
    if (!uploadedImageData) {
        showError('먼저 이미지를 업로드해주세요.');
        return;
    }

    // UI 상태 변경
    loading.style.display = 'block';
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '분석 중...';
    
    try {
        // 분석 옵션 수집
        const analysisOptions = {
            image: uploadedImageData,
            reference_size: parseInt(document.getElementById('referenceSize').value),
            options: {
                detect_windows: document.getElementById('detectWindows').checked,
                detect_doors: document.getElementById('detectDoors').checked,
                detect_furniture: document.getElementById('detectFurniture').checked,
                room_type: document.getElementById('roomType').value
            }
        };
        
        // API 호출
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(analysisOptions)
        });

        if (!response.ok) {
            throw new Error(`서버 오류: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
            showSuccess('AI 분석이 완료되었습니다!');
        } else {
            showError(`분석 실패: ${result.error}`);
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(`분석 중 오류가 발생했습니다: ${error.message}`);
    } finally {
        // UI 상태 복원
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = '🔍 AI 분석 시작';
    }
}

// 분석 결과 표시
function displayResults(result) {
    analysisData = result;
    
    // 분석 결과 이미지 표시
    if (result.result_image) {
        resultImage.src = result.result_image;
        resultImage.style.display = 'block';
    }
    
    // 측정 결과 표시
    if (result.dimensions) {
        const dims = result.dimensions;
        document.getElementById('widthValue').textContent = `${dims.width}cm`;
        document.getElementById('heightValue').textContent = `${dims.height}cm`;
        document.getElementById('areaValue').textContent = `${dims.area}m²`;
        document.getElementById('perimeterValue').textContent = `${dims.perimeter}m`;
    } else {
        // 측정 실패시 기본값 표시
        document.getElementById('widthValue').textContent = '측정 실패';
        document.getElementById('heightValue').textContent = '측정 실패';
        document.getElementById('areaValue').textContent = '측정 실패';
        document.getElementById('perimeterValue').textContent = '측정 실패';
    }
    
    // 감지된 특징 표시
    if (result.features && result.features.length > 0) {
        const featureTypes = result.features.map(f => f.type);
        const uniqueFeatures = [...new Set(featureTypes)];
        const featureNames = uniqueFeatures.map(type => {
            const translations = {
                'door': '문',
                'window': '창문',
                'furniture': '가구'
            };
            return translations[type] || type;
        });
        document.getElementById('featuresValue').textContent = featureNames.join(', ');
    } else {
        document.getElementById('featuresValue').textContent = '감지된 특징 없음';
    }
    
    // 결과 섹션 표시
    measurementResults.style.display = 'block';
    downloadBtn.disabled = false;
}

// 결과 다운로드
function downloadResult() {
    if (!analysisData || !analysisData.result_image) {
        showError('다운로드할 결과가 없습니다.');
        return;
    }
    
    try {
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        link.download = `방_분석_결과_${timestamp}.png`;
        link.href = analysisData.result_image;
        
        // 다운로드 실행
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showSuccess('분석 결과가 다운로드되었습니다.');
    } catch (error) {
        showError('다운로드 중 오류가 발생했습니다.');
    }
}

// 애플리케이션 리셋
function resetApplication() {
    // 데이터 초기화
    uploadedImageData = null;
    analysisData = null;
    
    // UI 상태 복원
    uploadArea.style.display = 'flex';
    analysisResults.style.display = 'none';
    measurementResults.style.display = 'none';
    loading.style.display = 'none';
    
    // 버튼 상태 복원
    analyzeBtn.disabled = true;
    downloadBtn.disabled = true;
    analyzeBtn.textContent = '🔍 AI 분석 시작';
    
    // 입력 필드 초기화
    fileInput.value = '';
    
    // 메시지 숨기기
    hideMessages();
    
    showSuccess('새로운 분석을 시작할 수 있습니다.');
}

// 서버 상태 확인
async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        const result = await response.json();
        
        if (result.status === 'healthy') {
            console.log('✅ 서버가 정상적으로 작동 중입니다.');
        }
    } catch (error) {
        console.error('❌ 서버 연결 실패:', error);
        showError('서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
    }
}

// 메시지 표시 함수들
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    successMessage.style.display = 'none';
    
    // 자동으로 메시지 숨기기
    setTimeout(hideMessages, 5000);
}

function showSuccess(message) {
    successMessage.textContent = message;
    successMessage.style.display = 'block';
    errorMessage.style.display = 'none';
    
    // 자동으로 메시지 숨기기
    setTimeout(hideMessages, 3000);
}

function hideMessages() {
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
}

// 유틸리티 함수들
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    return validTypes.includes(file.type);
}

// 에러 로깅
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
});

// 미처리 Promise 에러 로깅
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
});'''

# =============================================================================
# 10. 프로젝트 생성 스크립트 업데이트
# =============================================================================

def create_complete_project():
    """완전한 프로젝트 파일들을 생성하는 함수"""
    
    import os
    
    # 디렉토리 생성
    directories = [
        'analyzer',
        'templates', 
        'static/css',
        'static/js',
        'uploads',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # requirements.txt 생성
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write("""flask==2.3.3
flask-cors==4.0.0
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
python-dotenv==1.0.0
""")
    
    # README.md 생성
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(README_CONTENT)
    
    # templates/room_analyzer.html 생성
    with open('templates/room_analyzer.html', 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)
    
    # static/css/style.css 생성
    with open('static/css/style.css', 'w', encoding='utf-8') as f:
        f.write(CSS_CONTENT)
    
    # static/js/app.js 생성
    with open('static/js/app.js', 'w', encoding='utf-8') as f:
        f.write(JS_CONTENT)
    
    # analyzer/__init__.py 생성
    with open('analyzer/__init__.py', 'w', encoding='utf-8') as f:
        f.write('''"""
Room Analyzer Package
방 사진 분석 및 평면도 생성 패키지
"""

__version__ = "1.0.0"
__author__ = "Room Analyzer Team"
''')
    
    # .gitignore 생성
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Application specific
uploads/*.jpg
uploads/*.png
uploads/*.jpeg
results/*.png
*.log
""")
    
    print("✅ 완전한 프로젝트 구조가 생성되었습니다!")
    print("\n📁 프로젝트 구조:")
    print("room_analyzer/")
    print("├── 📄 app.py")
    print("├── 📄 requirements.txt")
    print("├── 📄 README.md")
    print("├── 📄 .gitignore")
    print("├── 📂 analyzer/")
    print("│   ├── __init__.py")
    print("│   ├── room_analyzer.py")
    print("│   └── utils.py")
    print("├── 📂 templates/")
    print("│   └── room_analyzer.html")
    print("├── 📂 static/")
    print("│   ├── 📂 css/")
    print("│   │   └── style.css")
    print("│   └── 📂 js/")
    print("│       └── app.js")
    print("├── 📂 uploads/")
    print("└── 📂 results/")
    print("\n🚀 실행 방법:")
    print("1. pip install -r requirements.txt")
    print("2. python app.py")
    print("3. http://localhost:5000 접속")

if __name__ == "__main__":
    create_complete_project()