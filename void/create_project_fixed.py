#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
방 사진 분석 프로젝트 생성 스크립트 (간단 버전)
"""

import os

def create_project():
    """프로젝트 파일들을 생성"""
    
    # 디렉토리 생성
    os.makedirs('analyzer', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write("""flask==2.3.3
flask-cors==4.0.0
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
""")
    
    # app.py
    app_content = """from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from datetime import datetime
from analyzer.room_analyzer import RoomAnalyzer

app = Flask(__name__)
CORS(app)

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
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("🚀 방 사진 분석 서버를 시작합니다...")
    print("📍 http://localhost:5000 에서 접속하세요")
    app.run(debug=True, host='0.0.0.0', port=5000)
"""
    
    with open('app.py', 'w') as f:
        f.write(app_content)
    
    # analyzer/__init__.py
    with open('analyzer/__init__.py', 'w') as f:
        f.write('# Room Analyzer Package\\n')
    
    # analyzer/room_analyzer.py
    analyzer_content = """import cv2
import numpy as np
import base64
import math
from datetime import datetime

class RoomAnalyzer:
    def __init__(self):
        self.reference_objects = {
            'door': {'width': 80, 'height': 200},
            'window': {'width': 120, 'height': 150}
        }
    
    def analyze_image(self, image_data, reference_size=200, options=None):
        if options is None:
            options = {'detect_windows': True, 'detect_doors': True}
            
        try:
            # base64 이미지 디코딩
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("이미지를 디코딩할 수 없습니다.")
            
            # 이미지 전처리
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 가장 큰 컨투어를 방 경계로 가정
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 근사화
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if len(approx) >= 4:
                    # 바운딩 박스 계산
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # 픽셀을 실제 크기로 변환
                    scale_factor = reference_size / max(w, h)
                    width_cm = w * scale_factor
                    height_cm = h * scale_factor
                    
                    dimensions = {
                        'width': round(width_cm, 1),
                        'height': round(height_cm, 1),
                        'area': round(width_cm * height_cm / 10000, 2),
                        'perimeter': round((width_cm + height_cm) * 2 / 100, 2)
                    }
                    
                    # 시각화
                    result_image = self.visualize_results(image, approx, dimensions)
                    
                    return {
                        'success': True,
                        'dimensions': dimensions,
                        'features': [],
                        'result_image': result_image
                    }
            
            return {'success': False, 'error': '방 경계를 찾을 수 없습니다.'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def visualize_results(self, image, contour, dimensions):
        result = image.copy()
        
        # 컨투어 그리기
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
        
        # 치수 표시
        cv2.putText(result, f"{dimensions['width']}cm", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f"{dimensions['height']}cm", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f"{dimensions['area']}m²", 
                   (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # base64 인코딩
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{result_base64}"
"""
    
    with open('analyzer/room_analyzer.py', 'w') as f:
        f.write(analyzer_content)
    
    # HTML 템플릿
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>방 사진 분석 시스템</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin-bottom: 20px; background: white; }
        .controls { background: white; padding: 20px; margin-bottom: 20px; }
        .results { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .result-box { background: white; padding: 20px; }
        img { max-width: 100%; height: auto; }
        button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        input, select { padding: 8px; margin: 5px; width: 100px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>방 사진 분석 시스템</h1>
            <p>방 사진을 업로드하여 크기를 측정하세요</p>
        </div>
        
        <div class="upload-area" id="uploadArea">
            <p>이미지를 드래그하거나 클릭하여 업로드</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        </div>
        
        <div class="controls">
            <label>기준 크기: <input type="number" id="referenceSize" value="200" min="50" max="500"> cm</label>
            <button id="analyzeBtn" disabled>분석 시작</button>
            <button id="resetBtn">리셋</button>
        </div>
        
        <div class="results hidden" id="results">
            <div class="result-box">
                <h3>원본 이미지</h3>
                <img id="originalImage" alt="원본 이미지">
            </div>
            <div class="result-box">
                <h3>분석 결과</h3>
                <img id="resultImage" alt="분석 결과">
                <div id="measurements"></div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const originalImage = document.getElementById('originalImage');
        const resultImage = document.getElementById('resultImage');
        const results = document.getElementById('results');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resetBtn = document.getElementById('resetBtn');
        const measurements = document.getElementById('measurements');
        
        let imageData = null;
        
        uploadArea.onclick = () => fileInput.click();
        
        fileInput.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imageData = e.target.result;
                    originalImage.src = imageData;
                    results.classList.remove('hidden');
                    analyzeBtn.disabled = false;
                };
                reader.readAsDataURL(file);
            }
        };
        
        analyzeBtn.onclick = async () => {
            analyzeBtn.textContent = '분석 중...';
            analyzeBtn.disabled = true;
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: imageData,
                        reference_size: parseInt(document.getElementById('referenceSize').value),
                        options: {}
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultImage.src = result.result_image;
                    const dims = result.dimensions;
                    measurements.innerHTML = `
                        <p>가로: ${dims.width}cm</p>
                        <p>세로: ${dims.height}cm</p>
                        <p>면적: ${dims.area}m²</p>
                        <p>둘레: ${dims.perimeter}m</p>
                    `;
                } else {
                    alert('분석 실패: ' + result.error);
                }
            } catch (error) {
                alert('오류: ' + error.message);
            }
            
            analyzeBtn.textContent = '분석 시작';
            analyzeBtn.disabled = false;
        };
        
        resetBtn.onclick = () => {
            imageData = null;
            results.classList.add('hidden');
            analyzeBtn.disabled = true;
            fileInput.value = '';
        };
    </script>
</body>
</html>"""
    
    with open('templates/room_analyzer.html', 'w') as f:
        f.write(html_content)
    
    # README.md
    readme_content = """# 방 사진 분석 시스템

## 설치
```bash
pip install -r requirements.txt
```

## 실행
```bash
python app.py
```

## 접속
http://localhost:5000

## 사용법