from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한
app.config['STATIC_FOLDER'] = 'static'

# 업로드 폴더가 없으면 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def calculate_floor_dimensions(image_path, ceiling_height=2.3):
    """
    방 사진에서 바닥 면적을 계산하는 함수
    :param image_path: 이미지 파일 경로
    :param ceiling_height: 천장 높이 (미터 단위)
    :return: (가로 길이, 세로 길이, 면적) 또는 (None, None, None)
    """
    try:
        # 1. 이미지 로드 및 전처리
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 에지 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 3. 직선 검출 (Hough Transform)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return None, None, None
            
        # 4. 천장-바닥 경계선 찾기
        floor_ceiling_lines = []
        for line in lines:
            rho, theta = line[0]
            # 수평선에 가까운 각도 필터링 (약간의 오차 허용)
            if np.pi/2 - 0.2 < theta < np.pi/2 + 0.2:
                floor_ceiling_lines.append(line)
        
        # 5. 가장 위쪽(천장)과 아래쪽(바닥) 선 찾기
        if len(floor_ceiling_lines) >= 2:
            floor_ceiling_lines.sort(key=lambda x: x[0][0])
            ceiling_line = floor_ceiling_lines[0]
            floor_line = floor_ceiling_lines[-1]
            
            # 픽셀당 미터 단위 계산
            pixel_height = abs(floor_line[0][0] - ceiling_line[0][0])
            if pixel_height == 0:
                return None, None, None
                
            pixel_to_meter = ceiling_height / pixel_height
            
            # 6. 방의 가로 길이 계산 (이미지 너비 기반)
            room_width_px = img.shape[1]  # 이미지 너비(픽셀)
            room_width_m = room_width_px * pixel_to_meter
            
            # 7. 방의 세로 길이 추정 (원근 보정 필요)
            # 여기서는 단순화를 위해 가로 길이의 0.7배로 가정
            room_length_m = room_width_m * 0.7
            
            # 8. 면적 계산
            area = room_width_m * room_length_m
            
            return round(room_width_m, 2), round(room_length_m, 2), round(area, 2)
            
        return None, None, None
        
    except Exception as e:
        print(f"Error in calculate_floor_dimensions: {str(e)}")
        return None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            width, length, area = calculate_floor_dimensions(filepath)
            if width and length and area:
                return jsonify({
                    'success': True,
                    'width': width,
                    'length': length,
                    'area': area,
                    'image_url': f'/static/uploads/{filename}'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '방 크기를 계산할 수 없습니다. 다른 각도에서 촬영해 주세요.'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'처리 중 오류가 발생했습니다: {str(e)}'
            }), 500

if __name__ == '__main__':
    # 정적 파일 서빙을 위한 심볼릭 링크 생성
    os.makedirs('static/uploads', exist_ok=True)
    if not os.path.exists('static/uploads'):
        os.symlink(os.path.abspath('uploads'), 'static/uploads')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
