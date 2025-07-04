import cv2
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
