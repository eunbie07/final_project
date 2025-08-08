"""
Colab ComfyUI Inpainting API 연동 클라이언트
MongoDB 좌표 → 픽셀 마스크 → 95%+ 정확도 Inpainting 생성
"""

import requests
import base64
import io
from PIL import Image
from typing import Dict, Any, Tuple, Optional
import os
from datetime import datetime
import numpy as np

class ColabInpaintingGenerator:
    """Colab ComfyUI Inpainting 생성기 클라이언트"""
    
    def __init__(self, colab_api_url: str):
        """
        Args:
            colab_api_url: Colab에서 실행 중인 API URL
                          예: "https://abc123.ngrok.io" 또는 "http://colab-server:5000"
        """
        self.api_url = colab_api_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 300  # 5분 타임아웃
        
        print(f"ColabInpaintingGenerator 초기화: {self.api_url}")
    
    def health_check(self) -> bool:
        """Colab API 서버 상태 확인"""
        try:
            # ComfyUI system_stats 엔드포인트 사용
            response = self.session.get(f"{self.api_url}/system_stats", timeout=10)
            if response.status_code == 200:
                print(f"Colab ComfyUI 서버 연결 성공")
                return True
            else:
                print(f"Colab 서버 응답 오류: {response.status_code}")
                return False
        except Exception as e:
            print(f"Colab 서버 연결 실패: {e}")
            return False
    
    def generate_interior_image(self, 
                              room_data: Dict[str, Any], 
                              style: str = "modern") -> Tuple[Optional[str], Dict[str, Any]]:
        """
        MongoDB 좌표를 사용해 95%+ 정확도로 가구 배치된 인테리어 이미지 생성
        
        Args:
            room_data: MongoDB에서 가져온 방 데이터
                      {
                          'dimensions': {'width_cm': 387, 'depth_cm': 465},
                          'furniture_3d': [
                              {'name': 'bed', 'position': [203.67, 0, 238.00]}
                          ]
                      }
            style: 인테리어 스타일 ('modern', 'scandinavian', 'industrial')
        
        Returns:
            image_path: 생성된 이미지 파일 경로
            metadata: 생성 메타데이터 (정확도, 분석 정보 등)
        """
        
        print(f"[COLAB] Inpainting 이미지 생성 시작: {style} 스타일")
        print(f"   방 크기: {room_data.get('dimensions', {}).get('width_cm')}x{room_data.get('dimensions', {}).get('depth_cm')}cm")
        print(f"   가구 개수: {len(room_data.get('furniture_3d', []))}개")
        
        try:
            # 1. 헬스 체크
            if not self.health_check():
                return None, {"error": "Colab 서버 연결 불가", "mock_mode": True}
            
            # 2. 완전한 워크플로우 요청
            print("[COLAB] MongoDB 좌표 → 픽셀 마스크 → Inpainting 실행...")
            
            response = self.session.post(
                f"{self.api_url}/generate-complete",
                json={
                    'room_data': room_data,
                    'style': style
                },
                timeout=300  # 5분 타임아웃 (Inpainting은 시간 소요)
            )
            
            if response.status_code != 200:
                error_msg = f"Colab API 오류: {response.status_code} - {response.text[:200]}"
                print(f"[ERROR] {error_msg}")
                return None, {"error": error_msg, "mock_mode": True}
            
            result = response.json()
            
            if not result.get('success'):
                return None, {"error": "Colab 생성 실패", "mock_mode": True}
            
            # 3. base64 이미지를 파일로 저장
            image_base64 = result['image_base64']
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"colab_inpaint_{style}_{timestamp}.png"
            image_path = os.path.join("generated_images", filename)
            
            # 디렉토리 생성
            os.makedirs("generated_images", exist_ok=True)
            
            # 이미지 저장
            image.save(image_path)
            
            # 메타데이터 준비
            metadata = {
                "generator": "Colab_ComfyUI_Inpainting",
                "style": style,
                "accuracy_score": result['accuracy_score'],
                "accuracy_percentage": result['accuracy_percentage'],
                "position_analysis": result['position_analysis'],
                "workflow_steps": result['workflow_steps'],
                "image_size": image.size,
                "mock_mode": False,
                "timestamp": result['timestamp']
            }
            
            print(f"[COLAB] 생성 완료! 정확도: {result['accuracy_percentage']}")
            print(f"   이미지 저장: {image_path}")
            print(f"   상태: {result['position_analysis']['accuracy_status']}")
            
            return image_path, metadata
            
        except requests.exceptions.Timeout:
            print("[ERROR] Colab API 타임아웃 (5분 초과)")
            return None, {"error": "타임아웃", "mock_mode": True}
        
        except Exception as e:
            print(f"[ERROR] Colab 연동 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, {"error": str(e), "mock_mode": True}
    
    def convert_coordinates_only(self, room_data: Dict[str, Any]) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """
        MongoDB 좌표를 픽셀 마스크로만 변환 (이미지 생성 없이)
        디버깅 및 테스트 용도
        
        Returns:
            mask_image: PIL Image (512x512)
            conversion_info: 변환 정보
        """
        try:
            response = self.session.post(
                f"{self.api_url}/convert-coordinates",
                json={'room_data': room_data},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # base64 마스크를 PIL 이미지로 변환
                mask_data = base64.b64decode(result['mask_base64'])
                mask_image = Image.open(io.BytesIO(mask_data))
                
                return mask_image, result['conversion_info']
            else:
                return None, {"error": f"변환 실패: {response.status_code}"}
                
        except Exception as e:
            return None, {"error": str(e)}


class LocalCoordinateConverter:
    """
    Colab 없이 로컬에서 좌표 변환 (백업용)
    """
    
    def __init__(self, image_size=512):
        self.image_size = image_size
    
    def convert_room_to_mask(self, room_data: Dict[str, Any]) -> Tuple[Image.Image, list]:
        """
        MongoDB cm 좌표를 512x512 픽셀 마스크로 변환
        """
        # 방 크기 (cm)
        room_width_cm = room_data['dimensions']['width_cm']
        room_depth_cm = room_data['dimensions']['depth_cm']
        
        # cm → pixel 변환 비율
        scale_x = self.image_size / room_width_cm
        scale_y = self.image_size / room_depth_cm
        
        print(f"로컬 좌표 변환: {room_width_cm}x{room_depth_cm}cm → {self.image_size}x{self.image_size}px")
        print(f"변환 비율: X={scale_x:.3f}, Y={scale_y:.3f} pixel/cm")
        
        # 마스크 이미지 생성
        from PIL import Image, ImageDraw
        mask = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        
        furniture_regions = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, furniture in enumerate(room_data.get('furniture_3d', [])):
            name = furniture['name']
            position = furniture['position']  # [x, y, z] cm
            
            # MongoDB 좌표 → 픽셀 좌표
            x_cm, y_cm, z_cm = position[0], position[1], position[2]
            pixel_x = int(x_cm * scale_x)
            pixel_z = int(z_cm * scale_y)
            
            # 가구 크기 (픽셀)
            if 'bed' in name.lower():
                w, h = 80, 60
            elif 'sofa' in name.lower():
                w, h = 60, 40
            elif 'table' in name.lower():
                w, h = 40, 40
            else:
                w, h = 30, 30
            
            # 가구 영역 계산
            x1 = max(0, pixel_x - w//2)
            y1 = max(0, pixel_z - h//2)
            x2 = min(self.image_size-1, pixel_x + w//2)
            y2 = min(self.image_size-1, pixel_z + h//2)
            
            # 마스크에 그리기
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255), width=2)
            draw.ellipse([pixel_x-3, pixel_z-3, pixel_x+3, pixel_z+3], 
                        fill=(255, 255, 255), outline=(0, 0, 0))
            
            furniture_regions.append({
                'name': name,
                'center': (pixel_x, pixel_z),
                'bbox': (x1, y1, x2, y2),
                'original_position_cm': (x_cm, z_cm),
                'color': color
            })
            
            print(f"가구 '{name}': ({x_cm}, {z_cm})cm → ({pixel_x}, {pixel_z})px")
        
        return mask, furniture_regions