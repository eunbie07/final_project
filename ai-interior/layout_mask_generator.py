"""
레이아웃 마스크 생성 시스템
RoomBox.jsx 좌표 → ControlNet 마스크 변환

정확한 가구 위치를 시각적으로 표현하여 AI가 해당 위치에 가구를 생성하도록 강제
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
import json
import os
from pathlib import Path
from datetime import datetime

class LayoutMaskGenerator:
    """레이아웃 마스크 생성기"""
    
    def __init__(self, 
                 image_width: int = 512,  # AMD CPU 최적화: 작은 해상도
                 image_height: int = 512, # AMD CPU 최적화: 작은 해상도
                 mask_color: Tuple[int, int, int] = (255, 255, 255),  # 흰색: 가구 위치
                 background_color: Tuple[int, int, int] = (0, 0, 0)):  # 검은색: 빈 공간
        
        self.image_width = image_width
        self.image_height = image_height
        self.mask_color = mask_color
        self.background_color = background_color
        
        # 가구 타입별 기본 크기 (미터 단위)
        self.furniture_sizes = {
            'bed': {'width': 2.0, 'height': 2.2},
            'desk': {'width': 1.2, 'height': 0.6},
            'chair': {'width': 0.5, 'height': 0.5},
            'wardrobe': {'width': 1.0, 'height': 0.6},
            'sofa': {'width': 2.0, 'height': 0.8},
            'table': {'width': 1.0, 'height': 1.0},
            'nightstand': {'width': 0.5, 'height': 0.4},
            'bookshelf': {'width': 0.8, 'height': 0.3}
        }
    
    def room_to_image_coordinates(self, 
                                room_x: float, 
                                room_z: float, 
                                room_width: float, 
                                room_height: float) -> Tuple[int, int]:
        """
        방 좌표계 → 이미지 픽셀 좌표계 변환
        
        Args:
            room_x: 방에서의 X 좌표 (미터)
            room_z: 방에서의 Z 좌표 (미터) 
            room_width: 방의 너비 (미터)
            room_height: 방의 높이 (미터)
            
        Returns:
            (pixel_x, pixel_y): 이미지에서의 픽셀 좌표
        """
        # 방 좌표를 이미지 좌표로 매핑
        pixel_x = int((room_x / room_width) * self.image_width)
        pixel_y = int((room_z / room_height) * self.image_height)
        
        return (pixel_x, pixel_y)
    
    def get_furniture_dimensions(self, 
                               furniture_type: str, 
                               room_width: float, 
                               room_height: float) -> Tuple[int, int]:
        """
        가구 타입별 이미지에서의 크기 계산
        
        Args:
            furniture_type: 가구 타입
            room_width: 방 너비 (미터)
            room_height: 방 높이 (미터)
            
        Returns:
            (width_pixels, height_pixels): 이미지에서의 가구 크기
        """
        if furniture_type.lower() not in self.furniture_sizes:
            # 기본 크기
            furniture_width = 1.0
            furniture_height = 1.0
        else:
            furniture_info = self.furniture_sizes[furniture_type.lower()]
            furniture_width = furniture_info['width']
            furniture_height = furniture_info['height']
        
        # 픽셀 크기로 변환
        width_pixels = int((furniture_width / room_width) * self.image_width)
        height_pixels = int((furniture_height / room_height) * self.image_height)
        
        return (width_pixels, height_pixels)
    
    def create_layout_mask(self, 
                          furniture_data: List[Dict],
                          room_dimensions: Dict) -> Image.Image:
        """
        가구 배치 데이터로부터 레이아웃 마스크 생성
        
        Args:
            furniture_data: 가구 배치 정보 리스트
                [{'type': 'bed', 'center_x': 1960, 'center_z': 2180, 'id': 'bed_001'}]
            room_dimensions: 방 크기 정보 {'width': 4.0, 'height': 4.0}
            
        Returns:
            PIL Image: 레이아웃 마스크 이미지
        """
        # 검은 배경으로 시작
        mask = Image.new('RGB', (self.image_width, self.image_height), self.background_color)
        draw = ImageDraw.Draw(mask)
        
        room_width = room_dimensions.get('width', 4.0)  # 기본 4m
        room_height = room_dimensions.get('height', 4.0)  # 기본 4m
        
        print(f"[MASK] 방 크기: {room_width}m x {room_height}m")
        print(f"[MASK] 이미지 크기: {self.image_width}x{self.image_height}px")
        
        for furniture in furniture_data:
            # 가구 중심 좌표 (미터 단위로 변환)
            center_x = furniture.get('center_x', 0) / 1000.0  # mm → m 변환
            center_z = furniture.get('center_z', 0) / 1000.0  # mm → m 변환
            furniture_type = furniture.get('type', 'unknown')
            
            # 이미지 좌표로 변환
            pixel_x, pixel_y = self.room_to_image_coordinates(
                center_x, center_z, room_width, room_height
            )
            
            # 가구 크기 계산
            width_pixels, height_pixels = self.get_furniture_dimensions(
                furniture_type, room_width, room_height
            )
            
            # 가구 영역을 흰색으로 그리기 (사각형)
            left = pixel_x - width_pixels // 2
            top = pixel_y - height_pixels // 2
            right = pixel_x + width_pixels // 2
            bottom = pixel_y + height_pixels // 2
            
            # 이미지 경계 내로 제한
            left = max(0, min(left, self.image_width))
            top = max(0, min(top, self.image_height))
            right = max(0, min(right, self.image_width))
            bottom = max(0, min(bottom, self.image_height))
            
            # 가구 위치를 흰색으로 표시
            draw.rectangle([left, top, right, bottom], 
                         fill=self.mask_color, outline=self.mask_color)
            
            print(f"[MASK] {furniture_type}: 실제좌표({center_x:.2f}m, {center_z:.2f}m) → "
                  f"픽셀({pixel_x}, {pixel_y}) → 영역({left}, {top}, {right}, {bottom})")
        
        return mask
    
    def create_canny_edge_mask(self, 
                             furniture_data: List[Dict],
                             room_dimensions: Dict) -> Image.Image:
        """
        Canny 엣지 감지를 사용한 윤곽선 마스크 생성
        (ControlNet Canny 모델 사용시)
        
        Args:
            furniture_data: 가구 배치 정보
            room_dimensions: 방 크기 정보
            
        Returns:
            PIL Image: Canny 엣지 마스크
        """
        # 기본 마스크 생성
        mask = self.create_layout_mask(furniture_data, room_dimensions)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY)
        
        # Canny 엣지 감지
        edges = cv2.Canny(gray, 50, 150)
        
        # 3채널로 변환 (RGB)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges_rgb)
    
    def create_depth_mask(self, 
                        furniture_data: List[Dict],
                        room_dimensions: Dict) -> Image.Image:
        """
        깊이 기반 마스크 생성 (ControlNet Depth 모델 사용시)
        
        Args:
            furniture_data: 가구 배치 정보
            room_dimensions: 방 크기 정보
            
        Returns:
            PIL Image: 깊이 마스크
        """
        # 기본 마스크 생성
        mask = self.create_layout_mask(furniture_data, room_dimensions)
        
        # 깊이 정보로 변환 (가구가 있는 곳은 더 밝게)
        mask_array = np.array(mask)
        depth_mask = np.zeros_like(mask_array)
        
        # 가구 위치는 높이 정보로 처리
        for furniture in furniture_data:
            furniture_type = furniture.get('type', 'unknown')
            
            # 가구 타입별 높이 설정
            height_values = {
                'bed': 128,
                'desk': 180, 
                'chair': 100,
                'wardrobe': 255,
                'sofa': 120,
                'table': 160
            }
            
            height_value = height_values.get(furniture_type, 140)
            
            # 해당 위치를 높이값으로 설정
            mask_indices = np.where(mask_array > 128)
            depth_mask[mask_indices] = height_value
        
        return Image.fromarray(depth_mask.astype(np.uint8))
    
    def save_mask_with_metadata(self, 
                              mask: Image.Image,
                              furniture_data: List[Dict],
                              room_dimensions: Dict,
                              save_path: str,
                              mask_type: str = "layout") -> str:
        """
        마스크 이미지를 메타데이터와 함께 저장
        
        Args:
            mask: 마스크 이미지
            furniture_data: 원본 가구 데이터
            room_dimensions: 방 크기 정보
            save_path: 저장 경로
            mask_type: 마스크 타입 ("layout", "canny", "depth")
            
        Returns:
            str: 저장된 파일 경로
        """
        # 마스크 이미지 저장
        mask.save(save_path)
        
        # 메타데이터 파일 생성
        metadata_path = save_path.replace('.png', '_metadata.json')
        metadata = {
            'mask_type': mask_type,
            'timestamp': datetime.now().isoformat(),
            'furniture_data': furniture_data,
            'room_dimensions': room_dimensions,
            'image_dimensions': {
                'width': self.image_width, 
                'height': self.image_height
            },
            'mask_colors': {
                'furniture': self.mask_color,
                'background': self.background_color
            },
            'furniture_count': len(furniture_data)
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVE] 마스크 저장: {save_path}")
        print(f"[META] 메타데이터: {metadata_path}")
        
        return save_path


def create_test_furniture_data():
    """테스트용 가구 배치 데이터 생성 (중앙 침대 시나리오)"""
    return [
        {
            'type': 'bed',
            'center_x': 2000,  # 2.0m (방 중앙)
            'center_z': 2000,  # 2.0m (방 중앙)
            'id': 'bed_center'
        }
    ]


def main():
    """테스트 실행"""
    print("="*50)
    print("레이아웃 마스크 생성기 테스트")
    print("="*50)
    
    # 생성기 초기화
    generator = LayoutMaskGenerator()
    
    # 테스트 데이터 (중앙에 침대 배치)
    furniture_data = create_test_furniture_data()
    room_dimensions = {'width': 4.0, 'height': 4.0}  # 4m x 4m 방
    
    print(f"테스트 시나리오: {room_dimensions['width']}m x {room_dimensions['height']}m 방에 침대를 중앙에 배치")
    
    # 저장 디렉토리 생성
    output_dir = Path(__file__).parent / 'generated_masks'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 기본 레이아웃 마스크
    print("\n1. 기본 레이아웃 마스크 생성...")
    layout_mask = generator.create_layout_mask(furniture_data, room_dimensions)
    layout_path = str(output_dir / f'layout_mask_{timestamp}.png')
    generator.save_mask_with_metadata(layout_mask, furniture_data, room_dimensions, layout_path, "layout")
    
    # 2. Canny 엣지 마스크
    print("\n2. Canny 엣지 마스크 생성...")
    canny_mask = generator.create_canny_edge_mask(furniture_data, room_dimensions)
    canny_path = str(output_dir / f'canny_mask_{timestamp}.png')
    generator.save_mask_with_metadata(canny_mask, furniture_data, room_dimensions, canny_path, "canny")
    
    # 3. 깊이 마스크
    print("\n3. 깊이 마스크 생성...")
    depth_mask = generator.create_depth_mask(furniture_data, room_dimensions)
    depth_path = str(output_dir / f'depth_mask_{timestamp}.png')
    generator.save_mask_with_metadata(depth_mask, furniture_data, room_dimensions, depth_path, "depth")
    
    print("\n" + "="*50)
    print("테스트 완료! 생성된 마스크들:")
    print(f"  - 레이아웃: {layout_path}")
    print(f"  - Canny:   {canny_path}")
    print(f"  - 깊이:    {depth_path}")
    print("="*50)


if __name__ == "__main__":
    main()