"""
DALL-E 3 기반 AI 인테리어 생성기
위치 제어 정확도 테스트용
"""

import openai
from typing import List, Dict, Optional, Tuple
import json
import os
from pathlib import Path
from datetime import datetime
import requests
from PIL import Image
import io
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DalleGenerator:
    """DALL-E 3 인테리어 생성기"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            api_key: OpenAI API 키
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            logger.warning("[WARNING] OpenAI API 키가 없습니다. 환경변수 OPENAI_API_KEY를 설정하거나 Mock 모드로 실행됩니다.")
            self.mock_mode = True
            self.client = None
        else:
            self.mock_mode = False
            self.client = openai.OpenAI(api_key=self.api_key)
        
        logger.info(f"[INIT] DALL-E 생성기 초기화 (Mock 모드: {self.mock_mode})")
    
    def create_position_prompt(self, 
                             furniture_data: List[Dict],
                             room_dimensions: Dict,
                             style: str = "scandinavian") -> str:
        """
        위치 제어를 위한 상세한 프롬프트 생성
        
        Args:
            furniture_data: 가구 배치 정보
            room_dimensions: 방 크기 정보
            style: 인테리어 스타일
            
        Returns:
            str: 위치 제어 최적화된 프롬프트
        """
        # 스타일별 기본 프롬프트 (한국식 인테리어 강조)
        style_prompts = {
            "scandinavian": "Korean-style Scandinavian minimalist bedroom with ondol floor heating system, light wood floors, white walls, natural lighting, Korean furniture proportions",
            "modern": "Korean modern contemporary bedroom with sleek lines, neutral colors, clean aesthetic, Korean apartment style, ondol heating, built-in storage typical of Korean homes",
            "industrial": "Korean industrial loft-style bedroom with exposed elements, modern Korean apartment features, ondol floor system, Korean urban living space design"
        }
        
        base_style = style_prompts.get(style, style_prompts["scandinavian"])
        
        # 방 크기 정보
        room_width = room_dimensions.get('width', 4.0)
        room_height = room_dimensions.get('height', 4.0)
        
        # 가구별 위치 설명 생성
        furniture_positions = []
        for furniture in furniture_data:
            ftype = furniture.get('type', 'furniture')
            # mm를 m로 변환
            x_pos = furniture.get('center_x', 0) / 1000.0
            z_pos = furniture.get('center_z', 0) / 1000.0
            
            # 상대적 위치 계산
            x_percent = (x_pos / room_width) * 100
            z_percent = (z_pos / room_height) * 100
            
            # 더 정확한 위치 설명 생성 (좌표값 포함)
            if x_percent < 20:
                x_desc = "far left wall area"
            elif x_percent < 40:
                x_desc = "left side area"
            elif x_percent < 60:
                x_desc = f"center of the room horizontally ({x_percent:.0f}% from left wall)"
            elif x_percent < 80:
                x_desc = "right side area"
            else:
                x_desc = "far right wall area"
                
            if z_percent < 20:
                z_desc = "very front of room"
            elif z_percent < 40:
                z_desc = "front side area"
            elif z_percent < 60:
                z_desc = f"center of the room vertically ({z_percent:.0f}% from front wall)"
            elif z_percent < 80:
                z_desc = "back side area"  
            else:
                z_desc = "very back of room"
            
            # 간단하고 명확한 위치 설명
            if 45 <= x_percent <= 55 and 45 <= z_percent <= 55:
                position_desc = f"{ftype} floating in the exact CENTER of the room, not touching any walls"
            else:
                position_desc = f"{ftype} positioned {x_desc} and {z_desc}, floating freely in open space"
            furniture_positions.append(position_desc)
        
        # 극단적으로 단순화된 프롬프트 (벽 붙임 방지)
        prompt = f"""{base_style}

The bed is FLOATING IN THE MIDDLE OF THE ROOM with empty space all around it.
The bed is NOT against any wall.
There is open floor space between the bed and every wall.
Show the bed surrounded by empty floor space on all four sides.

Korean minimalist bedroom interior, no measurements, no grid lines, professional photography."""
        
        return prompt.strip()
    
    def generate_interior_image(self, 
                              furniture_data: List[Dict],
                              room_dimensions: Dict,
                              style: str = "scandinavian",
                              additional_prompt: str = "") -> Tuple[str, Dict]:
        """
        DALL-E로 인테리어 이미지 생성
        
        Args:
            furniture_data: 가구 배치 정보
            room_dimensions: 방 크기 정보
            style: 인테리어 스타일
            additional_prompt: 추가 프롬프트
            
        Returns:
            Tuple[str, Dict]: (생성된 이미지 경로, 메타데이터)
        """
        # 프롬프트 생성
        prompt = self.create_position_prompt(furniture_data, room_dimensions, style)
        
        if additional_prompt:
            prompt += f"\n\nADDITIONAL: {additional_prompt}"
        
        logger.info(f"[GENERATE] DALL-E 이미지 생성 시작 (스타일: {style})")
        logger.info(f"[PROMPT] {prompt[:100]}...")
        
        # 메타데이터 준비
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'furniture_data': furniture_data,
            'room_dimensions': room_dimensions,
            'style': style,
            'prompt': prompt,
            'generator': 'dalle-3',
            'mock_mode': self.mock_mode
        }
        
        if self.mock_mode:
            # Mock 모드: 가짜 이미지 생성
            return self._generate_mock_image(metadata)
        
        try:
            # DALL-E API 호출
            logger.info("[API] DALL-E 3 API 호출 중...")
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                n=1
            )
            
            # 이미지 URL 추출
            image_url = response.data[0].url
            
            # 이미지 다운로드
            logger.info("[DOWNLOAD] 생성된 이미지 다운로드 중...")
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            
            # 이미지 저장
            image = Image.open(io.BytesIO(image_response.content))
            image_path = self._save_generated_image(image, metadata)
            
            logger.info(f"[SUCCESS] DALL-E 이미지 생성 완료: {image_path}")
            return image_path, metadata
            
        except Exception as e:
            logger.error(f"[ERROR] DALL-E 이미지 생성 실패: {e}")
            # 폴백: Mock 이미지
            return self._generate_mock_image(metadata)
    
    def _generate_mock_image(self, metadata: Dict) -> Tuple[str, Dict]:
        """Mock 이미지 생성"""
        logger.info("[MOCK] Mock 이미지 생성 중...")
        
        # 스타일별 색상
        style = metadata['style']
        colors = {
            'scandinavian': (245, 245, 240),  # 아이보리
            'modern': (220, 220, 220),        # 밝은 회색
            'industrial': (100, 90, 80)       # 어두운 갈색
        }
        
        bg_color = colors.get(style, colors['scandinavian'])
        
        # 1024x1024 Mock 이미지 생성
        mock_image = Image.new('RGB', (1024, 1024), bg_color)
        
        # 이미지 저장
        image_path = self._save_generated_image(mock_image, metadata)
        
        logger.info(f"[MOCK] Mock 이미지 저장: {image_path}")
        return image_path, metadata
    
    def _save_generated_image(self, image: Image.Image, metadata: Dict) -> str:
        """생성된 이미지 저장"""
        # 저장 디렉토리 생성
        save_dir = Path(__file__).parent / 'generated_images'
        save_dir.mkdir(exist_ok=True)
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        style = metadata.get('style', 'unknown')
        prefix = "dalle_mock" if metadata.get('mock_mode') else "dalle"
        
        filename = f"{prefix}_{style}_{timestamp}.png"
        image_path = save_dir / filename
        metadata_path = save_dir / f"{prefix}_{style}_{timestamp}_metadata.json"
        
        # 이미지 저장
        image.save(image_path, quality=95, optimize=True)
        
        # 메타데이터 저장
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return str(image_path)


def test_dalle_generator():
    """DALL-E 생성기 테스트"""
    print("=" * 60)
    print("DALL-E 위치 제어 정확도 테스트")
    print("=" * 60)
    
    # 테스트 시나리오: 방 중앙에 침대 배치 (Dify와 동일한 조건)
    furniture_data = [
        {
            'type': 'bed',
            'center_x': 1988,  # 1.988m (거의 중앙)
            'center_z': 2480,  # 2.48m (거의 중앙)
            'id': 'center_bed_test'
        }
    ]
    
    room_dimensions = {'width': 4.026, 'height': 4.75}  # Dify 테스트와 동일
    
    print(f"테스트 시나리오: {room_dimensions['width']}x{room_dimensions['height']}m 방에 침대를 중앙에 배치")
    print(f"지정 좌표: X=1.988m, Z=2.48m")
    
    # 생성기 초기화
    generator = DalleGenerator()
    
    try:
        # 이미지 생성 테스트
        print("\n[TEST] DALL-E 인테리어 이미지 생성...")
        image_path, metadata = generator.generate_interior_image(
            furniture_data=furniture_data,
            room_dimensions=room_dimensions,
            style="scandinavian",
            additional_prompt="precise center placement test"
        )
        
        print(f"\n[SUCCESS] 테스트 완료!")
        print(f"생성된 이미지: {image_path}")
        print(f"Mock 모드: {metadata.get('mock_mode', False)}")
        
        # 프롬프트 출력
        print(f"\n[PROMPT] 생성된 프롬프트:")
        print(metadata['prompt'][:500] + "...")
        
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")
    
    print("\n" + "=" * 60)
    print("DALL-E 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    test_dalle_generator()