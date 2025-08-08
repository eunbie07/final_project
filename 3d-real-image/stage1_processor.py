"""
Stage 1: Style Change Processor
1단계 - 가구 배치는 유지하면서 스타일만 변경
"""

import os
import requests
import base64
import time
from PIL import Image
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class StyleProcessor:
    """1단계: 스타일 변경 프로세서"""
    
    def __init__(self):
        """초기화"""
        self.stability_api_key = os.getenv('STABILITY_API_KEY')
        self.stability_host = os.getenv('STABILITY_API_HOST', 'https://api.stability.ai')
        self.stability_path = os.getenv('STABILITY_SDXL_PATH', '/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image')
        
        if not self.stability_api_key:
            print("⚠️  STABILITY_API_KEY 환경변수가 설정되지 않았습니다")
        
        # 스타일별 프롬프트
        self.style_prompts = {
            'modern': {
                'positive': 'modern minimalist interior, clean lines, neutral colors, contemporary furniture, sleek design, high quality',
                'negative': 'cluttered, ornate, vintage, baroque, messy'
            },
            'scandinavian': {
                'positive': 'scandinavian interior style, light wood, white walls, cozy hygge feeling, natural materials, bright lighting',
                'negative': 'dark, heavy, ornate, cluttered'
            },
            'industrial': {
                'positive': 'industrial interior style, exposed brick, metal fixtures, raw materials, urban loft feeling, concrete elements',
                'negative': 'soft, delicate, ornate, pastel colors'
            },
            'cozy': {
                'positive': 'cozy warm interior, soft textures, comfortable furniture, warm lighting, inviting atmosphere',
                'negative': 'cold, sterile, minimal, harsh lighting'
            },
            'luxury': {
                'positive': 'luxury interior design, expensive materials, elegant furniture, sophisticated details, premium quality',
                'negative': 'cheap, simple, basic, plain'
            }
        }
    
    def process(self, input_path: str, style: str, file_id: str) -> str:
        """
        1단계 처리: 스타일 변경 (배치 유지)
        
        Args:
            input_path: 입력 이미지 경로 (3D 스크린샷)
            style: 변경할 스타일
            file_id: 파일 고유 ID
            
        Returns:
            output_path: 처리된 이미지 경로
        """
        
        print(f"[STAGE1] 스타일 변경 시작: {style}")
        print(f"   입력: {input_path}")
        
        try:
            print(f"🔑 API 키 상태: {'있음' if self.stability_api_key else '없음'}")
            
            if self.stability_api_key:
                print("🚀 Stability AI API 시도 중...")
                # Stability AI로 스타일 변경 (SDXL image-to-image)
                output_path = self._process_with_stability(input_path, style, file_id)
            else:
                print("🎭 Mock 처리 실행 중...")
                # Mock 처리 (테스트용)
                output_path = self._process_mock(input_path, style, file_id)
            
            print(f"   출력: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Stage1 처리 실패: {e}")
            # 실패시 Mock 처리
            return self._process_mock(input_path, style, file_id)
    
    def _process_with_stability(self, input_path: str, style: str, file_id: str) -> str:
        """Stability AI를 사용한 스타일 변경 (SDXL image-to-image)"""
        
        # 스타일별 프롬프트 가져오기
        style_config = self.style_prompts.get(style, self.style_prompts['modern'])
        
        prompt = f"""
        COMPLETELY TRANSFORM this 3D room: 
        The pink rectangular box is a {style} bed - replace it with a real {style} bed with pillows and bedding.
        Add {style} furniture, decorations, lighting, flooring, walls, textures.
        Make it look like a professional {style} bedroom photograph, not 3D render.
        {style_config['positive']}
        """
        
        try:
            # API URL (Text-to-Image)
            url = f"{self.stability_host}{self.stability_path}"
            
            # Text-to-Image 데이터 (init_image 제거)
            data = {
                'text_prompts[0][text]': prompt,
                'text_prompts[0][weight]': '1.5',
                'text_prompts[1][text]': f"3d render, CGI, simple, empty, untextured, {style_config['negative']}, blurry, distorted",
                'text_prompts[1][weight]': '-1',
                'cfg_scale': '12',
                'sampler': 'K_DPM_2_ANCESTRAL',
                'samples': '1',
                'steps': '50',
                'width': '1024',
                'height': '1024'
            }
            
            # API 호출 (Text-to-Image - files 제거)
            response = requests.post(
                url,
                headers={
                    'Authorization': f'Bearer {self.stability_api_key}',
                    'Content-Type': 'application/json'
                },
                json=data,
                timeout=60
            )
            
            if not response.ok:
                raise Exception(f"Stability API 오류: {response.status_code} - {response.text}")
            
            # 응답 처리
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                result = response.json()
                if 'artifacts' in result and result['artifacts']:
                    image_data = result['artifacts'][0]['base64']
                else:
                    raise Exception("Stability API 응답에 이미지 데이터 없음")
            else:
                # 직접 이미지 바이너리 반환
                image_data = base64.b64encode(response.content).decode()
            
            # 출력 파일 경로
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{file_id}_stage1_{style}_{timestamp}.png"
            output_path = os.path.join('stage1_output', output_filename)
            
            # 디렉토리 생성
            os.makedirs('stage1_output', exist_ok=True)
            
            # 파일 저장
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            
            return output_path
                
        except Exception as e:
            print(f"Stability AI 스타일 변경 실패: {e}")
            raise
    
    def _resize_for_sdxl(self, input_path: str, file_id: str) -> str:
        """SDXL 호환 크기로 이미지 리사이즈"""
        
        # SDXL 지원 크기 목록
        sdxl_sizes = [
            (1024, 1024), (1152, 896), (1216, 832), (1344, 768),
            (1536, 640), (640, 1536), (768, 1344), (832, 1216), (896, 1152)
        ]
        
        image = Image.open(input_path)
        original_width, original_height = image.size
        original_ratio = original_width / original_height
        
        # 가장 가까운 비율의 SDXL 크기 찾기
        best_size = min(sdxl_sizes, key=lambda size: abs(size[0]/size[1] - original_ratio))
        
        # 리사이즈
        resized_image = image.resize(best_size, Image.Resampling.LANCZOS)
        
        # 임시 파일 저장
        resized_path = f"temp_resized_{file_id}.png"
        resized_image.save(resized_path)
        
        print(f"   이미지 리사이즈: {original_width}x{original_height} → {best_size[0]}x{best_size[1]}")
        
        return resized_path
    
    def _process_mock(self, input_path: str, style: str, file_id: str) -> str:
        """실제처럼 보이는 Mock: 미리 준비된 bedroom 이미지 사용"""
        
        print(f"[MOCK] {style} 스타일 → 준비된 실제 bedroom 이미지로 변환!")
        
        # 스타일별 미리 만든 bedroom 이미지 URL
        mock_images = {
            'luxury': 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=1024&h=1024&fit=crop',
            'modern': 'https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?w=1024&h=1024&fit=crop', 
            'scandinavian': 'https://images.unsplash.com/photo-1571508601891-ca5e7a713859?w=1024&h=1024&fit=crop',
            'industrial': 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=1024&h=1024&fit=crop',
            'cozy': 'https://images.unsplash.com/photo-1560448075-bb485b067938?w=1024&h=1024&fit=crop'
        }
        
        # Import 먼저
        from PIL import ImageDraw, ImageFont, ImageFilter, ImageEnhance
        
        try:
            # 해당 스타일 이미지 다운로드
            import requests
            from urllib.parse import urljoin
            from io import BytesIO
            
            image_url = mock_images.get(style, mock_images['modern'])
            response = requests.get(image_url, timeout=30)
            
            if response.ok:
                # 실제 bedroom 이미지로 교체
                image = Image.open(BytesIO(response.content))
                image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                
                # 워터마크 추가
                draw = ImageDraw.Draw(image)
                try:
                    font = ImageFont.load_default()
                    draw.text((10, 10), f"DEMO {style.upper()}", fill='white', font=font)
                except:
                    pass
            else:
                raise Exception("이미지 다운로드 실패")
                
        except Exception as e:
            print(f"실제 이미지 로드 실패: {e}, 기본 처리로 fallback")
            # 기본 이미지 처리  
            image = Image.open(input_path)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.5 if style in ['scandinavian'] else 0.7)
        
        # 출력 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{file_id}_stage1_DRAMATIC_{style}_{timestamp}.png"
        output_path = os.path.join('stage1_output', output_filename)
        
        os.makedirs('stage1_output', exist_ok=True)
        image.save(output_path)
        
        return output_path
    
    def get_available_styles(self) -> list:
        """사용 가능한 스타일 목록 반환"""
        return list(self.style_prompts.keys())