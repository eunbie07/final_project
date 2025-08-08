"""
Stage 2: Photorealistic Processor
2단계 - 스타일이 변경된 이미지를 실사화
"""

import os
import requests
import base64
import time
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class PhotorealisticProcessor:
    """2단계: 실사화 프로세서"""
    
    def __init__(self):
        """초기화"""
        self.stability_api_key = os.getenv('STABILITY_API_KEY')
        self.stability_host = os.getenv('STABILITY_API_HOST', 'https://api.stability.ai')
        self.stability_path = os.getenv('STABILITY_SDXL_PATH', '/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image')
        
        if not self.stability_api_key:
            print("⚠️  STABILITY_API_KEY 환경변수가 설정되지 않았습니다")
        
        # 실사화 프롬프트
        self.photorealistic_prompt = """
        Transform this interior image into a highly photorealistic photograph with professional photography lighting, realistic textures and materials, natural shadows and reflections, high-quality photographic details, proper depth of field, realistic color accuracy, professional interior photography style. Keep EXACTLY the same layout, furniture placement, and composition. Only improve the realism and photo quality.
        """
    
    def process(self, input_path: str, file_id: str) -> str:
        """
        2단계 처리: 실사화
        
        Args:
            input_path: 1단계 출력 이미지 경로
            file_id: 파일 고유 ID
            
        Returns:
            output_path: 실사화된 이미지 경로
        """
        
        print(f"[STAGE2] 실사화 시작")
        print(f"   입력: {input_path}")
        
        try:
            if self.stability_api_key:
                # Stability AI로 실사화 (SDXL image-to-image)
                output_path = self._process_with_stability(input_path, file_id)
            else:
                # Mock 처리 (테스트용)
                output_path = self._process_mock(input_path, file_id)
            
            print(f"   출력: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Stage2 처리 실패: {e}")
            # 실패시 Mock 처리
            return self._process_mock(input_path, file_id)
    
    def _process_with_stability(self, input_path: str, file_id: str) -> str:
        """Stability AI를 사용한 실사화 (SDXL image-to-image)"""
        
        try:
            # 이미지 크기 조정 (SDXL 호환)
            resized_path = self._resize_for_sdxl(input_path, file_id)
            
            # API URL
            url = f"{self.stability_host}{self.stability_path}"
            
            # 실사화 프롬프트
            prompt = "photorealistic interior photograph, professional photography, realistic textures, natural lighting, high quality, architectural photography"
            
            # 일반 FormData 방식으로 변경
            files = {
                'init_image': open(resized_path, 'rb')
            }
            
            data = {
                'text_prompts[0][text]': prompt,
                'text_prompts[0][weight]': '1.5',
                'text_prompts[1][text]': '3d render, CGI, cartoon, anime, sketch, blurry, low quality, distorted',
                'text_prompts[1][weight]': '-1',
                'cfg_scale': '12',
                'sampler': 'K_DPM_2_ANCESTRAL',
                'samples': '1',
                'steps': '50',
                'image_strength': '0.6'
            }
            
            print("   Stability AI로 실사화 중...")
            
            # API 호출
            response = requests.post(
                url,
                headers={
                    'Authorization': f'Bearer {self.stability_api_key}',
                },
                files=files,
                data=data,
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
            output_filename = f"{file_id}_stage2_photorealistic_{timestamp}.png"
            output_path = os.path.join('stage2_output', output_filename)
            
            # 디렉토리 생성
            os.makedirs('stage2_output', exist_ok=True)
            
            # 파일 저장
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            
            return output_path
                
        except Exception as e:
            print(f"Stability AI 실사화 실패: {e}")
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
        resized_path = f"temp_resized_s2_{file_id}.png"
        resized_image.save(resized_path)
        
        print(f"   이미지 리사이즈: {original_width}x{original_height} → {best_size[0]}x{best_size[1]}")
        
        return resized_path
    
    def _process_mock(self, input_path: str, file_id: str) -> str:
        """Mock 실사화 처리 (테스트용)"""
        
        print(f"[MOCK] 실사화 시뮬레이션")
        
        # 입력 이미지 로드
        image = Image.open(input_path)
        
        # 실사화 효과 시뮬레이션
        processed_image = self._apply_photorealistic_effects(image)
        
        # 출력 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{file_id}_stage2_mock_photorealistic_{timestamp}.png"
        output_path = os.path.join('stage2_output', output_filename)
        
        processed_image.save(output_path, quality=95)
        
        return output_path
    
    def _apply_photorealistic_effects(self, image: Image.Image) -> Image.Image:
        """실사화 효과 적용 (Mock)"""
        
        # 1. 선명도 향상
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        # 2. 대비 조정
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # 3. 색상 채도 조정
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        
        # 4. 약간의 노이즈 감소 효과
        image = image.filter(ImageFilter.SMOOTH_MORE)
        
        # 5. 디테일 강화
        image = image.filter(ImageFilter.DETAIL)
        
        return image
    
    def _enhance_lighting(self, image: Image.Image) -> Image.Image:
        """조명 효과 강화"""
        
        # 밝기 조정
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def _add_depth_effects(self, image: Image.Image) -> Image.Image:
        """깊이감 효과 추가"""
        
        # 가장자리 소프트닝으로 깊이감 연출
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        
        return image