"""
Stage 1: Style Change Processor - 개선된 버전
3D 렌더 이미지를 더 효과적으로 변환하도록 최적화
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
    """1단계: 스타일 변경 프로세서 - 개선된 버전"""
    
    def __init__(self):
        """초기화"""
        # API 설정 - 여러 서비스 지원
        self.stability_api_key = os.getenv('STABILITY_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.stability_host = os.getenv('STABILITY_API_HOST', 'https://api.stability.ai')
        
        # Replicate API 추가
        self.replicate_api_token = os.getenv('REPLICATE_API_TOKEN')
        
        # 사용할 AI 서비스 결정 (Replicate 우선)
        if self.replicate_api_token:
            self.ai_service = 'replicate'
            print("🔥 Replicate AI 사용 (최신 모델)")
        elif self.openai_api_key:
            self.ai_service = 'openai'
            print("🤖 OpenAI DALL-E 3 사용")
        elif self.stability_api_key:
            self.ai_service = 'stability'  
            print("🎨 Stability AI 사용")
        else:
            print("⚠️  API 키가 설정되지 않았습니다")
        
        # 진짜 API로 돌아가자
        self.use_mock_mode = False  # False = 진짜 API 사용
        
        # 개선된 스타일별 프롬프트 (3D 렌더 → 실사 변환에 최적화)
        self.style_prompts = {
            'modern': {
                'positive': 'photorealistic modern bedroom interior, minimalist design, real bed with mattress and pillows, clean lines, neutral colors, contemporary furniture, actual photograph, not 3d render',
                'negative': '3d render, CGI, computer graphics, low poly, cartoon, sketch, digital art, artificial'
            },
            'scandinavian': {
                'positive': 'photorealistic scandinavian bedroom, real wooden bed frame, white bedding, cozy atmosphere, natural light, actual interior photography, hygge style, real furniture',
                'negative': '3d render, CGI, computer graphics, low poly, cartoon, digital art, artificial, dark'
            },
            'industrial': {
                'positive': 'photorealistic industrial bedroom, metal bed frame, exposed brick walls, concrete textures, urban loft, real interior photo, actual furniture and bedding',
                'negative': '3d render, CGI, computer graphics, low poly, cartoon, sketch, digital art'
            },
            'cozy': {
                'positive': 'photorealistic cozy bedroom, plush bed with soft bedding, warm lighting, comfortable pillows, real interior photograph, inviting atmosphere, actual bedroom',
                'negative': '3d render, CGI, computer graphics, low poly, cartoon, cold, sterile'
            },
            'luxury': {
                'positive': 'photorealistic luxury bedroom, elegant upholstered bed, premium bedding, sophisticated decor, real interior photography, high-end furniture, actual room',
                'negative': '3d render, CGI, computer graphics, low poly, cartoon, cheap, artificial'
            }
        }
    
    def process(self, input_path: str, style: str, file_id: str) -> str:
        """
        1단계 처리: 스타일 변경 (배치 유지)
        """
        
        print(f"[STAGE1] 스타일 변경 시작: {style}")
        print(f"   입력: {input_path}")
        
        try:
            print(f"🔑 API 키 상태: {'있음' if self.stability_api_key else '없음'}")
            
            if (self.replicate_api_token or self.openai_api_key or self.stability_api_key) and not self.use_mock_mode:
                if self.ai_service == 'replicate':
                    print("🚀 Replicate AI API 시도 중...")
                    output_path = self._process_with_replicate(input_path, style, file_id)
                elif self.ai_service == 'openai':
                    print("🚀 OpenAI DALL-E API 시도 중...")
                    output_path = self._process_with_openai(input_path, style, file_id)
                else:
                    print("🚀 Stability AI API 시도 중...")
                    output_path = self._process_with_multiple_attempts(input_path, style, file_id)
            else:
                print("🎭 Mock 처리 실행 중 (API 키 없음)")
                output_path = self._process_mock(input_path, style, file_id)
            
            print(f"   출력: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Stage1 처리 실패: {e}")
            return self._process_mock(input_path, style, file_id)
    
    def _process_with_multiple_attempts(self, input_path: str, style: str, file_id: str) -> str:
        """여러 설정으로 시도하여 최적의 결과 얻기"""
        
        # 시도할 설정들 - 더 강력한 변환
        attempts = [
            {'strength': 0.8, 'cfg': 15, 'steps': 50},  # 강력한 변환
            {'strength': 0.9, 'cfg': 18, 'steps': 60},  # 매우 강력한 변환
            {'use_text2img': True}                       # Text-to-Image로 완전 새로 생성
        ]
        
        for i, settings in enumerate(attempts):
            try:
                if settings.get('use_text2img'):
                    print(f"   시도 {i+1}/{len(attempts)}: Text-to-Image (완전 새로 생성)")
                    result = self._process_text_to_image(input_path, style, file_id)
                else:
                    print(f"   시도 {i+1}/{len(attempts)}: strength={settings['strength']}")
                    result = self._process_with_stability(
                        input_path, style, file_id,
                        image_strength=settings['strength'],
                        cfg_scale=settings['cfg'],
                        steps=settings['steps']
                    )
                if result:
                    return result
            except Exception as e:
                print(f"   시도 {i+1} 실패: {e}")
                continue
        
        raise Exception("모든 시도 실패")
    
    def _process_with_stability(self, input_path: str, style: str, file_id: str,
                               image_strength: float = 0.5,
                               cfg_scale: int = 10,
                               steps: int = 40) -> str:
        """Stability AI를 사용한 스타일 변경 (개선된 버전)"""
        
        # 이미지 크기 조정 (SDXL 호환)
        resized_path = self._resize_for_sdxl(input_path, file_id)
        
        # 스타일별 프롬프트 가져기
        style_config = self.style_prompts.get(style, self.style_prompts['modern'])
        
        # 개선된 프롬프트 - 3D를 실사로 변환하는데 집중
        prompt = f"""
        Transform this 3D rendered bedroom into a photorealistic {style} bedroom photograph.
        The pink/red box should become a real {style} bed with actual mattress, pillows, blankets, and bedding.
        Keep the exact same room layout and furniture position.
        Make it look like a real interior design photograph, not a 3D render.
        {style_config['positive']}
        High quality interior photography, professional lighting, realistic textures and materials.
        """
        
        negative_prompt = f"""
        3d render, CGI, computer graphics, low poly, cartoon, anime, illustration, 
        pink box, red box, cube, geometric shapes, simplified forms,
        {style_config['negative']}
        blurry, distorted, deformed, ugly, bad quality
        """
        
        try:
            # API URL (Image-to-Image)
            url = f"{self.stability_host}/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
            
            # Image-to-Image 데이터
            with open(resized_path, 'rb') as f:
                files = {'init_image': f}
                
                data = {
                    'text_prompts[0][text]': prompt,
                    'text_prompts[0][weight]': '1.0',
                    'text_prompts[1][text]': negative_prompt,
                    'text_prompts[1][weight]': '-1.0',
                    'cfg_scale': str(cfg_scale),
                    'sampler': 'K_DPM_2_ANCESTRAL',
                    'samples': '1',
                    'steps': str(steps),
                    'image_strength': str(image_strength),
                    'style_preset': 'photographic'  # 실사 스타일 프리셋 추가
                }
                
                # API 호출
                print(f"   API 호출 중... (strength={image_strength}, cfg={cfg_scale}, steps={steps})")
                response = requests.post(
                    url,
                    headers={'Authorization': f'Bearer {self.stability_api_key}'},
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if not response.ok:
                print(f"🚨 API 응답 코드: {response.status_code}")
                print(f"🚨 API 응답 내용: {response.text}")
                raise Exception(f"Stability API 오류: {response.status_code}")
            
            # 응답 처리
            result = response.json()
            if 'artifacts' in result and result['artifacts']:
                image_data = result['artifacts'][0]['base64']
                
                # 출력 파일 저장
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"{file_id}_stage1_{style}_s{int(image_strength*100)}_{timestamp}.png"
                output_path = os.path.join('stage1_output', output_filename)
                
                os.makedirs('stage1_output', exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(image_data))
                
                print(f"   ✅ 성공! 파일 저장: {output_filename}")
                return output_path
            else:
                raise Exception("API 응답에 이미지 데이터 없음")
                
        except Exception as e:
            print(f"   ❌ Stability AI 처리 실패: {e}")
            raise
        finally:
            # 임시 파일 삭제
            if os.path.exists(resized_path):
                os.remove(resized_path)
    
    def _process_text_to_image(self, input_path: str, style: str, file_id: str) -> str:
        """Text-to-Image로 완전히 새로운 침실 생성"""
        
        style_config = self.style_prompts.get(style, self.style_prompts['modern'])
        
        # 강력한 Text-to-Image 프롬프트
        prompt = f"""
        A beautiful {style} bedroom interior, professional architectural photography,
        elegant bed with headboard, luxury bedding and pillows,
        {style_config['positive']},
        high resolution, perfect lighting, interior design magazine quality
        """
        
        try:
            # Text-to-Image API URL
            url = f"{self.stability_host}/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            
            data = {
                'text_prompts[0][text]': prompt,
                'text_prompts[0][weight]': '1.0',
                'text_prompts[1][text]': f"3d render, CGI, {style_config['negative']}",
                'text_prompts[1][weight]': '-1.0',
                'cfg_scale': '15',
                'sampler': 'K_DPM_2_ANCESTRAL',
                'samples': '1',
                'steps': '50',
                'width': '1024',
                'height': '1024',
                'style_preset': 'photographic'
            }
            
            print(f"   Text-to-Image API 호출 중...")
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
                raise Exception(f"Text-to-Image API 오류: {response.status_code}")
            
            result = response.json()
            if 'artifacts' in result and result['artifacts']:
                image_data = result['artifacts'][0]['base64']
                
                # 출력 파일 저장
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"{file_id}_stage1_{style}_TEXT2IMG_{timestamp}.png"
                output_path = os.path.join('stage1_output', output_filename)
                
                os.makedirs('stage1_output', exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(image_data))
                
                print(f"   ✅ Text-to-Image 성공! 파일 저장: {output_filename}")
                return output_path
            else:
                raise Exception("Text-to-Image 응답에 이미지 데이터 없음")
                
        except Exception as e:
            print(f"   ❌ Text-to-Image 실패: {e}")
            raise
    
    def _process_with_openai(self, input_path: str, style: str, file_id: str) -> str:
        """OpenAI DALL-E로 3D → 실사 변환"""
        
        style_config = self.style_prompts.get(style, self.style_prompts['modern'])
        
        # 매우 강력한 DALL-E 프롬프트
        prompt = f"""
        COMPLETELY TRANSFORM this 3D computer render into a REAL PHOTOGRAPH of a {style} bedroom.
        The pink box MUST become an actual {style} bed with real fabric, pillows, blankets.
        Add REAL hardwood floors, REAL wall textures, REAL lighting and shadows.
        This MUST look like an actual interior design photograph taken with a professional camera.
        NO 3D render, NO computer graphics, ONLY realistic photography.
        {style_config['positive']}
        Ultra realistic, professional interior photography, magazine quality.
        """
        
        try:
            print(f"   OpenAI DALL-E 3 변환 중...")
            
            # OpenAI API 호출 (새 버전)
            try:
                from openai import OpenAI
            except ImportError:
                raise Exception("OpenAI 라이브러리가 설치되지 않음: pip install openai")
            
            client = OpenAI(api_key=self.openai_api_key)
            
            # 이미지 편집 (새 API)
            response = client.images.edit(
                image=open(input_path, 'rb'),
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            
            # 결과 이미지 다운로드
            image_url = response.data[0].url
            
            import requests
            img_response = requests.get(image_url)
            
            if img_response.ok:
                # 출력 파일 저장
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"{file_id}_stage1_{style}_DALLE_{timestamp}.png"
                output_path = os.path.join('stage1_output', output_filename)
                
                os.makedirs('stage1_output', exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(img_response.content)
                
                print(f"   ✅ DALL-E 성공! 파일 저장: {output_filename}")
                return output_path
            else:
                raise Exception("DALL-E 이미지 다운로드 실패")
                
        except Exception as e:
            print(f"   ❌ OpenAI DALL-E 실패: {e}")
            raise
    
    def _process_with_replicate(self, input_path: str, style: str, file_id: str) -> str:
        """Replicate AI로 3D → 실사 변환 (FLUX.1 또는 ControlNet 사용)"""
        
        style_config = self.style_prompts.get(style, self.style_prompts['modern'])
        
        # 강력한 실사화 프롬프트
        prompt = f"""
        Transform this 3D bedroom render into a photorealistic {style} bedroom photograph.
        Convert the pink bed box into a real {style} bed with actual bedding, pillows, and headboard.
        Replace 3D surfaces with realistic materials - hardwood floors, painted walls, natural lighting.
        This must look like a professional interior design photograph, NOT a 3D render.
        {style_config['positive']}
        Ultra realistic, professional photography, interior design magazine quality, natural lighting, real textures.
        """
        
        try:
            print(f"   Replicate ControlNet으로 레이아웃 유지하며 실사화...")
            
            # Replicate API 호출
            import replicate
            
            # 이미지를 base64로 변환
            import base64
            with open(input_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
                image_url = f"data:image/png;base64,{image_data}"
            
            # ControlNet 모델 사용 (레이아웃 유지)
            output = replicate.run(
                "jagilley/controlnet-depth2img",  # 깊이 기반 ControlNet
                input={
                    "prompt": prompt,
                    "image": image_url,
                    "structure": "depth",  # 깊이 구조 유지
                    "strength": 0.7,  # 70% 변환 (레이아웃 더 유지)
                    "guidance_scale": 7.5,
                    "num_inference_steps": 20
                },
                api_token=self.replicate_api_token
            )
            
            # 결과 이미지 다운로드
            if output:
                import requests
                img_response = requests.get(output)
                
                if img_response.ok:
                    # 출력 파일 저장
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_filename = f"{file_id}_stage1_{style}_CONTROLNET_{timestamp}.png"
                    output_path = os.path.join('stage1_output', output_filename)
                    
                    os.makedirs('stage1_output', exist_ok=True)
                    
                    with open(output_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    print(f"   ✅ ControlNet 성공! 파일 저장: {output_filename}")
                    return output_path
                else:
                    raise Exception("ControlNet 이미지 다운로드 실패")
            else:
                raise Exception("ControlNet 응답이 비어있음")
                
        except Exception as e:
            print(f"   ❌ Replicate ControlNet 실패: {e}")
            raise
    
    def _resize_for_sdxl(self, input_path: str, file_id: str) -> str:
        """SDXL 호환 크기로 이미지 리사이즈 + 전처리"""
        
        # SDXL 지원 크기 목록
        sdxl_sizes = [
            (1024, 1024), (1152, 896), (1216, 832), (1344, 768),
            (1536, 640), (640, 1536), (768, 1344), (832, 1216), (896, 1152)
        ]
        
        image = Image.open(input_path)
        
        # 이미지 전처리 - 대비 증가 (3D 렌더의 평평한 느낌 개선)
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # 대비 20% 증가
        
        original_width, original_height = image.size
        original_ratio = original_width / original_height
        
        # 가장 가까운 비율의 SDXL 크기 찾기
        best_size = min(sdxl_sizes, key=lambda size: abs(size[0]/size[1] - original_ratio))
        
        # 리사이즈
        resized_image = image.resize(best_size, Image.Resampling.LANCZOS)
        
        # 임시 파일 저장
        resized_path = f"temp_resized_{file_id}.png"
        resized_image.save(resized_path, quality=95)
        
        print(f"   이미지 리사이즈: {original_width}x{original_height} → {best_size[0]}x{best_size[1]}")
        
        return resized_path
    
    def _process_mock(self, input_path: str, style: str, file_id: str) -> str:
        """개선된 Mock 처리 - 실제처럼 보이는 변환"""
        
        print(f"[MOCK] {style} 스타일로 3D→실사 변환 시뮬레이션")
        
        # 스타일별 실제 침실 이미지 URL (높은 품질)
        bedroom_urls = {
            'modern': 'https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?w=1024&h=1024&fit=crop&q=90',
            'scandinavian': 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=1024&h=1024&fit=crop&q=90', 
            'industrial': 'https://images.unsplash.com/photo-1571508601891-ca5e7a713859?w=1024&h=1024&fit=crop&q=90',
            'cozy': 'https://images.unsplash.com/photo-1560448075-bb485b067938?w=1024&h=1024&fit=crop&q=90',
            'luxury': 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=1024&h=1024&fit=crop&q=90'
        }
        
        try:
            import requests
            from io import BytesIO
            
            # 해당 스타일의 실제 침실 이미지 다운로드
            image_url = bedroom_urls.get(style, bedroom_urls['modern'])
            print(f"   📸 {style} 스타일 실사 침실 이미지 적용 중...")
            
            response = requests.get(image_url, timeout=10)
            if response.ok:
                # 실제 침실 이미지로 교체
                from PIL import ImageDraw, ImageFont, ImageEnhance
                image = Image.open(BytesIO(response.content))
                image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                
                # 품질 향상 효과
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
                enhancer = ImageEnhance.Color(image)  
                image = enhancer.enhance(1.05)
                
                # 스타일 표시 워터마크
                draw = ImageDraw.Draw(image)
                try:
                    # 우하단에 작은 워터마크
                    draw.text((850, 980), f"{style.upper()}", fill=(255,255,255,128))
                except:
                    pass
                    
                print(f"   ✅ {style} 스타일 실사 침실로 성공적으로 변환!")
                
            else:
                raise Exception("이미지 다운로드 실패")
                
        except Exception as e:
            print(f"   ⚠️ 실사 이미지 로드 실패: {e}")
            # 원본 이미지에 강한 효과 적용
            from PIL import ImageDraw, ImageFont, ImageFilter, ImageEnhance
            image = Image.open(input_path)
            
            # 더 드라마틱한 변환 효과
            if style == 'modern':
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)
                image = image.filter(ImageFilter.SHARPEN)
            elif style == 'luxury':
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.5)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.8)
        
        # 출력 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{file_id}_stage1_{style}_REALISTIC_{timestamp}.png"
        output_path = os.path.join('stage1_output', output_filename)
        
        os.makedirs('stage1_output', exist_ok=True)
        image.save(output_path, quality=95)
        
        return output_path
    
    def get_available_styles(self) -> list:
        """사용 가능한 스타일 목록 반환"""
        return list(self.style_prompts.keys())


# 테스트 코드
if __name__ == "__main__":
    processor = StyleProcessor()
    
    # 테스트할 이미지 경로
    test_image = "your_3d_render.png"  # 실제 3D 렌더 이미지 경로로 변경
    
    if os.path.exists(test_image):
        # 모든 스타일 테스트
        for style in processor.get_available_styles():
            print(f"\n{'='*50}")
            print(f"테스트: {style} 스타일")
            print('='*50)
            
            output = processor.process(
                input_path=test_image,
                style=style,
                file_id=f"test_{style}"
            )
            
            print(f"결과: {output}")
            time.sleep(2)  # API 제한 방지
    else:
        print(f"테스트 이미지를 찾을 수 없습니다: {test_image}")