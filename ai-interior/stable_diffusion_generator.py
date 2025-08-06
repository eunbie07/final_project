"""
Stable Diffusion + ControlNet 기반 AI 인테리어 생성기
정확한 가구 위치 제어를 위한 ControlNet 활용
"""

import torch
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
from datetime import datetime
import logging

# Diffusers 관련 import (조건부)
try:
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
        StableDiffusionPipeline
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# 기존 시스템 import
from layout_mask_generator import LayoutMaskGenerator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StableDiffusionGenerator:
    """Stable Diffusion + ControlNet 인테리어 생성기"""
    
    def __init__(self, 
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 controlnet_model: str = "lllyasviel/sd-controlnet-canny",
                 device: Optional[str] = None,
                 use_controlnet: bool = True,
                 enable_cpu_offload: bool = True):
        """
        초기화
        
        Args:
            model_id: Stable Diffusion 모델 ID
            controlnet_model: ControlNet 모델 ID
            device: 실행 디바이스 ("cuda", "cpu", None=auto)
            use_controlnet: ControlNet 사용 여부
            enable_cpu_offload: CPU 오프로드 활성화 여부
        """
        
        if not DIFFUSERS_AVAILABLE:
            logger.warning("[WARNING] diffusers 라이브러리가 없습니다. 모의(mock) 모드로 실행됩니다.")
            self.mock_mode = True
        else:
            self.mock_mode = False
        
        # 디바이스 자동 감지
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"[DEVICE] 사용 디바이스: {self.device}")
        
        # 모델 정보
        self.model_id = model_id
        self.controlnet_model = controlnet_model
        self.use_controlnet = use_controlnet and not self.mock_mode
        self.enable_cpu_offload = enable_cpu_offload
        
        # 파이프라인 (지연 로딩)
        self.pipe = None
        self.controlnet = None
        
        # 마스크 생성기
        self.mask_generator = LayoutMaskGenerator()
        
        # 생성 설정
        self.default_settings = {
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'controlnet_conditioning_scale': 1.0,
            'width': 1024,
            'height': 1024,
        }
        
        logger.info(f"[INIT] Stable Diffusion 생성기 초기화 (ControlNet: {self.use_controlnet})")
    
    def initialize_pipeline(self):
        """파이프라인 초기화 (지연 로딩)"""
        if self.mock_mode:
            logger.info("[MOCK] 모의 모드로 실행 - 실제 모델 로딩 건너뛰기")
            return
            
        if self.pipe is not None:
            return  # 이미 초기화됨
        
        try:
            if self.use_controlnet:
                logger.info("[LOADING] ControlNet 모델 로딩...")
                self.controlnet = ControlNetModel.from_pretrained(
                    self.controlnet_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                logger.info(f"[LOADING] Stable Diffusion + ControlNet 파이프라인 로딩...")
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    self.model_id,
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                logger.info(f"[LOADING] 기본 Stable Diffusion 파이프라인 로딩...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            
            # 디바이스 이동
            self.pipe = self.pipe.to(self.device)
            
            # 스케줄러 최적화
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # 메모리 최적화
            if self.enable_cpu_offload and self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                logger.info("[OPTIMIZE] CPU 오프로드 활성화")
            
            logger.info("[SUCCESS] 파이프라인 초기화 완료!")
            
        except Exception as e:
            logger.error(f"[ERROR] 파이프라인 초기화 실패: {e}")
            self.mock_mode = True
            logger.info("[FALLBACK] 모의 모드로 전환")
    
    def create_enhanced_prompt(self, 
                             furniture_data: List[Dict],
                             room_dimensions: Dict,
                             style: str = "scandinavian",
                             additional_prompt: str = "") -> str:
        """
        향상된 프롬프트 생성
        
        Args:
            furniture_data: 가구 배치 정보
            room_dimensions: 방 크기 정보  
            style: 인테리어 스타일
            additional_prompt: 추가 프롬프트
            
        Returns:
            str: 완성된 프롬프트
        """
        # 스타일별 기본 프롬프트
        style_prompts = {
            "scandinavian": "Scandinavian interior design, light wood floors, white walls, natural lighting, cozy minimalist atmosphere",
            "modern": "Modern interior design, sleek lines, contemporary furniture, clean aesthetic, neutral colors", 
            "industrial": "Industrial interior design, exposed brick walls, metal fixtures, concrete floors, urban loft style",
            "bohemian": "Bohemian interior design, warm earth tones, textured fabrics, eclectic decorative elements, cozy atmosphere",
            "cozy": "Cozy interior design, soft textures, warm lighting, comfortable furniture, homely atmosphere"
        }
        
        base_style = style_prompts.get(style, style_prompts["scandinavian"])
        
        # 가구 위치 정보 구성
        furniture_details = []
        for furniture in furniture_data:
            ftype = furniture.get('type', 'furniture')
            center_x = furniture.get('center_x', 0) / 1000.0  # mm → m
            center_z = furniture.get('center_z', 0) / 1000.0  # mm → m
            
            furniture_details.append(f"{ftype} positioned at coordinates X={center_x:.2f}m, Z={center_z:.2f}m")
        
        furniture_text = ", ".join(furniture_details)
        
        # 강화된 위치 제어 프롬프트
        enhanced_prompt = f"""{base_style}

CRITICAL FURNITURE PLACEMENT INSTRUCTIONS:
{furniture_text}

MANDATORY REQUIREMENTS:
- Place each furniture item at the EXACT coordinates specified above
- Do not follow conventional placement rules (e.g., beds against walls)
- Maintain realistic scale and proportions
- Ensure all furniture is clearly visible and positioned precisely
- Create photorealistic lighting and shadows

{additional_prompt}

High quality, masterpiece, photorealistic interior photography, professional lighting, 8k resolution""".strip()
        
        return enhanced_prompt
    
    def generate_interior_image(self, 
                              furniture_data: List[Dict],
                              room_dimensions: Dict,
                              style: str = "scandinavian",
                              additional_prompt: str = "",
                              use_mask: bool = True,
                              **generation_kwargs) -> Tuple[str, Dict]:
        """
        AI 인테리어 이미지 생성
        
        Args:
            furniture_data: 가구 배치 정보
            room_dimensions: 방 크기 정보
            style: 인테리어 스타일
            additional_prompt: 추가 프롬프트
            use_mask: 마스크 사용 여부
            **generation_kwargs: 추가 생성 파라미터
            
        Returns:
            Tuple[str, Dict]: (생성된 이미지 경로, 메타데이터)
        """
        # 설정 병합
        settings = {**self.default_settings, **generation_kwargs}
        
        # 프롬프트 생성
        prompt = self.create_enhanced_prompt(
            furniture_data, room_dimensions, style, additional_prompt
        )
        
        logger.info(f"[GENERATE] 이미지 생성 시작 (스타일: {style})")
        logger.info(f"[PROMPT] {prompt[:100]}...")
        
        # 메타데이터 준비
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'furniture_data': furniture_data,
            'room_dimensions': room_dimensions,
            'style': style,
            'prompt': prompt,
            'settings': settings,
            'model_id': self.model_id,
            'controlnet_model': self.controlnet_model if self.use_controlnet else None,
            'device': self.device,
            'use_controlnet': self.use_controlnet,
            'use_mask': use_mask,
            'mock_mode': self.mock_mode
        }
        
        if self.mock_mode:
            # 모의 모드: 가짜 이미지 생성
            return self._generate_mock_image(metadata)
        
        # 파이프라인 초기화
        self.initialize_pipeline()
        
        control_image = None
        if self.use_controlnet and use_mask:
            # ControlNet 마스크 생성
            logger.info("[MASK] 제어 마스크 생성 중...")
            control_image = self.mask_generator.create_canny_edge_mask(
                furniture_data, room_dimensions
            )
            
            # 마스크 저장 (디버깅용)
            mask_dir = Path(__file__).parent / 'generated_masks'
            mask_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mask_path = mask_dir / f'control_mask_{timestamp}.png'
            control_image.save(mask_path)
            metadata['control_mask_path'] = str(mask_path)
        
        try:
            # 이미지 생성
            logger.info("[AI] Stable Diffusion 이미지 생성 중...")
            
            with torch.autocast(self.device):
                if self.use_controlnet and control_image is not None:
                    result = self.pipe(
                        prompt=prompt,
                        image=control_image,
                        num_inference_steps=settings['num_inference_steps'],
                        guidance_scale=settings['guidance_scale'],
                        controlnet_conditioning_scale=settings['controlnet_conditioning_scale'],
                        width=settings['width'],
                        height=settings['height'],
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    )
                else:
                    result = self.pipe(
                        prompt=prompt,
                        num_inference_steps=settings['num_inference_steps'],
                        guidance_scale=settings['guidance_scale'],
                        width=settings['width'],
                        height=settings['height'],
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    )
            
            generated_image = result.images[0]
            
            # 이미지 저장
            image_path = self._save_generated_image(generated_image, metadata)
            
            logger.info(f"[SUCCESS] 이미지 생성 완료: {image_path}")
            return image_path, metadata
            
        except Exception as e:
            logger.error(f"[ERROR] 이미지 생성 실패: {e}")
            # 폴백: 모의 이미지
            return self._generate_mock_image(metadata)
    
    def _generate_mock_image(self, metadata: Dict) -> Tuple[str, Dict]:
        """모의 이미지 생성 (테스트용)"""
        logger.info("[MOCK] 모의 이미지 생성 중...")
        
        # 간단한 테스트 이미지 생성
        width = metadata['settings']['width']
        height = metadata['settings']['height']
        
        # 색상 기반 이미지 생성
        style = metadata['style']
        colors = {
            'scandinavian': (240, 240, 235),  # 연한 베이지
            'modern': (200, 200, 200),        # 회색
            'industrial': (80, 70, 60),       # 어두운 갈색
            'bohemian': (180, 150, 120),      # 따뜻한 베이지
            'cozy': (220, 180, 140)           # 따뜻한 크림색
        }
        
        bg_color = colors.get(style, colors['scandinavian'])
        mock_image = Image.new('RGB', (width, height), bg_color)
        
        # 이미지 저장
        image_path = self._save_generated_image(mock_image, metadata)
        
        logger.info(f"[MOCK] 모의 이미지 저장: {image_path}")
        return image_path, metadata
    
    def _save_generated_image(self, image: Image.Image, metadata: Dict) -> str:
        """생성된 이미지 저장"""
        # 저장 디렉토리 생성
        save_dir = Path(__file__).parent / 'generated_images'
        save_dir.mkdir(exist_ok=True)
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        style = metadata.get('style', 'unknown')
        prefix = "sd_mock" if metadata.get('mock_mode') else "sd"
        
        filename = f"{prefix}_{style}_{timestamp}.png"
        image_path = save_dir / filename
        metadata_path = save_dir / f"{prefix}_{style}_{timestamp}_metadata.json"
        
        # 이미지 저장
        image.save(image_path, quality=95, optimize=True)
        
        # 메타데이터 저장
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return str(image_path)


def test_stable_diffusion_generator():
    """Stable Diffusion 생성기 테스트"""
    print("="*60)
    print("Stable Diffusion 생성기 테스트")
    print("="*60)
    
    # 테스트 시나리오: 방 중앙에 침대 배치
    furniture_data = [
        {
            'type': 'bed',
            'center_x': 2000,  # 2.0m (중앙)
            'center_z': 2000,  # 2.0m (중앙)
            'id': 'center_bed'
        }
    ]
    
    room_dimensions = {'width': 4.0, 'height': 4.0}
    
    print(f"시나리오: 4x4m 방에 침대를 정중앙(2m, 2m)에 배치")
    
    # 생성기 초기화
    generator = StableDiffusionGenerator(
        use_controlnet=True,
        enable_cpu_offload=True
    )
    
    try:
        # 이미지 생성 테스트
        print("\n[TEST] AI 인테리어 이미지 생성...")
        image_path, metadata = generator.generate_interior_image(
            furniture_data=furniture_data,
            room_dimensions=room_dimensions,
            style="scandinavian",
            additional_prompt="bedroom with bed in center of room, unusual placement",
            num_inference_steps=10  # 빠른 테스트를 위해 단축
        )
        
        print(f"\n[SUCCESS] 테스트 완료!")
        print(f"생성된 이미지: {image_path}")
        print(f"모의 모드: {metadata.get('mock_mode', False)}")
        print(f"ControlNet 사용: {metadata.get('use_controlnet', False)}")
        
        # 추가 스타일 테스트
        styles_to_test = ["modern", "industrial"]
        for style in styles_to_test:
            print(f"\n[TEST] {style} 스타일 테스트...")
            image_path, _ = generator.generate_interior_image(
                furniture_data, room_dimensions, style,
                num_inference_steps=5
            )
            print(f"완료: {image_path}")
            
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)


if __name__ == "__main__":
    test_stable_diffusion_generator()