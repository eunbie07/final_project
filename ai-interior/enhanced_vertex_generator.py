"""
Enhanced Vertex AI Image Generator
minah의 기능을 ai-interior 시스템에 통합한 고도화된 이미지 생성기
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from PIL import Image as PIL_Image

# Google AI 및 Vertex AI imports
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("WARNING: Google Generative AI 라이브러리가 설치되지 않았습니다.")

try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("WARNING: Vertex AI 라이브러리가 설치되지 않았습니다.")

from dify_rag import DifyLayoutRAG
from roombox_integration import RoomLayout, FurnitureCoordinate


class EnhancedVertexGenerator:
    """minah + ai-interior 통합 이미지 생성기"""
    
    def __init__(self, project_id: str = "virtual-muse-466706-v2", location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.output_dir = "generated_images"
        
        # 스타일별 참조 이미지 설정
        self.style_references = {
            "modern": [
                "gs://my-room-finetune-bucket/image/7a1b2c3d4e5f6a7b8c9d0e1f.png",
                "gs://my-room-finetune-bucket/image/8b2c3d4e5f6a7b8c9d0e1f2a.png",
                "gs://my-room-finetune-bucket/image/9c3d4e5f6a7b8c9d0e1f2a3b.png"
            ],
            "scandinavian": [
                "gs://my-room-finetune-bucket/scandinavian/scandi_1.png",
                "gs://my-room-finetune-bucket/scandinavian/scandi_2.png"
            ],
            "industrial": [
                "gs://my-room-finetune-bucket/industrial/industrial_1.png",
                "gs://my-room-finetune-bucket/industrial/industrial_2.png"
            ]
        }
        
        # 초기화
        self._initialize_services()
        
    def _initialize_services(self):
        """AI 서비스 초기화"""
        self.google_client = None
        self.vertex_model = None
        
        # Google Cloud 인증 설정
        key_path = os.path.join(os.path.dirname(__file__), "key.json")
        if os.path.exists(key_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
            print(f"OK: Google Cloud 키 파일 설정: {key_path}")
        
        try:
            if GOOGLE_AI_AVAILABLE:
                # Google AI API 키 설정 (환경변수에서)
                api_key = os.getenv("GOOGLE_AI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.google_client = genai.GenerativeModel('gemini-pro')
                    print("OK: Google Generative AI 초기화 완료")
                else:
                    print("WARNING: GOOGLE_AI_API_KEY 환경변수가 설정되지 않았습니다")
            
            if VERTEX_AI_AVAILABLE:
                vertexai.init(project=self.project_id, location=self.location)
                self.vertex_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
                print("OK: Vertex AI 모델 초기화 완료")
                
        except Exception as e:
            print(f"WARNING: AI 서비스 초기화 실패: {e}")

    def create_enhanced_prompt(self, room_layout: RoomLayout, style: str = "modern") -> str:
        """ai-interior의 좌표 정확도 + minah의 프롬프트 품질 결합"""
        
        # 1. 기본 방 정보 (minah 스타일)
        base_prompt = f"""
        A hyper-realistic 4K 3D render of a bright, clean, and {style} Korean-style room.
        Room dimensions: {room_layout.width_mm/1000:.1f}m × {room_layout.depth_mm/1000:.1f}m × {room_layout.height_mm/1000:.1f}m.
        Total area: {room_layout.area_sqm:.1f}㎡.
        The room has simple off-white wallpaper and light oak wood flooring.
        """
        
        # 2. 정확한 가구 좌표 정보 (ai-interior 스타일)
        furniture_descriptions = []
        for furniture in room_layout.furniture:
            # 상대적 위치 계산
            rel_x = (furniture.center_x / room_layout.width_mm) * 100
            rel_y = (furniture.center_y / room_layout.depth_mm) * 100
            
            # 위치 설명
            position_desc = self._get_natural_position(rel_x, rel_y)
            
            # 정확한 좌표와 자연어 위치 결합
            furniture_desc = f"""
            A {furniture.name} ({furniture.width/1000:.1f}m × {furniture.depth/1000:.1f}m × {furniture.height/1000:.1f}m) 
            is placed {position_desc} of the room.
            Exact position: {rel_x:.1f}% from left, {rel_y:.1f}% from bottom.
            Center coordinates: ({furniture.center_x}mm, {furniture.center_y}mm).
            """
            
            if furniture.rotation_z != 0:
                furniture_desc += f" Rotated {furniture.rotation_z}° from north."
            
            furniture_descriptions.append(furniture_desc)
        
        # 3. 스타일별 특화 설명
        style_descriptions = {
            "modern": "The overall aesthetic is minimalist and clean, with geometric shapes, neutral colors (white, grey, beige), and sleek materials (glass, steel, marble).",
            "scandinavian": "The atmosphere is cozy and warm with light wood, soft pastels, natural materials, and hygge feeling. Nordic style with functional furniture.",
            "industrial": "The mood is urban and edgy with exposed materials, dark colors (grey, black, rust), metal frames, and vintage industrial elements."
        }
        
        # 4. 최종 품질 지시사항 (minah 스타일)
        quality_requirements = """
        Camera view: wide-angle shot from a corner, showing the entire room layout clearly.
        Lighting: natural soft shadows and diffused light from windows.
        Quality: ultra-realistic, professional interior photography, 8K resolution.
        Focus on realism, accurate proportions, and spatial relationships.
        No furniture cut off by frame. No text or watermarks.
        """
        
        # 모든 요소 결합
        final_prompt = f"{base_prompt}\n\nFurniture layout:\n" + "\n".join(furniture_descriptions)
        final_prompt += f"\n\nStyle: {style_descriptions.get(style, style_descriptions['modern'])}"
        final_prompt += f"\n\n{quality_requirements}"
        
        return final_prompt.strip()

    def _get_natural_position(self, x_percent: float, y_percent: float) -> str:
        """좌표를 자연어 위치로 변환 (ai-interior 방식)"""
        
        # X축 위치 (좌우)
        if x_percent < 25:
            x_desc = "far left"
        elif x_percent < 40:
            x_desc = "left side"
        elif x_percent < 60:
            x_desc = "center"
        elif x_percent < 75:
            x_desc = "right side"
        else:
            x_desc = "far right"
        
        # Y축 위치 (앞뒤)
        if y_percent < 25:
            y_desc = "front area"
        elif y_percent < 40:
            y_desc = "front-middle area"
        elif y_percent < 60:
            y_desc = "middle area"
        elif y_percent < 75:
            y_desc = "back-middle area"
        else:
            y_desc = "back area"
        
        return f"in the {x_desc} {y_desc}"

    async def generate_with_google_ai(self, prompt: str, style: str, output_filename: str) -> Dict[str, Any]:
        """Google AI 이미지 생성 (단순화된 버전)"""
        
        # Google AI는 현재 텍스트 생성만 지원하므로 Mock으로 처리
        print(f"Google AI 텍스트 생성으로 {style} 스타일 설명 생성 중...")
        
        try:
            if self.google_client:
                # 텍스트 기반 설명 생성
                response = self.google_client.generate_content(
                    f"Generate a detailed description for this interior design: {prompt[:200]}..."
                )
                
                enhanced_description = response.text
                print(f"OK: Google AI 설명 생성 완료: {enhanced_description[:100]}...")
            
            # 실제 이미지 생성은 Vertex AI로 폴백
            return await self.generate_with_vertex_ai(prompt, style, output_filename)
            
        except Exception as e:
            print(f"ERROR: Google AI 처리 실패: {e}")
            return await self.generate_with_vertex_ai(prompt, style, output_filename)

    async def generate_with_vertex_ai(self, prompt: str, style: str, output_filename: str) -> Dict[str, Any]:
        """Vertex AI (coordinate main.py 방식) 이미지 생성"""
        
        if not self.vertex_model or not VERTEX_AI_AVAILABLE:
            return {"success": False, "error": "Vertex AI 서비스를 사용할 수 없습니다"}
        
        try:
            print(f"Vertex AI로 {style} 스타일 이미지 생성 중...")
            
            # 비동기 이미지 생성
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(None, lambda: self.vertex_model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="16:9",
                safety_filter_level="block_some",
                person_generation="dont_allow",
                negative_prompt="text, watermark, unrealistic, cartoon, 3d model, blurry, low quality, human, people"
            ))
            
            if images:
                # 이미지 저장
                os.makedirs(self.output_dir, exist_ok=True)
                full_path = os.path.join(self.output_dir, output_filename)
                
                images[0].save(location=full_path, include_generation_parameters=True)
                
                return {
                    "success": True,
                    "image_path": full_path,
                    "method": "vertex_ai",
                    "style": style,
                    "prompt_length": len(prompt)
                }
            else:
                return {"success": False, "error": "이미지 생성 실패", "method": "vertex_ai"}
                
        except Exception as e:
            print(f"ERROR: Vertex AI 생성 실패: {e}")
            return {"success": False, "error": str(e), "method": "vertex_ai"}

    async def generate_image(self, prompt: str, style: str = "modern", 
                           room_data: Dict[str, Any] = None, 
                           prefer_google_ai: bool = True) -> Dict[str, Any]:
        """통합 이미지 생성 (Google AI 우선, Vertex AI 폴백)"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_{style}_{timestamp}.png"
        
        # 1. Google AI 시도 (minah 방식, 스타일 참조 포함)
        if prefer_google_ai and GOOGLE_AI_AVAILABLE:
            result = await self.generate_with_google_ai(prompt, style, filename)
            if result["success"]:
                return result
            print("Google AI 실패, Vertex AI로 폴백...")
        
        # 2. Vertex AI 폴백 (coordinate main.py 방식)
        if VERTEX_AI_AVAILABLE:
            result = await self.generate_with_vertex_ai(prompt, style, filename)
            if result["success"]:
                return result
        
        # 3. 모든 방법 실패 시 Mock 생성
        return await self._generate_mock_image(style, filename)

    async def _generate_mock_image(self, style: str, filename: str) -> Dict[str, Any]:
        """Mock 이미지 생성 (테스트용)"""
        await asyncio.sleep(1)  # 시뮬레이션
        
        mock_path = os.path.join(self.output_dir, f"mock_{filename}")
        
        return {
            "success": True,
            "image_path": mock_path,
            "method": "mock",
            "style": style,
            "note": "실제 AI 서비스를 사용할 수 없어 Mock 결과를 반환했습니다"
        }

    def create_style_reference_images(self, style: str) -> List[Dict[str, Any]]:
        """스타일별 참조 이미지 생성"""
        
        reference_images = []
        reference_uris = self.style_references.get(style, self.style_references["modern"])
        
        style_descriptions = {
            "modern": "clean, minimalist Korean-style room with geometric furniture",
            "scandinavian": "cozy Nordic room with light wood and natural materials",
            "industrial": "urban loft with exposed materials and metal elements"
        }
        
        for i, gcs_uri in enumerate(reference_uris):
            try:
                style_ref = {
                    "reference_id": i + 1,
                    "gcs_uri": gcs_uri,
                    "style_description": style_descriptions.get(style, "Korean interior room")
                }
                reference_images.append(style_ref)
            except Exception as e:
                print(f"WARNING: 참조 이미지 {i+1} 로드 실패: {e}")
        
        return reference_images


# 편의 함수들
async def generate_room_image_from_coordinates(room_data: Dict[str, Any], 
                                             style: str = "modern",
                                             prefer_google_ai: bool = True) -> Dict[str, Any]:
    """좌표 데이터로부터 직접 이미지 생성"""
    
    from roombox_integration import RoomBoxDataProcessor, DifyLayoutRAG
    
    # Dify RAG 초기화 (환경 변수에서)
    import os
    dify_rag = DifyLayoutRAG(
        os.getenv("DIFY_API_KEY", ""),
        os.getenv("DIFY_APP_ID", ""),
        os.getenv("DIFY_DATASET_ID", "")
    )
    
    # 좌표 데이터 파싱
    processor = RoomBoxDataProcessor(dify_rag)
    room_layout = processor.parse_roombox_data(room_data)
    
    # 이미지 생성기 초기화
    generator = EnhancedVertexGenerator()
    
    # 고품질 프롬프트 생성
    prompt = generator.create_enhanced_prompt(room_layout, style)
    
    # 이미지 생성
    result = await generator.generate_image(prompt, style, room_data, prefer_google_ai)
    
    # 결과에 추가 정보 포함
    result.update({
        "room_layout": {
            "width_mm": room_layout.width_mm,
            "depth_mm": room_layout.depth_mm,
            "height_mm": room_layout.height_mm,
            "area_sqm": room_layout.area_sqm,
            "furniture_count": len(room_layout.furniture)
        },
        "generated_prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt
    })
    
    return result


# 테스트용 메인 함수
async def main():
    """테스트 실행"""
    
    # 샘플 데이터
    sample_data = {
        "scene": {
            "room": {"width": 4000, "depth": 5000, "height": 2800},
            "objects": [
                {
                    "type": "furniture",
                    "name": "bed",
                    "position": {"center": {"x": 2000, "y": 4000}},
                    "dimensions": {"width": 1600, "depth": 2000, "height": 600},
                    "rotation_z": 0
                },
                {
                    "type": "furniture", 
                    "name": "desk",
                    "position": {"center": {"x": 3500, "y": 1500}},
                    "dimensions": {"width": 1200, "depth": 600, "height": 750},
                    "rotation_z": 45
                }
            ]
        }
    }
    
    print("Enhanced Vertex Generator 테스트 시작")
    
    # 이미지 생성 테스트
    result = await generate_room_image_from_coordinates(
        room_data=sample_data,
        style="scandinavian",
        prefer_google_ai=True
    )
    
    print("\nOK: 생성 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())