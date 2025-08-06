import json
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from dify_rag import DifyLayoutRAG


class IntegratedImageGenerator:
    def __init__(self, dify_api_key: str, dify_app_id: str, dify_dataset_id: str = None):
        self.dify_rag = DifyLayoutRAG(dify_api_key, dify_app_id, dify_dataset_id)
        
    async def generate_with_dify_optimization(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """Dify 최적화를 활용한 이미지 생성"""
        
        try:
            # 1. Dify RAG로 최적화된 프롬프트 생성
            optimized_prompt = self.dify_rag.generate_optimized_prompt(room_data)
            
            if not optimized_prompt:
                # Dify 실패 시 기존 방식으로 폴백
                return await self.generate_traditional(room_data)
            
            # 2. 레이아웃 해시로 일관된 시드 생성
            layout_hash = hash(json.dumps(room_data, sort_keys=True))
            seed = layout_hash % (2**32)
            
            # 3. 시드 고정으로 일관된 이미지 생성
            enhanced_prompt = f"{optimized_prompt} [SEED:{seed}]"
            
            # 4. 실제 이미지 생성 (구현 필요)
            image_path = await self.generate_image_with_ai_service(
                enhanced_prompt,
                f"dify_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            return {
                "success": True,
                "image_path": image_path,
                "prompt": enhanced_prompt,
                "seed": seed,
                "method": "dify_optimized"
            }
            
        except Exception as e:
            print(f"Dify 최적화 실패: {e}")
            return await self.generate_traditional(room_data)
    
    async def generate_traditional(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """기존 방식으로 이미지 생성 (폴백)"""
        try:
            # 기본 프롬프트 생성
            prompt = self.create_basic_prompt(room_data)
            
            # 이미지 생성
            image_path = await self.generate_image_with_ai_service(
                prompt,
                f"traditional_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            return {
                "success": True,
                "image_path": image_path,
                "prompt": prompt,
                "seed": None,
                "method": "traditional"
            }
            
        except Exception as e:
            print(f"전통적 방식 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "traditional"
            }
    
    def create_basic_prompt(self, room_data: Dict[str, Any]) -> str:
        """기본 프롬프트 생성"""
        room = room_data["scene"]["room"]
        objects = room_data["scene"]["objects"]
        
        prompt = f"Korean modern interior room, {room['width']}x{room['depth']}mm, "
        
        furniture_descriptions = []
        for obj in objects:
            if obj["type"] == "furniture":
                furniture_descriptions.append(obj["name"])
        
        if furniture_descriptions:
            prompt += f"with {', '.join(furniture_descriptions)}, "
        
        prompt += "professional interior photography, natural lighting, clean and minimalist design"
        
        return prompt
    
    async def generate_image_with_ai_service(self, prompt: str, filename: str) -> str:
        """AI 서비스로 실제 이미지 생성 (구현 필요)"""
        # 이 부분은 실제 AI 이미지 생성 서비스 연동이 필요
        # 예: OpenAI DALL-E, Vertex AI Imagen, Midjourney API 등
        
        # 임시 구현 - 실제로는 AI 서비스 호출
        print(f"Generating image with prompt: {prompt[:100]}...")
        
        # 가상의 이미지 경로 반환
        output_path = f"generated_images/{filename}"
        
        # 실제 구현에서는 여기서 AI 서비스를 호출하고 이미지를 저장
        await asyncio.sleep(1)  # 생성 시간 시뮬레이션
        
        return output_path
    
    async def collect_feedback_and_learn(self, room_data: Dict[str, Any], image_path: str, 
                                       user_rating: float, comments: str = "") -> Dict[str, Any]:
        """사용자 피드백 수집 및 학습"""
        
        if user_rating >= 4.0:
            success = self.dify_rag.add_successful_layout(
                room_data, 
                user_rating, 
                image_path
            )
            
            if success:
                print(f"OK: 성공 사례 학습 완료 (평점: {user_rating})")
            else:
                print(f"ERROR: 학습 실패")
        
        # 피드백 데이터 저장
        feedback_data = {
            "room_data": room_data,
            "image_path": image_path,
            "user_rating": user_rating,
            "comments": comments,
            "timestamp": datetime.now().isoformat(),
            "learned": user_rating >= 4.0
        }
        
        # 피드백 로그 저장 (구현 필요)
        await self.save_feedback_log(feedback_data)
                
        return {"learned": user_rating >= 4.0}
    
    async def save_feedback_log(self, feedback_data: Dict[str, Any]):
        """피드백 로그 저장"""
        # 실제 구현에서는 데이터베이스나 파일에 저장
        print(f"Feedback saved: Rating {feedback_data['user_rating']}/5.0")
    
    async def batch_generate_images(self, room_data_list: list) -> list:
        """여러 레이아웃 동시 처리"""
        tasks = []
        for room_data in room_data_list:
            task = self.generate_with_dify_optimization(room_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results