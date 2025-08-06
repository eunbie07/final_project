import requests
import json
from datetime import datetime
import os
from typing import Dict, Any, Optional


class DifyLayoutRAG:
    def __init__(self, api_key: str, app_id: str, dataset_id: str = None):
        self.api_key = api_key
        self.app_id = app_id
        self.dataset_id = dataset_id
        self.base_url = "https://api.dify.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_spatial_embedding(self, room_data: Dict[str, Any]) -> str:
        """방 레이아웃을 구조화된 텍스트로 변환"""
        room = room_data["scene"]["room"]
        objects = room_data["scene"]["objects"]
        
        # 구조화된 설명 생성
        description = f"""
        Room Layout Analysis:
        - Dimensions: {room['width']}×{room['depth']}×{room['height']}mm
        - Total Area: {(room['width'] * room['depth']) / 1000000:.2f}㎡
        - Room Ratio: {room['width']/room['depth']:.2f} (width/depth)
        
        Furniture Configuration:
        """
        
        for obj in objects:
            if obj["type"] == "furniture":
                pos = obj["position"]["center"]
                dims = obj["dimensions"]
                
                # 상대적 위치 계산
                rel_x = (pos["x"] / room["width"]) * 100
                rel_y = (pos["y"] / room["depth"]) * 100
                
                description += f"""
        - {obj['name']}:
          * Absolute Position: ({pos['x']}, {pos['y']})mm
          * Relative Position: {rel_x:.1f}% from left, {rel_y:.1f}% from bottom
          * Size: {dims['width']}×{dims['depth']}mm
          * Area Ratio: {(dims['width']*dims['depth'])/(room['width']*room['depth'])*100:.1f}%
          * Wall Distance: Left={pos['x']}mm, Right={room['width']-pos['x']}mm
                """
        
        return description.strip()
    
    def add_successful_layout(self, room_data: Dict[str, Any], user_rating: float, image_path: str = None) -> bool:
        """성공적인 레이아웃을 Knowledge Base에 추가"""
        
        if user_rating < 4.0:  # 좋은 결과만 학습
            return False
            
        # 구조화된 텍스트 생성
        layout_text = self.create_spatial_embedding(room_data)
        layout_text += f"""
        
        Success Metrics:
        - User Rating: {user_rating}/5.0
        - Generated At: {datetime.now().isoformat()}
        - Style: Korean Modern Interior
        """
        
        if not self.dataset_id:
            print("Warning: dataset_id not set, cannot add to knowledge base")
            return False
        
        # Dify Knowledge Base에 추가
        try:
            response = requests.post(
                f"{self.base_url}/datasets/{self.dataset_id}/documents",
                headers=self.headers,
                json={
                    "name": f"layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "text": layout_text,
                    "indexing_technique": "high_quality"
                }
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Error adding to knowledge base: {e}")
            return False
    
    def find_similar_layouts(self, room_data: Dict[str, Any], min_rating: float = 4.0) -> Optional[Dict[str, Any]]:
        """유사한 성공 레이아웃 검색"""
        
        # 현재 레이아웃을 쿼리로 변환
        current_layout = self.create_spatial_embedding(room_data)
        
        query = f"""
        Find similar room layouts to:
        {current_layout}
        
        Requirements:
        - Similar room size (±20%)
        - Similar furniture types and arrangement
        - High user rating (≥{min_rating})
        - Korean interior style
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/chat-messages",
                headers=self.headers,
                json={
                    "inputs": {},
                    "query": query,
                    "response_mode": "blocking",
                    "user": "layout_analyzer"
                }
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error finding similar layouts: {e}")
            return None
    
    def generate_optimized_prompt(self, room_data: Dict[str, Any]) -> Optional[str]:
        """RAG 결과 기반 최적화된 프롬프트 생성"""
        
        similar_layouts = self.find_similar_layouts(room_data)
        current_layout = self.create_spatial_embedding(room_data)
        
        optimization_query = f"""
        Based on similar successful room layouts, generate an optimized AI image generation prompt for:
        
        Current Room Layout:
        {current_layout}
        
        Similar Successful Cases:
        {similar_layouts.get('answer', 'No similar layouts found') if similar_layouts else 'No data'}
        
        Requirements:
        1. Precise furniture positioning using exact coordinates
        2. Consistent Korean modern interior style
        3. Realistic proportions and lighting
        4. Professional interior photography quality
        5. Specific camera angle and composition
        
        Generate a detailed, structured prompt that ensures accurate spatial relationships.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/chat-messages",
                headers=self.headers,
                json={
                    "inputs": {},
                    "query": optimization_query,
                    "response_mode": "blocking",
                    "user": "prompt_optimizer"
                }
            )
            
            if response.status_code == 200:
                return response.json().get('answer', '')
            return None
        except Exception as e:
            print(f"Error generating optimized prompt: {e}")
            return None