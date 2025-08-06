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
        """방 레이아웃을 구조화된 텍스트로 변환 (새로운 데이터 구조 지원)"""
        
        # 새로운 데이터 구조 처리 (furniture_3d 형식)
        if "dimensions" in room_data and "furniture_3d" in room_data:
            dimensions = room_data["dimensions"]
            furniture_list = room_data["furniture_3d"]
            
            description = f"""
ROOM SPATIAL ANALYSIS:
- Room Size: {dimensions['width_cm']:.1f}cm × {dimensions['depth_cm']:.1f}cm × {dimensions['height_cm']:.1f}cm
- Room Center Point: ({dimensions['width_cm']/2:.1f}cm, {dimensions['depth_cm']/2:.1f}cm)
- Total Floor Area: {(dimensions['width_cm'] * dimensions['depth_cm']) / 10000:.2f}㎡

CRITICAL FURNITURE POSITIONING:"""
            
            for furniture in furniture_list:
                name = furniture.get('name', 'furniture')
                pos = furniture["position"]
                pos_x_cm = pos[0]  
                pos_z_cm = pos[2]  # Z축이 depth
                
                # 중앙도 계산 (0.5가 완전 중앙)
                rel_x = pos_x_cm / dimensions['width_cm']
                rel_z = pos_z_cm / dimensions['depth_cm']
                center_distance = ((rel_x - 0.5) ** 2 + (rel_z - 0.5) ** 2) ** 0.5
                
                # 배치 유형 결정
                if center_distance < 0.1:
                    placement = "PERFECT CENTER"
                elif center_distance < 0.2:
                    placement = "NEAR CENTER"
                elif center_distance < 0.3:
                    placement = "OFF-CENTER"
                else:
                    placement = "EDGE/WALL"
                
                description += f"""
- {name.upper()} PLACEMENT:
  EXACT COORDINATES: X={pos_x_cm:.1f}cm, Z={pos_z_cm:.1f}cm
  RELATIVE POSITION: ({rel_x:.3f}, {rel_z:.3f}) where (0.5,0.5)=center
  CENTER DISTANCE: {center_distance:.3f} (0.0=perfect center, 0.5+=edge)
  PLACEMENT TYPE: {placement}
  WALL CLEARANCES: Left={pos_x_cm:.1f}cm, Right={dimensions['width_cm']-pos_x_cm:.1f}cm
  POSITIONING INSTRUCTION: Place {name} EXACTLY at coordinates ({pos_x_cm:.1f}, {pos_z_cm:.1f}) - THIS IS {placement}
"""
            
        else:
            # 기존 MongoDB 구조 처리 (fallback)
            room = room_data.get("scene", {}).get("room", {})
            objects = room_data.get("scene", {}).get("objects", [])
            
            description = "LEGACY FORMAT SPATIAL ANALYSIS:\n"
            for obj in objects:
                if obj.get("type") == "furniture":
                    description += f"- {obj.get('name', 'furniture')}: Legacy format detected\n"
        
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
            url = f"{self.base_url}/datasets/{self.dataset_id}/document/create_by_text"
            payload = {
                "name": f"layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "text": layout_text,
                "indexing_technique": "high_quality",
                "process_rule": {"mode": "automatic"}
            }
            
            print(f"DEBUG: API URL: {url}")
            print(f"DEBUG: Headers: {self.headers}")
            print(f"DEBUG: Payload name: {payload['name']}")
            print(f"DEBUG: Text length: {len(payload['text'])}")
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            print(f"DEBUG: Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"DEBUG: Response text: {response.text}")
            
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
                    "response_mode": "streaming",
                    "user": "layout_analyzer"
                }
            )
            
            if response.status_code == 200:
                response_text = response.text
                if len(response_text) > 100:
                    print(f"OK: Knowledge Base에서 {len(response_text)} 문자 응답")
                    return {"answer": response_text, "success": True}
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