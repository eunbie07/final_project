"""
RoomBox.jsx와 Dify 통합을 위한 정확한 좌표 처리 모듈
일관성 있는 AI 인테리어 이미지 생성을 위한 핵심 시스템
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dify_rag import DifyLayoutRAG
import math


@dataclass
class FurnitureCoordinate:
    """가구 좌표 정보 클래스"""
    name: str
    center_x: int  # mm 단위
    center_y: int  # mm 단위
    width: int     # mm 단위
    depth: int     # mm 단위
    height: int    # mm 단위
    rotation_z: int  # 도 단위
    
    @property
    def relative_position(self) -> Dict[str, float]:
        """방 크기 기준 상대적 위치 (0.0-1.0)"""
        return {
            "x_percent": self.center_x / self.room_width if hasattr(self, 'room_width') else 0.5,
            "y_percent": self.center_y / self.room_depth if hasattr(self, 'room_depth') else 0.5
        }


@dataclass 
class RoomLayout:
    """방 레이아웃 전체 정보"""
    width_mm: int
    depth_mm: int  
    height_mm: int
    furniture: List[FurnitureCoordinate]
    windows: List[Dict[str, Any]]
    
    @property
    def area_sqm(self) -> float:
        """방 면적 (제곱미터)"""
        return (self.width_mm * self.depth_mm) / 1_000_000
    
    @property
    def room_ratio(self) -> float:
        """방 비율 (width/depth)"""
        return self.width_mm / self.depth_mm


class CoordinateValidator:
    """좌표 검증 및 일관성 보장 클래스"""
    
    def __init__(self):
        self.tolerance_mm = 50  # 50mm 허용 오차
    
    def validate_furniture_position(self, furniture: FurnitureCoordinate, room_layout: RoomLayout) -> Dict[str, Any]:
        """가구 위치의 유효성 검증"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "corrections": {}
        }
        
        # 1. 방 경계 내부 검증
        if not self._is_within_room_bounds(furniture, room_layout):
            validation_result["valid"] = False
            validation_result["errors"].append("가구가 방 경계를 벗어남")
            
            # 자동 보정 제안
            corrected_pos = self._correct_position_to_bounds(furniture, room_layout)
            validation_result["corrections"]["position"] = corrected_pos
        
        # 2. 가구 크기 검증
        if not self._is_realistic_size(furniture):
            validation_result["warnings"].append("가구 크기가 비현실적")
        
        # 3. 벽과의 거리 검증
        wall_distances = self._calculate_wall_distances(furniture, room_layout)
        if any(dist < 100 for dist in wall_distances.values()):  # 10cm 미만
            validation_result["warnings"].append("벽과 너무 가까움")
        
        return validation_result
    
    def _is_within_room_bounds(self, furniture: FurnitureCoordinate, room_layout: RoomLayout) -> bool:
        """가구가 방 경계 내부에 있는지 확인"""
        
        half_width = furniture.width / 2
        half_depth = furniture.depth / 2
        
        # 가구의 네 모서리 좌표
        left = furniture.center_x - half_width
        right = furniture.center_x + half_width
        front = furniture.center_y - half_depth
        back = furniture.center_y + half_depth
        
        return (left >= 0 and right <= room_layout.width_mm and 
                front >= 0 and back <= room_layout.depth_mm)
    
    def _correct_position_to_bounds(self, furniture: FurnitureCoordinate, room_layout: RoomLayout) -> Dict[str, int]:
        """가구 위치를 방 경계 내로 보정"""
        
        half_width = furniture.width / 2
        half_depth = furniture.depth / 2
        
        # X 좌표 보정
        corrected_x = furniture.center_x
        if furniture.center_x - half_width < 0:
            corrected_x = half_width
        elif furniture.center_x + half_width > room_layout.width_mm:
            corrected_x = room_layout.width_mm - half_width
        
        # Y 좌표 보정
        corrected_y = furniture.center_y
        if furniture.center_y - half_depth < 0:
            corrected_y = half_depth
        elif furniture.center_y + half_depth > room_layout.depth_mm:
            corrected_y = room_layout.depth_mm - half_depth
        
        return {"x": int(corrected_x), "y": int(corrected_y)}
    
    def _is_realistic_size(self, furniture: FurnitureCoordinate) -> bool:
        """가구 크기가 현실적인지 확인"""
        
        # 최소/최대 크기 제한 (mm)
        size_limits = {
            "min_width": 200, "max_width": 3000,
            "min_depth": 200, "max_depth": 3000,  
            "min_height": 200, "max_height": 2500
        }
        
        return (size_limits["min_width"] <= furniture.width <= size_limits["max_width"] and
                size_limits["min_depth"] <= furniture.depth <= size_limits["max_depth"] and
                size_limits["min_height"] <= furniture.height <= size_limits["max_height"])
    
    def _calculate_wall_distances(self, furniture: FurnitureCoordinate, room_layout: RoomLayout) -> Dict[str, float]:
        """벽과의 거리 계산"""
        
        half_width = furniture.width / 2
        half_depth = furniture.depth / 2
        
        return {
            "left_wall": furniture.center_x - half_width,
            "right_wall": room_layout.width_mm - (furniture.center_x + half_width),
            "front_wall": furniture.center_y - half_depth,
            "back_wall": room_layout.depth_mm - (furniture.center_y + half_depth)
        }


class RoomBoxDataProcessor:
    """RoomBox.jsx 데이터를 Dify용으로 변환하는 처리기"""
    
    def __init__(self, dify_rag: DifyLayoutRAG):
        self.dify_rag = dify_rag
        self.coordinate_validator = CoordinateValidator()
        
    def parse_roombox_data(self, room_data: Dict[str, Any]) -> RoomLayout:
        """RoomBox.jsx에서 온 데이터를 파싱"""
        
        # 새로운 데이터 구조 지원 (dimensions, furniture_3d)
        dimensions = room_data.get("dimensions", {})
        furniture_3d = room_data.get("furniture_3d", [])
        
        # 기존 scene 구조도 지원 (하위 호환성)
        scene = room_data.get("scene", {})
        room_info = scene.get("room", {}) if scene else {}
        objects = scene.get("objects", []) if scene else []
        
        # 방 크기 정보 (새 구조 우선, 기존 구조 fallback)
        width_mm = int(dimensions.get("width_cm", room_info.get("width", 400)) * 10)
        depth_mm = int(dimensions.get("depth_cm", room_info.get("depth", 500)) * 10)
        height_mm = int(dimensions.get("height_cm", room_info.get("height", 280)) * 10)
        
        # 가구 정보 추출
        furniture_list = []
        windows_list = []
        
        # 임시 레이아웃 생성 (검증용)
        temp_layout = RoomLayout(width_mm, depth_mm, height_mm, [], windows_list)
        
        # 새로운 구조에서 가구 정보 처리
        furniture_data = furniture_3d if furniture_3d else objects
        
        for obj in furniture_data:
            # 새로운 구조는 모두 가구로 간주
            if furniture_3d or obj.get("type") == "furniture":
                furniture = self._extract_furniture_coordinate_new(obj, width_mm, depth_mm) if furniture_3d else self._extract_furniture_coordinate(obj, width_mm, depth_mm)
                
                # 좌표 검증 및 보정
                validation = self.coordinate_validator.validate_furniture_position(furniture, temp_layout)
                
                if not validation["valid"]:
                    print(f"WARNING: 가구 '{furniture.name}' 좌표 문제: {validation['errors']}")
                    
                    # 자동 보정 적용
                    if "position" in validation["corrections"]:
                        corrected = validation["corrections"]["position"]
                        furniture.center_x = corrected["x"]
                        furniture.center_y = corrected["y"]
                        print(f"OK: 좌표 자동 보정: ({corrected['x']}, {corrected['y']})")
                
                if validation["warnings"]:
                    print(f"WARNING: 가구 '{furniture.name}' 경고: {validation['warnings']}")
                
                furniture_list.append(furniture)
            elif obj.get("type") == "window":
                windows_list.append(obj)
        
        return RoomLayout(
            width_mm=width_mm,
            depth_mm=depth_mm,
            height_mm=height_mm,
            furniture=furniture_list,
            windows=windows_list
        )
    
    def _extract_furniture_coordinate_new(self, furniture_obj: Dict[str, Any], 
                                         room_width: int, room_depth: int) -> FurnitureCoordinate:
        """새로운 데이터 구조(furniture_3d)를 위한 좌표 추출"""
        
        # furniture_3d 구조에서 데이터 추출
        name = furniture_obj.get("name", furniture_obj.get("type", "furniture"))
        
        # position은 Three.js 형식 (x, y, z)
        position = furniture_obj.get("position", {})
        center_x = int(position.get("x", room_width // 2))  # mm 단위
        center_y = int(position.get("z", room_depth // 2))  # z가 depth
        
        # scale은 Three.js 스케일 (실제 크기는 기본값 * scale)
        scale = furniture_obj.get("scale", {})
        base_width = 1000   # 기본 1m
        base_depth = 800    # 기본 0.8m  
        base_height = 800   # 기본 0.8m
        
        width = int(base_width * scale.get("x", 1))
        depth = int(base_depth * scale.get("z", 1))
        height = int(base_height * scale.get("y", 1))
        
        # rotation은 Three.js 라디안 -> 도 변환
        rotation = furniture_obj.get("rotation", {})
        rotation_z = int(math.degrees(rotation.get("y", 0)))  # y축 회전이 Z축 회전
        
        furniture = FurnitureCoordinate(
            name=name,
            center_x=center_x,
            center_y=center_y,
            width=width,
            depth=depth,
            height=height,
            rotation_z=rotation_z
        )
        
        # 상대적 위치 계산을 위한 방 크기 저장
        furniture.room_width = room_width
        furniture.room_depth = room_depth
        
        return furniture
    
    def _extract_furniture_coordinate(self, furniture_obj: Dict[str, Any], 
                                    room_width: int, room_depth: int) -> FurnitureCoordinate:
        """RoomBox.jsx 좌표 시스템에 맞춘 정확한 좌표 추출"""
        
        # RoomBox.jsx에서는 여러 좌표 형식을 사용할 수 있음
        position = furniture_obj.get("position", {})
        dimensions = furniture_obj.get("dimensions", {})
        
        # 좌표 추출 - RoomBox.jsx의 다양한 형식 지원
        center_x, center_y = self._extract_center_coordinates(position, dimensions)
        
        # 크기 정보 - RoomBox.jsx 형식 지원
        width, depth, height = self._extract_dimensions(dimensions, furniture_obj)
        
        # 회전 정보 - 다양한 형식 지원
        rotation_z = self._extract_rotation(furniture_obj)
        
        furniture = FurnitureCoordinate(
            name=furniture_obj.get("name", "furniture"),
            center_x=int(center_x),
            center_y=int(center_y),
            width=int(width),
            depth=int(depth),
            height=int(height),
            rotation_z=int(rotation_z)
        )
        
        # 상대적 위치 계산을 위한 방 크기 저장
        furniture.room_width = room_width
        furniture.room_depth = room_depth
        
        return furniture
    
    def _extract_center_coordinates(self, position: Dict[str, Any], dimensions: Dict[str, Any]) -> tuple:
        """RoomBox.jsx의 다양한 좌표 형식에서 중심 좌표 추출"""
        
        # 형식 1: position.center
        if "center" in position:
            center = position["center"]
            return center.get("x", 0), center.get("y", 0)
        
        # 형식 2: position에 직접 x, y, z (RoomBox.jsx 3D 좌표)
        if "x" in position and "z" in position:
            # RoomBox.jsx는 3D 좌표계 사용: x=가로, z=세로, y=높이
            return position["x"], position["z"]
        
        # 형식 3: position에 직접 x, y (2D 좌표)
        if "x" in position and "y" in position:
            return position["x"], position["y"]
        
        # 기본값
        return 0, 0
    
    def _extract_dimensions(self, dimensions: Dict[str, Any], furniture_obj: Dict[str, Any]) -> tuple:
        """RoomBox.jsx의 다양한 크기 형식에서 dimensions 추출"""
        
        # 형식 1: dimensions 객체
        if dimensions:
            width = dimensions.get("width", 1000)
            depth = dimensions.get("depth", dimensions.get("height", 1000))  # depth가 없으면 height 사용
            height = dimensions.get("height", 800)
            return width, depth, height
        
        # 형식 2: furniture_obj에 직접 size 배열 (RoomBox.jsx 형식)
        if "size" in furniture_obj:
            size = furniture_obj["size"]
            if isinstance(size, list) and len(size) >= 3:
                return size[0], size[2], size[1]  # [width, height, depth] -> [width, depth, height]
        
        # 형식 3: furniture_obj에 직접 width, depth, height
        width = furniture_obj.get("width", 1000)
        depth = furniture_obj.get("depth", 1000)
        height = furniture_obj.get("height", 800)
        
        return width, depth, height
    
    def _extract_rotation(self, furniture_obj: Dict[str, Any]) -> int:
        """RoomBox.jsx의 다양한 회전 형식에서 회전값 추출"""
        
        # 형식 1: rotation_z 직접
        if "rotation_z" in furniture_obj:
            return furniture_obj["rotation_z"]
        
        # 형식 2: rotation 배열 (RoomBox.jsx 3D 형식)
        if "rotation" in furniture_obj:
            rotation = furniture_obj["rotation"]
            if isinstance(rotation, list) and len(rotation) >= 3:
                # Y축 회전을 Z축 회전으로 변환 (라디안 -> 도)
                return int((rotation[1] * 180 / 3.14159) % 360)
        
        # 형식 3: 직접 rotation (도 단위)
        if "rotation" in furniture_obj and isinstance(furniture_obj["rotation"], (int, float)):
            return int(furniture_obj["rotation"] % 360)
        
        return 0


class ConsistentStyleGenerator:
    """일관성 있는 스타일 프롬프트 생성기"""
    
    def __init__(self):
        # 스타일별 일관된 키워드 정의
        self.style_definitions = {
            "modern": {
                "color_palette": ["white", "grey", "black", "beige"],
                "materials": ["glass", "steel", "marble", "wood"],
                "lighting": "bright natural lighting, large windows",
                "furniture_style": "minimalist, clean lines, geometric shapes",
                "atmosphere": "spacious, uncluttered, sophisticated",
                "keywords": ["contemporary", "sleek", "minimal", "geometric", "neutral tones"]
            },
            "scandinavian": {
                "color_palette": ["white", "light grey", "natural wood", "soft pastels"],
                "materials": ["light wood", "wool", "cotton", "ceramics"],
                "lighting": "soft natural lighting, cozy ambient lights",
                "furniture_style": "functional, simple, organic shapes",
                "atmosphere": "cozy, warm, hygge feeling",
                "keywords": ["nordic", "hygge", "cozy", "functional", "natural materials"]
            },
            "industrial": {
                "color_palette": ["dark grey", "black", "rust", "raw metal"],
                "materials": ["exposed brick", "steel", "concrete", "leather"],
                "lighting": "dramatic industrial lighting, exposed bulbs",
                "furniture_style": "raw materials, metal frames, vintage",
                "atmosphere": "urban, edgy, raw character",
                "keywords": ["loft", "exposed", "metal", "vintage", "urban"]
            }
        }
    
    def generate_consistent_prompt(self, room_layout: RoomLayout, 
                                 style: str = "modern") -> str:
        """일관성 있는 스타일로 프롬프트 생성"""
        
        if style not in self.style_definitions:
            style = "modern"  # 기본값
        
        style_def = self.style_definitions[style]
        
        # 1. 방 기본 정보
        room_description = f"""
        {style.capitalize()} Korean interior room design:
        - Room dimensions: {room_layout.width_mm/1000:.1f}m × {room_layout.depth_mm/1000:.1f}m × {room_layout.height_mm/1000:.1f}m
        - Total area: {room_layout.area_sqm:.1f}㎡
        - Room ratio: {room_layout.room_ratio:.2f} (width/depth)
        """
        
        # 2. 가구 배치 정보 (정확한 좌표 + 공간 관계)
        furniture_description = "\nFurniture layout with precise positioning and spatial relationships:"
        
        for furniture in room_layout.furniture:
            rel_pos = furniture.relative_position
            
            # 위치를 자연어로 변환
            position_desc = self._get_position_description(rel_pos["x_percent"], rel_pos["y_percent"])
            
            # 벽과의 거리 계산
            wall_distances = self._calculate_furniture_wall_distances(furniture, room_layout)
            
            # 다른 가구와의 관계 분석
            spatial_relationships = self._analyze_spatial_relationships(furniture, room_layout.furniture)
            
            furniture_description += f"""
        - {furniture.name}:
          * Position: {position_desc} of the room
          * Exact coordinates: ({furniture.center_x}mm, {furniture.center_y}mm) from bottom-left corner
          * Size: {furniture.width/1000:.1f}m × {furniture.depth/1000:.1f}m × {furniture.height/1000:.1f}m
          * Rotation: {furniture.rotation_z}° from north
          * Relative position: {rel_pos['x_percent']*100:.1f}% from left, {rel_pos['y_percent']*100:.1f}% from bottom
          * Wall distances: Left {wall_distances['left']/10:.0f}cm, Right {wall_distances['right']/10:.0f}cm, Front {wall_distances['front']/10:.0f}cm, Back {wall_distances['back']/10:.0f}cm
          * Spatial context: {spatial_relationships}
            """
        
        # 3. 스타일 일관성 강화
        style_description = f"""
        
        Consistent {style} style requirements:
        - Color palette: {', '.join(style_def['color_palette'])}
        - Primary materials: {', '.join(style_def['materials'])}
        - Lighting: {style_def['lighting']}
        - Furniture characteristics: {style_def['furniture_style']}
        - Overall atmosphere: {style_def['atmosphere']}
        - Key style elements: {', '.join(style_def['keywords'])}
        """
        
        # 4. 이미지 품질 지시사항
        quality_instructions = """
        
        Image generation specifications:
        - Professional interior photography quality
        - 8K resolution, high detail
        - Perfect perspective and proportions
        - Accurate furniture placement matching exact coordinates
        - Consistent lighting and shadows
        - Korean apartment interior context
        - Camera angle: slightly elevated, showing room layout clearly
        - Depth of field: entire room in focus
        """
        
        return room_description + furniture_description + style_description + quality_instructions
    
    def _get_position_description(self, x_percent: float, y_percent: float) -> str:
        """좌표를 자연어 위치 설명으로 변환"""
        
        # X축 위치 (좌우)
        if x_percent < 0.3:
            x_desc = "left side"
        elif x_percent > 0.7:
            x_desc = "right side"
        else:
            x_desc = "center"
        
        # Y축 위치 (앞뒤)
        if y_percent < 0.3:
            y_desc = "front area"
        elif y_percent > 0.7:
            y_desc = "back area"
        else:
            y_desc = "middle area"
        
        return f"{x_desc} {y_desc}"
    
    def _calculate_furniture_wall_distances(self, furniture: FurnitureCoordinate, room_layout: RoomLayout) -> Dict[str, float]:
        """가구와 벽 사이의 거리 계산"""
        
        half_width = furniture.width / 2
        half_depth = furniture.depth / 2
        
        return {
            "left": furniture.center_x - half_width,
            "right": room_layout.width_mm - (furniture.center_x + half_width),
            "front": furniture.center_y - half_depth,
            "back": room_layout.depth_mm - (furniture.center_y + half_depth)
        }
    
    def _analyze_spatial_relationships(self, target_furniture: FurnitureCoordinate, all_furniture: List[FurnitureCoordinate]) -> str:
        """가구 간 공간적 관계 분석"""
        
        relationships = []
        
        for other_furniture in all_furniture:
            if other_furniture.name == target_furniture.name:
                continue
            
            # 거리 계산
            distance = math.sqrt(
                (target_furniture.center_x - other_furniture.center_x) ** 2 +
                (target_furniture.center_y - other_furniture.center_y) ** 2
            )
            
            # 방향 분석
            dx = other_furniture.center_x - target_furniture.center_x
            dy = other_furniture.center_y - target_furniture.center_y
            
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "behind" if dy > 0 else "in front"
            
            # 근접성 분석
            if distance < 1000:  # 1m 이내
                proximity = "very close"
            elif distance < 2000:  # 2m 이내
                proximity = "close"
            else:
                proximity = "distant"
            
            relationships.append(f"{proximity} to {other_furniture.name} ({direction}, {distance/10:.0f}cm)")
        
        return "; ".join(relationships) if relationships else "isolated placement"


class DifyRoomImageGenerator:
    """Dify를 활용한 일관성 있는 방 이미지 생성기"""
    
    def __init__(self, dify_api_key: str, dify_app_id: str, dify_dataset_id: str = None):
        self.dify_rag = DifyLayoutRAG(dify_api_key, dify_app_id, dify_dataset_id)
        self.data_processor = RoomBoxDataProcessor(self.dify_rag)
        self.style_generator = ConsistentStyleGenerator()
    
    async def generate_consistent_room_image(self, room_data: Dict[str, Any], 
                                           style: str = "modern",
                                           user_id: str = None) -> Dict[str, Any]:
        """RoomBox 데이터로부터 일관성 있는 방 이미지 생성"""
        
        try:
            # 1. 데이터 파싱
            room_layout = self.data_processor.parse_roombox_data(room_data)
            
            # 2. 일관성 있는 프롬프트 생성
            base_prompt = self.style_generator.generate_consistent_prompt(room_layout, style)
            
            # 3. Dify RAG로 프롬프트 최적화
            similar_layouts = self.dify_rag.find_similar_layouts(room_data)
            
            if similar_layouts:
                # 성공 사례가 있으면 참고하여 프롬프트 개선
                optimized_prompt = self.dify_rag.generate_optimized_prompt(room_data)
                if optimized_prompt:
                    # 스타일 일관성과 RAG 최적화 결합
                    final_prompt = f"{base_prompt}\n\nOptimization based on successful similar layouts:\n{optimized_prompt}"
                else:
                    final_prompt = base_prompt
            else:
                # 첫 생성이면 기본 프롬프트 사용
                final_prompt = base_prompt
            
            # 4. 이미지 생성 (실제 AI 서비스 연결 필요)
            result = await self._generate_image_with_ai_service(final_prompt, room_layout, style)
            
            return {
                "success": True,
                "image_path": result.get("image_path"),
                "prompt": final_prompt,
                "style": style,
                "room_layout": room_layout,
                "method": "dify_consistent"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "dify_consistent"
            }
    
    async def _generate_image_with_ai_service(self, prompt: str, room_layout: RoomLayout, 
                                            style: str) -> Dict[str, Any]:
        """Google Vertex AI로 실제 이미지 생성"""
        
        try:
            # Vertex AI 이미지 생성기 import
            from vertex_ai_generator import VertexAIImageGenerator
            
            # Vertex AI 생성기 초기화
            vertex_generator = VertexAIImageGenerator()
            
            # 실제 이미지 생성
            result = await vertex_generator.generate_image(prompt, style, room_layout.__dict__)
            
            if result["success"]:
                return {
                    "image_path": result["image_path"],
                    "prompt_used": prompt[:200] + "...",
                    "generation_time": result.get("generation_time", "Unknown"),
                    "service": "vertex_ai",
                    "parameters": result.get("parameters", {})
                }
            else:
                # Vertex AI 실패시 Mock으로 폴백
                print(f"⚠️ Vertex AI 실패, Mock으로 폴백: {result.get('error')}")
                return await self._generate_mock_image(prompt, room_layout, style)
                
        except ImportError:
            print("⚠️ Vertex AI 모듈 없음, Mock으로 폴백")
            return await self._generate_mock_image(prompt, room_layout, style)
        except Exception as e:
            print(f"⚠️ Vertex AI 오류, Mock으로 폴백: {e}")
            return await self._generate_mock_image(prompt, room_layout, style)
    
    async def _generate_mock_image(self, prompt: str, room_layout: RoomLayout, style: str) -> Dict[str, Any]:
        """Mock 이미지 생성 (폴백)"""
        import asyncio
        await asyncio.sleep(1)  # 시뮬레이션
        
        filename = f"mock_{style}_{room_layout.width_mm}x{room_layout.depth_mm}.png"
        
        return {
            "image_path": f"generated_images/{filename}",
            "prompt_used": prompt[:200] + "...",
            "generation_time": "1초 (Mock)",
            "service": "mock"
        }
    
    async def learn_from_feedback(self, room_data: Dict[str, Any], 
                                image_path: str, user_rating: float,
                                style: str, comments: str = "") -> Dict[str, Any]:
        """사용자 피드백을 통한 학습"""
        
        if user_rating >= 4.0:
            # 성공 사례로 학습
            success = self.dify_rag.add_successful_layout(room_data, user_rating, image_path)
            
            return {
                "learned": success,
                "rating": user_rating,
                "style": style,
                "message": f"High rating ({user_rating}/5.0) layout learned for {style} style"
            }
        
        return {
            "learned": False,
            "rating": user_rating,
            "style": style,
            "message": f"Low rating ({user_rating}/5.0) - not learned"
        }