"""
MongoDB에서 실제 eunbi 가구 좌표 데이터를 추출하는 모듈
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# MongoDB 관련 imports (선택적)
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("WARNING: PyMongo가 설치되지 않았습니다. pip install pymongo")


class MongoDBFurnitureExtractor:
    """eunbi MongoDB에서 실제 가구 좌표 데이터를 추출"""
    
    def __init__(self, 
                 mongo_uri: str = "mongodb://13.55.21.100:27017/",
                 database: str = "room_measure",
                 collection: str = "room_layouts"):
        
        self.mongo_uri = mongo_uri
        self.database_name = database
        self.collection_name = collection
        self.client = None
        self.db = None
        self.collection = None
        
        # MongoDB 연결 시도
        self._connect_mongodb()
    
    def _connect_mongodb(self):
        """MongoDB 연결"""
        
        if not MONGODB_AVAILABLE:
            print("WARNING: MongoDB 클라이언트를 사용할 수 없습니다")
            return
        
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # 연결 테스트
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            print(f"OK: MongoDB 연결 성공: {self.database_name}.{self.collection_name}")
            
        except ConnectionFailure as e:
            print(f"ERROR: MongoDB 연결 실패: {e}")
            self.client = None
    
    def extract_furniture_coordinates(self, limit: int = 50) -> List[Dict[str, Any]]:
        """MongoDB에서 실제 가구 좌표 데이터 추출"""
        
        if self.collection is None:
            print("WARNING: MongoDB 연결 없음 - 샘플 데이터 사용")
            return self._get_sample_furniture_data()
        
        try:
            # MongoDB에서 가구 데이터가 있는 레이아웃들을 가져오기
            cursor = self.collection.find({
                "furniture_3d": {"$exists": True, "$ne": []}
            }).sort("saved_at", -1).limit(limit)
            
            furniture_extracts = []
            
            for layout in cursor:
                layout_id = str(layout.get("_id", "unknown"))
                room_dimensions = layout.get("room_dimensions", {})
                furniture_3d = layout.get("furniture_3d", [])
                
                print(f"INFO: 처리 중 - {layout_id} - 가구 {len(furniture_3d)}개")
                
                # 각 가구의 좌표 정보 추출
                for i, furniture in enumerate(furniture_3d):
                    furniture_extract = self._process_furniture_item(
                        furniture, room_dimensions, layout_id, i
                    )
                    if furniture_extract:
                        furniture_extracts.append(furniture_extract)
            
            print(f"OK: 총 {len(furniture_extracts)}개 가구 좌표 추출 완료")
            return furniture_extracts
            
        except Exception as e:
            print(f"ERROR: MongoDB 쿼리 실패: {e}")
            return self._get_sample_furniture_data()
    
    def _process_furniture_item(self, furniture: Dict, room_dims: Dict, 
                               layout_id: str, furniture_index: int) -> Optional[Dict[str, Any]]:
        """개별 가구 아이템 처리"""
        
        try:
            # 좌표 정보 추출
            position = furniture.get("position", {})
            scale = furniture.get("scale", {})
            
            center_x = position.get("x", 0)
            center_y = position.get("z", 0)  # Three.js에서는 z가 depth
            center_z = position.get("y", 0)  # Three.js에서는 y가 height
            
            # 크기 정보 (scale에서 추출)
            width = scale.get("x", 1) * 1000  # mm 단위로 변환
            depth = scale.get("z", 1) * 1000
            height = scale.get("y", 1) * 1000
            
            # 방 크기 정보
            room_width = room_dims.get("width_cm", 400) * 10  # mm로 변환
            room_depth = room_dims.get("depth_cm", 400) * 10
            room_height = room_dims.get("height_cm", 230) * 10
            
            furniture_extract = {
                "layout_id": layout_id,
                "furniture_index": furniture_index,
                "name": furniture.get("name", f"가구_{furniture_index}"),
                "type": furniture.get("type", "unknown"),
                
                # 정확한 좌표 정보
                "coordinates": {
                    "center_x": float(center_x),
                    "center_y": float(center_y), 
                    "center_z": float(center_z),
                    "width": float(width),
                    "depth": float(depth),
                    "height": float(height)
                },
                
                # 방 정보
                "room_dimensions": {
                    "width_mm": float(room_width),
                    "depth_mm": float(room_depth), 
                    "height_mm": float(room_height)
                },
                
                # 상대적 위치 계산
                "relative_position": {
                    "x_ratio": float(center_x) / float(room_width) if room_width > 0 else 0.5,
                    "y_ratio": float(center_y) / float(room_depth) if room_depth > 0 else 0.5,
                    "position_description": self._get_position_description(
                        float(center_x) / float(room_width) if room_width > 0 else 0.5,
                        float(center_y) / float(room_depth) if room_depth > 0 else 0.5
                    )
                },
                
                # 메타데이터
                "extracted_at": datetime.now().isoformat(),
                "data_source": "eunbi_mongodb"
            }
            
            return furniture_extract
            
        except Exception as e:
            print(f"WARNING: 가구 처리 실패: {e}")
            return None
    
    def _get_position_description(self, x_ratio: float, y_ratio: float) -> str:
        """좌표 비율을 기반으로 위치 설명 생성"""
        
        # x축 위치 (좌우)
        if x_ratio < 0.2:
            x_desc = "왼쪽"
        elif x_ratio > 0.8:
            x_desc = "오른쪽"
        else:
            x_desc = "중앙"
        
        # y축 위치 (앞뒤)
        if y_ratio < 0.3:
            y_desc = "앞쪽"
        elif y_ratio > 0.7:
            y_desc = "뒤쪽" 
        else:
            y_desc = "중앙"
        
        if x_desc == "중앙" and y_desc == "중앙":
            return "중앙"
        elif x_ratio < 0.2 or x_ratio > 0.8 or y_ratio < 0.2 or y_ratio > 0.8:
            return f"{x_desc} {y_desc} 코너"
        else:
            return f"{x_desc} {y_desc}"
    
    def _get_sample_furniture_data(self) -> List[Dict[str, Any]]:
        """MongoDB 연결 실패 시 사용할 샘플 데이터"""
        
        sample_data = [
            {
                "layout_id": "sample_1",
                "furniture_index": 0,
                "name": "L자형 소파",
                "type": "sofa",
                "coordinates": {
                    "center_x": 1750.0,
                    "center_y": 2200.0,
                    "center_z": 400.0,
                    "width": 2200.0,
                    "depth": 900.0,
                    "height": 800.0
                },
                "room_dimensions": {
                    "width_mm": 3500.0,
                    "depth_mm": 4000.0,
                    "height_mm": 2300.0
                },
                "relative_position": {
                    "x_ratio": 0.5,
                    "y_ratio": 0.55,
                    "position_description": "중앙"
                },
                "extracted_at": datetime.now().isoformat(),
                "data_source": "sample"
            },
            {
                "layout_id": "sample_1", 
                "furniture_index": 1,
                "name": "커피테이블",
                "type": "table",
                "coordinates": {
                    "center_x": 1750.0,
                    "center_y": 1400.0,
                    "center_z": 200.0,
                    "width": 1000.0,
                    "depth": 500.0,
                    "height": 400.0
                },
                "room_dimensions": {
                    "width_mm": 3500.0,
                    "depth_mm": 4000.0,
                    "height_mm": 2300.0
                },
                "relative_position": {
                    "x_ratio": 0.5,
                    "y_ratio": 0.35,
                    "position_description": "중앙 앞쪽"
                },
                "extracted_at": datetime.now().isoformat(),
                "data_source": "sample"
            }
        ]
        
        return sample_data
    
    def save_extracted_data(self, furniture_data: List[Dict[str, Any]], 
                           output_file: str = None) -> str:
        """추출된 가구 좌표 데이터를 JSON 파일로 저장"""
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"extracted_furniture_coordinates_{timestamp}.json"
        
        # 결과 데이터 구성
        result = {
            "extraction_info": {
                "total_furniture": len(furniture_data),
                "extracted_at": datetime.now().isoformat(),
                "source_database": f"{self.database_name}.{self.collection_name}",
                "extractor_version": "1.0"
            },
            "furniture_coordinates": furniture_data
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"OK: 가구 좌표 데이터 저장 완료: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"ERROR: 파일 저장 실패: {e}")
            return ""


async def main():
    """메인 실행 함수"""
    
    print("INFO: MongoDB에서 실제 eunbi 가구 좌표 추출 시작...")
    
    # MongoDB 가구 좌표 추출기 초기화
    extractor = MongoDBFurnitureExtractor()
    
    # 가구 좌표 데이터 추출
    furniture_data = extractor.extract_furniture_coordinates(limit=100)
    
    if furniture_data:
        # 결과 저장
        output_file = extractor.save_extracted_data(furniture_data)
        
        print(f"\nRESULT: 추출 결과")
        print(f"   - 총 가구 수: {len(furniture_data)}개")
        print(f"   - 저장 파일: {output_file}")
        
        # 샘플 데이터 출력
        if furniture_data:
            print(f"\nSAMPLE: 첫 번째 가구 정보")
            first_furniture = furniture_data[0]
            coords = first_furniture["coordinates"]
            print(f"   - 이름: {first_furniture['name']}")
            print(f"   - 좌표: ({coords['center_x']:.0f}, {coords['center_y']:.0f})mm")
            print(f"   - 크기: {coords['width']:.0f}×{coords['depth']:.0f}×{coords['height']:.0f}mm")
            print(f"   - 위치: {first_furniture['relative_position']['position_description']}")
    
    else:
        print("ERROR: 가구 좌표 데이터를 추출할 수 없습니다.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())