"""
MongoDB 통합 모듈
eunbi -> MongoDB -> ai-interior 파이프라인을 위한 데이터 처리
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

# MongoDB 관련 imports (선택적)
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("WARNING: PyMongo가 설치되지 않았습니다. pip install pymongo")

from enhanced_vertex_generator import generate_room_image_from_coordinates
from dify_rag import DifyLayoutRAG
from roombox_integration import RoomBoxDataProcessor


class MongoDBRoomProcessor:
    """MongoDB에서 방 데이터를 가져와서 AI 이미지를 생성하는 통합 처리기"""
    
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
        
        # Dify RAG 초기화
        import os
        self.dify_rag = DifyLayoutRAG(
            os.getenv("DIFY_API_KEY", ""),
            os.getenv("DIFY_APP_ID", ""),
            os.getenv("DIFY_DATASET_ID", "")
        )
        
        self.processor = RoomBoxDataProcessor(self.dify_rag)
        
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
            print("INFO: 대안: 로컬 JSON 파일을 사용합니다")
            self.client = None
    
    async def fetch_latest_layouts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최신 레이아웃 데이터 가져오기"""
        
        if self.collection is not None:
            try:
                # MongoDB에서 최신 데이터 가져오기
                cursor = self.collection.find().sort("saved_at", -1).limit(limit)
                layouts = list(cursor)
                
                # ObjectId를 문자열로 변환
                for layout in layouts:
                    if "_id" in layout:
                        layout["_id"] = str(layout["_id"])
                
                print(f"OK: MongoDB에서 {len(layouts)}개 레이아웃 가져옴")
                return layouts
                
            except Exception as e:
                print(f"ERROR: MongoDB 조회 실패: {e}")
        
        # MongoDB 실패 시 샘플 데이터 반환
        return self._get_sample_layouts()
    
    def _get_sample_layouts(self) -> List[Dict[str, Any]]:
        """샘플 레이아웃 데이터 (MongoDB 연결 실패 시)"""
        
        return [
            {
                "_id": "sample_001",
                "scene": {
                    "description": "샘플 방 1 - 모던 스타일",
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
                            "rotation_z": 0
                        }
                    ]
                },
                "saved_at": datetime.now().isoformat(),
                "format_version": "2.0.0"
            },
            {
                "_id": "sample_002",
                "scene": {
                    "description": "샘플 방 2 - 스칸디나비아 스타일",
                    "room": {"width": 3500, "depth": 4000, "height": 2500},
                    "objects": [
                        {
                            "type": "furniture",
                            "name": "sofa",
                            "position": {"center": {"x": 1750, "y": 2000}},
                            "dimensions": {"width": 2200, "depth": 900, "height": 850},
                            "rotation_z": 0
                        },
                        {
                            "type": "furniture",
                            "name": "coffee_table",
                            "position": {"center": {"x": 1750, "y": 1200}},
                            "dimensions": {"width": 1200, "depth": 600, "height": 450},
                            "rotation_z": 0
                        },
                        {
                            "type": "window",
                            "name": "main_window",
                            "wall": 1,
                            "dimensions": {"width": 1500, "height": 1200},
                            "position": {"x": 1000, "y": 0, "z": 1800},
                            "rotation_z": 0
                        }
                    ]
                },
                "saved_at": datetime.now().isoformat(),
                "format_version": "2.0.0"
            }
        ]
    
    async def process_layout_to_image(self, 
                                    layout_data: Dict[str, Any], 
                                    style: str = "modern",
                                    auto_learn: bool = True) -> Dict[str, Any]:
        """레이아웃 데이터를 이미지로 변환"""
        
        try:
            # 1. 데이터 검증 및 파싱
            if "scene" not in layout_data:
                return {"success": False, "error": "Invalid layout data format"}
            
            room_layout = self.processor.parse_roombox_data(layout_data)
            
            print(f"ROOM: 방 처리 시작: {room_layout.width_mm}x{room_layout.depth_mm}mm")
            print(f"ITEM: 가구 개수: {len(room_layout.furniture)}")
            
            # 2. Dify RAG로 유사 레이아웃 검색
            similar_layouts = None
            if self.dify_rag.dataset_id:
                similar_layouts = self.dify_rag.find_similar_layouts(layout_data)
            
            # 3. 이미지 생성
            result = await generate_room_image_from_coordinates(
                room_data=layout_data,
                style=style,
                prefer_google_ai=True
            )
            
            # 4. 결과에 추가 정보 포함
            result.update({
                "layout_id": layout_data.get("_id", "unknown"),
                "processing_time": datetime.now().isoformat(),
                "similar_layouts_found": similar_layouts is not None,
                "style_applied": style,
                "coordinate_accuracy": "enhanced"
            })
            
            # 5. 성공 시 Dify에 학습 데이터 추가 (선택적)
            if auto_learn and result.get("success") and self.dify_rag.dataset_id:
                try:
                    self.dify_rag.add_successful_layout(
                        layout_data, 
                        4.2,  # 기본 높은 점수
                        result.get("image_path", "")
                    )
                    print("LEARN: Dify Knowledge Base에 자동 학습 완료")
                except Exception as e:
                    print(f"WARNING: 자동 학습 실패: {e}")
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "layout_id": layout_data.get("_id", "unknown")
            }
    
    async def batch_process_layouts(self, 
                                  limit: int = 5,
                                  style: str = "modern") -> List[Dict[str, Any]]:
        """여러 레이아웃을 배치로 처리"""
        
        print(f"LAUNCH: 배치 처리 시작: {limit}개 레이아웃")
        
        # 1. MongoDB에서 최신 레이아웃들 가져오기
        layouts = await self.fetch_latest_layouts(limit)
        
        if not layouts:
            return [{"success": False, "error": "No layouts found"}]
        
        # 2. 각 레이아웃을 비동기로 처리
        tasks = []
        for layout in layouts:
            task = self.process_layout_to_image(layout, style)
            tasks.append(task)
        
        # 3. 모든 작업 동시 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 4. 결과 정리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "layout_index": i
                })
            else:
                processed_results.append(result)
        
        success_count = sum(1 for r in processed_results if r.get("success", False))
        print(f"OK: 배치 처리 완료: {success_count}/{len(layouts)} 성공")
        
        return processed_results
    
    async def monitor_new_layouts(self, 
                                style: str = "modern", 
                                check_interval: int = 30):
        """새로운 레이아웃 모니터링 및 자동 처리"""
        
        print(f"WATCH: 새 레이아웃 모니터링 시작 (간격: {check_interval}초)")
        
        last_check_time = datetime.now()
        processed_ids = set()
        
        while True:
            try:
                # 최근 추가된 레이아웃 확인
                if self.collection is not None:
                    new_layouts = list(self.collection.find({
                        "saved_at": {"$gte": last_check_time.isoformat()}
                    }))
                else:
                    new_layouts = []
                
                for layout in new_layouts:
                    layout_id = str(layout.get("_id", ""))
                    
                    if layout_id not in processed_ids:
                        print(f"NEW: 새 레이아웃 발견: {layout_id}")
                        
                        result = await self.process_layout_to_image(layout, style)
                        
                        if result.get("success"):
                            print(f"OK: 자동 처리 완료: {result.get('image_path')}")
                        else:
                            print(f"ERROR: 자동 처리 실패: {result.get('error')}")
                        
                        processed_ids.add(layout_id)
                
                last_check_time = datetime.now()
                await asyncio.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\nSTOP: 모니터링 중단")
                break
            except Exception as e:
                print(f"WARNING: 모니터링 오류: {e}")
                await asyncio.sleep(check_interval)
    
    def close(self):
        """MongoDB 연결 종료"""
        if self.client:
            self.client.close()
            print("OK: MongoDB 연결 종료")


class RoomImagePipeline:
    """eunbi -> MongoDB -> ai-interior 전체 파이프라인"""
    
    def __init__(self, mongo_uri: str = None):
        self.mongo_processor = MongoDBRoomProcessor(mongo_uri) if mongo_uri else MongoDBRoomProcessor()
        
    async def run_pipeline(self, 
                          mode: str = "batch",
                          style: str = "modern", 
                          limit: int = 5) -> Dict[str, Any]:
        """파이프라인 실행"""
        
        print(f"LAUNCH: Room Image Pipeline 시작 (모드: {mode}, 스타일: {style})")
        
        if mode == "batch":
            # 배치 처리 모드
            results = await self.mongo_processor.batch_process_layouts(limit, style)
            
            summary = {
                "mode": "batch",
                "total_processed": len(results),
                "successful": sum(1 for r in results if r.get("success", False)),
                "failed": sum(1 for r in results if not r.get("success", False)),
                "style": style,
                "results": results
            }
            
            return summary
            
        elif mode == "monitor":
            # 모니터링 모드
            await self.mongo_processor.monitor_new_layouts(style)
            return {"mode": "monitor", "status": "stopped"}
        
        else:
            return {"error": f"Unknown mode: {mode}"}
    
    def close(self):
        """리소스 정리"""
        self.mongo_processor.close()


# CLI 진입점들
async def run_batch_processing():
    """배치 처리 실행"""
    pipeline = RoomImagePipeline()
    
    try:
        result = await pipeline.run_pipeline(mode="batch", style="scandinavian", limit=3)
        print("\n" + "="*50)
        print("STATS: 처리 결과 요약:")
        print(f"  - 총 처리: {result['total_processed']}개")
        print(f"  - 성공: {result['successful']}개")
        print(f"  - 실패: {result['failed']}개")
        print(f"  - 스타일: {result['style']}")
        
        for i, res in enumerate(result['results']):
            if res.get('success'):
                print(f"  OK: {i+1}: {res.get('image_path', 'Unknown')}")
            else:
                print(f"  ERROR: {i+1}: {res.get('error', 'Unknown error')}")
                
    finally:
        pipeline.close()

async def run_monitoring():
    """모니터링 실행"""
    pipeline = RoomImagePipeline()
    
    try:
        await pipeline.run_pipeline(mode="monitor", style="modern")
    finally:
        pipeline.close()


# uv run 명령어들
def batch_main():
    """uv run batch-process"""
    asyncio.run(run_batch_processing())

def monitor_main():
    """uv run monitor-layouts"""
    asyncio.run(run_monitoring())


if __name__ == "__main__":
    # 기본 실행: 배치 처리
    asyncio.run(run_batch_processing())