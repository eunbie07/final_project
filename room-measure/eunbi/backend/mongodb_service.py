# room-measure/backend/mongodb_service.py

from pymongo import MongoClient
from datetime import datetime
from pydantic import BaseModel
from fastapi import HTTPException
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# MongoDB용 3D Scene 모델 (새로운 형식)
class SceneData(BaseModel):
    scene: dict

# 기존 호환성을 위한 모델 (deprecated)
class RoomLayoutData(BaseModel):
    roomInfo: dict = None
    furniture: list = None
    windows: list = None
    statistics: dict = None
    metadata: dict = None
    scene: dict = None  # 새로운 형식 지원

class MongoDBService:
    def __init__(self, mongo_url: str = "mongodb://13.55.21.100:27017", db_name: str = "room_measure"):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.client = None
        self.db = None
        self.room_layouts_collection = None
        self.connect()
    
    def connect(self):
        """MongoDB에 연결"""
        try:
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.room_layouts_collection = self.db.room_layouts
            logger.info("MongoDB 연결 성공")
            return True
        except Exception as e:
            logger.error(f"MongoDB 연결 실패: {e}")
            self.client = None
            self.db = None
            self.room_layouts_collection = None
            return False
    
    def is_connected(self) -> bool:
        """MongoDB 연결 상태 확인"""
        return self.room_layouts_collection is not None
    
    async def save_room_layout(self, layout_data: RoomLayoutData) -> dict:
        """방 레이아웃 데이터를 MongoDB에 저장 (기존 + 새로운 형식 지원)"""
        if not self.is_connected():
            raise HTTPException(status_code=500, detail="MongoDB 연결이 없습니다")
        
        try:
            # 저장할 데이터 준비
            save_data = layout_data.model_dump()
            
            # None 값 제거 (새로운 형식에서는 불필요한 필드)
            save_data = {k: v for k, v in save_data.items() if v is not None}
            
            # 저장 시간 추가 (한국 시간)
            save_data["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 데이터 형식 감지 및 로깅
            if "scene" in save_data:
                logger.info("새로운 3D Scene 형식으로 저장")
                save_data["format_version"] = "2.0.0"
            else:
                logger.info("기존 RoomLayout 형식으로 저장")
                save_data["format_version"] = "1.0.0"
            
            # MongoDB에 저장
            result = self.room_layouts_collection.insert_one(save_data)
            
            logger.info(f"Room layout saved with ID: {result.inserted_id}")
            
            return {
                "success": True,
                "message": "방 레이아웃이 성공적으로 저장되었습니다",
                "id": str(result.inserted_id),
                "format_version": save_data["format_version"]
            }
            
        except Exception as e:
            logger.error(f"MongoDB 저장 오류: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"저장 중 오류가 발생했습니다: {str(e)}"
            )
    
    async def get_room_layouts(self, limit: int = 10, skip: int = 0) -> dict:
        """저장된 방 레이아웃 목록 조회"""
        if not self.is_connected():
            raise HTTPException(status_code=500, detail="MongoDB 연결이 없습니다")
        
        try:
            # MongoDB에서 데이터 조회 (최신순으로 정렬)
            cursor = self.room_layouts_collection.find().sort("saved_at", -1).skip(skip).limit(limit)
            layouts = []
            
            for layout in cursor:
                # ObjectId를 문자열로 변환
                layout["_id"] = str(layout["_id"])
                layouts.append(layout)
            
            # 전체 개수 조회
            total_count = self.room_layouts_collection.count_documents({})
            
            return {
                "success": True,
                "layouts": layouts,
                "total_count": total_count,
                "returned_count": len(layouts)
            }
            
        except Exception as e:
            logger.error(f"MongoDB 조회 오류: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"조회 중 오류가 발생했습니다: {str(e)}"
            )
    
    async def get_room_layout_by_id(self, layout_id: str) -> dict:
        """특정 방 레이아웃 조회"""
        if not self.is_connected():
            raise HTTPException(status_code=500, detail="MongoDB 연결이 없습니다")
        
        try:
            from bson import ObjectId
            
            # MongoDB에서 특정 레이아웃 조회
            layout = self.room_layouts_collection.find_one({"_id": ObjectId(layout_id)})
            
            if not layout:
                raise HTTPException(status_code=404, detail="레이아웃을 찾을 수 없습니다")
            
            # ObjectId를 문자열로 변환
            layout["_id"] = str(layout["_id"])
            
            return {
                "success": True,
                "layout": layout
            }
            
        except Exception as e:
            logger.error(f"MongoDB 조회 오류: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"조회 중 오류가 발생했습니다: {str(e)}"
            )
    
    def close_connection(self):
        """MongoDB 연결 종료"""
        if self.client:
            self.client.close()
            logger.info("MongoDB 연결 종료")

# 글로벌 MongoDB 서비스 인스턴스
mongodb_service = MongoDBService()