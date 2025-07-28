# EC2 배포용 경량 백엔드 (포트 3000)
# 데이터 저장/조회만 담당

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from typing import List

# 모델 정의
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Point3D(BaseModel):
    x: float
    y: float
    z: float

class FurniturePosition3D(BaseModel):
    id: str
    type: str
    position: Point3D
    rotation: List[float]
    size: List[float]

class RoomLayoutData(BaseModel):
    scene: dict  # 프론트엔드에서 보내는 전체 scene 구조를 받음
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class FurniturePosition2D(BaseModel):
    id: str
    type: str
    x: float
    y: float
    rotation: float
    size: List[float]

class FurnitureCoordinateConversionRequest(BaseModel):
    furniture_2d: List[FurniturePosition2D]
    room_size: dict

class FurnitureCoordinateConversionResponse(BaseModel):
    furniture_3d: List[FurniturePosition3D]

# MongoDB 서비스 (간단 버전)
import json

class SimpleStorageService:
    def __init__(self):
        self.storage_file = "room_layouts.json"
        self.data = self.load_data()
    
    def load_data(self):
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception:
            return []
    
    def save_data(self):
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)
            return True
        except Exception:
            return False
    
    def save_room_layout(self, layout_data: dict):
        layout_data['created_at'] = datetime.now().isoformat()
        layout_data['updated_at'] = datetime.now().isoformat()
        self.data.append(layout_data)
        return self.save_data()
    
    def get_all_layouts(self):
        return self.data
    
    def get_layout_by_id(self, layout_id: str):
        for layout in self.data:
            if layout.get('scene', {}).get('room', {}).get('id') == layout_id:
                return layout
        return None

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Room Measure Cloud API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 스토리지 서비스 초기화
storage_service = SimpleStorageService()

@app.get("/")
async def root():
    return {"message": "Room Measure Cloud API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cloud"}

@app.post("/convert-furniture-coordinates", response_model=FurnitureCoordinateConversionResponse)
async def convert_furniture_coordinates(request: FurnitureCoordinateConversionRequest):
    """2D 가구 좌표를 3D 좌표로 변환"""
    try:
        furniture_3d = []
        room_width = request.room_size.get('width', 400)
        room_depth = request.room_size.get('depth', 400)
        
        for furniture_2d in request.furniture_2d:
            # 2D 좌표를 3D 좌표로 변환
            # x: 2D x 좌표를 3D x 좌표로 직접 매핑
            # y: 가구 높이의 절반 (바닥에서 중심까지)
            # z: 2D y 좌표를 3D z 좌표로 매핑
            
            furniture_3d_pos = FurniturePosition3D(
                id=furniture_2d.id,
                type=furniture_2d.type,
                position=Point3D(
                    x=furniture_2d.x,
                    y=furniture_2d.size[1] / 2,  # 높이의 절반
                    z=furniture_2d.y
                ),
                rotation=[0, furniture_2d.rotation, 0],
                size=furniture_2d.size
            )
            furniture_3d.append(furniture_3d_pos)
        
        return FurnitureCoordinateConversionResponse(furniture_3d=furniture_3d)
        
    except Exception as e:
        logger.error(f"좌표 변환 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "좌표 변환 중 오류가 발생했습니다"}
        )

@app.post("/save-room-layout")
async def save_room_layout(layout_data: RoomLayoutData):
    """방 레이아웃 저장"""
    try:
        layout_dict = layout_data.dict()
        success = storage_service.save_room_layout(layout_dict)
        
        if success:
            logger.info(f"방 레이아웃 저장 완료")
            return {"success": True, "message": "방 레이아웃이 저장되었습니다"}
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "레이아웃 저장에 실패했습니다"}
            )
            
    except Exception as e:
        logger.error(f"레이아웃 저장 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "레이아웃 저장 중 오류가 발생했습니다"}
        )

@app.get("/room-layouts")
async def get_all_room_layouts():
    """모든 방 레이아웃 조회"""
    try:
        layouts = storage_service.get_all_layouts()
        return {"success": True, "layouts": layouts}
        
    except Exception as e:
        logger.error(f"레이아웃 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "레이아웃 조회 중 오류가 발생했습니다"}
        )

@app.get("/room-layout/{layout_id}")
async def get_room_layout(layout_id: str):
    """특정 방 레이아웃 조회"""
    try:
        layout = storage_service.get_layout_by_id(layout_id)
        
        if layout:
            return {"success": True, "layout": layout}
        else:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "레이아웃을 찾을 수 없습니다"}
            )
            
    except Exception as e:
        logger.error(f"레이아웃 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "레이아웃 조회 중 오류가 발생했습니다"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)