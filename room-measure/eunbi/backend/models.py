# room-measure/backend/models.py

from pydantic import BaseModel
from typing import List, Optional

# ---------------------------
# 3D 포인트 및 방 관련 모델
# ---------------------------
class Point3D(BaseModel):
    x: float
    y: float
    z: float

class RoomPoints(BaseModel):
    points: List[Point3D]
    target_height: Optional[float] = 2.3  # m 단위, 기본값 2.3m

class PixelPoint(BaseModel):
    x: int
    y: int

class DepthDistanceRequest(BaseModel):
    point1: PixelPoint
    point2: PixelPoint

class RoomNetRequest(BaseModel):
    use_roomnet: bool = True
    confidence_threshold: float = 0.7

# ---------------------------
# 창문 감지 관련 모델
# ---------------------------
class WindowInfo(BaseModel):
    wall_position: str  # "front", "back", "left", "right"
    x_position: float   # 벽에서의 상대적 위치 (0-1)
    y_position: float   # 높이 위치 (0-1)
    width: float        # 창문 너비 (상대적)
    height: float       # 창문 높이 (상대적)
    confidence: float   # 감지 신뢰도
    width_meters: float  # 실제 창문 너비 (미터)
    height_meters: float # 실제 창문 높이 (미터)

class RoomAnalysis(BaseModel):
    room_dimensions: dict
    windows: List[WindowInfo]

# ---------------------------
# 가구 좌표 변환 관련 모델
# ---------------------------
class FurniturePosition2D(BaseModel):
    x: float  # 2D x 좌표 (cm)
    z: float  # 2D y 좌표 (z 필드에 저장)

class FurniturePosition3D(BaseModel):
    x: float  # 3D x 좌표 (cm)
    y: float  # 3D y 좌표 (cm)
    z: float  # 3D z 좌표 (cm)

class FurnitureCoordinateConversionRequest(BaseModel):
    furniture_id: str
    position_3d: FurniturePosition3D
    furniture_size: List[float]  # [width, height, depth] in cm
    room_size: List[float]       # [width, height, depth] in cm

class FurnitureCoordinateConversionResponse(BaseModel):
    furniture_id: str
    position_2d: FurniturePosition2D