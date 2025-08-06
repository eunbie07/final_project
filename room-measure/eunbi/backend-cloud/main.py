# EC2 배포용 경량 백엔드 (포트 3000)
# 데이터 저장/조회만 담당

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
import logging
import os
from typing import List
import jwt
import bcrypt
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from dotenv import load_dotenv
import traceback

# 환경변수 로드
import pathlib
env_path = pathlib.Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# 디버그: 환경변수 확인
print(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST', 'localhost')}")
print(f"POSTGRES_USER: {os.getenv('POSTGRES_USER', 'postgres')}")
print(f"POSTGRES_DB: {os.getenv('POSTGRES_DB', 'user_auth')}")

# 모델 정의
from pydantic import BaseModel, EmailStr
from typing import Optional

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

# 사용자 인증 관련 모델
class UserSignup(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

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

# 환경변수 설정
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-here")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# PostgreSQL 데이터베이스 연결
class DatabaseService:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
                database=os.getenv("POSTGRES_DB", "user_auth"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
            print("PostgreSQL 데이터베이스 연결 성공")
        except Exception as e:
            print(f"데이터베이스 연결 실패: {e}")
            self.connection = None
    
    def get_connection(self):
        if not self.connection or self.connection.closed:
            self.connect()
        return self.connection

# 인증 관련 유틸리티
security = HTTPBearer()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="토큰이 유효하지 않습니다")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="토큰이 유효하지 않습니다")

# MongoDB 서비스
from pymongo import MongoClient

class MongoDBService:
    def __init__(self):
        self.mongo_url = f"mongodb://{os.getenv('MONGO_HOST', '13.55.21.100')}:27017"
        self.db_name = "room_measure"
        self.client = None
        self.db = None
        self.room_layouts_collection = None
        self.connect()
    
    def connect(self):
        try:
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.room_layouts_collection = self.db.room_layouts
            print("MongoDB 연결 성공")
            return True
        except Exception as e:
            print(f"MongoDB 연결 실패: {e}")
            self.client = None
            self.db = None
            self.room_layouts_collection = None
            return False
    
    def is_connected(self):
        return self.room_layouts_collection is not None
    
    def save_room_layout(self, layout_data: dict):
        if not self.is_connected():
            return False, None
        
        try:
            layout_data["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            layout_data["format_version"] = "2.0.0"
            
            result = self.room_layouts_collection.insert_one(layout_data)
            logger.info(f"MongoDB에 저장 완료: {result.inserted_id}")
            return True, str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB 저장 오류: {e}")
            return False, None

# 스토리지 서비스 (JSON 파일 기반 - 백업용)
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('room_measure_cloud.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Room Measure Cloud API",
    version="1.0.0",
    description="""
    방 크기 측정 및 가구 배치 서비스의 클라우드 API

    ## 주요 기능
    - 사용자 인증 (회원가입/로그인)
    - 방 레이아웃 저장/조회
    - 가구 좌표 변환

    ## 포트 구분
    - **포트 3000**: 클라우드 데이터 저장/조회 (이 API)
    - **포트 3010**: 로컬 이미지/AI 처리
    """,
    contact={
        "name": "Room Measure Team",
        "email": "support@roommeasure.com"
    },
    license_info={
        "name": "MIT License"
    },
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 ID 추적을 위한 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    request_id = id(request)
    
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Request {request_id} completed in {process_time:.3f}s - Status: {response.status_code}")
    
    return response

# 전역 예외 핸들러
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """입력 검증 오류 처리"""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "입력 데이터가 올바르지 않습니다",
            "details": exc.errors()
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """서버 내부 오류 처리"""
    logger.error(f"Internal server error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "request_id": id(request)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 처리"""
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "예기치 않은 오류가 발생했습니다",
            "request_id": id(request)
        }
    )

# 서비스 초기화
db_service = DatabaseService()
mongodb_service = MongoDBService()
storage_service = SimpleStorageService()

@app.get("/")
async def root():
    return {"message": "Room Measure Cloud API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cloud"}

# 사용자 인증 API
@app.post(
    "/signup", 
    response_model=dict,
    summary="사용자 회원가입",
    description="새 사용자 계정을 생성합니다",
    responses={
        200: {"description": "회원가입 성공"},
        400: {"description": "이미 가입된 이메일"},
        422: {"description": "입력 데이터 검증 실패"},
        500: {"description": "서버 오류"}
    }
)
async def signup(user_data: UserSignup):
    """
    새 사용자 계정을 생성합니다.
    
    - **email**: 유효한 이메일 주소
    - **password**: 6자 이상의 비밀번호
    """
    try:
        conn = db_service.get_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="데이터베이스 연결 실패")
        
        with conn.cursor() as cursor:
            # 이메일 중복 확인
            cursor.execute("SELECT id FROM users WHERE email = %s", (user_data.email,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="이미 가입된 이메일입니다")
            
            # 비밀번호 해싱
            hashed_password = hash_password(user_data.password)
            
            # 사용자 생성
            cursor.execute(
                "INSERT INTO users (email, password, created_at) VALUES (%s, %s, %s)",
                (user_data.email, hashed_password, datetime.now())
            )
            
        return {"message": "회원가입이 완료되었습니다"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"회원가입 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다")

@app.post(
    "/login", 
    response_model=Token,
    summary="사용자 로그인",
    description="사용자 인증 후 JWT 토큰을 발급합니다",
    responses={
        200: {"description": "로그인 성공, JWT 토큰 반환"},
        401: {"description": "이메일 또는 비밀번호 불일치"},
        422: {"description": "입력 데이터 검증 실패"},
        500: {"description": "서버 오류"}
    }
)
async def login(user_data: UserLogin):
    """
    사용자 로그인을 처리합니다.
    
    - **email**: 등록된 이메일 주소
    - **password**: 계정 비밀번호
    
    성공시 JWT 토큰과 사용자 정보를 반환합니다.
    """
    try:
        conn = db_service.get_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="데이터베이스 연결 실패")
        
        with conn.cursor() as cursor:
            # 사용자 조회
            cursor.execute("SELECT * FROM users WHERE email = %s", (user_data.email,))
            user = cursor.fetchone()
            
            if not user or not verify_password(user_data.password, user['password']):
                raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 일치하지 않습니다")
            
            # JWT 토큰 생성
            access_token = create_access_token({"user_id": user['id']})
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": user['id'],
                    "email": user['email'],
                    "created_at": user['created_at']
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"로그인 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다")

@app.get("/me", response_model=UserResponse)
async def get_current_user(current_user_id: int = Depends(verify_token)):
    """현재 사용자 정보 조회"""
    try:
        conn = db_service.get_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="데이터베이스 연결 실패")
        
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (current_user_id,))
            user = cursor.fetchone()
            
            if not user:
                raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
            
            return {
                "id": user['id'],
                "email": user['email'],
                "created_at": user['created_at']
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"사용자 조회 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 오류가 발생했습니다")

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
async def save_room_layout(
    layout_data: RoomLayoutData, 
    current_user_id: int = Depends(verify_token)
):
    """방 레이아웃 저장 (로그인 사용자)"""
    try:
        conn = db_service.get_connection()
        if not conn:
            # 데이터베이스 연결 실패시 JSON 파일로 저장
            layout_dict = layout_data.dict()
            layout_dict['user_id'] = current_user_id
            success = storage_service.save_room_layout(layout_dict)
            
            if success:
                logger.info(f"방 레이아웃 저장 완료 (JSON): 사용자 {current_user_id}")
                return {"success": True, "message": "방 레이아웃이 저장되었습니다", "layout_id": None}
            else:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": "레이아웃 저장에 실패했습니다"}
                )
        
        # MySQL에 저장
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO room_layouts (user_id, layout_data, created_at) VALUES (%s, %s, %s)",
                (current_user_id, json.dumps(layout_data.dict()), datetime.now())
            )
            layout_id = cursor.lastrowid
            
        logger.info(f"방 레이아웃 저장 완료 (DB): 사용자 {current_user_id}")
        return {"success": True, "message": "방 레이아웃이 저장되었습니다", "layout_id": str(layout_id)}
        
    except Exception as e:
        logger.error(f"레이아웃 저장 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "레이아웃 저장 중 오류가 발생했습니다"}
        )

@app.post("/save-room-layout-guest")
async def save_room_layout_guest(layout_data: RoomLayoutData):
    """방 레이아웃 저장 (게스트 사용자)"""
    try:
        layout_dict = layout_data.dict()
        layout_dict['user_id'] = None  # 게스트 사용자
        
        # MongoDB 우선 저장 시도
        mongodb_success, mongodb_result = mongodb_service.save_room_layout(layout_dict.copy())
        
        # JSON 파일에도 백업 저장
        json_success = storage_service.save_room_layout(layout_dict)
        
        if mongodb_success:
            logger.info("방 레이아웃 저장 완료 (게스트 - MongoDB)")
            return {"success": True, "message": "방 레이아웃이 MongoDB에 저장되었습니다", "layout_id": mongodb_result}
        elif json_success:
            logger.info("방 레이아웃 저장 완료 (게스트 - JSON 백업)")
            return {"success": True, "message": "방 레이아웃이 JSON 파일에 저장되었습니다", "layout_id": None}
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
async def get_user_room_layouts(current_user_id: int = Depends(verify_token)):
    """사용자별 방 레이아웃 조회"""
    try:
        conn = db_service.get_connection()
        if not conn:
            # 데이터베이스 연결 실패시 JSON 파일에서 조회
            layouts = storage_service.get_all_layouts()
            user_layouts = [layout for layout in layouts if layout.get('user_id') == current_user_id]
            return {"success": True, "layouts": user_layouts}
        
        # MySQL에서 조회
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM room_layouts WHERE user_id = %s ORDER BY created_at DESC",
                (current_user_id,)
            )
            layouts = cursor.fetchall()
            
            # JSON 데이터 파싱
            for layout in layouts:
                if isinstance(layout['layout_data'], str):
                    layout['layout_data'] = json.loads(layout['layout_data'])
            
        return {"success": True, "layouts": layouts}
        
    except Exception as e:
        logger.error(f"레이아웃 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "레이아웃 조회 중 오류가 발생했습니다"}
        )

@app.get("/room-layouts-guest")
async def get_guest_room_layouts():
    """게스트 사용자 방 레이아웃 조회"""
    try:
        layouts = storage_service.get_all_layouts()
        guest_layouts = [layout for layout in layouts if layout.get('user_id') is None]
        return {"success": True, "layouts": guest_layouts}
        
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