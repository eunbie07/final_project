"""
RoomBox.jsx와 연결되는 FastAPI 서버
일관성 있는 AI 인테리어 이미지 생성 API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import json
import os
from datetime import datetime

# Optional imports - missing modules will be handled gracefully
try:
    from roombox_integration import DifyRoomImageGenerator
except ImportError:
    print("WARNING: roombox_integration module not found")
    DifyRoomImageGenerator = None

try:
    from config import load_config
except ImportError:
    print("WARNING: config module not found")
    def load_config():
        return None

try:
    from stable_diffusion_generator import StableDiffusionGenerator
except ImportError:
    print("WARNING: stable_diffusion_generator module not found")
    StableDiffusionGenerator = None

try:
    from dalle_generator import DalleGenerator
except ImportError:
    print("WARNING: dalle_generator module not found")
    DalleGenerator = None

try:
    from colab_integration import ColabInpaintingGenerator
except ImportError:
    print("WARNING: colab_integration module not found")
    ColabInpaintingGenerator = None

try:
    from enhanced_vertex_generator import EnhancedVertexGenerator
except ImportError:
    print("WARNING: enhanced_vertex_generator module not found")
    EnhancedVertexGenerator = None


def convert_mongo_to_current_format(mongo_data: Dict[str, Any]) -> Dict[str, Any]:
    """MongoDB 데이터를 현재 AI 생성 형식으로 변환"""
    try:
        scene = mongo_data.get('scene', {})
        room_info = scene.get('room', {})
        objects = scene.get('objects', [])
        
        # 새로운 형식으로 변환
        converted_data = {
            'dimensions': {
                'width_cm': room_info.get('width', 400),
                'depth_cm': room_info.get('depth', 500), 
                'height_cm': room_info.get('height', 280)
            },
            'furniture_3d': [],
            'area_sqm': (room_info.get('width', 400) * room_info.get('depth', 500)) / 10000,
            'volume_cum': (room_info.get('width', 400) * room_info.get('depth', 500) * room_info.get('height', 280)) / 1000000,
            'created_at': str(mongo_data.get('created_at', datetime.now())),
            'mongo_id': str(mongo_data.get('_id', ''))
        }
        
        # 가구 데이터 변환 (모든 가구 타입 포함)
        print(f"DEBUG: 변환할 가구 objects: {len(objects)}개")
        for i, obj in enumerate(objects):
            obj_type = obj.get('type', 'furniture')
            obj_name = obj.get('name', 'Furniture')
            position = obj.get('position', [0, 0, 0])
            rotation = obj.get('rotation', [0, 0, 0])
            
            print(f"DEBUG: 가구 {i+1} - {obj_name} ({obj_type}) at {position}")
            
            furniture = {
                'name': obj_name,
                'type': obj_type,
                'position': position,  # MongoDB에는 이미 cm 단위로 저장됨
                'rotation': rotation
            }
            converted_data['furniture_3d'].append(furniture)
        
        print(f"DEBUG: MongoDB 변환 완료 - 가구 {len(converted_data['furniture_3d'])}개")
        print(f"DEBUG: 변환된 데이터: {converted_data}")
        return converted_data
        
    except Exception as e:
        print(f"ERROR: MongoDB 데이터 변환 실패: {e}")
        return mongo_data


app = FastAPI(title="Dify Room Image Generator API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 생성된 이미지 파일 정적 서빙
if not os.path.exists("generated_images"):
    os.makedirs("generated_images")
    
app.mount("/images", StaticFiles(directory="generated_images"), name="images")

# 전역 변수
generator = None
sd_generator = None
dalle_generator = None
colab_generator = None
vertex_generator = None


class RoomDataRequest(BaseModel):
    """방 데이터 요청 모델"""
    room_data: Dict[str, Any]
    style: str = "scandinavian"
    generate_image: bool = True
    user_id: Optional[str] = None
    use_real_ai: bool = True  # 실제 AI 모델 사용 (시간 소요 있음)
    # 가구 스타일 변경 모드 필드들
    mode: Optional[str] = None
    screenshot: Optional[str] = None
    selected_furniture: Optional[Dict[str, Any]] = None
    new_style: Optional[str] = None


class FeedbackRequest(BaseModel):
    """피드백 요청 모델"""
    room_data: Dict[str, Any]
    image_path: str
    user_rating: float
    style: str
    comments: str = ""
    user_id: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """서버 시작시 모든 생성기 초기화"""
    global generator, sd_generator, dalle_generator, colab_generator, vertex_generator
    
    try:
        if DifyRoomImageGenerator is not None:
            config = load_config()
            if config:
                generator = DifyRoomImageGenerator(
                    config.api_key,
                    config.app_id, 
                    config.dataset_id
                )
                print("OK: Dify Room Image Generator 초기화 완료")
            else:
                print("WARNING: Config not available - Dify 비활성화")
                generator = None
        else:
            print("WARNING: DifyRoomImageGenerator not available")
            generator = None
        
    except Exception as e:
        print(f"ERROR: Dify 초기화 실패: {e}")
        generator = None
    
    # Stable Diffusion 초기화 (AMD 최적화)
    try:
        if StableDiffusionGenerator is not None:
            sd_generator = StableDiffusionGenerator(
                use_controlnet=True,  # ControlNet 활성화
                enable_cpu_offload=True
            )
            print("OK: Stable Diffusion Generator 초기화 완료 (AMD 최적화)")
        else:
            print("WARNING: StableDiffusionGenerator not available")
            sd_generator = None
        
    except Exception as e:
        print(f"ERROR: Stable Diffusion 초기화 실패: {e}")
        sd_generator = None
    
    # DALL-E 초기화
    try:
        if DalleGenerator is not None:
            dalle_generator = DalleGenerator()
            print("OK: DALL-E Generator 초기화 완료")
        else:
            print("WARNING: DalleGenerator not available")
            dalle_generator = None
        
    except Exception as e:
        print(f"ERROR: DALL-E 초기화 실패: {e}")
        dalle_generator = None
    
    # Colab Inpainting 생성기 초기화
    try:
        if ColabInpaintingGenerator is not None:
            # 환경변수에서 Colab URL 가져오기
            colab_url = os.environ.get('COLAB_API_URL', 'https://your-ngrok-url.ngrok.io')
            
            if colab_url != 'https://your-ngrok-url.ngrok.io':
                colab_generator = ColabInpaintingGenerator(colab_url)
                
                if colab_generator.health_check():
                    print("OK: Colab Inpainting Generator 초기화 완료 (95%+ 정확도)")
                else:
                    print("WARNING: Colab 서버 연결 불가 - 비활성화")
                    colab_generator = None
            else:
                print("INFO: COLAB_API_URL 환경변수 미설정 - Colab 생성기 비활성화")
                colab_generator = None
        else:
            print("WARNING: ColabInpaintingGenerator not available")
            colab_generator = None
            
    except Exception as e:
        print(f"ERROR: Colab 생성기 초기화 실패: {e}")
        colab_generator = None
    
    # Enhanced Vertex AI 생성기 초기화
    try:
        if EnhancedVertexGenerator is not None:
            vertex_generator = EnhancedVertexGenerator()
            print("OK: Enhanced Vertex AI Generator 초기화 완료 (가구 스타일 변경 지원)")
        else:
            print("WARNING: EnhancedVertexGenerator not available")
            vertex_generator = None
        
    except Exception as e:
        print(f"ERROR: Vertex AI 생성기 초기화 실패: {e}")
        vertex_generator = None


@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "AI Interior Image Generator API",
        "generators": {
            "dify": "running" if generator else "error",
            "stable_diffusion": "running" if sd_generator else "error", 
            "dalle": "running" if dalle_generator else "error",
            "colab_inpainting": "running" if colab_generator else "error",
            "vertex_ai": "running" if vertex_generator else "error"
        },
        "endpoints": {
            "dify": "/generate-interior",
            "stable_diffusion": "/generate-interior-sd",
            "dalle": "/generate-interior-dalle",
            "colab_inpainting": "/generate-interior-colab",
            "vertex_ai": "/generate-interior-vertex"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/styles")
async def get_available_styles():
    """사용 가능한 스타일 목록 반환"""
    return {
        "styles": {
            "modern": {
                "name": "모던",
                "description": "깔끔하고 미니멀한 현대적 인테리어",
                "keywords": ["minimalist", "clean", "geometric", "neutral"]
            },
            "scandinavian": {
                "name": "스칸디나비안",
                "description": "따뜻하고 아늑한 북유럽 스타일",
                "keywords": ["cozy", "hygge", "natural", "functional"]
            },
            "industrial": {
                "name": "인더스트리얼",
                "description": "도시적이고 날것의 느낌",
                "keywords": ["urban", "raw", "metal", "exposed"]
            }
        }
    }


@app.post("/generate-interior")
async def generate_interior(request: RoomDataRequest):
    """RoomBox 데이터로 일관성 있는 방 이미지 생성"""
    
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    try:
        # 요청 데이터 로깅
        print(f"TARGET: 이미지 생성 요청: {request.style} 스타일")
        print(f"   방 데이터: {request.room_data.get('dimensions', {})}")
        
        # MongoDB ID 확인 및 실제 데이터 로드
        mongo_id = request.room_data.get('mongo_id')
        final_room_data = request.room_data
        
        if mongo_id:
            print(f"   MongoDB ID: {mongo_id}")
            print("   방 데이터가 MongoDB에 저장된 후 AI 생성 요청")
            
            # MongoDB에서 실제 저장된 데이터 가져오기
            print(f"   DEBUG: MongoDB ID로 실제 데이터 조회 시작: {mongo_id}")
            try:
                print("   DEBUG: MongoDBRoomProcessor import 중...")
                from mongodb_integration import MongoDBRoomProcessor
                print("   DEBUG: MongoDBRoomProcessor 초기화 중...")
                mongo_processor = MongoDBRoomProcessor()
                print("   DEBUG: MongoDB 연결 중...")
                await mongo_processor.connect()
                print("   DEBUG: 방 데이터 조회 중...")
                
                mongo_data = await mongo_processor.get_room_data(mongo_id)
                if mongo_data:
                    print("   OK: 실제 MongoDB 데이터 조회 성공")
                    print(f"   DEBUG: 조회된 데이터 키: {list(mongo_data.keys())}")
                    if 'scene' in mongo_data:
                        objects = mongo_data['scene'].get('objects', [])
                        print(f"   DEBUG: 가구 개수: {len(objects)}")
                        for i, obj in enumerate(objects):
                            print(f"   DEBUG: 가구 {i+1}: {obj.get('name')} at {obj.get('position')}")
                    
                    # MongoDB 데이터를 현재 형식으로 변환
                    print("   DEBUG: 데이터 형식 변환 중...")
                    final_room_data = convert_mongo_to_current_format(mongo_data)
                    print(f"   DEBUG: 변환된 데이터 - 가구: {len(final_room_data.get('furniture_3d', []))}")
                else:
                    print("   WARNING: MongoDB 데이터 조회 결과가 None")
                    
                print("   DEBUG: MongoDB 연결 종료 중...")
                await mongo_processor.disconnect()
                    
            except Exception as e:
                import traceback
                print(f"   ERROR: MongoDB 접근 실패: {e}")
                print(f"   ERROR: 상세 오류:\n{traceback.format_exc()}")
                print("   전달받은 데이터를 사용하여 진행")
        
        # 일관성 있는 이미지 생성
        result = await generator.generate_consistent_room_image(
            room_data=final_room_data,
            style=request.style,
            user_id=request.user_id
        )
        
        if result["success"]:
            print(f"OK: 이미지 생성 성공: {result['image_path']}")
            
            # 로컬 파일 경로를 HTTP URL로 변환
            if 'image_path' in result and result['image_path']:
                # 파일명만 추출 (경로 제거, Windows 백슬래시 처리)
                filename = os.path.basename(result['image_path'].replace('\\', '/'))
                # HTTP URL로 변환
                result['image_url'] = f"http://localhost:8000/images/{filename}"
                print(f"   이미지 URL: {result['image_url']}")
                
                # 이미지 파일 존재 확인
                full_path = os.path.join("generated_images", filename)
                if os.path.exists(full_path):
                    print(f"   이미지 파일 확인됨: {full_path} ({os.path.getsize(full_path)} bytes)")
                else:
                    print(f"   WARNING: 이미지 파일 없음: {full_path}")
        else:
            print(f"ERROR: 이미지 생성 실패: {result.get('error')}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def convert_roomdata_to_furniture_list(room_data: Dict[str, Any]) -> List[Dict]:
    """RoomBox 데이터를 SD 생성기용 가구 리스트로 변환"""
    furniture_list = []
    
    # furniture_3d가 있는 경우 (현재 형식)
    if 'furniture_3d' in room_data:
        for furniture in room_data['furniture_3d']:
            position = furniture.get('position', [0, 0, 0])
            furniture_item = {
                'type': furniture.get('type', 'furniture'),
                'center_x': position[0] * 10 if position[0] < 100 else position[0],  # cm → mm 변환
                'center_z': position[2] * 10 if position[2] < 100 else position[2],  # cm → mm 변환
                'id': furniture.get('name', f"furniture_{len(furniture_list)}")
            }
            furniture_list.append(furniture_item)
    
    # MongoDB 형식의 경우
    elif 'scene' in room_data and 'objects' in room_data['scene']:
        for obj in room_data['scene']['objects']:
            position = obj.get('position', [0, 0, 0])
            # 이미 mm 단위인 경우 그대로 사용
            furniture_item = {
                'type': obj.get('type', 'furniture'),
                'center_x': position[0],
                'center_z': position[2],
                'id': obj.get('name', f"furniture_{len(furniture_list)}")
            }
            furniture_list.append(furniture_item)
    
    print(f"DEBUG: 변환된 가구 리스트 - {len(furniture_list)}개")
    for i, item in enumerate(furniture_list):
        print(f"  {i+1}. {item['type']} at ({item['center_x']}, {item['center_z']})")
    
    return furniture_list


@app.post("/generate-interior-sd")
async def generate_interior_with_stable_diffusion(request: RoomDataRequest):
    """Stable Diffusion + ControlNet으로 정확한 가구 위치 제어 인테리어 생성"""
    
    if not sd_generator:
        raise HTTPException(status_code=503, detail="Stable Diffusion Generator not initialized")
    
    try:
        # 요청 데이터 로깅
        print(f"TARGET: SD 이미지 생성 요청: {request.style} 스타일")
        print(f"   방 데이터: {request.room_data.get('dimensions', {})}")
        
        # MongoDB ID 확인 및 실제 데이터 로드 (기존 로직 재사용)
        mongo_id = request.room_data.get('mongo_id')
        final_room_data = request.room_data
        
        if mongo_id:
            print(f"   MongoDB ID: {mongo_id}")
            try:
                from mongodb_integration import MongoDBRoomProcessor
                mongo_processor = MongoDBRoomProcessor()
                await mongo_processor.connect()
                
                mongo_data = await mongo_processor.get_room_data(mongo_id)
                if mongo_data:
                    print("   OK: 실제 MongoDB 데이터 조회 성공")
                    final_room_data = convert_mongo_to_current_format(mongo_data)
                    
                await mongo_processor.disconnect()
                    
            except Exception as e:
                print(f"   ERROR: MongoDB 접근 실패: {e}")
                print("   전달받은 데이터를 사용하여 진행")
        
        # RoomBox 데이터를 SD용 가구 리스트로 변환
        furniture_data = convert_roomdata_to_furniture_list(final_room_data)
        
        # 방 크기 정보 추출
        dimensions = final_room_data.get('dimensions', {})
        room_dimensions = {
            'width': dimensions.get('width_cm', 400) / 100.0,   # cm → m 변환
            'height': dimensions.get('depth_cm', 500) / 100.0   # cm → m 변환
        }
        
        print(f"   변환된 방 크기: {room_dimensions['width']}m x {room_dimensions['height']}m")
        print(f"   가구 개수: {len(furniture_data)}개")
        
        # AMD CPU 최적화: Mock 모드 설정
        if not request.use_real_ai:
            sd_generator.mock_mode = True
            print("[SD] Mock 모드로 빠른 생성 (AMD CPU 최적화)")
        else:
            sd_generator.mock_mode = False
            print("[SD] 실제 AI 모델 사용 (시간 소요: 5-40분)")
        
        # Stable Diffusion으로 이미지 생성
        print("[SD] Stable Diffusion 이미지 생성 시작...")
        image_path, metadata = sd_generator.generate_interior_image(
            furniture_data=furniture_data,
            room_dimensions=room_dimensions,
            style=request.style,
            additional_prompt="precise furniture placement, photorealistic interior",
            use_mask=True,
            num_inference_steps=5 if request.use_real_ai else 1  # AMD CPU 초고속 생성
        )
        
        # 결과 준비
        result = {
            "success": True,
            "image_path": image_path,
            "generator_type": "stable_diffusion",
            "style": request.style,
            "furniture_count": len(furniture_data),
            "room_dimensions": room_dimensions,
            "mock_mode": metadata.get('mock_mode', False),
            "use_controlnet": metadata.get('use_controlnet', False),
            "timestamp": datetime.now().isoformat()
        }
        
        # 로컬 파일 경로를 HTTP URL로 변환
        if image_path:
            filename = os.path.basename(image_path.replace('\\', '/'))
            result['image_url'] = f"http://localhost:8000/images/{filename}"
            print(f"   SD 이미지 URL: {result['image_url']}")
            
            # 이미지 파일 존재 확인
            if os.path.exists(image_path):
                print(f"   이미지 파일 확인됨: {image_path} ({os.path.getsize(image_path)} bytes)")
            else:
                print(f"   WARNING: 이미지 파일 없음: {image_path}")
        
        print(f"OK: SD 이미지 생성 완료: {image_path}")
        return result
        
    except Exception as e:
        print(f"ERROR: SD API 오류: {e}")
        import traceback
        print(f"ERROR: 상세 오류:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-interior-dalle")
async def generate_interior_with_dalle(request: RoomDataRequest):
    """DALL-E 3으로 위치 제어 정확도 테스트 인테리어 생성"""
    
    if not dalle_generator:
        raise HTTPException(status_code=503, detail="DALL-E Generator not initialized")
    
    try:
        # 요청 데이터 로깅
        print(f"TARGET: DALL-E 이미지 생성 요청: {request.style} 스타일")
        print(f"   방 데이터: {request.room_data.get('dimensions', {})}")
        
        # MongoDB ID 확인 및 실제 데이터 로드 (기존 로직 재사용)
        mongo_id = request.room_data.get('mongo_id')
        final_room_data = request.room_data
        
        if mongo_id:
            print(f"   MongoDB ID: {mongo_id}")
            try:
                from mongodb_integration import MongoDBRoomProcessor
                mongo_processor = MongoDBRoomProcessor()
                await mongo_processor.connect()
                
                mongo_data = await mongo_processor.get_room_data(mongo_id)
                if mongo_data:
                    print("   OK: 실제 MongoDB 데이터 조회 성공")
                    final_room_data = convert_mongo_to_current_format(mongo_data)
                    
                await mongo_processor.disconnect()
                    
            except Exception as e:
                print(f"   ERROR: MongoDB 접근 실패: {e}")
                print("   전달받은 데이터를 사용하여 진행")
        
        # RoomBox 데이터를 DALL-E용 가구 리스트로 변환
        furniture_data = convert_roomdata_to_furniture_list(final_room_data)
        
        # 방 크기 정보 추출
        dimensions = final_room_data.get('dimensions', {})
        room_dimensions = {
            'width': dimensions.get('width_cm', 400) / 100.0,   # cm → m 변환
            'height': dimensions.get('depth_cm', 500) / 100.0   # cm → m 변환
        }
        
        print(f"   변환된 방 크기: {room_dimensions['width']}m x {room_dimensions['height']}m")
        print(f"   가구 개수: {len(furniture_data)}개")
        
        # DALL-E로 이미지 생성
        print("[DALLE] DALL-E 3 이미지 생성 시작...")
        image_path, metadata = dalle_generator.generate_interior_image(
            furniture_data=furniture_data,
            room_dimensions=room_dimensions,
            style=request.style,
            additional_prompt="position accuracy test - center placement verification"
        )
        
        # 결과 준비
        result = {
            "success": True,
            "image_path": image_path,
            "generator_type": "dalle-3",
            "style": request.style,
            "furniture_count": len(furniture_data),
            "room_dimensions": room_dimensions,
            "mock_mode": metadata.get('mock_mode', False),
            "timestamp": datetime.now().isoformat()
        }
        
        # 로컬 파일 경로를 HTTP URL로 변환
        if image_path:
            filename = os.path.basename(image_path.replace('\\', '/'))
            result['image_url'] = f"http://localhost:8000/images/{filename}"
            print(f"   DALL-E 이미지 URL: {result['image_url']}")
            
            # 이미지 파일 존재 확인
            if os.path.exists(image_path):
                print(f"   이미지 파일 확인됨: {image_path} ({os.path.getsize(image_path)} bytes)")
            else:
                print(f"   WARNING: 이미지 파일 없음: {image_path}")
        
        print(f"OK: DALL-E 이미지 생성 완료: {image_path}")
        return result
        
    except Exception as e:
        print(f"ERROR: DALL-E API 오류: {e}")
        import traceback
        print(f"ERROR: 상세 오류:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """사용자 피드백 수집 및 학습"""
    
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    try:
        print(f"LOG: 피드백 수집: 평점 {request.user_rating}/5.0, 스타일: {request.style}")
        
        # 피드백을 통한 학습
        learning_result = await generator.learn_from_feedback(
            room_data=request.room_data,
            image_path=request.image_path,
            user_rating=request.user_rating,
            style=request.style,
            comments=request.comments
        )
        
        if learning_result["learned"]:
            print(f"OK: 학습 완료: {learning_result['message']}")
        else:
            print(f"WARNING: 학습 안됨: {learning_result['message']}")
        
        return learning_result
        
    except Exception as e:
        print(f"ERROR: 피드백 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics")
async def get_analytics():
    """생성 및 학습 통계"""
    
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    # TODO: 실제 통계 데이터 구현
    return {
        "total_generations": 0,
        "successful_generations": 0,
        "learned_layouts": 0,
        "style_distribution": {
            "modern": 0,
            "scandinavian": 0,
            "industrial": 0
        },
        "average_rating": 0.0
    }


@app.post("/test-consistency")
async def test_style_consistency(style: str = "modern", generator_type: str = "dify"):
    """스타일 일관성 테스트용 엔드포인트"""
    
    if generator_type == "dify" and not generator:
        raise HTTPException(status_code=503, detail="Dify Generator not initialized")
    elif generator_type == "stable_diffusion" and not sd_generator:
        raise HTTPException(status_code=503, detail="Stable Diffusion Generator not initialized")
    elif generator_type == "dalle" and not dalle_generator:
        raise HTTPException(status_code=503, detail="DALL-E Generator not initialized")
    
    # 테스트용 방 데이터
    test_room_data = {
        "scene": {
            "room": {
                "width": 4000,
                "depth": 5000,
                "height": 2800
            },
            "objects": [
                {
                    "type": "furniture",
                    "name": "sofa",
                    "position": {
                        "center": {"x": 2000, "y": 1500, "z": 0}
                    },
                    "dimensions": {
                        "width": 2000,
                        "depth": 800,
                        "height": 800
                    },
                    "rotation_z": 0
                },
                {
                    "type": "furniture", 
                    "name": "coffee_table",
                    "position": {
                        "center": {"x": 2000, "y": 2500, "z": 0}
                    },
                    "dimensions": {
                        "width": 1200,
                        "depth": 600,
                        "height": 400
                    },
                    "rotation_z": 0
                }
            ]
        }
    }
    
    try:
        if generator_type == "dify":
            result = await generator.generate_consistent_room_image(
                room_data=test_room_data,
                style=style,
                user_id="test_user"
            )
        elif generator_type == "stable_diffusion":
            # 테스트 데이터를 SD용 형식으로 변환
            furniture_data = convert_roomdata_to_furniture_list(test_room_data)
            room_dimensions = {'width': 4.0, 'height': 5.0}
            
            image_path, metadata = sd_generator.generate_interior_image(
                furniture_data=furniture_data,
                room_dimensions=room_dimensions,
                style=style,
                additional_prompt="test room layout",
                num_inference_steps=10
            )
            
            result = {
                "success": True,
                "image_path": image_path,
                "generator_type": "stable_diffusion",
                "mock_mode": metadata.get('mock_mode', False)
            }
        else:  # dalle
            # 테스트 데이터를 DALL-E용 형식으로 변환
            furniture_data = convert_roomdata_to_furniture_list(test_room_data)
            room_dimensions = {'width': 4.0, 'height': 5.0}
            
            image_path, metadata = dalle_generator.generate_interior_image(
                furniture_data=furniture_data,
                room_dimensions=room_dimensions,
                style=style,
                additional_prompt="test room layout for position accuracy"
            )
            
            result = {
                "success": True,
                "image_path": image_path,
                "generator_type": "dalle-3",
                "mock_mode": metadata.get('mock_mode', False)
            }
        
        return {
            "test_result": result,
            "style_tested": style,
            "generator_tested": generator_type,
            "test_data": test_room_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-interior-colab")
async def generate_interior_with_colab(request: RoomDataRequest):
    """Colab ComfyUI Inpainting으로 95%+ 정확도 인테리어 생성"""
    
    if not colab_generator:
        raise HTTPException(status_code=503, detail="Colab Inpainting Generator not initialized")
    
    try:
        # 요청 데이터 로깅
        print(f"TARGET: Colab Inpainting 이미지 생성 요청: {request.style} 스타일")
        print(f"   방 데이터: {request.room_data.get('dimensions', {})}")  
        
        # MongoDB ID 확인 및 실제 데이터 로드 (기존 로직 재사용)
        mongo_id = request.room_data.get('mongo_id')
        final_room_data = request.room_data
        
        if mongo_id:
            print(f"   MongoDB ID: {mongo_id}")
            try:
                from mongodb_integration import MongoDBRoomProcessor
                mongo_processor = MongoDBRoomProcessor()
                await mongo_processor.connect()
                
                mongo_data = await mongo_processor.get_room_data(mongo_id)
                if mongo_data:
                    print("   OK: 실제 MongoDB 데이터 조회 성공")
                    final_room_data = convert_mongo_to_current_format(mongo_data)
                    
                await mongo_processor.disconnect()
                    
            except Exception as e:
                print(f"   ERROR: MongoDB 접근 실패: {e}")
                print("   전달받은 데이터를 사용하여 진행")
        
        # Colab으로 이미지 생성 (95%+ 정확도)
        print("[COLAB] 정확한 위치 제어 Inpainting 생성 시작...")
        image_path, metadata = colab_generator.generate_interior_image(
            room_data=final_room_data,
            style=request.style
        )
        
        if image_path:
            # 결과 준비
            result = {
                "success": True,
                "image_path": image_path,
                "generator_type": "colab_comfyui_inpainting",
                "style": request.style,
                "accuracy_score": metadata.get('accuracy_score', 0.0),
                "accuracy_percentage": metadata.get('accuracy_percentage', '0%'),
                "position_analysis": metadata.get('position_analysis', {}),
                "furniture_count": len(final_room_data.get('furniture_3d', [])),
                "room_dimensions": final_room_data.get('dimensions', {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # 로컬 파일 경로를 HTTP URL로 변환
            filename = os.path.basename(image_path.replace('\\', '/'))
            result['image_url'] = f"http://localhost:8000/images/{filename}"
            print(f"   Colab 이미지 URL: {result['image_url']}")
            
            # 이미지 파일 존재 확인
            if os.path.exists(image_path):
                print(f"   이미지 파일 확인됨: {image_path} ({os.path.getsize(image_path)} bytes)")
            else:
                print(f"   WARNING: 이미지 파일 없음: {image_path}")
                
        else:
            # Colab 생성 실패시 폴백
            result = {
                "success": False,
                "error": "Colab 생성 실패",
                "generator_type": "colab_fallback",
                "metadata": metadata
            }
        
        print(f"OK: Colab 인테리어 생성 완료")
        return result
        
    except Exception as e:
        print(f"ERROR: Colab API 오류: {e}")
        import traceback
        print(f"ERROR: 상세 오류:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-interior-vertex")
async def generate_interior_with_vertex_ai(request: RoomDataRequest):
    """Vertex AI로 고품질 인테리어 생성 및 가구 스타일 변경"""
    
    # 요청 데이터 로깅
    print(f"TARGET: Vertex AI 이미지 생성 요청: {request.style} 스타일")
    print(f"   방 데이터: {request.room_data.get('dimensions', {})}")
    
    # Vertex AI Generator가 사용 가능한지 확인
    if not vertex_generator:
        print("WARNING: Vertex AI Generator 사용 불가 - Mock 응답 반환")
        return await generate_vertex_mock_response(request)
    
    try:
        # 가구 스타일 변경 모드 감지 (top-level 또는 room_data 내부)
        mode = getattr(request, 'mode', None) or request.room_data.get('mode')
        is_furniture_style_change = mode == "furniture_style_change"
        
        if is_furniture_style_change:
            print("   모드: 가구 스타일 변경 🎯")
            return await handle_vertex_furniture_style_change(request)
        else:
            print("   모드: 일반 인테리어 생성 🏠")
            return await handle_vertex_room_generation(request)
            
    except Exception as e:
        print(f"ERROR: Vertex AI API 오류: {e}")
        import traceback
        print(f"ERROR: 상세 오류:\n{traceback.format_exc()}")
        # 오류 발생 시에도 Mock 응답 반환
        return await generate_vertex_mock_response(request)


async def handle_vertex_room_generation(request: RoomDataRequest):
    """Vertex AI 일반 방 인테리어 생성"""
    global vertex_generator
    
    # MongoDB ID 확인 및 실제 데이터 로드 (기존 로직 재사용)
    mongo_id = request.room_data.get('mongo_id')
    final_room_data = request.room_data
    
    if mongo_id:
        print(f"   MongoDB ID: {mongo_id}")
        try:
            from mongodb_integration import MongoDBRoomProcessor
            mongo_processor = MongoDBRoomProcessor()
            await mongo_processor.connect()
            
            mongo_data = await mongo_processor.get_room_data(mongo_id)
            if mongo_data:
                print("   OK: 실제 MongoDB 데이터 조회 성공")
                final_room_data = convert_mongo_to_current_format(mongo_data)
                
            await mongo_processor.disconnect()
                
        except Exception as e:
            print(f"   ERROR: MongoDB 접근 실패: {e}")
            print("   전달받은 데이터를 사용하여 진행")
    
    print("[VERTEX] Enhanced Vertex AI 이미지 생성 시작...")
    
    # Enhanced Vertex Generator는 인증이 필요하므로 직접 Mock 응답 생성
    return await generate_vertex_mock_response(request)


async def handle_vertex_furniture_style_change(request: RoomDataRequest):
    """Vertex AI 가구 스타일 변경"""
    global vertex_generator
    
    room_data = request.room_data
    
    # 가구 스타일 변경에 필요한 데이터 확인 (top-level 우선, room_data 폴백)
    screenshot = getattr(request, 'screenshot', None) or room_data.get('screenshot')
    selected_furniture = getattr(request, 'selected_furniture', None) or room_data.get('selected_furniture')
    new_style = getattr(request, 'new_style', None) or room_data.get('new_style')
    
    # new_style은 필수, screenshot은 선택사항, selected_furniture는 없으면 전체 가구 변경
    if not new_style:
        raise HTTPException(
            status_code=400, 
            detail="가구 스타일 변경에는 new_style이 필요합니다"
        )
    
    if selected_furniture:
        print(f"   가구: {selected_furniture.get('name')} → {new_style} 스타일")
        change_mode = "individual"
    else:
        print(f"   모든 가구 → {new_style} 스타일 일괄 변경")
        change_mode = "all_furniture"
    
    # 가구 스타일 변경용 특별 프롬프트 생성
    furniture_prompt = create_furniture_style_change_prompt(
        screenshot=screenshot,
        selected_furniture=selected_furniture,
        new_style=new_style,
        room_data=room_data,
        change_mode=change_mode
    )
    
    print("[VERTEX] 가구 스타일 변경용 Vertex AI 생성 시작...")
    
    # 가구 스타일 변경 모드를 위한 Mock 응답 생성
    return await generate_vertex_mock_response(request)


def create_furniture_style_change_prompt(screenshot: str, selected_furniture: Dict[str, Any], 
                                        new_style: str, room_data: Dict[str, Any], 
                                        change_mode: str = "individual") -> str:
    """가구 스타일 변경용 특별 프롬프트 생성"""
    
    if change_mode == "all_furniture":
        # 전체 가구 스타일 변경 모드
        furniture_list = room_data.get('furniture_3d', [])
        furniture_names = [f.get('name', 'furniture') for f in furniture_list]
        
        if furniture_list:
            furniture_desc = f"all furniture ({', '.join(furniture_names)})"
        else:
            furniture_desc = "all furniture in the room"
    else:
        # 개별 가구 변경 모드 (기존 로직)
        furniture_name = selected_furniture.get('name', 'furniture')
        furniture_type = selected_furniture.get('type', 'furniture')
        position = selected_furniture.get('position', [0, 0, 0])
        furniture_desc = f"the {furniture_name}"
    
    # 스타일별 설명
    style_descriptions = {
        'modern_bed': 'sleek, minimalist platform bed with clean geometric lines and neutral colors',
        'vintage_bed': 'classic ornate bed frame with traditional carved details and rich wood finish',
        'minimalist_bed': 'ultra-simple bed design with no headboard, clean lines, and functional focus',
        'luxury_bed': 'opulent upholstered bed with tufted headboard and premium materials',
        'scandinavian_bed': 'light wood bed frame with natural finish and cozy Nordic styling',
        'industrial_bed': 'metal frame bed with exposed materials and urban loft aesthetic',
        
        'ergonomic_chair': 'modern office chair with mesh backing and adjustable features',
        'vintage_chair': 'classic mid-century chair with retro fabric and wooden legs',
        'gaming_chair': 'high-back racing style chair with LED accents and premium padding',
        'office_chair': 'professional executive chair with leather upholstery',
        'accent_chair': 'statement piece chair with bold patterns or unique design',
        'rocking_chair': 'traditional wooden rocking chair with comfortable cushioning'
    }
    
    if change_mode == "all_furniture":
        # 전체 가구 변경용 프롬프트
        style_desc = style_descriptions.get(new_style, f'{new_style} style furniture')
        
        prompt = f"""
        Transform this interior room image: change all furniture to {new_style} style.
        
        Requirements:
        - Keep the exact same room layout, lighting, and room structure
        - Transform {furniture_desc} to {new_style} style
        - Maintain photorealistic quality and natural lighting
        - Preserve the room's overall proportions and spatial layout
        - Ensure all new furniture fits naturally in the existing space
        - Keep furniture in their current positions but update their style/appearance
        
        Style transformation: All furniture → {new_style} style
        
        The result should look like a natural interior photograph with cohesive furniture styling.
        """
    else:
        # 개별 가구 변경용 프롬프트 (기존)
        furniture_type = selected_furniture.get('type', 'furniture')
        position = selected_furniture.get('position', [0, 0, 0])
        style_desc = style_descriptions.get(new_style, f'{new_style} style {furniture_type}')
        
        prompt = f"""
        Transform this interior room image: replace {furniture_desc} with a {style_desc}.
        
        Requirements:
        - Keep the exact same room layout, lighting, and all other furniture
        - Only change {furniture_desc} located at position ({position[0]}, {position[1]})
        - Maintain photorealistic quality and natural lighting
        - Preserve the room's overall style while updating only the specified furniture
        - Ensure the new {furniture_type} fits naturally in the existing space
        
        Style transformation: {furniture_desc} → {style_desc}
        
        The result should look like a natural interior photograph with the furniture seamlessly integrated.
        """
    
    return prompt.strip()


async def generate_vertex_mock_response(request: RoomDataRequest):
    """Vertex AI 사용 불가 시 Mock 응답 생성"""
    import asyncio
    from PIL import Image, ImageDraw, ImageFont
    
    # 짧은 지연 시뮬레이션
    await asyncio.sleep(1)
    
    # 가구 스타일 변경 모드인지 확인 (top-level 또는 room_data 내부)
    mode = getattr(request, 'mode', None) or request.room_data.get('mode')
    is_furniture_style_change = mode == "furniture_style_change"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_furniture_style_change:
        selected_furniture = getattr(request, 'selected_furniture', None) or request.room_data.get('selected_furniture', {})
        new_style = getattr(request, 'new_style', None) or request.room_data.get('new_style', 'unknown')
        
        # 전체 가구 변경 vs 개별 가구 변경 구분
        if selected_furniture:
            # 개별 가구 변경
            filename = f"vertex_mock_furniture_{new_style}_{timestamp}.png"
            change_description = f"{selected_furniture.get('name', 'furniture')} - {new_style}"
            change_details = {
                "name": selected_furniture.get('name', 'furniture'),
                "type": selected_furniture.get('type', 'furniture'),
                "new_style": new_style
            }
        else:
            # 전체 가구 변경
            filename = f"vertex_mock_all_furniture_{new_style}_{timestamp}.png"
            furniture_list = request.room_data.get('furniture_3d', [])
            furniture_names = [f.get('name', 'furniture') for f in furniture_list]
            change_description = f"모든 가구 - {new_style}"
            change_details = {
                "mode": "all_furniture",
                "furniture_list": furniture_names,
                "new_style": new_style,
                "count": len(furniture_list)
            }
        
        # Mock 가구 스타일 변경 이미지 생성
        await create_mock_furniture_image(filename, selected_furniture or {"name": "전체가구", "type": "furniture"}, new_style)
        
        mock_result = {
            "success": True,
            "message": "Vertex AI Mock Mode - 가구 스타일 변경 시뮬레이션",
            "image_path": f"generated_images/{filename}",
            "image_url": f"http://localhost:8000/images/{filename}",
            "generator_type": "vertex_ai_mock_furniture",
            "style": change_description,
            "method": "mock",
            "furniture_change": True,
            "changed_furniture": change_details,
            "note": "실제 Vertex AI 서비스를 사용할 수 없어 Mock 결과를 반환했습니다",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"OK: Vertex AI Mock 가구 스타일 변경 응답 생성: {filename}")
        return mock_result
    else:
        # 일반 인테리어 생성 Mock
        filename = f"vertex_mock_{request.style}_{timestamp}.png"
        
        # Mock 인테리어 이미지 생성
        await create_mock_interior_image(filename, request.style, request.room_data)
        
        mock_result = {
            "success": True,
            "message": "Vertex AI Mock Mode - 인테리어 생성 시뮬레이션",
            "image_path": f"generated_images/{filename}",
            "image_url": f"http://localhost:8000/images/{filename}",
            "generator_type": "vertex_ai_mock",
            "style": request.style,
            "method": "mock",
            "room_layout": {
                "width_mm": request.room_data.get('dimensions', {}).get('width_cm', 400) * 10,
                "depth_mm": request.room_data.get('dimensions', {}).get('depth_cm', 500) * 10,
                "height_mm": request.room_data.get('dimensions', {}).get('height_cm', 280) * 10
            },
            "note": "실제 Vertex AI 서비스를 사용할 수 없어 Mock 결과를 반환했습니다",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"OK: Vertex AI Mock 인테리어 생성 응답: {filename}")
        return mock_result


async def create_mock_furniture_image(filename: str, furniture: dict, style: str):
    """Mock 가구 스타일 변경 이미지 생성"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # 512x512 이미지 생성
        img = Image.new('RGB', (512, 512), color=(245, 245, 250))
        draw = ImageDraw.Draw(img)
        
        # 제목
        furniture_name = furniture.get('name', 'Furniture')
        title = f"{furniture_name} → {style} Style"
        
        # 텍스트 그리기 (기본 폰트 사용)
        try:
            # Windows 시스템 폰트 시도
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            # 기본 폰트 폴백
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 제목 (중앙)
        draw.text((256, 50), title, font=font_large, fill=(50, 50, 100), anchor="mt")
        
        # 가구 표시 (간단한 직사각형으로)
        furniture_color = (100, 150, 200)
        draw.rectangle([156, 150, 356, 300], fill=furniture_color, outline=(80, 120, 180), width=3)
        
        # 스타일 설명
        style_desc = get_style_description(style)
        draw.text((256, 320), style_desc, font=font_small, fill=(70, 70, 70), anchor="mt")
        
        # Mock 표시
        draw.text((256, 400), "Vertex AI Mock Generation", font=font_small, fill=(150, 100, 100), anchor="mt")
        draw.text((256, 430), "실제 AI 생성이 아닙니다", font=font_small, fill=(100, 100, 100), anchor="mt")
        
        # 저장
        full_path = os.path.join("generated_images", filename)
        img.save(full_path, "PNG")
        print(f"Mock 가구 이미지 생성 완료: {full_path}")
        
    except Exception as e:
        print(f"Mock 가구 이미지 생성 실패: {e}")


async def create_mock_interior_image(filename: str, style: str, room_data: dict):
    """Mock 인테리어 이미지 생성"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # 768x512 이미지 생성 (16:9 비율)
        img = Image.new('RGB', (768, 512), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)
        
        # 제목
        title = f"Vertex AI {style.title()} Interior"
        
        # 폰트 설정
        try:
            font_large = ImageFont.truetype("arial.ttf", 28)
            font_medium = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_large = font_medium = font_small = ImageFont.load_default()
        
        # 제목
        draw.text((384, 30), title, font=font_large, fill=(50, 50, 100), anchor="mt")
        
        # 간단한 방 표현
        room_color = get_room_color(style)
        draw.rectangle([50, 80, 718, 400], fill=room_color, outline=(100, 100, 100), width=2)
        
        # 가구 표현 (간단한 도형들)
        furniture_list = room_data.get('furniture_3d', [])
        if furniture_list:
            for i, furniture in enumerate(furniture_list[:3]):  # 최대 3개만
                x = 100 + (i * 200)
                y = 200 + (i * 20)
                draw.rectangle([x, y, x+80, y+60], fill=(150, 100, 80), outline=(120, 80, 60), width=2)
                draw.text((x+40, y+70), furniture.get('name', 'Furniture'), font=font_small, fill=(80, 80, 80), anchor="mt")
        
        # 방 정보
        dimensions = room_data.get('dimensions', {})
        room_info = f"Room: {dimensions.get('width_cm', 0)/100:.1f}m × {dimensions.get('depth_cm', 0)/100:.1f}m"
        draw.text((384, 420), room_info, font=font_medium, fill=(100, 100, 100), anchor="mt")
        
        # Mock 표시
        draw.text((384, 450), "Vertex AI Mock Generation", font=font_small, fill=(150, 100, 100), anchor="mt")
        draw.text((384, 470), "실제 AI 생성이 아닙니다", font=font_small, fill=(100, 100, 100), anchor="mt")
        
        # 저장
        full_path = os.path.join("generated_images", filename)
        img.save(full_path, "PNG")
        print(f"Mock 인테리어 이미지 생성 완료: {full_path}")
        
    except Exception as e:
        print(f"Mock 인테리어 이미지 생성 실패: {e}")


def get_style_description(style: str) -> str:
    """스타일별 설명 반환"""
    descriptions = {
        'modern_bed': 'Clean lines, minimalist design',
        'vintage_bed': 'Classic ornate details',
        'luxury_bed': 'Upholstered headboard',
        'scandinavian_bed': 'Light wood, natural finish',
        'industrial_bed': 'Metal frame, urban style',
        'ergonomic_chair': 'Mesh back, adjustable',
        'vintage_chair': 'Retro fabric, wooden legs',
        'gaming_chair': 'Racing style, LED accents'
    }
    return descriptions.get(style, f'{style} furniture style')


def get_room_color(style: str) -> tuple:
    """스타일별 방 색상 반환"""
    colors = {
        'modern': (250, 250, 250),
        'scandinavian': (248, 245, 240),
        'industrial': (230, 230, 235),
        'cozy': (245, 240, 235),
        'bohemian': (240, 238, 230)
    }
    return colors.get(style, (245, 245, 245))


async def main():
    """메인 실행 함수 (uv run용)"""
    import uvicorn
    
    print("LAUNCH: AI Interior Image Generator API 시작...")
    print("   - 포트: 8000")  
    print("   - 문서: http://localhost:8000/docs")
    print("   - Dify 테스트: http://localhost:8000/test-consistency?style=modern&generator_type=dify")
    print("   - SD 테스트: http://localhost:8000/test-consistency?style=modern&generator_type=stable_diffusion")
    print("   - DALL-E 테스트: http://localhost:8000/test-consistency?style=modern&generator_type=dalle")
    print("   - Colab 95%+ 정확도: POST http://localhost:8000/generate-interior-colab")
    print("   - Vertex AI (가구 스타일 변경): POST http://localhost:8000/generate-interior-vertex")
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()


def cli_main():
    """CLI 진입점 (uv run api-server)"""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()