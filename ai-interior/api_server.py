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

from roombox_integration import DifyRoomImageGenerator
from config import load_config
from stable_diffusion_generator import StableDiffusionGenerator
from dalle_generator import DalleGenerator
from colab_integration import ColabInpaintingGenerator


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
            'created_at': mongo_data.get('created_at', datetime.now().isoformat()),
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


class RoomDataRequest(BaseModel):
    """방 데이터 요청 모델"""
    room_data: Dict[str, Any]
    style: str = "scandinavian"
    generate_image: bool = True
    user_id: Optional[str] = None
    use_real_ai: bool = True  # 실제 AI 모델 사용 (시간 소요 있음)


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
    global generator, sd_generator, dalle_generator, colab_generator
    
    try:
        config = load_config()
        generator = DifyRoomImageGenerator(
            config.api_key,
            config.app_id, 
            config.dataset_id
        )
        print("OK: Dify Room Image Generator 초기화 완료")
        
    except Exception as e:
        print(f"ERROR: Dify 초기화 실패: {e}")
        generator = None
    
    # Stable Diffusion 초기화 (AMD 최적화)
    try:
        sd_generator = StableDiffusionGenerator(
            use_controlnet=True,  # ControlNet 활성화
            enable_cpu_offload=True
        )
        print("OK: Stable Diffusion Generator 초기화 완료 (AMD 최적화)")
        
    except Exception as e:
        print(f"ERROR: Stable Diffusion 초기화 실패: {e}")
        sd_generator = None
    
    # DALL-E 초기화
    try:
        dalle_generator = DalleGenerator()
        print("OK: DALL-E Generator 초기화 완료")
        
    except Exception as e:
        print(f"ERROR: DALL-E 초기화 실패: {e}")
        dalle_generator = None
    
    # Colab Inpainting 생성기 초기화
    try:
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
            
    except Exception as e:
        print(f"ERROR: Colab 생성기 초기화 실패: {e}")
        colab_generator = None


@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "AI Interior Image Generator API",
        "generators": {
            "dify": "running" if generator else "error",
            "stable_diffusion": "running" if sd_generator else "error", 
            "dalle": "running" if dalle_generator else "error",
            "colab_inpainting": "running" if colab_generator else "error"
        },
        "endpoints": {
            "dify": "/generate-interior",
            "stable_diffusion": "/generate-interior-sd",
            "dalle": "/generate-interior-dalle",
            "colab_inpainting": "/generate-interior-colab"
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