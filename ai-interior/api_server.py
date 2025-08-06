"""
RoomBox.jsx와 연결되는 FastAPI 서버
일관성 있는 AI 인테리어 이미지 생성 API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import json
from datetime import datetime

from roombox_integration import DifyRoomImageGenerator
from config import load_config


app = FastAPI(title="Dify Room Image Generator API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
generator = None


class RoomDataRequest(BaseModel):
    """방 데이터 요청 모델"""
    room_data: Dict[str, Any]
    style: str = "scandinavian"
    generate_image: bool = True
    user_id: Optional[str] = None


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
    """서버 시작시 Dify 시스템 초기화"""
    global generator
    
    try:
        config = load_config()
        generator = DifyRoomImageGenerator(
            config.api_key,
            config.app_id, 
            config.dataset_id
        )
        print("OK: Dify Room Image Generator 초기화 완료")
        
    except Exception as e:
        print(f"ERROR: 초기화 실패: {e}")
        generator = None


@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "Dify Room Image Generator API",
        "status": "running" if generator else "error",
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
        
        # MongoDB ID 확인
        mongo_id = request.room_data.get('mongo_id')
        if mongo_id:
            print(f"   MongoDB ID: {mongo_id}")
            print("   방 데이터가 MongoDB에 저장된 후 AI 생성 요청")
        
        # 일관성 있는 이미지 생성
        result = await generator.generate_consistent_room_image(
            room_data=request.room_data,
            style=request.style,
            user_id=request.user_id
        )
        
        if result["success"]:
            print(f"OK: 이미지 생성 성공: {result['image_path']}")
        else:
            print(f"ERROR: 이미지 생성 실패: {result.get('error')}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: API 오류: {e}")
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
async def test_style_consistency(style: str = "modern"):
    """스타일 일관성 테스트용 엔드포인트"""
    
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
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
        result = await generator.generate_consistent_room_image(
            room_data=test_room_data,
            style=style,
            user_id="test_user"
        )
        
        return {
            "test_result": result,
            "style_tested": style,
            "test_data": test_room_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def main():
    """메인 실행 함수 (uv run용)"""
    import uvicorn
    
    print("LAUNCH: Dify Room Image Generator API 시작...")
    print("   - 포트: 8080")  
    print("   - 문서: http://localhost:8080/docs")
    print("   - 테스트: http://localhost:8080/test-consistency?style=modern")
    
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