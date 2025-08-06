# 단계별 구현 가이드

## 🎯 전체 구현 로드맵

### 타임라인 개요
- **Phase 1**: Dify 기본 연동 (1주)
- **Phase 2**: 피드백 시스템 구축 (1주)  
- **Phase 3**: 고도화 및 최적화 (2주)
- **Phase 4**: 성능 모니터링 및 운영 (1주)

---

## 📋 Phase 1: Dify 기본 연동 (1주)

### Day 1-2: Dify 환경 설정

#### 1.1 Dify 계정 및 프로젝트 설정
```bash
# 1. Dify 가입 및 로그인
# https://cloud.dify.ai/apps

# 2. 새 앱 생성
앱 이름: "Room Layout RAG"
앱 타입: "Chat App" 선택
템플릿: "Blank App" 사용
```

#### 1.2 Knowledge Base 생성
```bash
# Dify UI에서 진행
1. Knowledge → "Create Knowledge" 클릭
2. 설정값:
   - Name: "Korean Room Layouts"
   - Description: "Interior room layouts with coordinates and success metrics"
   - Indexing Technique: "High Quality"
   - Embedding Model: "text-embedding-ada-002"
   - Chunk Strategy: "Automatic"
   - Chunk Size: 500 tokens
   - Chunk Overlap: 50 tokens
```

#### 1.3 API 키 및 설정 정보 수집
```python
# .env 파일 생성
DIFY_API_KEY="app-xxxxxxxxxxxxxxxxxx"
DIFY_APP_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  
DIFY_DATASET_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
DIFY_BASE_URL="https://api.dify.ai/v1"
```

### Day 3-4: 기본 API 연동

#### 1.4 Dify 클라이언트 구현
```python
# minah/backend/python/dify_client/base_client.py
import requests
import json
from typing import Dict, Any, Optional

class DifyClient:
    def __init__(self, api_key: str, base_url: str = "https://api.dify.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def test_connection(self) -> bool:
        """Dify API 연결 테스트"""
        try:
            response = requests.get(
                f"{self.base_url}/apps",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Dify 연결 테스트 실패: {e}")
            return False
    
    def chat_completion(self, query: str, conversation_id: str = "") -> Optional[Dict]:
        """Dify Chat Completion API 호출"""
        try:
            response = requests.post(
                f"{self.base_url}/chat-messages",
                headers=self.headers,
                json={
                    "inputs": {},
                    "query": query,
                    "response_mode": "blocking",
                    "conversation_id": conversation_id,
                    "user": "room_designer"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Chat completion 실패: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Chat completion 오류: {e}")
            return None
```

#### 1.5 기본 연동 테스트
```python
# test_dify_integration.py
from dify_client.base_client import DifyClient
import os
from dotenv import load_dotenv

load_dotenv()

def test_dify_basic_integration():
    """Dify 기본 연동 테스트"""
    
    client = DifyClient(os.getenv("DIFY_API_KEY"))
    
    # 1. 연결 테스트
    if not client.test_connection():
        print("❌ Dify 연결 실패")
        return False
    
    print("✅ Dify 연결 성공")
    
    # 2. 간단한 쿼리 테스트
    test_query = "Generate a room layout description for a 4m x 5m room with a desk and bed"
    
    result = client.chat_completion(test_query)
    
    if result and result.get('answer'):
        print("✅ Chat completion 성공")
        print(f"응답: {result['answer'][:100]}...")
        return True
    else:
        print("❌ Chat completion 실패")
        return False

if __name__ == "__main__":
    test_dify_basic_integration()
```

### Day 5-7: 레이아웃 처리 시스템 구현

#### 1.6 레이아웃 → 텍스트 변환기
```python
# minah/backend/python/dify_client/layout_processor.py
from typing import Dict, List, Any
import json

class LayoutProcessor:
    def __init__(self):
        self.furniture_categories = {
            "desk": "작업용 가구",
            "bed": "수면용 가구", 
            "chair": "좌석용 가구",
            "sofa": "휴식용 가구",
            "wardrobe": "수납용 가구",
            "table": "식탁용 가구"
        }
    
    def convert_to_structured_text(self, room_data: Dict[Any, Any]) -> str:
        """방 레이아웃을 구조화된 텍스트로 변환"""
        
        scene = room_data["scene"]
        room = scene["room"] 
        objects = scene["objects"]
        
        # 방 기본 정보
        text_parts = [
            "=== 방 정보 ===",
            f"크기: {room['width']}mm × {room['depth']}mm × {room['height']}mm",
            f"면적: {(room['width'] * room['depth']) / 1000000:.1f}㎡",
            f"비율: {room['width']/room['depth']:.2f} (가로/세로)",
            ""
        ]
        
        # 가구 배치 정보
        furniture_objects = [obj for obj in objects if obj.get("type") == "furniture"]
        
        if furniture_objects:
            text_parts.append("=== 가구 배치 ===")
            
            for i, obj in enumerate(furniture_objects, 1):
                pos = obj["position"]["center"]
                dims = obj["dimensions"]
                
                # 상대적 위치 계산
                rel_x = (pos["x"] / room["width"]) * 100
                rel_y = (pos["y"] / room["depth"]) * 100
                
                # 벽으로부터의 거리
                dist_left = pos["x"]
                dist_right = room["width"] - pos["x"]
                dist_bottom = pos["y"] 
                dist_top = room["depth"] - pos["y"]
                
                furniture_info = f"""
{i}. {obj['name']}:
   - 절대 좌표: ({pos['x']}, {pos['y']})mm
   - 상대 위치: 왼쪽에서 {rel_x:.1f}%, 아래에서 {rel_y:.1f}%
   - 크기: {dims['width']}×{dims['depth']}×{dims['height']}mm
   - 벽 거리: 왼쪽 {dist_left}mm, 오른쪽 {dist_right}mm, 아래 {dist_bottom}mm, 위 {dist_top}mm
   - 면적 비율: {(dims['width']*dims['depth'])/(room['width']*room['depth'])*100:.1f}%
                """
                text_parts.append(furniture_info.strip())
        
        # 창문 정보
        window_objects = [obj for obj in objects if obj.get("type") == "window"]
        
        if window_objects:
            text_parts.append("\\n=== 창문 정보 ===")
            for i, window in enumerate(window_objects, 1):
                window_info = f"{i}. {window['name']}: {window.get('details', '위치 정보 없음')}"
                text_parts.append(window_info)
        
        return "\\n".join(text_parts)
    
    def extract_key_features(self, room_data: Dict[Any, Any]) -> Dict[str, Any]:
        """레이아웃의 핵심 특징 추출"""
        
        scene = room_data["scene"]
        room = scene["room"]
        objects = scene["objects"]
        
        furniture_list = [obj for obj in objects if obj.get("type") == "furniture"]
        
        features = {
            "room_size_category": self._categorize_room_size(room),
            "furniture_count": len(furniture_list),
            "furniture_types": [obj["name"] for obj in furniture_list],
            "room_ratio": round(room["width"] / room["depth"], 2),
            "total_furniture_area_ratio": self._calculate_furniture_area_ratio(furniture_list, room),
            "layout_density": self._calculate_layout_density(furniture_list, room)
        }
        
        return features
    
    def _categorize_room_size(self, room: Dict[str, int]) -> str:
        """방 크기 카테고리 분류"""
        area_sqm = (room["width"] * room["depth"]) / 1000000
        
        if area_sqm < 10:
            return "소형"
        elif area_sqm < 20:
            return "중형" 
        else:
            return "대형"
    
    def _calculate_furniture_area_ratio(self, furniture_list: List[Dict], room: Dict) -> float:
        """가구가 차지하는 면적 비율 계산"""
        total_furniture_area = sum(
            obj["dimensions"]["width"] * obj["dimensions"]["depth"] 
            for obj in furniture_list
        )
        room_area = room["width"] * room["depth"]
        
        return round((total_furniture_area / room_area) * 100, 1)
    
    def _calculate_layout_density(self, furniture_list: List[Dict], room: Dict) -> str:
        """레이아웃 밀도 계산"""
        furniture_area_ratio = self._calculate_furniture_area_ratio(furniture_list, room)
        
        if furniture_area_ratio < 15:
            return "여유로움"
        elif furniture_area_ratio < 30:
            return "적절함"
        else:
            return "빽빽함"
```

---

## 📋 Phase 2: 피드백 시스템 구축 (1주)

### Day 8-10: 학습 시스템 구현

#### 2.1 Knowledge Base 관리자
```python
# minah/backend/python/dify_client/knowledge_manager.py
from .base_client import DifyClient
from .layout_processor import LayoutProcessor
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

class KnowledgeManager:
    def __init__(self, dify_client: DifyClient, dataset_id: str):
        self.client = dify_client
        self.dataset_id = dataset_id
        self.processor = LayoutProcessor()
    
    def add_successful_layout(self, room_data: Dict[Any, Any], 
                            user_rating: float, 
                            image_path: str = None,
                            user_comments: str = "") -> bool:
        """성공적인 레이아웃을 Knowledge Base에 추가"""
        
        if user_rating < 4.0:  # 4점 이상만 학습
            return False
        
        try:
            # 구조화된 텍스트 생성
            layout_text = self.processor.convert_to_structured_text(room_data)
            
            # 성공 지표 추가
            success_info = f"""
            
=== 성공 지표 ===
- 사용자 평점: {user_rating}/5.0
- 생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 스타일: 한국 모던 인테리어
- 이미지 경로: {image_path or '없음'}
- 사용자 코멘트: {user_comments or '없음'}
            """
            
            final_text = layout_text + success_info
            
            # Dify Knowledge Base에 문서 추가
            response = requests.post(
                f"{self.client.base_url}/datasets/{self.dataset_id}/documents",
                headers=self.client.headers,
                json={
                    "name": f"success_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                    "text": final_text,
                    "indexing_technique": "high_quality",
                    "process_rule": {
                        "mode": "custom",
                        "rules": {
                            "pre_processing_rules": [
                                {"id": "remove_extra_spaces", "enabled": True}
                            ],
                            "segmentation": {
                                "separator": "===",
                                "max_tokens": 800
                            }
                        }
                    }
                }
            )
            
            success = response.status_code == 200
            
            if success:
                print(f"✅ 성공 사례 학습 완료 (평점: {user_rating})")
            else:
                print(f"❌ 학습 실패: {response.status_code}")
                
            return success
            
        except Exception as e:
            print(f"❌ Knowledge Base 추가 중 오류: {e}")
            return False
    
    def search_similar_layouts(self, room_data: Dict[Any, Any], 
                             min_rating: float = 4.0) -> Optional[str]:
        """유사한 성공 레이아웃 검색"""
        
        # 현재 레이아웃 특징 추출
        features = self.processor.extract_key_features(room_data)
        layout_text = self.processor.convert_to_structured_text(room_data)
        
        # 검색 쿼리 생성
        search_query = f"""
다음과 비슷한 성공적인 방 레이아웃을 찾아주세요:

현재 레이아웃:
{layout_text}

핵심 특징:
- 방 크기: {features['room_size_category']}
- 가구 개수: {features['furniture_count']}개
- 가구 종류: {', '.join(features['furniture_types'])}
- 방 비율: {features['room_ratio']}
- 레이아웃 밀도: {features['layout_density']}

요구사항:
- 사용자 평점 {min_rating}점 이상
- 비슷한 방 크기와 가구 구성
- 한국 인테리어 스타일
- 성공적인 배치 사례
        """
        
        result = self.client.chat_completion(search_query)
        
        if result and result.get('answer'):
            return result['answer']
        
        return None
```

#### 2.2 프롬프트 최적화 엔진
```python
# minah/backend/python/dify_client/prompt_optimizer.py
from .knowledge_manager import KnowledgeManager
from typing import Dict, Any, Optional

class PromptOptimizer:
    def __init__(self, knowledge_manager: KnowledgeManager):
        self.knowledge_manager = knowledge_manager
        
    def generate_optimized_prompt(self, room_data: Dict[Any, Any]) -> str:
        """RAG 기반 최적화된 프롬프트 생성"""
        
        # 1. 유사한 성공 사례 검색
        similar_layouts = self.knowledge_manager.search_similar_layouts(room_data)
        
        # 2. 현재 레이아웃 분석
        current_layout = self.knowledge_manager.processor.convert_to_structured_text(room_data)
        
        # 3. 최적화 쿼리 생성
        optimization_query = f"""
다음 정보를 바탕으로 AI 이미지 생성을 위한 최적화된 프롬프트를 만들어주세요:

=== 현재 생성할 방 레이아웃 ===
{current_layout}

=== 유사한 성공 사례들 ===
{similar_layouts or '유사한 성공 사례를 찾을 수 없습니다.'}

=== 프롬프트 생성 요구사항 ===
1. 정확한 가구 위치 지정 (좌표 기반)
2. 일관된 한국 모던 인테리어 스타일
3. 현실적인 비율과 조명
4. 전문적인 인테리어 사진 품질
5. 명확한 카메라 앵글과 구도

=== 출력 형식 ===
다음 형식으로 구조화된 프롬프트를 생성해주세요:

**기본 스타일**: [한국 모던 인테리어 기본 설명]
**방 구조**: [정확한 치수와 비율]
**가구 배치**: [각 가구의 정확한 위치와 크기]
**조명과 분위기**: [조명 설정과 전체적인 분위기]
**카메라 설정**: [촬영 각도와 구도]
**품질 지시사항**: [해상도, 스타일 일관성 등]
        """
        
        # 4. Dify로 최적화된 프롬프트 생성
        result = self.knowledge_manager.client.chat_completion(optimization_query)
        
        if result and result.get('answer'):
            optimized_prompt = result['answer']
            
            # 5. 일관성을 위한 시드 추가
            layout_hash = hash(json.dumps(room_data, sort_keys=True))
            seed = layout_hash % (2**32)
            
            final_prompt = f"{optimized_prompt}\\n\\n[CONSISTENCY_SEED: {seed}]"
            
            return final_prompt
        
        # 폴백: 기본 프롬프트 생성
        return self._generate_fallback_prompt(room_data)
    
    def _generate_fallback_prompt(self, room_data: Dict[Any, Any]) -> str:
        """Dify 실패 시 기본 프롬프트 생성"""
        
        scene = room_data["scene"]
        room = scene["room"]
        furniture_objects = [obj for obj in scene["objects"] if obj.get("type") == "furniture"]
        
        prompt_parts = [
            "A hyper-realistic Korean modern interior room photograph",
            f"Room dimensions: {room['width']}mm × {room['depth']}mm × {room['height']}mm",
        ]
        
        for obj in furniture_objects:
            pos = obj["position"]["center"]
            dims = obj["dimensions"]
            prompt_parts.append(
                f"{obj['name']}: positioned at ({pos['x']}, {pos['y']})mm, "
                f"size {dims['width']}×{dims['depth']}mm"
            )
        
        prompt_parts.extend([
            "Professional interior photography, 4K resolution",
            "Soft natural lighting, clean modern aesthetic",
            "Wide-angle view showing complete room layout"
        ])
        
        return " | ".join(prompt_parts)
```

### Day 11-14: 사용자 피드백 UI 구현

#### 2.3 FastAPI 엔드포인트 확장
```python
# minah/backend/python/fastapi/enhanced_main.py
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

# 기존 import에 추가
from ..dify_client.base_client import DifyClient
from ..dify_client.knowledge_manager import KnowledgeManager
from ..dify_client.prompt_optimizer import PromptOptimizer

app = FastAPI()

# Dify 클라이언트 초기화
dify_client = DifyClient(os.getenv("DIFY_API_KEY"))
knowledge_manager = KnowledgeManager(dify_client, os.getenv("DIFY_DATASET_ID"))
prompt_optimizer = PromptOptimizer(knowledge_manager)

class FeedbackRequest(BaseModel):
    layout_id: str
    user_rating: float
    comments: Optional[str] = ""
    image_quality: Optional[int] = None
    layout_accuracy: Optional[int] = None

class GenerationRequest(BaseModel):
    scene: Dict[Any, Any]
    use_dify: bool = True
    collect_feedback: bool = True

@app.post("/generate-image-v2/")
async def generate_room_image_enhanced(request: GenerationRequest):
    """Dify 통합 이미지 생성 (피드백 시스템 포함)"""
    
    try:
        layout_id = str(uuid.uuid4())
        
        if request.use_dify:
            # Dify로 최적화된 프롬프트 생성
            optimized_prompt = prompt_optimizer.generate_optimized_prompt(request.scene)
            
            # 기존 이미지 생성 로직 사용
            image_path = await generate_image_with_imagen(
                optimized_prompt,
                f"dify_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            method = "dify_optimized"
        else:
            # 기존 방식
            traditional_prompt = create_dynamic_prompt_from_json({"scene": request.scene})
            image_path = await generate_image_with_imagen(
                traditional_prompt,
                f"traditional_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            method = "traditional"
        
        response_data = {
            "success": True,
            "layout_id": layout_id,
            "image_path": image_path,
            "method": method,
            "feedback_enabled": request.collect_feedback
        }
        
        if request.collect_feedback:
            response_data["feedback_url"] = f"/feedback/{layout_id}"
        
        # 생성 정보를 임시 저장 (피드백 수집용)
        await store_generation_info(layout_id, request.scene, image_path, method)
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 생성 실패: {str(e)}")

@app.post("/feedback/{layout_id}")
async def collect_user_feedback(layout_id: str, feedback: FeedbackRequest):
    """사용자 피드백 수집 및 학습"""
    
    try:
        # 생성 정보 조회
        generation_info = await get_generation_info(layout_id)
        
        if not generation_info:
            raise HTTPException(status_code=404, detail="레이아웃 정보를 찾을 수 없습니다")
        
        # 좋은 평가인 경우 Knowledge Base에 학습
        if feedback.user_rating >= 4.0:
            success = knowledge_manager.add_successful_layout(
                generation_info["room_data"],
                feedback.user_rating,
                generation_info["image_path"],
                feedback.comments
            )
            
            learning_status = "학습 완료" if success else "학습 실패"
        else:
            learning_status = "낮은 평점으로 학습 제외"
        
        # 피드백 데이터 저장
        feedback_data = {
            "layout_id": layout_id,
            "user_rating": feedback.user_rating,
            "comments": feedback.comments,
            "image_quality": feedback.image_quality,
            "layout_accuracy": feedback.layout_accuracy,
            "method": generation_info["method"],
            "learning_status": learning_status,
            "created_at": datetime.now().isoformat()
        }
        
        await store_feedback_data(feedback_data)
        
        return {
            "success": True,
            "message": "피드백이 성공적으로 수집되었습니다",
            "learning_status": learning_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"피드백 처리 실패: {str(e)}")

@app.get("/analytics/performance")
async def get_performance_analytics():
    """성능 분석 데이터 조회"""
    
    try:
        analytics = await calculate_performance_metrics()
        
        return {
            "success": True,
            "data": analytics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 데이터 조회 실패: {str(e)}")

# 헬퍼 함수들
async def store_generation_info(layout_id: str, room_data: Dict, image_path: str, method: str):
    """생성 정보 임시 저장"""
    # Redis 또는 데이터베이스에 저장
    pass

async def get_generation_info(layout_id: str) -> Optional[Dict]:
    """생성 정보 조회"""
    # Redis 또는 데이터베이스에서 조회
    pass

async def store_feedback_data(feedback_data: Dict):
    """피드백 데이터 저장"""
    # 데이터베이스에 저장
    pass

async def calculate_performance_metrics() -> Dict:
    """성능 지표 계산"""
    # 데이터베이스에서 집계하여 계산
    return {
        "total_generations": 0,
        "dify_success_rate": 0.0,
        "average_rating": 0.0,
        "learning_events": 0
    }
```

---

## 📋 Phase 3: 고도화 및 최적화 (2주)

### Day 15-21: 고급 기능 구현

#### 3.1 A/B 테스트 시스템
```python
# minah/backend/python/ab_testing/test_manager.py
import random
import hashlib
from typing import Dict, Any, Optional
from enum import Enum

class TestGroup(Enum):
    CONTROL = "traditional"  # 기존 방식
    TREATMENT = "dify"       # Dify 사용

class ABTestManager:
    def __init__(self, treatment_ratio: float = 0.7):
        self.treatment_ratio = treatment_ratio
        self.user_assignments = {}  # 사용자별 그룹 고정
        
    def assign_user_to_group(self, user_id: str = None, session_id: str = None) -> TestGroup:
        """사용자를 테스트 그룹에 배정"""
        
        # 식별자가 있는 경우 일관된 배정
        if user_id or session_id:
            identifier = user_id or session_id
            
            # 이미 배정된 경우 기존 그룹 반환
            if identifier in self.user_assignments:
                return self.user_assignments[identifier]
            
            # 해시 기반 일관된 배정
            hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
            is_treatment = (hash_value % 100) < (self.treatment_ratio * 100)
            
            group = TestGroup.TREATMENT if is_treatment else TestGroup.CONTROL
            self.user_assignments[identifier] = group
            
            return group
        
        # 식별자가 없는 경우 랜덤 배정
        return TestGroup.TREATMENT if random.random() < self.treatment_ratio else TestGroup.CONTROL
    
    def track_result(self, user_id: str, group: TestGroup, 
                    rating: float, generation_time: float):
        """A/B 테스트 결과 추적"""
        
        result = {
            "user_id": user_id,
            "group": group.value,
            "rating": rating,
            "generation_time": generation_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # 결과 저장 (데이터베이스나 로그 파일)
        self._save_ab_result(result)
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """A/B 테스트 통계 조회"""
        # 데이터베이스에서 집계
        return {
            "control_group": {
                "count": 0,
                "avg_rating": 0.0,
                "avg_generation_time": 0.0
            },
            "treatment_group": {
                "count": 0, 
                "avg_rating": 0.0,
                "avg_generation_time": 0.0
            },
            "statistical_significance": False
        }
    
    def _save_ab_result(self, result: Dict):
        """A/B 테스트 결과 저장"""
        # 구현: 데이터베이스나 파일에 저장
        pass
```

#### 3.2 캐싱 시스템 구현
```python
# minah/backend/python/caching/layout_cache.py
from functools import lru_cache
import hashlib
import json
import redis
from typing import Dict, Any, Optional

class LayoutCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.cache_ttl = 3600 * 24  # 24시간
    
    def get_layout_hash(self, room_data: Dict[Any, Any]) -> str:
        """레이아웃 해시 생성"""
        # 중요한 정보만 해시 대상에 포함
        essential_data = {
            "room": room_data["scene"]["room"],
            "furniture": [
                {
                    "name": obj.get("name"),
                    "position": obj.get("position", {}).get("center"),
                    "dimensions": obj.get("dimensions")
                }
                for obj in room_data["scene"]["objects"] 
                if obj.get("type") == "furniture"
            ]
        }
        
        layout_str = json.dumps(essential_data, sort_keys=True)
        return hashlib.md5(layout_str.encode()).hexdigest()
    
    def get_cached_prompt(self, room_data: Dict[Any, Any]) -> Optional[str]:
        """캐시된 프롬프트 조회"""
        layout_hash = self.get_layout_hash(room_data)
        cache_key = f"prompt:{layout_hash}"
        
        try:
            cached_prompt = self.redis_client.get(cache_key)
            if cached_prompt:
                return cached_prompt.decode('utf-8')
        except Exception as e:
            print(f"캐시 조회 실패: {e}")
        
        return None
    
    def cache_prompt(self, room_data: Dict[Any, Any], prompt: str):
        """프롬프트 캐시 저장"""
        layout_hash = self.get_layout_hash(room_data)
        cache_key = f"prompt:{layout_hash}"
        
        try:
            self.redis_client.setex(cache_key, self.cache_ttl, prompt)
        except Exception as e:
            print(f"캐시 저장 실패: {e}")
    
    def get_cached_similar_layouts(self, room_data: Dict[Any, Any]) -> Optional[str]:
        """캐시된 유사 레이아웃 검색 결과 조회"""
        layout_hash = self.get_layout_hash(room_data)
        cache_key = f"similar:{layout_hash}"
        
        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return cached_result.decode('utf-8')
        except Exception as e:
            print(f"유사 레이아웃 캐시 조회 실패: {e}")
        
        return None
    
    def cache_similar_layouts(self, room_data: Dict[Any, Any], similar_layouts: str):
        """유사 레이아웃 검색 결과 캐시 저장"""
        layout_hash = self.get_layout_hash(room_data)
        cache_key = f"similar:{layout_hash}"
        
        try:
            # 유사 레이아웃은 더 오래 캐시 (1주일)
            self.redis_client.setex(cache_key, self.cache_ttl * 7, similar_layouts)
        except Exception as e:
            print(f"유사 레이아웃 캐시 저장 실패: {e}")
```

#### 3.3 성능 모니터링 시스템
```python
# minah/backend/python/monitoring/performance_monitor.py
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
from dataclasses import dataclass

@dataclass
class GenerationMetrics:
    method: str
    generation_time: float
    user_rating: float
    success: bool
    timestamp: datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics_buffer: List[GenerationMetrics] = []
        self.buffer_size = 1000
        
    def record_generation(self, method: str, generation_time: float, 
                         user_rating: float = None, success: bool = True):
        """이미지 생성 성능 기록"""
        
        metric = GenerationMetrics(
            method=method,
            generation_time=generation_time,
            user_rating=user_rating or 0.0,
            success=success,
            timestamp=datetime.now()
        )
        
        self.metrics_buffer.append(metric)
        
        # 버퍼 크기 관리
        if len(self.metrics_buffer) > self.buffer_size:
            self.metrics_buffer = self.metrics_buffer[-self.buffer_size:]
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """성능 요약 통계"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_buffer 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"message": "데이터 없음"}
        
        # 방법별 그룹화
        dify_metrics = [m for m in recent_metrics if m.method == "dify_optimized"]
        traditional_metrics = [m for m in recent_metrics if m.method == "traditional"]
        
        summary = {
            "time_range": f"최근 {hours}시간",
            "total_generations": len(recent_metrics),
            "dify_optimized": self._calculate_method_stats(dify_metrics),
            "traditional": self._calculate_method_stats(traditional_metrics),
            "comparison": self._compare_methods(dify_metrics, traditional_metrics)
        }
        
        return summary
    
    def _calculate_method_stats(self, metrics: List[GenerationMetrics]) -> Dict[str, Any]:
        """특정 방법에 대한 통계 계산"""
        
        if not metrics:
            return {"count": 0}
        
        success_metrics = [m for m in metrics if m.success]
        rated_metrics = [m for m in metrics if m.user_rating > 0]
        
        stats = {
            "count": len(metrics),
            "success_count": len(success_metrics),
            "success_rate": len(success_metrics) / len(metrics) * 100,
            "avg_generation_time": sum(m.generation_time for m in metrics) / len(metrics),
            "min_generation_time": min(m.generation_time for m in metrics),
            "max_generation_time": max(m.generation_time for m in metrics)
        }
        
        if rated_metrics:
            stats["avg_user_rating"] = sum(m.user_rating for m in rated_metrics) / len(rated_metrics)
            stats["rated_count"] = len(rated_metrics)
        
        return stats
    
    def _compare_methods(self, dify_metrics: List[GenerationMetrics], 
                        traditional_metrics: List[GenerationMetrics]) -> Dict[str, Any]:
        """두 방법 비교"""
        
        if not dify_metrics or not traditional_metrics:
            return {"comparison": "비교 데이터 부족"}
        
        dify_avg_rating = sum(m.user_rating for m in dify_metrics if m.user_rating > 0) / max(1, len([m for m in dify_metrics if m.user_rating > 0]))
        traditional_avg_rating = sum(m.user_rating for m in traditional_metrics if m.user_rating > 0) / max(1, len([m for m in traditional_metrics if m.user_rating > 0]))
        
        dify_avg_time = sum(m.generation_time for m in dify_metrics) / len(dify_metrics)
        traditional_avg_time = sum(m.generation_time for m in traditional_metrics) / len(traditional_metrics)
        
        rating_improvement = ((dify_avg_rating - traditional_avg_rating) / traditional_avg_rating) * 100 if traditional_avg_rating > 0 else 0
        time_change = ((dify_avg_time - traditional_avg_time) / traditional_avg_time) * 100 if traditional_avg_time > 0 else 0
        
        return {
            "rating_improvement_percent": round(rating_improvement, 1),
            "time_change_percent": round(time_change, 1),
            "dify_better_rating": dify_avg_rating > traditional_avg_rating,
            "dify_faster": dify_avg_time < traditional_avg_time
        }
```

---

## 📋 Phase 4: 성능 모니터링 및 운영 (1주)

### Day 22-28: 운영 안정화

#### 4.1 대시보드 구현
```python
# minah/backend/python/dashboard/analytics_api.py
from fastapi import APIRouter
from ..monitoring.performance_monitor import PerformanceMonitor
from ..ab_testing.test_manager import ABTestManager

router = APIRouter(prefix="/analytics")

performance_monitor = PerformanceMonitor()
ab_test_manager = ABTestManager()

@router.get("/dashboard")
async def get_dashboard_data():
    """실시간 대시보드 데이터"""
    
    performance_data = performance_monitor.get_performance_summary(hours=24)
    ab_test_data = ab_test_manager.get_test_statistics()
    
    return {
        "performance": performance_data,
        "ab_test": ab_test_data,
        "system_health": await check_system_health()
    }

@router.get("/trends")
async def get_performance_trends(days: int = 7):
    """성능 트렌드 데이터"""
    
    trends = await calculate_performance_trends(days)
    
    return {
        "trends": trends,
        "recommendations": generate_recommendations(trends)
    }

async def check_system_health():
    """시스템 상태 확인"""
    return {
        "dify_api": await test_dify_connection(),
        "image_generation": await test_image_generation(),
        "cache": await test_cache_connection(),
        "database": await test_database_connection()
    }

def generate_recommendations(trends):
    """성능 트렌드 기반 권장사항 생성"""
    recommendations = []
    
    # 여기에 트렌드 분석 로직 구현
    
    return recommendations
```

#### 4.2 배포 및 운영 설정
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  minah-api:
    build: 
      context: ./minah/backend/python
      dockerfile: Dockerfile.production
    environment:
      - DIFY_API_KEY=${DIFY_API_KEY}
      - DIFY_APP_ID=${DIFY_APP_ID}
      - DIFY_DATASET_ID=${DIFY_DATASET_ID}
      - REDIS_URL=${REDIS_URL}
      - MONGODB_URL=${MONGODB_URL}
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - mongodb
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGO_USERNAME}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_PASSWORD}

volumes:
  redis_data:
  mongodb_data:
```

#### 4.3 모니터링 및 알림 설정
```python
# minah/backend/python/alerts/alert_manager.py
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from typing import List, Dict, Any

class AlertManager:
    def __init__(self, smtp_config: Dict[str, str]):
        self.smtp_config = smtp_config
        self.alert_thresholds = {
            "error_rate": 5.0,  # 5% 이상 에러율
            "avg_rating_drop": 0.5,  # 평점 0.5점 이상 하락
            "generation_time": 30.0,  # 30초 이상 생성 시간
            "dify_api_failures": 10  # 연속 10회 실패
        }
    
    def check_and_send_alerts(self, metrics: Dict[str, Any]):
        """지표 확인 후 필요시 알림 발송"""
        
        alerts = []
        
        # 에러율 확인
        if metrics.get("error_rate", 0) > self.alert_thresholds["error_rate"]:
            alerts.append({
                "type": "error_rate",
                "message": f"에러율이 {metrics['error_rate']:.1f}%로 임계값({self.alert_thresholds['error_rate']}%)을 초과했습니다.",
                "severity": "high"
            })
        
        # 평점 하락 확인
        if metrics.get("rating_drop", 0) > self.alert_thresholds["avg_rating_drop"]:
            alerts.append({
                "type": "rating_drop", 
                "message": f"평균 평점이 {metrics['rating_drop']:.2f}점 하락했습니다.",
                "severity": "medium"
            })
        
        # 생성 시간 확인
        if metrics.get("avg_generation_time", 0) > self.alert_thresholds["generation_time"]:
            alerts.append({
                "type": "slow_generation",
                "message": f"평균 생성 시간이 {metrics['avg_generation_time']:.1f}초로 지연되고 있습니다.",
                "severity": "medium"
            })
        
        # 알림 발송
        for alert in alerts:
            self.send_alert(alert)
    
    def send_alert(self, alert: Dict[str, str]):
        """이메일 알림 발송"""
        
        subject = f"[AI Interior] {alert['severity'].upper()} - {alert['type']}"
        body = f"""
알림 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
알림 유형: {alert['type']}
심각도: {alert['severity']}
메시지: {alert['message']}

시스템을 확인해 주세요.
        """
        
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = self.smtp_config['to_email']
            
            server = smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port'])
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"✅ 알림 발송 완료: {alert['type']}")
            
        except Exception as e:
            print(f"❌ 알림 발송 실패: {e}")
```

---

## ✅ 완료 체크리스트

### Phase 1 완료 기준
- [ ] Dify 계정 생성 및 앱 설정
- [ ] Knowledge Base 구축
- [ ] 기본 API 연동 테스트 성공
- [ ] 레이아웃 → 텍스트 변환 동작 확인

### Phase 2 완료 기준  
- [ ] 사용자 피드백 수집 UI 구현
- [ ] Knowledge Base 자동 학습 기능
- [ ] 프롬프트 최적화 엔진 동작
- [ ] A/B 테스트 시스템 구축

### Phase 3 완료 기준
- [ ] 캐싱 시스템 구현 및 성능 향상 확인
- [ ] 성능 모니터링 대시보드 구축
- [ ] 배치 처리 기능 구현
- [ ] 다양한 스타일 지원

### Phase 4 완료 기준
- [ ] 프로덕션 배포 환경 구축
- [ ] 실시간 모니터링 시스템 동작
- [ ] 알림 시스템 구축
- [ ] 성능 최적화 완료

---

**🎯 성공 지표**: 
- 좌표 정확도 85% 이상
- 이미지 일관성 80% 이상  
- 사용자 평점 4.5/5 이상
- 생성 시간 15초 이내