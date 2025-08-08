# API 통합 예제 코드

## 🚀 빠른 시작 예제

### 1. 환경 설정

```bash
# .env 파일 생성
DIFY_API_KEY=app-xxxxxxxxxxxxxxxxxx
DIFY_APP_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
DIFY_DATASET_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxx
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

```python
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
requests==2.31.0
python-dotenv==1.0.0
pydantic==2.5.0
numpy==1.24.3
Pillow==10.1.0
redis==5.0.1
```

### 2. 기본 Dify 클라이언트

```python
# dify_client.py
import requests
import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class DifyClient:
    """Dify API 클라이언트"""
    
    def __init__(self):
        self.api_key = os.getenv("DIFY_API_KEY")
        self.app_id = os.getenv("DIFY_APP_ID") 
        self.dataset_id = os.getenv("DIFY_DATASET_ID")
        self.base_url = "https://api.dify.ai/v1"
        
        if not all([self.api_key, self.app_id, self.dataset_id]):
            raise ValueError("Dify 환경변수가 설정되지 않았습니다")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            response = requests.get(
                f"{self.base_url}/apps", 
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def chat_completion(self, query: str, user_id: str = "default") -> Optional[str]:
        """Chat Completion API 호출"""
        try:
            response = requests.post(
                f"{self.base_url}/chat-messages",
                headers=self.headers,
                json={
                    "inputs": {},
                    "query": query,
                    "response_mode": "blocking",
                    "user": user_id
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("answer", "")
            else:
                print(f"Chat completion 실패: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Chat completion 오류: {e}")
            return None
    
    def add_document(self, name: str, text: str) -> bool:
        """Knowledge Base에 문서 추가"""
        try:
            response = requests.post(
                f"{self.base_url}/datasets/{self.dataset_id}/documents",
                headers=self.headers,
                json={
                    "name": name,
                    "text": text,
                    "indexing_technique": "high_quality"
                },
                timeout=30
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"문서 추가 오류: {e}")
            return False

# 사용 예제
if __name__ == "__main__":
    client = DifyClient()
    
    # 연결 테스트
    if client.test_connection():
        print("✅ Dify 연결 성공")
        
        # 테스트 쿼리
        result = client.chat_completion("안녕하세요!")
        print(f"응답: {result}")
    else:
        print("❌ Dify 연결 실패")
```

### 3. 레이아웃 처리 예제

```python
# layout_processor.py
import json
import hashlib
from typing import Dict, Any, List

class LayoutProcessor:
    """방 레이아웃 데이터 처리기"""
    
    def __init__(self):
        self.furniture_types = {
            "desk": "책상",
            "bed": "침대", 
            "chair": "의자",
            "sofa": "소파",
            "wardrobe": "옷장",
            "table": "테이블"
        }
    
    def convert_to_text(self, room_data: Dict[Any, Any]) -> str:
        """JSON 레이아웃을 구조화된 텍스트로 변환"""
        
        scene = room_data["scene"]
        room = scene["room"]
        objects = scene["objects"]
        
        # 방 정보
        room_info = f"""
=== 방 기본 정보 ===
• 크기: {room['width']}mm × {room['depth']}mm × {room['height']}mm
• 면적: {(room['width'] * room['depth']) / 1000000:.1f}㎡  
• 가로세로 비율: {room['width']/room['depth']:.2f}
        """
        
        # 가구 정보
        furniture_list = [obj for obj in objects if obj.get("type") == "furniture"]
        
        if furniture_list:
            furniture_info = "\\n=== 가구 배치 ==="
            
            for i, furniture in enumerate(furniture_list, 1):
                pos = furniture["position"]["center"]
                dims = furniture["dimensions"]
                
                # 상대적 위치 계산
                rel_x = (pos["x"] / room["width"]) * 100
                rel_y = (pos["y"] / room["depth"]) * 100
                
                furniture_detail = f"""
{i}. {furniture['name']}:
   • 위치: ({pos['x']}, {pos['y']})mm (왼쪽에서 {rel_x:.1f}%, 아래에서 {rel_y:.1f}%)
   • 크기: {dims['width']}×{dims['depth']}×{dims['height']}mm
   • 면적 비율: {(dims['width']*dims['depth'])/(room['width']*room['depth'])*100:.1f}%
                """
                furniture_info += furniture_detail
        else:
            furniture_info = "\\n=== 가구 없음 ==="
        
        return room_info + furniture_info
    
    def extract_features(self, room_data: Dict[Any, Any]) -> Dict[str, Any]:
        """레이아웃 핵심 특징 추출"""
        
        scene = room_data["scene"]
        room = scene["room"]
        furniture_list = [obj for obj in scene["objects"] if obj.get("type") == "furniture"]
        
        # 방 크기 카테고리
        area_sqm = (room["width"] * room["depth"]) / 1000000
        if area_sqm < 10:
            size_category = "소형"
        elif area_sqm < 20: 
            size_category = "중형"
        else:
            size_category = "대형"
        
        # 가구 밀도 계산
        total_furniture_area = sum(
            obj["dimensions"]["width"] * obj["dimensions"]["depth"]
            for obj in furniture_list
        )
        furniture_ratio = (total_furniture_area / (room["width"] * room["depth"])) * 100
        
        return {
            "size_category": size_category,
            "area_sqm": round(area_sqm, 1),
            "furniture_count": len(furniture_list),
            "furniture_types": [obj["name"] for obj in furniture_list],
            "furniture_density": round(furniture_ratio, 1),
            "room_ratio": round(room["width"] / room["depth"], 2)
        }
    
    def create_hash(self, room_data: Dict[Any, Any]) -> str:
        """레이아웃 고유 해시 생성"""
        essential_data = {
            "room": room_data["scene"]["room"],
            "furniture": [
                {
                    "name": obj["name"],
                    "position": obj["position"]["center"],
                    "dimensions": obj["dimensions"]
                }
                for obj in room_data["scene"]["objects"]
                if obj.get("type") == "furniture"
            ]
        }
        
        data_str = json.dumps(essential_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

# 사용 예제
if __name__ == "__main__":
    # 테스트 데이터
    test_data = {
        "scene": {
            "room": {"width": 4000, "depth": 5000, "height": 2800},
            "objects": [
                {
                    "type": "furniture",
                    "name": "책상",
                    "position": {"center": {"x": 3000, "y": 2500, "z": 0}},
                    "dimensions": {"width": 1200, "depth": 600, "height": 750}
                },
                {
                    "type": "furniture", 
                    "name": "침대",
                    "position": {"center": {"x": 1000, "y": 4000, "z": 0}},
                    "dimensions": {"width": 1500, "depth": 2000, "height": 600}
                }
            ]
        }
    }
    
    processor = LayoutProcessor()
    
    # 텍스트 변환
    text = processor.convert_to_text(test_data)
    print("=== 레이아웃 텍스트 ===")
    print(text)
    
    # 특징 추출
    features = processor.extract_features(test_data)
    print("\\n=== 레이아웃 특징 ===")
    print(json.dumps(features, indent=2, ensure_ascii=False))
    
    # 해시 생성
    layout_hash = processor.create_hash(test_data)
    print(f"\\n=== 레이아웃 해시 ===")
    print(layout_hash)
```

### 4. 통합 이미지 생성기

```python
# integrated_generator.py
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dify_client import DifyClient
from layout_processor import LayoutProcessor

class IntegratedImageGenerator:
    """Dify 통합 이미지 생성기"""
    
    def __init__(self):
        self.dify_client = DifyClient()
        self.processor = LayoutProcessor()
        
        # Dify 연결 확인
        if not self.dify_client.test_connection():
            raise ConnectionError("Dify API 연결 실패")
    
    async def generate_image(self, room_data: Dict[Any, Any], 
                           use_dify: bool = True) -> Dict[str, Any]:
        """통합 이미지 생성 메인 함수"""
        
        start_time = datetime.now()
        
        try:
            if use_dify:
                result = await self._generate_with_dify(room_data)
            else:
                result = await self._generate_traditional(room_data)
            
            # 생성 시간 계산
            generation_time = (datetime.now() - start_time).total_seconds()
            result["generation_time"] = generation_time
            
            return result
            
        except Exception as e:
            # 오류 시 fallback
            print(f"이미지 생성 오류: {e}")
            if use_dify:
                print("기존 방식으로 재시도...")
                return await self._generate_traditional(room_data)
            else:
                raise e
    
    async def _generate_with_dify(self, room_data: Dict[Any, Any]) -> Dict[str, Any]:
        """Dify 최적화 방식으로 생성"""
        
        # 1. 유사한 성공 사례 검색
        similar_layouts = await self._search_similar_layouts(room_data)
        
        # 2. 최적화된 프롬프트 생성
        optimized_prompt = await self._generate_optimized_prompt(
            room_data, similar_layouts
        )
        
        # 3. 일관성을 위한 시드 생성
        layout_hash = self.processor.create_hash(room_data)
        seed = int(layout_hash[:8], 16) % (2**32)
        
        # 4. 실제 이미지 생성
        image_path = await self._call_image_generation_api(
            optimized_prompt, seed
        )
        
        return {
            "success": True,
            "method": "dify_optimized",
            "image_path": image_path,
            "prompt": optimized_prompt,
            "seed": seed,
            "layout_hash": layout_hash
        }
    
    async def _search_similar_layouts(self, room_data: Dict[Any, Any]) -> str:
        """유사한 레이아웃 검색"""
        
        layout_text = self.processor.convert_to_text(room_data)
        features = self.processor.extract_features(room_data)
        
        search_query = f"""
다음과 비슷한 성공적인 방 레이아웃 사례를 찾아주세요:

{layout_text}

핵심 특징:
- 방 크기: {features['size_category']} ({features['area_sqm']}㎡)
- 가구 개수: {features['furniture_count']}개  
- 가구 종류: {', '.join(features['furniture_types'])}
- 가구 밀도: {features['furniture_density']}%
- 방 비율: {features['room_ratio']}

조건:
- 사용자 평점 4점 이상
- 한국 인테리어 스타일
- 비슷한 방 크기와 가구 구성
        """
        
        result = self.dify_client.chat_completion(search_query)
        return result or "유사한 사례를 찾을 수 없습니다."
    
    async def _generate_optimized_prompt(self, room_data: Dict[Any, Any], 
                                       similar_layouts: str) -> str:
        """최적화된 프롬프트 생성"""
        
        layout_text = self.processor.convert_to_text(room_data)
        
        optimization_query = f"""
다음 정보를 바탕으로 AI 이미지 생성용 최적화된 프롬프트를 만들어주세요:

=== 생성할 방 레이아웃 ===
{layout_text}

=== 유사한 성공 사례 ===
{similar_layouts}

=== 요구사항 ===
1. 정확한 가구 위치 명시 (좌표 기반)
2. 일관된 한국 모던 인테리어 스타일
3. 현실적인 비율과 조명
4. 전문적인 인테리어 사진 품질
5. 명확한 카메라 앵글

=== 출력 형식 ===
다음과 같이 구조화된 프롬프트를 생성해주세요:

**스타일**: [한국 모던 인테리어 설명]
**방 구조**: [정확한 치수와 비율]  
**가구 배치**: [각 가구의 정확한 위치]
**조명**: [조명 설정과 분위기]
**카메라**: [촬영 각도와 구도]
**품질**: [해상도, 스타일 일관성]

영어로 된 완성된 프롬프트만 출력해주세요.
        """
        
        result = self.dify_client.chat_completion(optimization_query)
        
        if result:
            return result
        else:
            # 폴백 프롬프트
            return self._create_fallback_prompt(room_data)
    
    def _create_fallback_prompt(self, room_data: Dict[Any, Any]) -> str:
        """기본 프롬프트 생성 (Dify 실패시)"""
        
        scene = room_data["scene"]
        room = scene["room"]
        furniture_list = [obj for obj in scene["objects"] if obj.get("type") == "furniture"]
        
        prompt_parts = [
            "A hyper-realistic Korean modern interior room photograph",
            f"Room: {room['width']}×{room['depth']}×{room['height']}mm"
        ]
        
        for furniture in furniture_list:
            pos = furniture["position"]["center"]
            dims = furniture["dimensions"]
            prompt_parts.append(
                f"{furniture['name']}: at ({pos['x']},{pos['y']})mm, "
                f"size {dims['width']}×{dims['depth']}mm"
            )
        
        prompt_parts.extend([
            "Professional interior photography, 4K resolution",
            "Soft natural lighting, clean aesthetic",
            "Wide-angle view, complete room layout visible"
        ])
        
        return " | ".join(prompt_parts)
    
    async def _generate_traditional(self, room_data: Dict[Any, Any]) -> Dict[str, Any]:
        """기존 방식으로 생성"""
        
        prompt = self._create_fallback_prompt(room_data)
        image_path = await self._call_image_generation_api(prompt)
        
        return {
            "success": True,
            "method": "traditional",
            "image_path": image_path,
            "prompt": prompt
        }
    
    async def _call_image_generation_api(self, prompt: str, seed: int = None) -> str:
        """실제 이미지 생성 API 호출"""
        
        # 여기에 OpenAI DALL-E나 Vertex AI 호출 로직 구현
        # 예제에서는 모의 응답 반환
        
        filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # 실제 구현에서는 이 부분을 AI 이미지 생성 API로 대체
        print(f"🎨 이미지 생성 중... (프롬프트: {prompt[:100]}...)")
        
        # 모의 지연시간
        await asyncio.sleep(2)
        
        return filename

# 사용 예제  
async def main():
    # 테스트 데이터
    test_room = {
        "scene": {
            "room": {"width": 4000, "depth": 5000, "height": 2800},
            "objects": [
                {
                    "type": "furniture",
                    "name": "desk",
                    "position": {"center": {"x": 3200, "y": 2500, "z": 0}},
                    "dimensions": {"width": 1200, "depth": 600, "height": 750}
                }
            ]
        }
    }
    
    # 생성기 초기화
    generator = IntegratedImageGenerator()
    
    # Dify 방식으로 생성
    print("=== Dify 최적화 방식 ===")
    dify_result = await generator.generate_image(test_room, use_dify=True)
    print(json.dumps(dify_result, indent=2, ensure_ascii=False))
    
    # 기존 방식으로 생성
    print("\\n=== 기존 방식 ===") 
    traditional_result = await generator.generate_image(test_room, use_dify=False)
    print(json.dumps(traditional_result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. FastAPI 통합 예제

```python
# main.py - FastAPI 메인 애플리케이션
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import asyncio

from integrated_generator import IntegratedImageGenerator
from layout_processor import LayoutProcessor

app = FastAPI(
    title="AI Interior API",
    description="Dify 통합 AI 인테리어 이미지 생성 API",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 인스턴스
generator = IntegratedImageGenerator()
processor = LayoutProcessor()

# 요청/응답 모델
class GenerationRequest(BaseModel):
    scene: Dict[Any, Any]
    use_dify: bool = True
    collect_feedback: bool = True
    user_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    rating: float
    comments: Optional[str] = ""
    image_quality: Optional[int] = None
    layout_accuracy: Optional[int] = None

# 임시 저장소 (프로덕션에서는 Redis나 DB 사용)
generation_store = {}

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "AI Interior API v2.0",
        "status": "running",
        "dify_connected": generator.dify_client.test_connection()
    }

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """이미지 생성 메인 엔드포인트"""
    
    generation_id = str(uuid.uuid4())
    
    try:
        # 이미지 생성
        result = await generator.generate_image(
            request.scene, 
            use_dify=request.use_dify
        )
        
        if result.get("success"):
            # 피드백 수집을 위한 정보 저장
            if request.collect_feedback:
                generation_store[generation_id] = {
                    "room_data": request.scene,
                    "result": result,
                    "user_id": request.user_id,
                    "created_at": datetime.now().isoformat()
                }
            
            response = {
                "success": True,
                "generation_id": generation_id,
                "image_path": result["image_path"],
                "method": result["method"],
                "generation_time": result.get("generation_time", 0),
                "feedback_enabled": request.collect_feedback
            }
            
            # 레이아웃 분석 정보 추가
            features = processor.extract_features(request.scene)
            response["layout_analysis"] = features
            
            return response
        else:
            raise HTTPException(status_code=500, detail="이미지 생성 실패")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@app.post("/feedback/{generation_id}")
async def submit_feedback(generation_id: str, feedback: FeedbackRequest,
                         background_tasks: BackgroundTasks):
    """사용자 피드백 제출"""
    
    # 생성 정보 조회
    generation_info = generation_store.get(generation_id)
    
    if not generation_info:
        raise HTTPException(status_code=404, detail="생성 정보를 찾을 수 없습니다")
    
    try:
        # 피드백 처리를 백그라운드에서 실행
        background_tasks.add_task(
            process_feedback,
            generation_info,
            feedback.dict()
        )
        
        return {
            "success": True,
            "message": "피드백이 접수되었습니다",
            "generation_id": generation_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"피드백 처리 실패: {str(e)}")

@app.get("/analytics")
async def get_analytics():
    """간단한 분석 데이터"""
    
    total_generations = len(generation_store)
    
    methods = {}
    avg_times = {}
    
    for info in generation_store.values():
        method = info["result"].get("method", "unknown")
        gen_time = info["result"].get("generation_time", 0)
        
        if method not in methods:
            methods[method] = 0
            avg_times[method] = []
        
        methods[method] += 1
        avg_times[method].append(gen_time)
    
    # 평균 생성 시간 계산
    for method in avg_times:
        if avg_times[method]:
            avg_times[method] = sum(avg_times[method]) / len(avg_times[method])
        else:
            avg_times[method] = 0
    
    return {
        "total_generations": total_generations,
        "methods_used": methods,
        "average_generation_times": avg_times,
        "dify_status": generator.dify_client.test_connection()
    }

async def process_feedback(generation_info: Dict, feedback_data: Dict):
    """피드백 처리 (백그라운드 작업)"""
    
    try:
        # 높은 평점인 경우 Dify Knowledge Base에 학습
        if feedback_data["rating"] >= 4.0:
            layout_text = processor.convert_to_text(generation_info["room_data"])
            
            # 성공 사례 텍스트 생성
            success_text = f"""
{layout_text}

=== 성공 지표 ===
• 사용자 평점: {feedback_data['rating']}/5.0
• 생성 방식: {generation_info['result']['method']}
• 생성 시간: {generation_info['result'].get('generation_time', 0):.1f}초
• 사용자 코멘트: {feedback_data.get('comments', '없음')}
• 생성 일시: {generation_info['created_at']}
            """
            
            # Dify Knowledge Base에 추가
            success = generator.dify_client.add_document(
                name=f"success_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                text=success_text
            )
            
            if success:
                print(f"✅ 성공 사례 학습 완료 (평점: {feedback_data['rating']})")
            else:
                print(f"❌ 학습 실패")
        
        # 여기에 추가적인 피드백 처리 로직 구현
        # (데이터베이스 저장, 분석 등)
        
    except Exception as e:
        print(f"피드백 처리 중 오류: {e}")

# 개발 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

### 6. 프론트엔드 연동 예제

```javascript
// frontend-integration.js - React/JavaScript 연동 예제
class AIInteriorAPI {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    async generateImage(roomData, useDify = true, userId = null) {
        try {
            const response = await fetch(`${this.baseURL}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    scene: roomData,
                    use_dify: useDify,
                    collect_feedback: true,
                    user_id: userId
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            console.log('✅ 이미지 생성 완료:', result);
            
            return result;
            
        } catch (error) {
            console.error('❌ 이미지 생성 실패:', error);
            throw error;
        }
    }

    async submitFeedback(generationId, rating, comments = '') {
        try {
            const response = await fetch(`${this.baseURL}/feedback/${generationId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    rating: rating,
                    comments: comments
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            console.log('✅ 피드백 제출 완료:', result);
            
            return result;
            
        } catch (error) {
            console.error('❌ 피드백 제출 실패:', error);
            throw error;
        }
    }

    async getAnalytics() {
        try {
            const response = await fetch(`${this.baseURL}/analytics`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
            
        } catch (error) {
            console.error('❌ 분석 데이터 조회 실패:', error);
            throw error;
        }
    }
}

// React 컴포넌트 예제
import React, { useState } from 'react';

const AIInteriorGenerator = () => {
    const [api] = useState(new AIInteriorAPI());
    const [isGenerating, setIsGenerating] = useState(false);
    const [result, setResult] = useState(null);
    const [feedback, setFeedback] = useState({ rating: 5, comments: '' });

    // 테스트용 방 데이터
    const testRoomData = {
        room: { width: 4000, depth: 5000, height: 2800 },
        objects: [
            {
                type: "furniture",
                name: "책상",
                position: { center: { x: 3200, y: 2500, z: 0 } },
                dimensions: { width: 1200, depth: 600, height: 750 }
            },
            {
                type: "furniture", 
                name: "침대",
                position: { center: { x: 1200, y: 4000, z: 0 } },
                dimensions: { width: 1500, depth: 2000, height: 600 }
            }
        ]
    };

    const handleGenerate = async (useDify = true) => {
        setIsGenerating(true);
        
        try {
            const result = await api.generateImage(testRoomData, useDify);
            setResult(result);
        } catch (error) {
            alert('이미지 생성 실패: ' + error.message);
        } finally {
            setIsGenerating(false);
        }
    };

    const handleFeedbackSubmit = async () => {
        if (!result?.generation_id) return;

        try {
            await api.submitFeedback(
                result.generation_id,
                feedback.rating,
                feedback.comments
            );
            alert('피드백이 제출되었습니다!');
        } catch (error) {
            alert('피드백 제출 실패: ' + error.message);
        }
    };

    return (
        <div className="p-6 max-w-4xl mx-auto">
            <h1 className="text-3xl font-bold mb-6">AI 인테리어 생성기</h1>
            
            {/* 생성 버튼들 */}
            <div className="mb-6 space-x-4">
                <button 
                    onClick={() => handleGenerate(true)}
                    disabled={isGenerating}
                    className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                >
                    {isGenerating ? '생성 중...' : 'Dify 최적화 생성'}
                </button>
                
                <button 
                    onClick={() => handleGenerate(false)}
                    disabled={isGenerating}
                    className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50"
                >
                    {isGenerating ? '생성 중...' : '기존 방식 생성'}
                </button>
            </div>

            {/* 결과 표시 */}
            {result && (
                <div className="mb-6 p-4 border rounded-lg">
                    <h3 className="text-xl font-semibold mb-2">생성 결과</h3>
                    <p><strong>방식:</strong> {result.method}</p>
                    <p><strong>생성 시간:</strong> {result.generation_time?.toFixed(1)}초</p>
                    <p><strong>이미지:</strong> {result.image_path}</p>
                    
                    {result.layout_analysis && (
                        <div className="mt-2">
                            <strong>레이아웃 분석:</strong>
                            <ul className="list-disc list-inside ml-4">
                                <li>방 크기: {result.layout_analysis.size_category} ({result.layout_analysis.area_sqm}㎡)</li>
                                <li>가구 개수: {result.layout_analysis.furniture_count}개</li>
                                <li>가구 밀도: {result.layout_analysis.furniture_density}%</li>
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {/* 피드백 폼 */}
            {result?.feedback_enabled && (
                <div className="p-4 border rounded-lg">
                    <h3 className="text-xl font-semibold mb-4">피드백</h3>
                    
                    <div className="mb-4">
                        <label className="block text-sm font-medium mb-2">평점</label>
                        <select 
                            value={feedback.rating}
                            onChange={(e) => setFeedback({...feedback, rating: Number(e.target.value)})}
                            className="w-full p-2 border rounded"
                        >
                            <option value={5}>5점 - 매우 좋음</option>
                            <option value={4}>4점 - 좋음</option>
                            <option value={3}>3점 - 보통</option>
                            <option value={2}>2점 - 나쁨</option>
                            <option value={1}>1점 - 매우 나쁨</option>
                        </select>
                    </div>
                    
                    <div className="mb-4">
                        <label className="block text-sm font-medium mb-2">코멘트</label>
                        <textarea
                            value={feedback.comments}
                            onChange={(e) => setFeedback({...feedback, comments: e.target.value})}
                            placeholder="이미지에 대한 의견을 남겨주세요..."
                            className="w-full p-2 border rounded h-20"
                        />
                    </div>
                    
                    <button
                        onClick={handleFeedbackSubmit}
                        className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                    >
                        피드백 제출
                    </button>
                </div>
            )}
        </div>
    );
};

export default AIInteriorGenerator;
```

### 7. 테스트 스크립트

```python
# test_integration.py - 통합 테스트 스크립트
import asyncio
import json
import time
from integrated_generator import IntegratedImageGenerator

async def run_integration_tests():
    """통합 테스트 실행"""
    
    print("🧪 AI Interior 통합 테스트 시작")
    print("=" * 50)
    
    # 테스트 데이터
    test_rooms = [
        {
            "name": "소형 원룸",
            "data": {
                "scene": {
                    "room": {"width": 3000, "depth": 4000, "height": 2500},
                    "objects": [
                        {
                            "type": "furniture",
                            "name": "책상",
                            "position": {"center": {"x": 2500, "y": 1500, "z": 0}},
                            "dimensions": {"width": 1000, "depth": 500, "height": 750}
                        }
                    ]
                }
            }
        },
        {
            "name": "중형 침실",
            "data": {
                "scene": {
                    "room": {"width": 4000, "depth": 5000, "height": 2800},
                    "objects": [
                        {
                            "type": "furniture",
                            "name": "침대",
                            "position": {"center": {"x": 2000, "y": 4000, "z": 0}},
                            "dimensions": {"width": 1600, "depth": 2000, "height": 600}
                        },
                        {
                            "type": "furniture",
                            "name": "옷장",
                            "position": {"center": {"x": 300, "y": 2500, "z": 0}},
                            "dimensions": {"width": 600, "depth": 1800, "height": 2200}
                        }
                    ]
                }
            }
        }
    ]
    
    try:
        # 생성기 초기화
        generator = IntegratedImageGenerator()
        
        results = []
        
        for test_room in test_rooms:
            print(f"\\n📋 테스트: {test_room['name']}")
            print("-" * 30)
            
            # Dify 방식 테스트
            print("🔵 Dify 최적화 방식...")
            start_time = time.time()
            
            dify_result = await generator.generate_image(
                test_room["data"], 
                use_dify=True
            )
            
            dify_time = time.time() - start_time
            
            # 기존 방식 테스트
            print("🔘 기존 방식...")
            start_time = time.time()
            
            traditional_result = await generator.generate_image(
                test_room["data"],
                use_dify=False
            )
            
            traditional_time = time.time() - start_time
            
            # 결과 저장
            test_result = {
                "room_name": test_room["name"],
                "dify": {
                    "success": dify_result.get("success", False),
                    "time": dify_time,
                    "method": dify_result.get("method", "unknown")
                },
                "traditional": {
                    "success": traditional_result.get("success", False),
                    "time": traditional_time,
                    "method": traditional_result.get("method", "unknown")
                }
            }
            
            results.append(test_result)
            
            # 결과 출력
            print(f"✅ Dify 방식: {dify_time:.2f}초")
            print(f"✅ 기존 방식: {traditional_time:.2f}초")
            print(f"⚡ 속도 차이: {abs(dify_time - traditional_time):.2f}초")
        
        # 전체 결과 요약
        print("\\n" + "=" * 50)
        print("📊 테스트 결과 요약")
        print("=" * 50)
        
        dify_times = [r["dify"]["time"] for r in results if r["dify"]["success"]]
        traditional_times = [r["traditional"]["time"] for r in results if r["traditional"]["success"]]
        
        if dify_times and traditional_times:
            avg_dify = sum(dify_times) / len(dify_times)
            avg_traditional = sum(traditional_times) / len(traditional_times)
            
            print(f"평균 생성 시간:")
            print(f"  • Dify 방식: {avg_dify:.2f}초")
            print(f"  • 기존 방식: {avg_traditional:.2f}초")
            print(f"  • 성능 개선: {((avg_traditional - avg_dify) / avg_traditional * 100):.1f}%")
        
        success_rate = len([r for r in results if r["dify"]["success"] and r["traditional"]["success"]]) / len(results) * 100
        print(f"전체 성공률: {success_rate:.1f}%")
        
        # 상세 결과를 파일로 저장
        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\\n✅ 모든 테스트 완료!")
        print("📄 상세 결과는 test_results.json 파일을 확인하세요.")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        raise e

if __name__ == "__main__":
    asyncio.run(run_integration_tests())
```

이 예제 코드들을 통해 Dify 통합 시스템을 단계적으로 구현하고 테스트할 수 있습니다. 각 파일을 순서대로 구현하면서 점진적으로 기능을 확장해 나가시면 됩니다.