# Dify 통합 상세 가이드

## 🎯 Dify 선택 이유

### 비교 분석

| 기능 | 자체 구현 | **Dify 활용** | 차이점 |
|------|-----------|---------------|--------|
| **개발 시간** | 2-4주 | **1주** | 75% 단축 |
| **임베딩 모델** | 직접 선택/구축 | **다양한 모델 지원** | 전문성 ↑ |
| **벡터 DB** | ChromaDB 설정 | **내장 (Qdrant/Pinecone)** | 관리 부담 ↓ |
| **RAG 시스템** | 직접 구현 | **즉시 사용 가능** | 안정성 ↑ |
| **확장성** | 수동 스케일링 | **Auto-scaling** | 운영 편의성 ↑ |
| **학습 시스템** | 별도 구축 | **Knowledge Base 내장** | 통합성 ↑ |

## 🛠️ Dify 설정 가이드

### 1. 계정 생성 및 프로젝트 설정

1. **Dify 가입**: [https://cloud.dify.ai/apps](https://cloud.dify.ai/apps)
2. **새 앱 생성**: "Room Layout RAG" 앱 생성
3. **앱 타입 선택**: "Chat App" 또는 "Agent" 선택

### 2. Knowledge Base 구성

#### 2.1 Knowledge Base 생성
```bash
# Dify UI에서 진행
1. Knowledge → Create Knowledge
2. Name: "Room Layout Database"  
3. Description: "Korean interior room layouts with coordinates"
4. Indexing: "High Quality" 선택
5. Embedding Model: "text-embedding-ada-002" 권장
```

#### 2.2 데이터 구조 설계
```json
{
  "layout_id": "layout_20250101_001",
  "room_specs": {
    "width": 4000,
    "depth": 5000, 
    "height": 2800,
    "area_sqm": 20.0
  },
  "furniture_list": [
    {
      "name": "책상",
      "position": {"x": 3250, "y": 3250, "z": 0},
      "dimensions": {"width": 1200, "depth": 600, "height": 750},
      "relative_position": "75% from left, 65% from bottom"
    }
  ],
  "success_metrics": {
    "user_rating": 4.5,
    "generation_attempts": 1,
    "style_consistency": "high"
  }
}
```

### 3. 임베딩 전략

#### 3.1 공간 임베딩 생성
```python
def create_spatial_embedding(room_data):
    """방 레이아웃을 구조화된 텍스트로 변환"""
    room = room_data["scene"]["room"]
    objects = room_data["scene"]["objects"]
    
    # 구조화된 설명 생성
    description = f"""
    Room Layout Analysis:
    - Dimensions: {room['width']}×{room['depth']}×{room['height']}mm
    - Total Area: {(room['width'] * room['depth']) / 1000000:.2f}㎡
    - Room Ratio: {room['width']/room['depth']:.2f} (width/depth)
    
    Furniture Configuration:
    """
    
    for obj in objects:
        if obj["type"] == "furniture":
            pos = obj["position"]["center"]
            dims = obj["dimensions"]
            
            # 상대적 위치 계산
            rel_x = (pos["x"] / room["width"]) * 100
            rel_y = (pos["y"] / room["depth"]) * 100
            
            description += f"""
    - {obj['name']}:
      * Absolute Position: ({pos['x']}, {pos['y']})mm
      * Relative Position: {rel_x:.1f}% from left, {rel_y:.1f}% from bottom
      * Size: {dims['width']}×{dims['depth']}mm
      * Area Ratio: {(dims['width']*dims['depth'])/(room['width']*room['depth'])*100:.1f}%
      * Wall Distance: Left={pos['x']}mm, Right={room['width']-pos['x']}mm
            """
    
    return description.strip()
```

## 🔗 API 통합 방법

### 1. Dify RAG 클래스 구현

```python
import requests
import json
from datetime import datetime

class DifyLayoutRAG:
    def __init__(self, api_key, app_id):
        self.api_key = api_key
        self.app_id = app_id
        self.base_url = "https://api.dify.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def add_successful_layout(self, room_data, user_rating, image_path=None):
        """성공적인 레이아웃을 Knowledge Base에 추가"""
        
        if user_rating < 4.0:  # 좋은 결과만 학습
            return False
            
        # 구조화된 텍스트 생성
        layout_text = self.create_spatial_embedding(room_data)
        layout_text += f"""
        
        Success Metrics:
        - User Rating: {user_rating}/5.0
        - Generated At: {datetime.now().isoformat()}
        - Style: Korean Modern Interior
        """
        
        # Dify Knowledge Base에 추가
        response = requests.post(
            f"{self.base_url}/datasets/{self.dataset_id}/documents",
            headers=self.headers,
            json={
                "name": f"layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "text": layout_text,
                "indexing_technique": "high_quality"
            }
        )
        
        return response.status_code == 200
    
    def find_similar_layouts(self, room_data, min_rating=4.0):
        """유사한 성공 레이아웃 검색"""
        
        # 현재 레이아웃을 쿼리로 변환
        current_layout = self.create_spatial_embedding(room_data)
        
        query = f"""
        Find similar room layouts to:
        {current_layout}
        
        Requirements:
        - Similar room size (±20%)
        - Similar furniture types and arrangement
        - High user rating (≥{min_rating})
        - Korean interior style
        """
        
        response = requests.post(
            f"{self.base_url}/chat-messages",
            headers=self.headers,
            json={
                "inputs": {},
                "query": query,
                "response_mode": "blocking",
                "user": "layout_analyzer"
            }
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    
    def generate_optimized_prompt(self, room_data):
        """RAG 결과 기반 최적화된 프롬프트 생성"""
        
        similar_layouts = self.find_similar_layouts(room_data)
        current_layout = self.create_spatial_embedding(room_data)
        
        optimization_query = f"""
        Based on similar successful room layouts, generate an optimized AI image generation prompt for:
        
        Current Room Layout:
        {current_layout}
        
        Similar Successful Cases:
        {similar_layouts.get('answer', 'No similar layouts found') if similar_layouts else 'No data'}
        
        Requirements:
        1. Precise furniture positioning using exact coordinates
        2. Consistent Korean modern interior style
        3. Realistic proportions and lighting
        4. Professional interior photography quality
        5. Specific camera angle and composition
        
        Generate a detailed, structured prompt that ensures accurate spatial relationships.
        """
        
        response = requests.post(
            f"{self.base_url}/chat-messages",
            headers=self.headers,
            json={
                "inputs": {},
                "query": optimization_query,
                "response_mode": "blocking",
                "user": "prompt_optimizer"
            }
        )
        
        if response.status_code == 200:
            return response.json().get('answer', '')
        return None
```

### 2. 기존 시스템과 통합

```python
# minah/backend/python/dify_integration/integrated_generator.py
from dify_rag import DifyLayoutRAG
from ..fastapi.main import generate_image_with_imagen

class IntegratedImageGenerator:
    def __init__(self, dify_api_key, dify_app_id):
        self.dify_rag = DifyLayoutRAG(dify_api_key, dify_app_id)
        
    async def generate_with_dify_optimization(self, room_data):
        """Dify 최적화를 활용한 이미지 생성"""
        
        try:
            # 1. Dify RAG로 최적화된 프롬프트 생성
            optimized_prompt = self.dify_rag.generate_optimized_prompt(room_data)
            
            if not optimized_prompt:
                # Dify 실패 시 기존 방식으로 폴백
                return await self.generate_traditional(room_data)
            
            # 2. 레이아웃 해시로 일관된 시드 생성
            layout_hash = hash(json.dumps(room_data, sort_keys=True))
            seed = layout_hash % (2**32)
            
            # 3. 시드 고정으로 일관된 이미지 생성
            enhanced_prompt = f"{optimized_prompt} [SEED:{seed}]"
            
            # 4. Vertex AI/OpenAI로 실제 이미지 생성
            image_path = await generate_image_with_imagen(
                enhanced_prompt,
                f"dify_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            return {
                "success": True,
                "image_path": image_path,
                "prompt": enhanced_prompt,
                "seed": seed,
                "method": "dify_optimized"
            }
            
        except Exception as e:
            print(f"Dify 최적화 실패: {e}")
            return await self.generate_traditional(room_data)
    
    async def collect_feedback_and_learn(self, room_data, image_path, user_rating, comments=""):
        """사용자 피드백 수집 및 학습"""
        
        if user_rating >= 4.0:
            success = self.dify_rag.add_successful_layout(
                room_data, 
                user_rating, 
                image_path
            )
            
            if success:
                print(f"✅ 성공 사례 학습 완료 (평점: {user_rating})")
            else:
                print(f"❌ 학습 실패")
                
        return {"learned": user_rating >= 4.0}
```

## 📊 성능 모니터링

### 1. 핵심 지표 추적

```python
class DifyPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "dify_requests": 0,
            "dify_successes": 0, 
            "prompt_optimizations": 0,
            "learning_events": 0,
            "average_rating": 0.0
        }
    
    def track_generation(self, method, success, rating=None):
        """이미지 생성 성능 추적"""
        self.metrics["dify_requests"] += 1
        
        if success:
            self.metrics["dify_successes"] += 1
            
        if rating:
            # 이동 평균으로 평점 업데이트
            current_avg = self.metrics["average_rating"]
            new_avg = (current_avg * 0.9) + (rating * 0.1)
            self.metrics["average_rating"] = new_avg
    
    def get_performance_report(self):
        """성능 리포트 생성"""
        success_rate = (self.metrics["dify_successes"] / max(1, self.metrics["dify_requests"])) * 100
        
        return {
            "success_rate": f"{success_rate:.1f}%",
            "total_requests": self.metrics["dify_requests"],
            "average_rating": f"{self.metrics['average_rating']:.2f}/5.0",
            "learning_events": self.metrics["learning_events"]
        }
```

### 2. A/B 테스트 설정

```python
import random

class ABTestManager:
    def __init__(self, dify_ratio=0.7):
        self.dify_ratio = dify_ratio  # 70%는 Dify 사용
        
    def should_use_dify(self, user_id=None):
        """사용자별 A/B 테스트 그룹 결정"""
        if user_id:
            # 사용자 ID 기반 일관된 그룹 배정
            return hash(user_id) % 100 < (self.dify_ratio * 100)
        else:
            # 랜덤 배정
            return random.random() < self.dify_ratio
    
    def track_result(self, user_id, method, rating):
        """A/B 테스트 결과 추적"""
        # 결과를 데이터베이스나 파일에 저장
        result = {
            "user_id": user_id,
            "method": method,  # "dify" or "traditional"
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }
        
        # 결과 저장 로직
        self.save_ab_result(result)
```

## 🔧 환경 설정

### 1. 환경 변수 설정

```bash
# .env 파일
DIFY_API_KEY=your_dify_api_key_here
DIFY_APP_ID=your_dify_app_id_here
DIFY_DATASET_ID=your_knowledge_base_id_here
ENABLE_DIFY=true
DIFY_SUCCESS_THRESHOLD=4.0
AB_TEST_DIFY_RATIO=0.7
```

### 2. 의존성 설치

```bash
# requirements.txt에 추가
requests>=2.31.0
python-dotenv>=1.0.0
```

## ⚡ 최적화 팁

### 1. 캐싱 전략
```python
from functools import lru_cache
import hashlib

class DifyCacheManager:
    @lru_cache(maxsize=100)
    def cached_similarity_search(self, layout_hash):
        """유사 레이아웃 검색 결과 캐싱"""
        return self.dify_rag.find_similar_layouts(layout_hash)
    
    def get_layout_hash(self, room_data):
        """레이아웃 해시 생성"""
        layout_str = json.dumps(room_data, sort_keys=True)
        return hashlib.md5(layout_str.encode()).hexdigest()
```

### 2. 배치 처리
```python
async def batch_generate_images(self, room_data_list):
    """여러 레이아웃 동시 처리"""
    tasks = []
    for room_data in room_data_list:
        task = self.generate_with_dify_optimization(room_data)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

## 🎯 다음 단계

1. **Dify 계정 생성** 및 첫 번째 Knowledge Base 구축
2. **기본 API 연동** 테스트
3. **성공 사례 데이터** 수집 시작
4. **A/B 테스트** 통해 성능 비교
5. **점진적 확장** - 더 많은 스타일과 기능 추가

---

**💡 핵심 포인트**: Dify의 RAG 시스템을 활용하면 성공적인 레이아웃 사례들을 자동으로 학습하여, 점진적으로 이미지 생성 품질을 향상시킬 수 있습니다.