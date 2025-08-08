# 문제 해결 및 최적화 가이드

## 🚨 일반적인 문제들

### 1. Dify API 연결 문제

#### 문제: "Dify API 연결 실패"
```python
# 에러 메시지
ConnectionError: Dify API 연결 실패
# 또는
requests.exceptions.Timeout: Request timed out
```

**해결 방법:**
```python
# 1. API 키 확인
import os
from dotenv import load_dotenv

load_dotenv()

def check_dify_credentials():
    api_key = os.getenv("DIFY_API_KEY")
    app_id = os.getenv("DIFY_APP_ID") 
    dataset_id = os.getenv("DIFY_DATASET_ID")
    
    print(f"API Key: {'✅ 설정됨' if api_key else '❌ 없음'}")
    print(f"App ID: {'✅ 설정됨' if app_id else '❌ 없음'}")
    print(f"Dataset ID: {'✅ 설정됨' if dataset_id else '❌ 없음'}")
    
    if api_key:
        print(f"API Key 앞 10글자: {api_key[:10]}...")
    
    return all([api_key, app_id, dataset_id])

# 2. 네트워크 연결 테스트
import requests

def test_dify_connection():
    try:
        response = requests.get(
            "https://api.dify.ai/v1/apps",
            headers={"Authorization": f"Bearer {os.getenv('DIFY_API_KEY')}"},
            timeout=10
        )
        
        if response.status_code == 401:
            print("❌ API 키가 잘못되었습니다")
            return False
        elif response.status_code == 200:
            print("✅ Dify API 연결 성공")
            return True
        else:
            print(f"⚠️ 예상치 못한 응답: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 연결 시간 초과 - 네트워크 확인 필요")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류 - 인터넷 연결 확인 필요")
        return False

# 사용법
if not check_dify_credentials():
    print("환경변수를 먼저 설정하세요")
else:
    test_dify_connection()
```

#### 문제: "Rate Limit 초과"
```python
# 에러 메시지  
{"error": "Rate limit exceeded", "code": 429}
```

**해결 방법:**
```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1):
    """지수 백오프를 사용한 재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            print(f"Rate limit 도달, {delay:.2f}초 대기 중... (시도 {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                    raise e
            return None
        return wrapper
    return decorator

# 사용 예제
class RateLimitedDifyClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    @retry_with_backoff(max_retries=5, base_delay=2)
    def chat_completion(self, query):
        response = requests.post(
            "https://api.dify.ai/v1/chat-messages",
            headers=self.headers,
            json={"query": query, "response_mode": "blocking", "user": "user"},
            timeout=30
        )
        
        if response.status_code == 429:
            raise Exception("Rate limit exceeded")
        
        return response.json()
```

### 2. 이미지 생성 품질 문제

#### 문제: "생성된 이미지가 부정확함"
**원인:** 프롬프트가 모호하거나 좌표 정보가 부정확

**해결 방법:**
```python
class ImprovedPromptGenerator:
    def __init__(self):
        self.style_templates = {
            "korean_modern": {
                "base": "A hyper-realistic Korean modern interior photograph",
                "lighting": "soft natural lighting from large windows",
                "materials": "light oak flooring, white walls, minimalist furniture",
                "camera": "professional interior photography, wide-angle lens"
            },
            "korean_traditional": {
                "base": "A traditional Korean interior (hanok style)",
                "lighting": "warm ambient lighting",
                "materials": "wooden flooring, paper screens, traditional furniture",
                "camera": "professional architectural photography"
            }
        }
    
    def generate_detailed_prompt(self, room_data, style="korean_modern"):
        """더 상세하고 정확한 프롬프트 생성"""
        
        room = room_data["scene"]["room"]
        furniture_list = [obj for obj in room_data["scene"]["objects"] if obj.get("type") == "furniture"]
        
        template = self.style_templates.get(style, self.style_templates["korean_modern"])
        
        # 기본 스타일 설정
        prompt_parts = [
            template["base"],
            f"Room dimensions: {room['width']/1000:.1f}m × {room['depth']/1000:.1f}m × {room['height']/1000:.1f}m"
        ]
        
        # 정확한 가구 배치 정보
        for furniture in furniture_list:
            pos = furniture["position"]["center"]
            dims = furniture["dimensions"]
            
            # 상대적 위치를 더 정확히 표현
            rel_x = pos["x"] / room["width"]
            rel_y = pos["y"] / room["depth"]
            
            # 벽과의 거리 정보
            distance_info = self._calculate_wall_distances(pos, dims, room)
            
            furniture_desc = f"{furniture['name']}: {distance_info}, size {dims['width']/1000:.1f}m×{dims['depth']/1000:.1f}m"
            prompt_parts.append(furniture_desc)
        
        # 스타일별 추가 정보
        prompt_parts.extend([
            template["lighting"],
            template["materials"], 
            template["camera"],
            "Ultra-realistic rendering, 4K resolution, professional composition",
            "No text, watermarks, or people in the image"
        ])
        
        return " | ".join(prompt_parts)
    
    def _calculate_wall_distances(self, pos, dims, room):
        """벽과의 거리를 자연어로 표현"""
        
        center_x, center_y = pos["x"], pos["y"]
        furniture_width, furniture_depth = dims["width"], dims["depth"]
        
        # 가구 경계 계산
        left_edge = center_x - furniture_width / 2
        right_edge = center_x + furniture_width / 2
        bottom_edge = center_y - furniture_depth / 2
        top_edge = center_y + furniture_depth / 2
        
        # 벽과의 거리
        dist_to_left_wall = left_edge
        dist_to_right_wall = room["width"] - right_edge
        dist_to_bottom_wall = bottom_edge
        dist_to_top_wall = room["depth"] - top_edge
        
        # 가장 가까운 벽 기준으로 설명
        distances = {
            "left": dist_to_left_wall,
            "right": dist_to_right_wall,
            "bottom": dist_to_bottom_wall,
            "top": dist_to_top_wall
        }
        
        closest_wall = min(distances, key=distances.get)
        closest_distance = distances[closest_wall]
        
        if closest_distance < 300:  # 30cm 이내
            return f"against the {closest_wall} wall"
        elif closest_distance < 1000:  # 1m 이내
            return f"near the {closest_wall} wall ({closest_distance/1000:.1f}m away)"
        else:
            # 중앙 영역
            if 0.3 < center_x/room["width"] < 0.7 and 0.3 < center_y/room["depth"] < 0.7:
                return "in the center area"
            else:
                return f"positioned at {center_x/1000:.1f}m from left, {center_y/1000:.1f}m from bottom"
```

#### 문제: "이미지 일관성 부족"
**해결 방법:**
```python
class ConsistencyManager:
    def __init__(self):
        self.seed_cache = {}
        self.style_cache = {}
    
    def get_consistent_seed(self, room_data):
        """레이아웃 기반 일관된 시드 생성"""
        
        # 레이아웃의 핵심 요소만 추출
        essential_elements = {
            "room_size": f"{room_data['scene']['room']['width']}x{room_data['scene']['room']['depth']}",
            "furniture": []
        }
        
        for obj in room_data["scene"]["objects"]:
            if obj.get("type") == "furniture":
                essential_elements["furniture"].append({
                    "name": obj["name"],
                    "x": round(obj["position"]["center"]["x"] / 100) * 100,  # 100mm 단위로 반올림
                    "y": round(obj["position"]["center"]["y"] / 100) * 100
                })
        
        # 정렬하여 일관성 보장
        essential_elements["furniture"].sort(key=lambda x: (x["x"], x["y"]))
        
        # 해시 생성
        import hashlib
        import json
        
        layout_str = json.dumps(essential_elements, sort_keys=True)
        layout_hash = hashlib.md5(layout_str.encode()).hexdigest()
        
        # 시드로 변환 (32bit 정수)
        seed = int(layout_hash[:8], 16) % (2**32)
        
        return seed
    
    def get_style_consistency_prompt(self, base_prompt, previous_results=None):
        """스타일 일관성을 위한 프롬프트 보강"""
        
        consistency_elements = [
            "Maintain consistent lighting temperature (5500K daylight)",
            "Use consistent material textures throughout",
            "Apply uniform color grading and contrast",
            "Ensure consistent perspective and camera height (1.6m)",
            "Maintain consistent shadow density and direction"
        ]
        
        if previous_results:
            # 이전 결과의 스타일 특성을 반영
            style_consistency = "Style reference: maintain the same lighting, materials, and color palette as previous generations"
            consistency_elements.insert(0, style_consistency)
        
        enhanced_prompt = base_prompt + " | " + " | ".join(consistency_elements)
        
        return enhanced_prompt
```

### 3. 성능 최적화 문제

#### 문제: "응답 시간이 너무 오래 걸림"
**해결 방법:**
```python
import asyncio
import aioredis
from functools import lru_cache
import pickle

class PerformanceOptimizer:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis = None
        self.redis_url = redis_url
        self.local_cache = {}
        self.cache_ttl = 3600  # 1시간
    
    async def init_redis(self):
        """Redis 연결 초기화"""
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            await self.redis.ping()
            print("✅ Redis 연결 성공")
        except Exception as e:
            print(f"⚠️ Redis 연결 실패, 로컬 캐시 사용: {e}")
            self.redis = None
    
    async def get_cached_result(self, cache_key):
        """캐시된 결과 조회"""
        
        # Redis 캐시 확인
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return pickle.loads(cached)
            except Exception as e:
                print(f"Redis 캐시 조회 실패: {e}")
        
        # 로컬 캐시 확인
        return self.local_cache.get(cache_key)
    
    async def set_cached_result(self, cache_key, result):
        """결과 캐시 저장"""
        
        # Redis 캐시 저장
        if self.redis:
            try:
                await self.redis.setex(
                    cache_key, 
                    self.cache_ttl, 
                    pickle.dumps(result)
                )
            except Exception as e:
                print(f"Redis 캐시 저장 실패: {e}")
        
        # 로컬 캐시 저장 (크기 제한)
        if len(self.local_cache) > 100:
            # 오래된 항목 제거
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
        
        self.local_cache[cache_key] = result
    
    @lru_cache(maxsize=50)
    def get_layout_fingerprint(self, room_data_str):
        """레이아웃 지문 생성 (캐시 키용)"""
        import hashlib
        return hashlib.md5(room_data_str.encode()).hexdigest()
    
    async def optimized_generation(self, room_data, generator):
        """최적화된 이미지 생성"""
        
        # 캐시 키 생성
        room_data_str = json.dumps(room_data, sort_keys=True)
        cache_key = f"image_gen:{self.get_layout_fingerprint(room_data_str)}"
        
        # 캐시 확인
        cached_result = await self.get_cached_result(cache_key)
        if cached_result:
            print("🚀 캐시된 결과 사용")
            cached_result["from_cache"] = True
            return cached_result
        
        # 새로 생성
        print("🎨 새 이미지 생성 중...")
        result = await generator.generate_image(room_data)
        
        # 성공한 경우만 캐시
        if result.get("success"):
            await self.set_cached_result(cache_key, result)
        
        result["from_cache"] = False
        return result

# 병렬 처리 최적화
class BatchProcessor:
    def __init__(self, max_concurrent=3):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(self, room_data_list, generator):
        """여러 방 레이아웃 병렬 처리"""
        
        async def process_single(room_data):
            async with self.semaphore:
                return await generator.generate_image(room_data)
        
        tasks = [process_single(room_data) for room_data in room_data_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 오류 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "index": i
                })
            else:
                processed_results.append(result)
        
        return processed_results

# 사용 예제
async def optimized_example():
    optimizer = PerformanceOptimizer()
    await optimizer.init_redis()
    
    batch_processor = BatchProcessor(max_concurrent=2)
    
    # 테스트 데이터
    test_rooms = [
        {"scene": {"room": {"width": 4000, "depth": 5000, "height": 2800}, "objects": []}},
        {"scene": {"room": {"width": 3000, "depth": 4000, "height": 2500}, "objects": []}}
    ]
    
    # 병렬 처리
    from integrated_generator import IntegratedImageGenerator
    generator = IntegratedImageGenerator()
    
    results = await batch_processor.process_batch(test_rooms, generator)
    
    for i, result in enumerate(results):
        if result.get("success"):
            print(f"✅ 방 {i+1}: 생성 완료 ({result.get('generation_time', 0):.1f}초)")
        else:
            print(f"❌ 방 {i+1}: 생성 실패 - {result.get('error', 'Unknown error')}")
```

### 4. Knowledge Base 관리 문제

#### 문제: "학습 데이터가 누적되지 않음"
**해결 방법:**
```python
class ImprovedKnowledgeManager:
    def __init__(self, dify_client, dataset_id):
        self.client = dify_client
        self.dataset_id = dataset_id
        self.learning_queue = []
        self.batch_size = 5
    
    async def add_to_learning_queue(self, room_data, rating, comments=""):
        """학습 큐에 추가"""
        
        if rating >= 4.0:  # 좋은 평가만 학습
            learning_item = {
                "room_data": room_data,
                "rating": rating,
                "comments": comments,
                "timestamp": datetime.now().isoformat()
            }
            
            self.learning_queue.append(learning_item)
            
            # 배치 크기에 도달하면 일괄 처리
            if len(self.learning_queue) >= self.batch_size:
                await self.process_learning_batch()
    
    async def process_learning_batch(self):
        """학습 데이터 일괄 처리"""
        
        if not self.learning_queue:
            return
        
        print(f"📚 {len(self.learning_queue)}개 학습 데이터 처리 중...")
        
        success_count = 0
        
        for item in self.learning_queue:
            try:
                success_text = self._create_enriched_success_text(item)
                
                success = self.client.add_document(
                    name=f"success_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{success_count}",
                    text=success_text
                )
                
                if success:
                    success_count += 1
                    
            except Exception as e:
                print(f"학습 데이터 추가 실패: {e}")
        
        print(f"✅ {success_count}/{len(self.learning_queue)}개 학습 완료")
        
        # 큐 초기화
        self.learning_queue = []
    
    def _create_enriched_success_text(self, item):
        """풍부한 학습 데이터 생성"""
        
        from layout_processor import LayoutProcessor
        processor = LayoutProcessor()
        
        layout_text = processor.convert_to_text(item["room_data"])
        features = processor.extract_features(item["room_data"])
        
        # 성공 패턴 분석
        success_patterns = self._analyze_success_patterns(item["room_data"], item["rating"])
        
        enriched_text = f"""
{layout_text}

=== 성공 지표 ===
• 사용자 평점: {item['rating']}/5.0
• 생성 일시: {item['timestamp']}
• 사용자 코멘트: {item['comments'] or '없음'}

=== 레이아웃 특성 ===
• 방 크기 카테고리: {features['size_category']}
• 가구 밀도: {features['furniture_density']}%
• 가구 배치 유형: {', '.join(features['furniture_types'])}

=== 성공 패턴 분석 ===
{success_patterns}

=== 추천 스타일 ===
한국 모던 인테리어, 미니멀 디자인, 자연 채광 활용
        """
        
        return enriched_text.strip()
    
    def _analyze_success_patterns(self, room_data, rating):
        """성공 패턴 분석"""
        
        patterns = []
        
        room = room_data["scene"]["room"]
        furniture_list = [obj for obj in room_data["scene"]["objects"] if obj.get("type") == "furniture"]
        
        # 방 크기별 성공 패턴
        area_sqm = (room["width"] * room["depth"]) / 1000000
        if area_sqm < 15 and len(furniture_list) <= 3:
            patterns.append("소형 공간에서 가구 개수 제한으로 개방감 확보")
        
        # 가구 배치 패턴
        center_furniture = [f for f in furniture_list 
                          if 0.3 < f["position"]["center"]["x"]/room["width"] < 0.7 
                          and 0.3 < f["position"]["center"]["y"]/room["depth"] < 0.7]
        
        if len(center_furniture) == 0:
            patterns.append("벽면 활용으로 중앙 공간 확보")
        
        # 동선 분석
        if self._has_good_traffic_flow(furniture_list, room):
            patterns.append("효율적인 동선 확보")
        
        return " | ".join(patterns) if patterns else "일반적인 배치 패턴"
    
    def _has_good_traffic_flow(self, furniture_list, room):
        """동선 효율성 분석"""
        
        # 간단한 동선 분석 로직
        # 실제로는 더 복잡한 알고리즘 필요
        
        occupied_areas = []
        for furniture in furniture_list:
            pos = furniture["position"]["center"]
            dims = furniture["dimensions"]
            
            area = {
                "x1": pos["x"] - dims["width"]/2,
                "x2": pos["x"] + dims["width"]/2,
                "y1": pos["y"] - dims["depth"]/2,
                "y2": pos["y"] + dims["depth"]/2
            }
            occupied_areas.append(area)
        
        # 주요 동선(문에서 중앙으로) 확인
        # 실제 구현에서는 더 정교한 알고리즘 사용
        return len(occupied_areas) <= 3  # 단순화된 조건
```

### 5. 모니터링 및 디버깅

#### 문제: "시스템 상태 파악이 어려움"
**해결 방법:**
```python
import logging
import sys
from datetime import datetime, timedelta
import json

class SystemMonitor:
    def __init__(self, log_level=logging.INFO):
        self.setup_logging(log_level)
        self.metrics = {
            "requests": 0,
            "successes": 0,
            "errors": 0,
            "dify_calls": 0,
            "cache_hits": 0,
            "avg_response_time": 0
        }
        self.error_log = []
        
    def setup_logging(self, log_level):
        """로깅 설정"""
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ai_interior.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('AIInterior')
    
    def log_request(self, method, room_data, user_id=None):
        """요청 로깅"""
        
        self.metrics["requests"] += 1
        
        self.logger.info(f"새 요청 - 방식: {method}, 사용자: {user_id or 'anonymous'}")
        self.logger.debug(f"방 데이터: {json.dumps(room_data, ensure_ascii=False)}")
    
    def log_success(self, method, generation_time, from_cache=False):
        """성공 로깅"""
        
        self.metrics["successes"] += 1
        
        if from_cache:
            self.metrics["cache_hits"] += 1
        
        # 평균 응답 시간 업데이트
        current_avg = self.metrics["avg_response_time"]
        total_requests = self.metrics["requests"]
        self.metrics["avg_response_time"] = (current_avg * (total_requests - 1) + generation_time) / total_requests
        
        cache_status = "캐시" if from_cache else "새 생성"
        self.logger.info(f"생성 성공 - 방식: {method}, 시간: {generation_time:.2f}초, 소스: {cache_status}")
    
    def log_error(self, method, error, room_data=None):
        """오류 로깅"""
        
        self.metrics["errors"] += 1
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "error": str(error),
            "error_type": type(error).__name__
        }
        
        self.error_log.append(error_info)
        
        # 오류 로그는 최대 100개만 유지
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]
        
        self.logger.error(f"생성 실패 - 방식: {method}, 오류: {error}")
        
        if room_data:
            self.logger.debug(f"실패한 방 데이터: {json.dumps(room_data, ensure_ascii=False)}")
    
    def log_dify_call(self, operation, success=True, response_time=None):
        """Dify API 호출 로깅"""
        
        self.metrics["dify_calls"] += 1
        
        status = "성공" if success else "실패"
        time_info = f", 응답시간: {response_time:.2f}초" if response_time else ""
        
        self.logger.info(f"Dify API 호출 - 작업: {operation}, 상태: {status}{time_info}")
    
    def get_health_status(self):
        """시스템 상태 확인"""
        
        if self.metrics["requests"] == 0:
            return {"status": "대기 중", "details": "요청 없음"}
        
        success_rate = (self.metrics["successes"] / self.metrics["requests"]) * 100
        cache_hit_rate = (self.metrics["cache_hits"] / self.metrics["requests"]) * 100 if self.metrics["requests"] > 0 else 0
        
        # 최근 오류 확인 (최근 1시간)
        recent_errors = [
            err for err in self.error_log 
            if datetime.fromisoformat(err["timestamp"]) > datetime.now() - timedelta(hours=1)
        ]
        
        health_status = {
            "status": "건강함" if success_rate > 80 else "주의 필요" if success_rate > 60 else "문제 있음",
            "metrics": {
                "총 요청": self.metrics["requests"],
                "성공률": f"{success_rate:.1f}%",
                "평균 응답시간": f"{self.metrics['avg_response_time']:.2f}초", 
                "캐시 적중률": f"{cache_hit_rate:.1f}%",
                "Dify API 호출": self.metrics["dify_calls"],
                "최근 1시간 오류": len(recent_errors)
            }
        }
        
        if recent_errors:
            health_status["recent_errors"] = recent_errors[-3:]  # 최근 3개 오류만
        
        return health_status
    
    def generate_report(self):
        """상세 리포트 생성"""
        
        health = self.get_health_status()
        
        # 오류 유형별 집계
        error_types = {}
        for error in self.error_log:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        report = {
            "생성 시간": datetime.now().isoformat(),
            "시스템 상태": health,
            "오류 유형별 통계": error_types,
            "권장사항": self._generate_recommendations(health, error_types)
        }
        
        return report
    
    def _generate_recommendations(self, health, error_types):
        """개선 권장사항 생성"""
        
        recommendations = []
        
        success_rate = float(health["metrics"]["성공률"].replace("%", ""))
        
        if success_rate < 80:
            recommendations.append("성공률이 낮습니다. 오류 로그를 확인하고 Dify API 상태를 점검하세요.")
        
        if "ConnectionError" in error_types:
            recommendations.append("네트워크 연결 오류가 발생하고 있습니다. 인터넷 연결과 방화벽을 확인하세요.")
        
        if "TimeoutError" in error_types:
            recommendations.append("시간 초과 오류가 발생하고 있습니다. 타임아웃 설정을 늘리거나 서버 성능을 확인하세요.")
        
        avg_time = float(health["metrics"]["평균 응답시간"].replace("초", ""))
        if avg_time > 30:
            recommendations.append("응답 시간이 깁니다. 캐싱 시스템 활용을 늘리거나 API 성능을 최적화하세요.")
        
        cache_rate = float(health["metrics"]["캐시 적중률"].replace("%", ""))
        if cache_rate < 20:
            recommendations.append("캐시 적중률이 낮습니다. 캐시 전략을 개선하세요.")
        
        if not recommendations:
            recommendations.append("시스템이 정상적으로 동작하고 있습니다.")
        
        return recommendations

# 사용법
monitor = SystemMonitor()

# 모니터링과 함께 사용하는 래퍼 함수
async def monitored_generation(room_data, generator, method="dify", user_id=None):
    """모니터링을 포함한 이미지 생성"""
    
    monitor.log_request(method, room_data, user_id)
    
    start_time = time.time()
    
    try:
        result = await generator.generate_image(room_data, use_dify=(method=="dify"))
        
        generation_time = time.time() - start_time
        from_cache = result.get("from_cache", False)
        
        monitor.log_success(method, generation_time, from_cache)
        
        return result
        
    except Exception as e:
        monitor.log_error(method, e, room_data)
        raise e

# 주기적 상태 체크
import asyncio

async def periodic_health_check(interval=300):  # 5분마다
    """주기적 상태 체크"""
    
    while True:
        try:
            health = monitor.get_health_status()
            print(f"\\n🏥 시스템 상태: {health['status']}")
            
            if health["status"] != "건강함":
                print("⚠️ 주의가 필요한 상태입니다:")
                for key, value in health["metrics"].items():
                    print(f"  • {key}: {value}")
            
            await asyncio.sleep(interval)
            
        except Exception as e:
            print(f"상태 체크 중 오류: {e}")
            await asyncio.sleep(60)  # 오류 시 1분 후 재시도
```

## 📊 성능 최적화 체크리스트

### ✅ 완료해야 할 최적화 사항

#### 1. API 호출 최적화
- [ ] Dify API 호출 시 재시도 로직 구현
- [ ] Rate limiting 처리
- [ ] 연결 풀링 사용
- [ ] 타임아웃 설정 최적화

#### 2. 캐싱 전략
- [ ] Redis 캐시 구현
- [ ] 로컬 메모리 캐시 구현
- [ ] 캐시 무효화 전략 수립
- [ ] 캐시 히트율 모니터링

#### 3. 데이터 처리 최적화
- [ ] 배치 처리 구현
- [ ] 비동기 처리 적용
- [ ] 데이터 압축 적용
- [ ] 메모리 사용량 최적화

#### 4. 모니터링 및 로깅
- [ ] 상세 로깅 시스템 구축
- [ ] 성능 지표 수집
- [ ] 알림 시스템 구축
- [ ] 대시보드 구현

#### 5. 오류 처리
- [ ] 예외 처리 강화
- [ ] 폴백 메커니즘 구현
- [ ] 오류 복구 로직
- [ ] 사용자 친화적 오류 메시지

---

**💡 문제 해결 팁**: 문제가 발생하면 로그를 먼저 확인하고, 단계별로 격리하여 테스트해보세요. 대부분의 문제는 설정이나 네트워크 연결 문제입니다.