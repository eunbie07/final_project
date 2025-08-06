import json
import hashlib
import time
from functools import lru_cache
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class DifyCacheManager:
    """Dify 요청 결과 캐싱 관리자"""
    
    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get_layout_hash(self, room_data: Dict[str, Any]) -> str:
        """레이아웃 해시 생성"""
        # 레이아웃의 핵심 요소만 해싱에 사용
        hash_data = {
            "room": room_data["scene"]["room"],
            "objects": sorted(room_data["scene"]["objects"], key=lambda x: x.get("name", ""))
        }
        layout_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(layout_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """캐시 만료 확인"""
        return time.time() - timestamp > self.ttl_seconds
    
    def _evict_lru(self):
        """LRU 정책으로 캐시 항목 제거"""
        if len(self.cache) >= self.max_size:
            # 가장 오래된 접근 시간의 항목 제거
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
    
    def get_similar_layouts(self, room_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """캐시된 유사 레이아웃 검색 결과 조회"""
        layout_hash = self.get_layout_hash(room_data)
        cache_key = f"similar_{layout_hash}"
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            
            if not self._is_expired(timestamp):
                self.access_times[cache_key] = time.time()
                self.hit_count += 1
                return cached_data
            else:
                # 만료된 캐시 제거
                del self.cache[cache_key]
                del self.access_times[cache_key]
        
        self.miss_count += 1
        return None
    
    def set_similar_layouts(self, room_data: Dict[str, Any], result: Dict[str, Any]):
        """유사 레이아웃 검색 결과 캐시"""
        layout_hash = self.get_layout_hash(room_data)
        cache_key = f"similar_{layout_hash}"
        
        self._evict_lru()
        
        self.cache[cache_key] = (result, time.time())
        self.access_times[cache_key] = time.time()
    
    def get_optimized_prompt(self, room_data: Dict[str, Any]) -> Optional[str]:
        """캐시된 최적화 프롬프트 조회"""
        layout_hash = self.get_layout_hash(room_data)
        cache_key = f"prompt_{layout_hash}"
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            
            if not self._is_expired(timestamp):
                self.access_times[cache_key] = time.time()
                self.hit_count += 1
                return cached_data
            else:
                del self.cache[cache_key]
                del self.access_times[cache_key]
        
        self.miss_count += 1
        return None
    
    def set_optimized_prompt(self, room_data: Dict[str, Any], prompt: str):
        """최적화된 프롬프트 캐시"""
        layout_hash = self.get_layout_hash(room_data)
        cache_key = f"prompt_{layout_hash}"
        
        self._evict_lru()
        
        self.cache[cache_key] = (prompt, time.time())
        self.access_times[cache_key] = time.time()
    
    def clear_expired(self):
        """만료된 캐시 항목 제거"""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, timestamp) in self.cache.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl_hours": self.ttl_seconds / 3600
        }
    
    def clear_all(self):
        """모든 캐시 제거"""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0


class BatchProcessor:
    """배치 처리 최적화 유틸리티"""
    
    def __init__(self, batch_size: int = 5, delay_seconds: float = 0.1):
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds
        self.pending_requests = []
    
    async def add_request(self, room_data: Dict[str, Any], callback):
        """배치 처리 요청 추가"""
        self.pending_requests.append((room_data, callback))
        
        if len(self.pending_requests) >= self.batch_size:
            await self.process_batch()
    
    async def process_batch(self):
        """배치 처리 실행"""
        if not self.pending_requests:
            return
        
        print(f"배치 처리 시작: {len(self.pending_requests)}개 요청")
        
        # 요청들을 동시에 처리
        import asyncio
        tasks = []
        
        for room_data, callback in self.pending_requests:
            task = asyncio.create_task(callback(room_data))
            tasks.append(task)
        
        # 모든 작업 완료 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 처리
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"배치 작업 {i} 실패: {result}")
        
        self.pending_requests.clear()
        
        # 요청 간 지연
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
    
    async def flush(self):
        """대기 중인 모든 요청 처리"""
        if self.pending_requests:
            await self.process_batch()


class RateLimiter:
    """API 요청 속도 제한"""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def can_make_request(self) -> bool:
        """요청 가능 여부 확인"""
        current_time = time.time()
        
        # 시간 윈도우를 벗어난 요청 기록 제거
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < self.time_window]
        
        return len(self.requests) < self.max_requests
    
    def record_request(self):
        """요청 기록"""
        self.requests.append(time.time())
    
    def get_wait_time(self) -> float:
        """다음 요청까지 대기 시간 (초)"""
        if len(self.requests) < self.max_requests:
            return 0
        
        # 가장 오래된 요청이 시간 윈도우를 벗어날 때까지의 시간
        oldest_request = min(self.requests)
        wait_time = self.time_window - (time.time() - oldest_request)
        
        return max(0, wait_time)
    
    async def wait_if_needed(self):
        """필요시 대기"""
        if not self.can_make_request():
            wait_time = self.get_wait_time()
            if wait_time > 0:
                print(f"Rate limit: {wait_time:.1f}초 대기")
                import asyncio
                await asyncio.sleep(wait_time)