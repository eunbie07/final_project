import json
import random
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class TestGroup(Enum):
    DIFY = "dify"
    TRADITIONAL = "traditional"


class ABTestManager:
    """A/B 테스트 관리자"""
    
    def __init__(self, dify_ratio: float = 0.7, persistent_assignment: bool = True):
        self.dify_ratio = dify_ratio  # Dify 그룹 비율
        self.persistent_assignment = persistent_assignment
        self.test_results = []
        self.user_assignments = {}  # 사용자별 그룹 고정 할당
        
    def assign_test_group(self, user_id: str = None, session_id: str = None) -> TestGroup:
        """사용자를 테스트 그룹에 배정"""
        
        # 식별자 생성 (user_id 우선, 없으면 session_id 사용)
        identifier = user_id or session_id
        
        if identifier and self.persistent_assignment:
            # 이미 배정된 사용자인지 확인
            if identifier in self.user_assignments:
                return self.user_assignments[identifier]
            
            # 사용자 ID 기반 일관된 그룹 배정
            hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
            is_dify_group = (hash_value % 100) < (self.dify_ratio * 100)
            
            group = TestGroup.DIFY if is_dify_group else TestGroup.TRADITIONAL
            self.user_assignments[identifier] = group
            
            return group
        else:
            # 랜덤 배정
            return TestGroup.DIFY if random.random() < self.dify_ratio else TestGroup.TRADITIONAL
    
    def should_use_dify(self, user_id: str = None, session_id: str = None) -> bool:
        """Dify 사용 여부 결정"""
        group = self.assign_test_group(user_id, session_id)
        return group == TestGroup.DIFY
    
    def record_result(self, user_id: str, session_id: str, group: TestGroup, 
                     rating: float, generation_time: float = None, 
                     room_data: Dict[str, Any] = None, error: str = None):
        """A/B 테스트 결과 기록"""
        
        result = {
            "user_id": user_id,
            "session_id": session_id,
            "group": group.value,
            "rating": rating,
            "generation_time": generation_time,
            "timestamp": datetime.now().isoformat(),
            "room_size": None,
            "furniture_count": None,
            "error": error,
            "success": error is None
        }
        
        # 방 데이터에서 추가 정보 추출
        if room_data:
            room = room_data.get("scene", {}).get("room", {})
            objects = room_data.get("scene", {}).get("objects", [])
            
            result["room_size"] = f"{room.get('width', 0)}x{room.get('depth', 0)}"
            result["furniture_count"] = len([obj for obj in objects if obj.get("type") == "furniture"])
        
        self.test_results.append(result)
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """A/B 테스트 통계 분석"""
        
        if not self.test_results:
            return {
                "error": "테스트 결과가 없음",
                "total_tests": 0
            }
        
        # 그룹별 결과 분리
        dify_results = [r for r in self.test_results if r["group"] == "dify"]
        traditional_results = [r for r in self.test_results if r["group"] == "traditional"]
        
        def calculate_group_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not results:
                return {
                    "count": 0,
                    "success_rate": 0,
                    "avg_rating": 0,
                    "avg_generation_time": 0
                }
            
            successful = [r for r in results if r["success"]]
            ratings = [r["rating"] for r in results if r["rating"] is not None]
            times = [r["generation_time"] for r in results if r["generation_time"] is not None]
            
            return {
                "count": len(results),
                "success_rate": len(successful) / len(results) * 100,
                "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
                "avg_generation_time": sum(times) / len(times) if times else 0,
                "high_rating_count": len([r for r in ratings if r >= 4.0])
            }
        
        dify_stats = calculate_group_stats(dify_results)
        traditional_stats = calculate_group_stats(traditional_results)
        
        # 통계적 유의성 계산 (간단한 버전)
        total_dify = dify_stats["count"]
        total_traditional = traditional_stats["count"]
        
        # 평점 차이 계산
        rating_difference = dify_stats["avg_rating"] - traditional_stats["avg_rating"]
        
        # 승률 계산
        dify_wins = len([r for r in dify_results if r["rating"] >= 4.0])
        traditional_wins = len([r for r in traditional_results if r["rating"] >= 4.0])
        
        return {
            "total_tests": len(self.test_results),
            "test_ratio": f"Dify {total_dify}:{total_traditional} Traditional",
            "dify_group": dify_stats,
            "traditional_group": traditional_stats,
            "comparison": {
                "rating_difference": f"{rating_difference:+.2f}",
                "dify_better": rating_difference > 0,
                "success_rate_difference": f"{dify_stats['success_rate'] - traditional_stats['success_rate']:+.1f}%",
                "high_rating_ratio": {
                    "dify": f"{dify_wins}/{total_dify}" if total_dify > 0 else "0/0",
                    "traditional": f"{traditional_wins}/{total_traditional}" if total_traditional > 0 else "0/0"
                }
            },
            "recommendation": self._get_recommendation(dify_stats, traditional_stats)
        }
    
    def _get_recommendation(self, dify_stats: Dict[str, Any], traditional_stats: Dict[str, Any]) -> str:
        """테스트 결과 기반 추천"""
        
        if dify_stats["count"] < 10 or traditional_stats["count"] < 10:
            return "더 많은 테스트 데이터가 필요합니다 (각 그룹 최소 10개)"
        
        dify_score = (dify_stats["avg_rating"] * 0.6 + 
                     dify_stats["success_rate"] * 0.04 + 
                     (100 - dify_stats["avg_generation_time"]) * 0.01)
        
        traditional_score = (traditional_stats["avg_rating"] * 0.6 + 
                           traditional_stats["success_rate"] * 0.04 + 
                           (100 - traditional_stats["avg_generation_time"]) * 0.01)
        
        if dify_score > traditional_score + 0.3:
            return "Dify 방식을 더 많이 사용하는 것을 권장합니다"
        elif traditional_score > dify_score + 0.3:
            return "기존 방식을 유지하는 것을 권장합니다"
        else:
            return "두 방식이 비슷한 성능을 보입니다. 현재 비율을 유지하세요"
    
    def export_results(self, filepath: str = None) -> str:
        """테스트 결과 내보내기"""
        
        if not filepath:
            filepath = f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "config": {
                "dify_ratio": self.dify_ratio,
                "persistent_assignment": self.persistent_assignment
            },
            "statistics": self.get_test_statistics(),
            "raw_results": self.test_results,
            "user_assignments": {k: v.value for k, v in self.user_assignments.items()},
            "export_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"A/B 테스트 결과 내보내기 완료: {filepath}")
            return filepath
        except Exception as e:
            print(f"내보내기 실패: {e}")
            return None
    
    def load_results(self, filepath: str) -> bool:
        """이전 테스트 결과 불러오기"""
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.test_results.extend(data.get("raw_results", []))
            
            # 사용자 배정 정보 복원
            assignments = data.get("user_assignments", {})
            for user_id, group_str in assignments.items():
                self.user_assignments[user_id] = TestGroup(group_str)
            
            print(f"A/B 테스트 결과 불러오기 완료: {len(data.get('raw_results', []))}개 결과")
            return True
        except Exception as e:
            print(f"불러오기 실패: {e}")
            return False
    
    def clear_results(self):
        """모든 테스트 결과 초기화"""
        self.test_results.clear()
        self.user_assignments.clear()
        print("A/B 테스트 결과가 초기화되었습니다")
    
    def get_user_group(self, user_id: str) -> Optional[TestGroup]:
        """사용자의 현재 테스트 그룹 조회"""
        return self.user_assignments.get(user_id)
    
    def force_assign_group(self, user_id: str, group: TestGroup):
        """사용자를 특정 그룹에 강제 배정"""
        self.user_assignments[user_id] = group
        print(f"사용자 {user_id}를 {group.value} 그룹에 배정했습니다")