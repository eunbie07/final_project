import json
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict


class DifyPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "dify_requests": 0,
            "dify_successes": 0, 
            "prompt_optimizations": 0,
            "learning_events": 0,
            "average_rating": 0.0,
            "total_ratings": 0,
            "generation_times": [],
            "error_count": 0
        }
        self.session_start = datetime.now()
        self.detailed_logs = []
    
    def track_generation(self, method: str, success: bool, rating: float = None, 
                        generation_time: float = None, error_msg: str = None):
        """이미지 생성 성능 추적"""
        self.metrics["dify_requests"] += 1
        
        if success:
            self.metrics["dify_successes"] += 1
        else:
            self.metrics["error_count"] += 1
            
        if rating:
            # 이동 평균으로 평점 업데이트
            total_ratings = self.metrics["total_ratings"]
            current_avg = self.metrics["average_rating"]
            
            new_avg = (current_avg * total_ratings + rating) / (total_ratings + 1)
            self.metrics["average_rating"] = new_avg
            self.metrics["total_ratings"] += 1
        
        if generation_time:
            self.metrics["generation_times"].append(generation_time)
        
        # 상세 로그 기록
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "success": success,
            "rating": rating,
            "generation_time": generation_time,
            "error": error_msg
        }
        self.detailed_logs.append(log_entry)
    
    def track_learning_event(self, room_data: Dict[str, Any], rating: float, learned: bool):
        """학습 이벤트 추적"""
        if learned:
            self.metrics["learning_events"] += 1
        
        # 학습 이벤트 로그
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "learning",
            "rating": rating,
            "learned": learned,
            "room_size": f"{room_data['scene']['room']['width']}x{room_data['scene']['room']['depth']}"
        }
        self.detailed_logs.append(log_entry)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        total_requests = max(1, self.metrics["dify_requests"])
        success_rate = (self.metrics["dify_successes"] / total_requests) * 100
        
        # 생성 시간 통계
        generation_times = self.metrics["generation_times"]
        avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
        
        # 세션 지속 시간
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            "success_rate": f"{success_rate:.1f}%",
            "total_requests": self.metrics["dify_requests"],
            "successful_requests": self.metrics["dify_successes"],
            "error_count": self.metrics["error_count"],
            "average_rating": f"{self.metrics['average_rating']:.2f}/5.0",
            "total_ratings": self.metrics["total_ratings"],
            "learning_events": self.metrics["learning_events"],
            "average_generation_time": f"{avg_generation_time:.2f}s",
            "session_duration": f"{session_duration:.1f}s",
            "requests_per_minute": f"{(self.metrics['dify_requests'] / (session_duration / 60)):.1f}" if session_duration > 0 else "0"
        }
    
    def get_detailed_analytics(self) -> Dict[str, Any]:
        """상세 분석 데이터"""
        # 시간대별 성능 분석
        hourly_stats = defaultdict(lambda: {"requests": 0, "successes": 0, "ratings": []})
        
        for log in self.detailed_logs:
            if log.get("method"):
                hour = datetime.fromisoformat(log["timestamp"]).hour
                hourly_stats[hour]["requests"] += 1
                if log["success"]:
                    hourly_stats[hour]["successes"] += 1
                if log.get("rating"):
                    hourly_stats[hour]["ratings"].append(log["rating"])
        
        # 에러 분석
        error_analysis = defaultdict(int)
        for log in self.detailed_logs:
            if log.get("error"):
                error_analysis[log["error"]] += 1
        
        return {
            "hourly_performance": dict(hourly_stats),
            "error_analysis": dict(error_analysis),
            "recent_logs": self.detailed_logs[-10:],  # 최근 10개 로그
            "total_logs": len(self.detailed_logs)
        }
    
    def export_metrics(self, filepath: str = None):
        """메트릭스 내보내기"""
        if not filepath:
            filepath = f"dify_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "metrics": self.metrics,
            "performance_report": self.get_performance_report(),
            "detailed_analytics": self.get_detailed_analytics(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"메트릭스 내보내기 완료: {filepath}")
            return filepath
        except Exception as e:
            print(f"메트릭스 내보내기 실패: {e}")
            return None
    
    def reset_metrics(self):
        """메트릭스 초기화"""
        self.__init__()
        print("메트릭스가 초기화되었습니다.")


class PerformanceTimer:
    """성능 측정을 위한 컨텍스트 매니저"""
    
    def __init__(self, monitor: DifyPerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        
        success = exc_type is None
        error_msg = str(exc_val) if exc_val else None
        
        self.monitor.track_generation(
            method=self.operation_name,
            success=success,
            generation_time=duration,
            error_msg=error_msg
        )