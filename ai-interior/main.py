"""
Dify Integration Main Module
AI 인테리어 디자인을 위한 Dify RAG 시스템 통합 모듈
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional

from config import load_config, validate_config, print_config_status
from dify_rag import DifyLayoutRAG
from integrated_generator import IntegratedImageGenerator
from performance_monitor import DifyPerformanceMonitor, PerformanceTimer
from cache_manager import DifyCacheManager
from ab_test_manager import ABTestManager, TestGroup


class DifyIntegrationSystem:
    """Dify 통합 시스템 메인 클래스"""
    
    def __init__(self):
        # 설정 로드
        self.config = load_config()
        
        if not validate_config(self.config):
            raise ValueError("설정 유효성 검사 실패")
        
        # 컴포넌트 초기화
        self.generator = IntegratedImageGenerator(
            self.config.api_key,
            self.config.app_id,
            self.config.dataset_id
        )
        
        self.monitor = DifyPerformanceMonitor()
        self.cache = DifyCacheManager(max_size=self.config.cache_size)
        self.ab_test = ABTestManager(dify_ratio=self.config.ab_test_ratio)
        
        print_config_status(self.config)
    
    async def generate_room_image(self, room_data: Dict[str, Any], 
                                user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """방 이미지 생성 (A/B 테스트 포함)"""
        
        # A/B 테스트 그룹 결정
        should_use_dify = self.ab_test.should_use_dify(user_id, session_id)
        group = TestGroup.DIFY if should_use_dify else TestGroup.TRADITIONAL
        
        print(f"사용자 {user_id or session_id}: {group.value} 방식 사용")
        
        # 성능 측정 시작
        method_name = "dify_optimized" if should_use_dify else "traditional"
        
        with PerformanceTimer(self.monitor, method_name) as timer:
            try:
                if should_use_dify and self.config.enable_dify:
                    result = await self.generator.generate_with_dify_optimization(room_data)
                else:
                    result = await self.generator.generate_traditional(room_data)
                
                # A/B 테스트 결과 기록 (일단 기본 성공으로 기록)
                self.ab_test.record_result(
                    user_id=user_id or "anonymous",
                    session_id=session_id or "default",
                    group=group,
                    rating=4.0 if result.get("success") else 1.0,  # 임시 평점
                    room_data=room_data,
                    error=result.get("error")
                )
                
                return result
                
            except Exception as e:
                # 에러 발생 시 A/B 테스트에 기록
                self.ab_test.record_result(
                    user_id=user_id or "anonymous",
                    session_id=session_id or "default",
                    group=group,
                    rating=1.0,
                    room_data=room_data,
                    error=str(e)
                )
                
                return {
                    "success": False,
                    "error": str(e),
                    "method": method_name
                }
    
    async def collect_user_feedback(self, room_data: Dict[str, Any], image_path: str, 
                                  user_rating: float, comments: str = "",
                                  user_id: str = None, session_id: str = None):
        """사용자 피드백 수집 및 학습"""
        
        # 피드백을 통한 학습
        feedback_result = await self.generator.collect_feedback_and_learn(
            room_data, image_path, user_rating, comments
        )
        
        # A/B 테스트에 실제 평점 업데이트
        # (최근 결과 찾아서 업데이트)
        identifier = user_id or session_id or "anonymous"
        recent_results = [r for r in self.ab_test.test_results 
                         if r["user_id"] == identifier or r["session_id"] == identifier]
        
        if recent_results:
            recent_results[-1]["rating"] = user_rating
            recent_results[-1]["comments"] = comments
        
        # 성능 모니터링에 피드백 반영
        self.monitor.track_learning_event(room_data, user_rating, feedback_result["learned"])
        
        return feedback_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 전체 상태 조회"""
        
        return {
            "config": {
                "dify_enabled": self.config.enable_dify,
                "success_threshold": self.config.success_threshold,
                "ab_test_ratio": self.config.ab_test_ratio
            },
            "performance": self.monitor.get_performance_report(),
            "cache": self.cache.get_stats(),
            "ab_test": self.ab_test.get_test_statistics(),
            "timestamp": datetime.now().isoformat()
        }
    
    def export_all_data(self, base_filename: str = None) -> Dict[str, str]:
        """모든 데이터 내보내기"""
        
        if not base_filename:
            base_filename = f"dify_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        exported_files = {}
        
        # 성능 메트릭스 내보내기
        perf_file = self.monitor.export_metrics(f"{base_filename}_performance.json")
        if perf_file:
            exported_files["performance"] = perf_file
        
        # A/B 테스트 결과 내보내기
        ab_file = self.ab_test.export_results(f"{base_filename}_ab_test.json")
        if ab_file:
            exported_files["ab_test"] = ab_file
        
        # 시스템 상태 내보내기
        status_file = f"{base_filename}_status.json"
        try:
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(self.get_system_status(), f, indent=2, ensure_ascii=False)
            exported_files["status"] = status_file
        except Exception as e:
            print(f"상태 내보내기 실패: {e}")
        
        return exported_files


async def main():
    """메인 실행 함수 - 테스트용"""
    
    # 시스템 초기화
    system = DifyIntegrationSystem()
    
    # 테스트 방 데이터
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
                    "name": "책상",
                    "position": {"center": {"x": 3250, "y": 3250, "z": 0}},
                    "dimensions": {"width": 1200, "depth": 600, "height": 750}
                },
                {
                    "type": "furniture", 
                    "name": "의자",
                    "position": {"center": {"x": 3250, "y": 2500, "z": 0}},
                    "dimensions": {"width": 600, "depth": 600, "height": 1000}
                }
            ]
        }
    }
    
    print("=== Dify 통합 시스템 테스트 ===")
    
    # 이미지 생성 테스트
    result = await system.generate_room_image(
        test_room_data, 
        user_id="test_user_1",
        session_id="test_session_1"
    )
    
    print(f"생성 결과: {result}")
    
    # 피드백 시뮬레이션
    if result.get("success"):
        await system.collect_user_feedback(
            test_room_data,
            result.get("image_path", ""),
            user_rating=4.5,
            comments="매우 만족스럽습니다",
            user_id="test_user_1"
        )
    
    # 시스템 상태 출력
    status = system.get_system_status()
    print("\n=== 시스템 상태 ===")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    
    # 데이터 내보내기
    exported = system.export_all_data("test_export")
    print(f"\n내보낸 파일들: {exported}")


if __name__ == "__main__":
    asyncio.run(main())