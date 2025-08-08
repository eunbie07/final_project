"""
Dify Integration Usage Example
사용 예제 - AI 인테리어 디자인 Dify 통합
"""

import asyncio
import os
from main import DifyIntegrationSystem


async def example_usage():
    """기본 사용 예제"""
    
    # 환경 변수 설정 (실제 사용 시 .env 파일에 설정)
    os.environ["DIFY_API_KEY"] = "your_dify_api_key_here"
    os.environ["DIFY_APP_ID"] = "your_dify_app_id_here"
    os.environ["DIFY_DATASET_ID"] = "your_dataset_id_here"
    
    try:
        # 시스템 초기화
        system = DifyIntegrationSystem()
        
        # 방 데이터 예제
        room_data = {
            "scene": {
                "room": {
                    "width": 4000,  # 4m
                    "depth": 5000,  # 5m
                    "height": 2800  # 2.8m
                },
                "objects": [
                    {
                        "type": "furniture",
                        "name": "소파",
                        "position": {"center": {"x": 2000, "y": 1500, "z": 0}},
                        "dimensions": {"width": 2000, "depth": 800, "height": 800}
                    },
                    {
                        "type": "furniture",
                        "name": "커피테이블",
                        "position": {"center": {"x": 2000, "y": 2500, "z": 0}},
                        "dimensions": {"width": 1200, "depth": 600, "height": 400}
                    },
                    {
                        "type": "furniture",
                        "name": "TV",
                        "position": {"center": {"x": 2000, "y": 4500, "z": 1200}},
                        "dimensions": {"width": 1400, "depth": 100, "height": 800}
                    }
                ]
            }
        }
        
        print("ROOM: AI 인테리어 이미지 생성 시작...")
        
        # 1. 이미지 생성
        result = await system.generate_room_image(
            room_data=room_data,
            user_id="user123",
            session_id="session456"
        )
        
        if result["success"]:
            print(f"OK: 이미지 생성 성공!")
            print(f"   - 방식: {result['method']}")
            print(f"   - 경로: {result['image_path']}")
            print(f"   - 프롬프트: {result['prompt'][:100]}...")
            
            # 2. 사용자 피드백 시뮬레이션
            print("\nFEEDBACK: 사용자 피드백 수집...")
            feedback = await system.collect_user_feedback(
                room_data=room_data,
                image_path=result["image_path"],
                user_rating=4.8,
                comments="정말 마음에 드는 레이아웃입니다!",
                user_id="user123"
            )
            print(f"   - 학습 여부: {'OK: 학습됨' if feedback['learned'] else 'SKIP: 학습 안됨'}")
            
        else:
            print(f"ERROR: 이미지 생성 실패: {result.get('error')}")
        
        # 3. 시스템 상태 확인
        print("\nSTATS: 시스템 상태:")
        status = system.get_system_status()
        
        print(f"   - Dify 활성화: {status['config']['dify_enabled']}")
        print(f"   - 성공률: {status['performance']['success_rate']}")
        print(f"   - 평균 평점: {status['performance']['average_rating']}")
        print(f"   - 캐시 적중률: {status['cache']['hit_rate']}")
        
        # A/B 테스트 결과
        ab_stats = status['ab_test']
        if ab_stats.get('total_tests', 0) > 0:
            print(f"   - A/B 테스트: {ab_stats['test_ratio']}")
            print(f"   - 권장사항: {ab_stats['recommendation']}")
        
        # 4. 추가 테스트 (다른 사용자)
        print("\nTEST: 다른 사용자로 추가 테스트...")
        
        # 방 크기를 다르게 해서 테스트
        smaller_room = {
            "scene": {
                "room": {"width": 3000, "depth": 3500, "height": 2800},
                "objects": [
                    {
                        "type": "furniture",
                        "name": "침대",
                        "position": {"center": {"x": 1500, "y": 1000, "z": 0}},
                        "dimensions": {"width": 2000, "depth": 1200, "height": 500}
                    }
                ]
            }
        }
        
        result2 = await system.generate_room_image(
            room_data=smaller_room,
            user_id="user789"
        )
        
        if result2["success"]:
            print(f"OK: 두 번째 생성 성공 ({result2['method']})")
            
            # 낮은 평점 피드백
            await system.collect_user_feedback(
                room_data=smaller_room,
                image_path=result2["image_path"],
                user_rating=2.5,
                comments="레이아웃이 마음에 들지 않습니다",
                user_id="user789"
            )
        
        # 5. 최종 통계 및 데이터 내보내기
        print("\n💾 데이터 내보내기...")
        exported_files = system.export_all_data("example_export")
        
        for data_type, filepath in exported_files.items():
            print(f"   - {data_type}: {filepath}")
        
        print("\n🎉 예제 실행 완료!")
        
    except Exception as e:
        print(f"ERROR: 오류 발생: {e}")
        print("   환경 변수 설정을 확인해주세요.")


async def batch_test_example():
    """배치 처리 예제"""
    
    print("BATCH: 배치 처리 테스트...")
    
    system = DifyIntegrationSystem()
    
    # 여러 개의 방 데이터 생성
    room_variations = []
    
    for i in range(5):
        room_data = {
            "scene": {
                "room": {
                    "width": 3000 + i * 500,
                    "depth": 4000 + i * 300, 
                    "height": 2800
                },
                "objects": [
                    {
                        "type": "furniture",
                        "name": f"가구_{i}",
                        "position": {"center": {"x": 1500, "y": 2000, "z": 0}},
                        "dimensions": {"width": 1000, "depth": 600, "height": 750}
                    }
                ]
            }
        }
        room_variations.append(room_data)
    
    # 배치 처리
    results = await system.generator.batch_generate_images(room_variations)
    
    print(f"OK: 배치 처리 완료: {len(results)}개 결과")
    
    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    print(f"   - 성공: {success_count}/{len(results)}")


if __name__ == "__main__":
    print("=== Dify Integration 사용 예제 ===\n")
    
    # 기본 예제 실행
    asyncio.run(example_usage())
    
    print("\n" + "="*50 + "\n")
    
    # 배치 처리 예제 실행
    asyncio.run(batch_test_example())