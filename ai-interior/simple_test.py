"""
Simple Dify API Test (Windows Compatible)
"""
import asyncio
from main import DifyIntegrationSystem

async def simple_test():
    print("Dify API 실제 연동 테스트")
    print("=" * 40)
    
    try:
        # 시스템 초기화
        system = DifyIntegrationSystem()
        print("시스템 초기화 성공")
        
        # 테스트용 방 데이터
        test_room = {
            "scene": {
                "room": {
                    "width": 4000,
                    "depth": 5000, 
                    "height": 2800
                },
                "objects": [
                    {
                        "type": "furniture",
                        "name": "소파",
                        "position": {"center": {"x": 2000, "y": 1500, "z": 0}},
                        "dimensions": {"width": 2000, "depth": 800, "height": 800}
                    }
                ]
            }
        }
        
        print("방 데이터 준비 완료")
        print(f"방 크기: {test_room['scene']['room']['width']}x{test_room['scene']['room']['depth']}mm")
        
        # 이미지 생성 테스트
        print("이미지 생성 시작...")
        result = await system.generate_room_image(test_room, "test_user")
        
        if result.get("success"):
            print("SUCCESS: 이미지 생성 성공!")
            print(f"방식: {result['method']}")
            print(f"경로: {result.get('image_path', 'N/A')}")
        else:
            print("FAILED: 이미지 생성 실패")
            print(f"에러: {result.get('error', 'Unknown error')}")
        
        # 시스템 상태 확인
        print("\n시스템 상태:")
        status = system.get_system_status()
        print(f"성공률: {status['performance']['success_rate']}")
        print(f"총 요청: {status['performance']['total_requests']}")
        
        return result
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

if __name__ == "__main__":
    result = asyncio.run(simple_test())
    print("\n테스트 완료!")