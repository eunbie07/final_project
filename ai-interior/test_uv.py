"""
UV 실행 테스트 스크립트
"""

import asyncio
import sys
import traceback


def test_imports():
    """모듈 import 테스트"""
    print("TEST: 모듈 import 테스트...")
    
    try:
        from roombox_integration import DifyRoomImageGenerator
        print("OK: roombox_integration 모듈 로드 성공")
    except Exception as e:
        print(f"ERROR: roombox_integration 모듈 로드 실패: {e}")
        return False
    
    try:
        from realtime_sync import RealtimeCoordinateSync
        print("OK: realtime_sync 모듈 로드 성공")
    except Exception as e:
        print(f"ERROR: realtime_sync 모듈 로드 실패: {e}")
        return False
    
    try:
        from dify_rag import DifyLayoutRAG
        print("OK: dify_rag 모듈 로드 성공")
    except Exception as e:
        print(f"ERROR: dify_rag 모듈 로드 실패: {e}")
        return False
    
    return True


def test_config():
    """설정 테스트"""
    print("\nCONFIG: 설정 테스트...")
    
    try:
        from config import load_config, validate_config, print_config_status
        print("OK: config 모듈 로드 성공")
        
        # Mock 환경 변수 설정
        import os
        os.environ["DIFY_API_KEY"] = "test-api-key-for-uv-test"
        os.environ["DIFY_APP_ID"] = "test-app-id"
        
        config = load_config()
        print("OK: 설정 로드 성공")
        
        # 설정 상태 출력
        print_config_status(config)
        
        return True
        
    except Exception as e:
        print(f"ERROR: 설정 테스트 실패: {e}")
        traceback.print_exc()
        return False


async def test_async_functions():
    """비동기 함수 테스트"""
    print("\n⚡ 비동기 함수 테스트...")
    
    try:
        # 간단한 비동기 테스트
        await asyncio.sleep(0.1)
        print("OK: asyncio 테스트 성공")
        
        # 실제 함수 테스트 (Mock)
        from roombox_integration import DifyRoomImageGenerator
        
        # Mock 데이터로 테스트
        generator = DifyRoomImageGenerator(
            "mock-api-key",
            "mock-app-id", 
            "mock-dataset-id"
        )
        
        print("OK: DifyRoomImageGenerator 초기화 성공")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 비동기 함수 테스트 실패: {e}")
        traceback.print_exc()
        return False


def test_sample_data():
    """샘플 데이터 처리 테스트"""
    print("\nDATA: 샘플 데이터 처리 테스트...")
    
    try:
        from roombox_integration import RoomBoxDataProcessor, DifyLayoutRAG
        
        # Mock Dify RAG
        mock_dify = DifyLayoutRAG("mock-key", "mock-app", "mock-dataset")
        processor = RoomBoxDataProcessor(mock_dify)
        
        # 샘플 데이터
        sample_data = {
            "scene": {
                "room": {"width": 4000, "depth": 5000, "height": 2800},
                "objects": [
                    {
                        "type": "furniture",
                        "name": "test_bed",
                        "position": {"x": 2000, "z": 4500},
                        "dimensions": {"width": 1600, "depth": 2100, "height": 1000}
                    }
                ]
            }
        }
        
        # 데이터 파싱 테스트
        room_layout = processor.parse_roombox_data(sample_data)
        print(f"OK: 데이터 파싱 성공: {room_layout.width_mm}×{room_layout.depth_mm}mm")
        print(f"   가구 개수: {len(room_layout.furniture)}개")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 데이터 처리 테스트 실패: {e}")
        traceback.print_exc()
        return False


async def main():
    """UV 실행 테스트 메인 함수"""
    print("LAUNCH: UV 실행 테스트 시작")
    print(f"   Python 버전: {sys.version}")
    print(f"   실행 경로: {sys.executable}")
    print("=" * 50)
    
    tests = [
        ("모듈 Import", test_imports),
        ("설정 로드", test_config),
        ("비동기 함수", test_async_functions),
        ("샘플 데이터", test_sample_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🧪 {name} 테스트 실행:")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"OK: {name} 테스트 통과")
            else:
                print(f"ERROR: {name} 테스트 실패")
                
        except Exception as e:
            print(f"ERROR: {name} 테스트 오류: {e}")
    
    print("\n" + "=" * 50)
    print(f"STATS: 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! UV 실행 환경이 올바르게 설정되었습니다.")
        return True
    else:
        print("WARNING: 일부 테스트 실패. 의존성이나 설정을 확인해주세요.")
        return False


def cli_main():
    """CLI 진입점"""
    result = asyncio.run(main())
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    cli_main()