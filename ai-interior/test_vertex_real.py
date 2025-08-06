"""
실제 Vertex AI 이미지 생성 테스트 (이모지 없음)
"""

import asyncio
from vertex_ai_generator import VertexAIImageGenerator


async def test_vertex_ai():
    """Vertex AI 연결 테스트"""
    
    print("Vertex AI 이미지 생성 테스트 시작...")
    
    try:
        # Vertex AI 생성기 초기화
        generator = VertexAIImageGenerator()
        
        # 테스트 프롬프트
        test_prompt = """
        Modern Korean interior room design:
        - Room dimensions: 4.0m × 5.0m × 2.8m
        - Sofa positioned in the center-left area of the room
        - Coffee table in the middle area of the room  
        - TV stand against the back wall
        - Clean white walls with light wood flooring
        - Bright natural lighting from large windows
        - Professional interior photography, 8K quality, realistic lighting
        """
        
        print("Modern 스타일 이미지 생성 중...")
        result = await generator.generate_image(test_prompt, "modern")
        
        if result["success"]:
            print(f"SUCCESS: {result['image_path']}")
            print(f"Service: {result['service']}")
            print(f"Generation time: {result['generation_time']}")
        else:
            print(f"FAILED: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        return {"success": False, "error": str(e)}


async def test_full_integration():
    """전체 Dify + Vertex AI 통합 테스트"""
    
    print("\n" + "="*50)
    print("전체 통합 테스트 시작...")
    
    try:
        from roombox_integration import DifyRoomImageGenerator
        from config import load_config
        
        # 설정 로드
        config = load_config()
        
        # 통합 생성기 초기화
        integrated_generator = DifyRoomImageGenerator(
            config.api_key,
            config.app_id,
            config.dataset_id
        )
        
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
                        "name": "sofa",
                        "position": {
                            "center": {"x": 2000, "y": 1500, "z": 0}
                        },
                        "dimensions": {
                            "width": 2000,
                            "depth": 800,
                            "height": 800
                        },
                        "rotation_z": 0
                    }
                ]
            }
        }
        
        print("Dify + Vertex AI 이미지 생성 중...")
        result = await integrated_generator.generate_consistent_room_image(
            room_data=test_room_data,
            style="modern",
            user_id="vertex_test_user"
        )
        
        if result["success"]:
            print(f"SUCCESS: {result['image_path']}")
            print(f"Method: {result['method']}")
            print(f"Style: {result['style']}")
        else:
            print(f"FAILED: {result.get('error')}")
        
        return result
        
    except Exception as e:
        print(f"통합 테스트 실패: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("=== Vertex AI 연결 테스트 ===")
    
    # 1. Vertex AI 단독 테스트
    vertex_result = asyncio.run(test_vertex_ai())
    
    # 2. 전체 통합 테스트 (Vertex AI 성공시에만)
    if vertex_result.get("success"):
        print("\nVertex AI 성공! 전체 통합 테스트 진행...")
        integration_result = asyncio.run(test_full_integration())
        
        if integration_result.get("success"):
            print("\n" + "="*50)
            print("완벽! 모든 테스트 성공!")
            print("이제 실제 Google Vertex AI로 일관성 있는 인테리어 이미지를 생성할 수 있습니다!")
        else:
            print("\nVertex AI는 작동하지만 통합에서 문제 발생")
    else:
        print("\nVertex AI 연결 실패 - 설정을 확인해주세요")
        print("Google Cloud 인증 및 프로젝트 설정이 필요합니다")