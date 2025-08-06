"""
일관성 있는 AI 인테리어 이미지 생성 테스트
"""

import asyncio
import json
from roombox_integration import DifyRoomImageGenerator
from config import load_config


async def test_style_consistency():
    """스타일 일관성 테스트"""
    
    print("스타일 일관성 테스트 시작...")
    
    try:
        # Dify 설정 로드
        config = load_config()
        generator = DifyRoomImageGenerator(
            config.api_key,
            config.app_id,
            config.dataset_id
        )
        
        # 테스트용 방 데이터 (RoomBox.jsx 형태)
        test_room_data = {
            "scene": {
                "room": {
                    "width": 4000,  # 4m
                    "depth": 5000,  # 5m
                    "height": 2800  # 2.8m
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
                    },
                    {
                        "type": "furniture",
                        "name": "coffee_table", 
                        "position": {
                            "center": {"x": 2000, "y": 2500, "z": 0}
                        },
                        "dimensions": {
                            "width": 1200,
                            "depth": 600,
                            "height": 400
                        },
                        "rotation_z": 0
                    },
                    {
                        "type": "furniture",
                        "name": "tv_stand",
                        "position": {
                            "center": {"x": 2000, "y": 4500, "z": 0}
                        },
                        "dimensions": {
                            "width": 1400,
                            "depth": 400,
                            "height": 600
                        },
                        "rotation_z": 0
                    }
                ]
            }
        }
        
        # 각 스타일별 테스트
        styles = ["modern", "scandinavian", "industrial"]
        
        for style in styles:
            print(f"\n{style.upper()} 스타일 테스트:")
            print("-" * 50)
            
            result = await generator.generate_consistent_room_image(
                room_data=test_room_data,
                style=style,
                user_id="test_user"
            )
            
            if result["success"]:
                print(f"SUCCESS: {style} 스타일 이미지 생성 성공")
                print(f"   - 이미지: {result['image_path']}")
                print(f"   - 방식: {result['method']}")
                
                # 프롬프트 미리보기 (처음 200자)
                prompt_preview = result['prompt'][:200] + "..." if len(result['prompt']) > 200 else result['prompt']
                print(f"   - 프롬프트 미리보기: {prompt_preview}")
                
                # 테스트용 높은 평점 피드백
                feedback_result = await generator.learn_from_feedback(
                    room_data=test_room_data,
                    image_path=result['image_path'],
                    user_rating=4.8,
                    style=style,
                    comments=f"{style} 스타일이 매우 일관성 있게 잘 나왔습니다!"
                )
                
                if feedback_result["learned"]:
                    print(f"   - 학습: SUCCESS {feedback_result['message']}")
                else:
                    print(f"   - 학습: WARNING {feedback_result['message']}")
                    
            else:
                print(f"FAILED: {style} 스타일 이미지 생성 실패")
                print(f"   - 오류: {result.get('error')}")
        
        print("\n" + "="*60)
        print("스타일 일관성 테스트 완료!")
        print("\n다음 단계:")
        print("   1. API 서버 실행: uv run python api_server.py")
        print("   2. 브라우저에서 테스트: http://localhost:8000/docs")
        print("   3. RoomBox.jsx에 클라이언트 연결")
        
    except Exception as e:
        print(f"테스트 실패: {e}")


if __name__ == "__main__":
    asyncio.run(test_style_consistency())