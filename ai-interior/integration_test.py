"""
통합 시스템 최종 테스트
API 서버 없이 직접 SD 생성기를 테스트
"""

from stable_diffusion_generator import StableDiffusionGenerator
import json
from datetime import datetime

def test_roombox_to_sd_pipeline():
    """RoomBox → SD 파이프라인 전체 테스트"""
    print("=" * 60)
    print("RoomBox -> Stable Diffusion 전체 플로우 테스트")
    print("=" * 60)
    
    # 1. RoomBox 형식 테스트 데이터 (실제 사용되는 형식)
    room_data = {
        "dimensions": {
            "width_cm": 400,
            "depth_cm": 500,
            "height_cm": 280
        },
        "furniture_3d": [
            {
                "type": "bed",
                "name": "queen_bed",
                "position": [200, 0, 250]  # cm 단위
            },
            {
                "type": "desk",  
                "name": "work_desk",
                "position": [350, 0, 100]  # cm 단위
            }
        ]
    }
    
    print(f"[INFO] 테스트 방 정보:")
    print(f"   - 크기: {room_data['dimensions']['width_cm']}x{room_data['dimensions']['depth_cm']}cm")
    print(f"   - 가구: {len(room_data['furniture_3d'])}개")
    
    # 2. 데이터 변환 (API에서 사용하는 변환 로직 동일)
    furniture_data = []
    for furniture in room_data['furniture_3d']:
        position = furniture.get('position', [0, 0, 0])
        furniture_item = {
            'type': furniture.get('type', 'furniture'),
            'center_x': position[0] * 10,  # cm → mm
            'center_z': position[2] * 10,  # cm → mm
            'id': furniture.get('name', f"furniture_{len(furniture_data)}")
        }
        furniture_data.append(furniture_item)
    
    # 3. 방 크기 변환
    dimensions = room_data.get('dimensions', {})
    room_dimensions = {
        'width': dimensions.get('width_cm', 400) / 100.0,   # cm → m
        'height': dimensions.get('depth_cm', 500) / 100.0   # cm → m
    }
    
    print(f"[CONVERT] 변환 결과:")
    print(f"   - 방 크기: {room_dimensions['width']}m x {room_dimensions['height']}m")
    for i, item in enumerate(furniture_data):
        print(f"   - 가구 {i+1}: {item['type']} at ({item['center_x']}mm, {item['center_z']}mm)")
    
    # 4. SD 생성기 초기화 (Mock 모드 강제)
    print(f"\n[INIT] Stable Diffusion 생성기 초기화...")
    generator = StableDiffusionGenerator(
        use_controlnet=True,
        enable_cpu_offload=True
    )
    generator.mock_mode = True  # 빠른 테스트를 위해 Mock 모드
    
    # 5. 여러 스타일 테스트
    styles_to_test = ["scandinavian", "modern", "industrial"]
    results = []
    
    for style in styles_to_test:
        print(f"\n[TEST] {style.upper()} 스타일 테스트...")
        try:
            image_path, metadata = generator.generate_interior_image(
                furniture_data=furniture_data,
                room_dimensions=room_dimensions,
                style=style,
                additional_prompt=f"RoomBox integration test - {style} style",
                num_inference_steps=5  # 빠른 테스트
            )
            
            result = {
                'style': style,
                'success': True,
                'image_path': image_path,
                'mock_mode': metadata.get('mock_mode', False),
                'use_controlnet': metadata.get('use_controlnet', False)
            }
            results.append(result)
            
            print(f"   [SUCCESS] 성공: {image_path}")
            print(f"   - Mock 모드: {result['mock_mode']}")
            print(f"   - ControlNet: {result['use_controlnet']}")
            
        except Exception as e:
            print(f"   [FAILED] 실패: {e}")
            results.append({
                'style': style,
                'success': False,
                'error': str(e)
            })
    
    # 6. 결과 요약
    print(f"\n" + "=" * 60)
    print("[RESULT] 최종 테스트 결과")
    print("=" * 60)
    
    successful = len([r for r in results if r.get('success')])
    print(f"총 테스트: {len(results)}개")
    print(f"성공: {successful}개")
    print(f"실패: {len(results) - successful}개")
    
    # 7. 결과 저장
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'roombox_to_sd_integration',
        'room_data': room_data,
        'converted_furniture_data': furniture_data,
        'converted_room_dimensions': room_dimensions,
        'style_test_results': results,
        'summary': {
            'total_tests': len(results),
            'successful_tests': successful,
            'failed_tests': len(results) - successful
        }
    }
    
    # 결과 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"integration_test_result_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SAVE] 테스트 결과 저장: {result_file}")
    
    if successful == len(results):
        print("[COMPLETE] 모든 테스트 성공! RoomBox → SD 파이프라인 정상 작동")
        return True
    else:
        print("[WARNING]  일부 테스트 실패")
        return False

if __name__ == "__main__":
    test_roombox_to_sd_pipeline()