"""
API 서버 없이 직접 SD 엔드포인트 로직 테스트
AMD 최적화된 Mock 모드
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_diffusion_generator import StableDiffusionGenerator
import json
from datetime import datetime

def test_direct_sd_api():
    """API 서버 로직을 직접 테스트"""
    print("=" * 60)
    print("Direct SD API Logic Test (AMD 최적화)")
    print("=" * 60)
    
    # API 요청과 동일한 데이터 구조
    request_data = {
        "room_data": {
            "dimensions": {"width_cm": 400, "depth_cm": 500, "height_cm": 280},
            "furniture_3d": [
                {"type": "bed", "name": "amd_test_bed", "position": [200, 0, 250]},
                {"type": "chair", "name": "desk_chair", "position": [350, 0, 100]}
            ]
        },
        "style": "scandinavian",
        "use_real_ai": False  # Mock 모드
    }
    
    print(f"테스트 데이터:")
    print(f"- 방 크기: {request_data['room_data']['dimensions']['width_cm']}x{request_data['room_data']['dimensions']['depth_cm']}cm")
    print(f"- 가구 개수: {len(request_data['room_data']['furniture_3d'])}개")
    print(f"- 스타일: {request_data['style']}")
    print(f"- 실제 AI 사용: {request_data['use_real_ai']}")
    
    # 1. RoomBox 데이터를 SD용 가구 리스트로 변환 (API와 동일한 로직)
    def convert_roomdata_to_furniture_list(room_data):
        furniture_list = []
        
        if 'furniture_3d' in room_data:
            for furniture in room_data['furniture_3d']:
                position = furniture.get('position', [0, 0, 0])
                furniture_item = {
                    'type': furniture.get('type', 'furniture'),
                    'center_x': position[0] * 10 if position[0] < 100 else position[0],  # cm → mm 변환
                    'center_z': position[2] * 10 if position[2] < 100 else position[2],  # cm → mm 변환
                    'id': furniture.get('name', f"furniture_{len(furniture_list)}")
                }
                furniture_list.append(furniture_item)
        
        return furniture_list
    
    furniture_data = convert_roomdata_to_furniture_list(request_data['room_data'])
    
    # 2. 방 크기 정보 추출
    dimensions = request_data['room_data'].get('dimensions', {})
    room_dimensions = {
        'width': dimensions.get('width_cm', 400) / 100.0,   # cm → m 변환
        'height': dimensions.get('depth_cm', 500) / 100.0   # cm → m 변환
    }
    
    print(f"\n변환된 데이터:")
    print(f"- 방 크기: {room_dimensions['width']}m x {room_dimensions['height']}m")
    for i, item in enumerate(furniture_data):
        print(f"- 가구 {i+1}: {item['type']} at ({item['center_x']}, {item['center_z']})")
    
    # 3. SD 생성기 초기화
    print(f"\n[INIT] SD 생성기 초기화 중...")
    sd_generator = StableDiffusionGenerator(
        use_controlnet=True,
        enable_cpu_offload=True
    )
    
    # AMD CPU 최적화: Mock 모드 설정
    if not request_data['use_real_ai']:
        sd_generator.mock_mode = True
        print("[SD] Mock 모드로 빠른 생성 (AMD CPU 최적화)")
    else:
        sd_generator.mock_mode = False
        print("[SD] 실제 AI 모델 사용 (시간 소요: 5-40분)")
    
    # 4. 이미지 생성
    print(f"\n[GENERATE] 이미지 생성 시작...")
    start_time = datetime.now()
    
    try:
        image_path, metadata = sd_generator.generate_interior_image(
            furniture_data=furniture_data,
            room_dimensions=room_dimensions,
            style=request_data['style'],
            additional_prompt="AMD API test - precise furniture placement",
            use_mask=True,
            num_inference_steps=5 if request_data['use_real_ai'] else 1
        )
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # 5. 결과 준비 (API 응답과 동일한 형식)
        result = {
            "success": True,
            "image_path": image_path,
            "generator_type": "stable_diffusion",
            "style": request_data['style'],
            "furniture_count": len(furniture_data),
            "room_dimensions": room_dimensions,
            "mock_mode": metadata.get('mock_mode', False),
            "use_controlnet": metadata.get('use_controlnet', False),
            "generation_time_seconds": generation_time,
            "timestamp": end_time.isoformat()
        }
        
        # HTTP URL 변환 시뮬레이션
        if image_path:
            filename = os.path.basename(image_path.replace('\\', '/'))
            result['image_url'] = f"http://localhost:8000/images/{filename}"
        
        # 6. 결과 출력
        print(f"\n" + "=" * 60)
        print("생성 결과")
        print("=" * 60)
        print(f"[SUCCESS] 성공: {result['success']}")
        print(f"[PATH] 이미지 경로: {result['image_path']}")
        print(f"[URL] 이미지 URL: {result.get('image_url', 'N/A')}")
        print(f"[STYLE] 스타일: {result['style']}")
        print(f"[FURNITURE] 가구 개수: {result['furniture_count']}")
        print(f"[MODE] Mock 모드: {result['mock_mode']}")
        print(f"[CONTROLNET] ControlNet: {result['use_controlnet']}")
        print(f"[TIME] 생성 시간: {result['generation_time_seconds']:.2f}초")
        
        # 7. 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"api_test_result_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'request_data': request_data,
                'converted_data': {
                    'furniture_data': furniture_data,
                    'room_dimensions': room_dimensions
                },
                'result': result
            }, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVE] 결과 저장: {result_file}")
        
        # 파일 존재 확인
        if os.path.exists(image_path):
            file_size = os.path.getsize(image_path)
            print(f"[SIZE] 이미지 파일: {file_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 생성 실패: {e}")
        import traceback
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_direct_sd_api()
    if success:
        print(f"\n[COMPLETE] Direct SD API 테스트 성공!")
    else:
        print(f"\n[FAILED] Direct SD API 테스트 실패")