"""
간단한 SD 생성기 테스트
"""

from stable_diffusion_generator import StableDiffusionGenerator

def test_quick():
    """빠른 테스트"""
    print("=" * 50)
    print("간단한 SD 생성기 테스트")
    print("=" * 50)
    
    # 간단한 테스트 데이터
    furniture_data = [
        {
            'type': 'bed',
            'center_x': 2000,  # 2.0m
            'center_z': 2000,  # 2.0m
            'id': 'test_bed'
        }
    ]
    
    room_dimensions = {'width': 4.0, 'height': 4.0}
    
    # 생성기 초기화 (빠른 설정)
    generator = StableDiffusionGenerator(
        use_controlnet=False,  # ControlNet 비활성화로 빠른 테스트
        enable_cpu_offload=True
    )
    
    try:
        print("[TEST] Mock 모드 이미지 생성...")
        image_path, metadata = generator.generate_interior_image(
            furniture_data=furniture_data,
            room_dimensions=room_dimensions,
            style="scandinavian",
            additional_prompt="simple test",
            num_inference_steps=1  # 매우 빠른 테스트
        )
        
        print(f"[SUCCESS] 테스트 완료!")
        print(f"생성된 이미지: {image_path}")
        print(f"Mock 모드: {metadata.get('mock_mode', False)}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    test_quick()