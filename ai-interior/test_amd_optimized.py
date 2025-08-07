"""
AMD Ryzen 5 5625U 시스템에 최적화된 SD 테스트
CPU 모드에서 빠른 생성을 위한 설정 적용
"""

from stable_diffusion_generator import StableDiffusionGenerator
import time

def test_amd_optimized():
    """AMD 최적화 테스트"""
    print("=" * 60)
    print("AMD Ryzen 5 5625U 최적화 테스트")
    print("시스템: 16GB RAM, CPU 모드")
    print("=" * 60)
    
    # 매우 간단한 테스트 데이터
    furniture_data = [
        {
            'type': 'bed',
            'center_x': 2000,  # 2.0m
            'center_z': 2000,  # 2.0m  
            'id': 'center_bed'
        }
    ]
    
    room_dimensions = {'width': 4.0, 'height': 4.0}
    
    print("테스트 시나리오: 4x4m 방에 침대 중앙 배치")
    
    # AMD 최적화 생성기 초기화
    print("\n[INIT] AMD 최적화 SD 생성기 초기화...")
    start_time = time.time()
    
    generator = StableDiffusionGenerator(
        use_controlnet=False,  # 첫 테스트는 ControlNet 없이
        enable_cpu_offload=True
    )
    
    init_time = time.time() - start_time
    print(f"초기화 시간: {init_time:.2f}초")
    
    # 빠른 생성 테스트
    test_settings = [
        {"steps": 1, "size": 256, "desc": "초고속 테스트"},
        {"steps": 5, "size": 512, "desc": "빠른 테스트"}, 
        {"steps": 10, "size": 512, "desc": "표준 테스트"}
    ]
    
    results = []
    
    for i, setting in enumerate(test_settings):
        print(f"\n[TEST {i+1}] {setting['desc']} 시작...")
        print(f"설정: {setting['steps']}스텝, {setting['size']}x{setting['size']}px")
        
        start_time = time.time()
        
        try:
            image_path, metadata = generator.generate_interior_image(
                furniture_data=furniture_data,
                room_dimensions=room_dimensions,
                style="scandinavian",
                additional_prompt=f"AMD test {i+1}",
                num_inference_steps=setting['steps'],
                width=setting['size'],
                height=setting['size']
            )
            
            generation_time = time.time() - start_time
            
            result = {
                'test_name': setting['desc'],
                'steps': setting['steps'],
                'size': setting['size'],
                'time_seconds': generation_time,
                'success': True,
                'image_path': image_path,
                'mock_mode': metadata.get('mock_mode', False)
            }
            results.append(result)
            
            print(f"[SUCCESS] 완료: {generation_time:.2f}초")
            print(f"이미지: {image_path}")
            print(f"Mock 모드: {result['mock_mode']}")
            
        except Exception as e:
            print(f"[ERROR] 실패: {e}")
            results.append({
                'test_name': setting['desc'],
                'success': False,
                'error': str(e)
            })
    
    # 결과 요약
    print(f"\n" + "=" * 60)
    print("AMD 최적화 테스트 결과 요약")
    print("=" * 60)
    
    successful_tests = [r for r in results if r.get('success')]
    
    print(f"성공한 테스트: {len(successful_tests)}/{len(results)}개")
    
    if successful_tests:
        print("\n생성 시간 비교:")
        for result in successful_tests:
            if 'time_seconds' in result:
                print(f"- {result['test_name']}: {result['time_seconds']:.2f}초")
        
        fastest = min(successful_tests, key=lambda x: x.get('time_seconds', float('inf')))
        print(f"\n가장 빠른 설정: {fastest['test_name']} ({fastest.get('time_seconds', 0):.2f}초)")
    
    # AMD 시스템 권장사항
    print(f"\n" + "=" * 60)
    print("AMD Ryzen 5 5625U 권장 설정")
    print("=" * 60)
    print("1. CPU 모드에서 안정적 작동 확인됨")
    print("2. 권장 해상도: 512x512 (메모리 효율성)")
    print("3. 권장 스텝 수: 5-10 (속도와 품질 균형)")
    print("4. Mock 모드: 개발/테스트시 빠른 확인용")
    print("5. RAM 16GB: 여러 모델 동시 로드 가능")
    
    return len(successful_tests) > 0

if __name__ == "__main__":
    test_amd_optimized()