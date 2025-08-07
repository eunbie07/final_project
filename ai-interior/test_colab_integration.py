#!/usr/bin/env python3
"""
Colab Inpainting 시스템 통합 테스트
MongoDB 좌표 → 픽셀 마스크 → 95%+ 정확도 검증
"""

import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, Any

# 실제 MongoDB 데이터 (프로젝트에서 사용하는 정확한 데이터)
REAL_ROOM_DATA = {
    'dimensions': {
        'width_cm': 387,   # 실제 방 폭
        'depth_cm': 465,   # 실제 방 깊이  
        'height_cm': 280
    },
    'furniture_3d': [
        {
            'name': 'bed',
            'type': 'bed',
            'position': [203.67, 0, 238.00]  # 실제 침대 위치 (cm)
        }
    ]
}

def test_local_coordinate_conversion():
    """로컬 좌표 변환 시스템 테스트"""
    print("[1/4] 로컬 좌표 변환 테스트")
    
    try:
        from colab_integration import LocalCoordinateConverter
        
        converter = LocalCoordinateConverter()
        mask_image, regions = converter.convert_room_to_mask(REAL_ROOM_DATA)
        
        print(f"OK 좌표 변환 성공")
        print(f"   방 크기: {REAL_ROOM_DATA['dimensions']['width_cm']}x{REAL_ROOM_DATA['dimensions']['depth_cm']}cm")
        print(f"   가구 개수: {len(regions)}개")
        
        for region in regions:
            original_cm = region['original_position_cm']
            pixel_pos = region['center']
            print(f"   - {region['name']}: ({original_cm[0]}, {original_cm[1]})cm → ({pixel_pos[0]}, {pixel_pos[1]})px")
        
        # 마스크 이미지 저장
        mask_path = "generated_images/test_coordinate_mask.png"
        os.makedirs("generated_images", exist_ok=True)
        mask_image.save(mask_path)
        print(f"   마스크 저장: {mask_path}")
        
        return True, {"mask_path": mask_path, "regions": regions}
        
    except Exception as e:
        print(f"ERROR 로컬 좌표 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(e)}

def test_api_server_status():
    """현재 API 서버 상태 확인"""
    print("\n[2/4] API 서버 상태 확인")
    
    try:
        response = requests.get("http://localhost:8000/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("OK API 서버 실행 중")
            print(f"   생성기 상태:")
            for name, status in data['generators'].items():
                emoji = "OK" if status == "running" else "ERROR"
                print(f"     {emoji} {name}: {status}")
            
            colab_available = data['generators'].get('colab_inpainting') == 'running'
            return True, {"colab_available": colab_available, "generators": data['generators']}
        else:
            print(f"ERROR API 서버 응답 오류: {response.status_code}")
            return False, {"error": "API server error"}
            
    except requests.exceptions.ConnectionError:
        print("ERROR API 서버 연결 불가 - 서버를 먼저 시작하세요:")
        print("   uv run api_server.py")
        return False, {"error": "Connection refused"}
    except Exception as e:
        print(f"ERROR API 서버 테스트 실패: {e}")
        return False, {"error": str(e)}

def test_colab_endpoint():
    """Colab Inpainting 엔드포인트 테스트"""
    print("\n[3/4] Colab Inpainting 엔드포인트 테스트")
    
    # 환경변수 확인
    colab_url = os.environ.get('COLAB_API_URL')
    if not colab_url or colab_url == 'https://your-ngrok-url.ngrok.io':
        print("WARNING COLAB_API_URL 환경변수가 설정되지 않았습니다")
        print("   실제 Colab URL로 설정하세요:")
        print("   set COLAB_API_URL=https://your-actual-ngrok-url.ngrok.io")
        print("   또는 export COLAB_API_URL=https://your-actual-ngrok-url.ngrok.io")
        return False, {"error": "COLAB_API_URL not configured"}
    
    print(f"   Colab URL: {colab_url}")
    
    try:
        # Colab 엔드포인트 호출
        response = requests.post(
            "http://localhost:8000/generate-interior-colab",
            json={
                "room_data": REAL_ROOM_DATA,
                "style": "scandinavian",
                "generate_image": True,
                "use_real_ai": True
            },
            timeout=300  # 5분 타임아웃
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                accuracy = result.get('accuracy_score', 0.0)
                accuracy_pct = result.get('accuracy_percentage', '0%')
                
                print(f"OK Colab Inpainting 생성 성공!")
                print(f"   정확도: {accuracy_pct} (목표: 95%+)")
                print(f"   이미지: {result.get('image_url', 'N/A')}")
                print(f"   생성기: {result.get('generator_type')}")
                
                # 위치 분석 정보
                if 'position_analysis' in result:
                    analysis = result['position_analysis']
                    print(f"   상태: {analysis.get('accuracy_status', 'N/A')}")
                    
                    for pos in analysis.get('furniture_positions', []):
                        print(f"     - {pos['name']}: {pos['original_cm']}cm → {pos['converted_px']}px")
                
                # 목표 달성 확인
                target_achieved = accuracy >= 0.95
                print(f"   목표 달성: {'YES' if target_achieved else 'NO'}")
                
                return True, {
                    "accuracy_score": accuracy,
                    "target_achieved": target_achieved,
                    "result": result
                }
            else:
                print(f"ERROR Colab 생성 실패: {result.get('error', 'Unknown error')}")
                return False, {"error": result.get('error')}
        
        elif response.status_code == 503:
            print("ERROR Colab 생성기가 초기화되지 않았습니다")
            print("   COLAB_API_URL 환경변수를 확인하고 서버를 재시작하세요")
            return False, {"error": "Service unavailable"}
        else:
            print(f"ERROR API 호출 실패: {response.status_code}")
            print(f"   응답: {response.text[:200]}...")
            return False, {"error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.Timeout:
        print("ERROR 요청 타임아웃 (5분 초과)")
        print("   Colab에서 Inpainting 생성이 시간이 오래 걸릴 수 있습니다")
        return False, {"error": "Timeout"}
    except Exception as e:
        print(f"ERROR Colab 엔드포인트 테스트 실패: {e}")
        return False, {"error": str(e)}

def test_accuracy_verification():
    """정확도 검증 및 비교 분석"""
    print("\n[4/4] 정확도 검증 및 비교 분석")
    
    # 원본 MongoDB 좌표 분석
    bed_pos = REAL_ROOM_DATA['furniture_3d'][0]['position']  # [203.67, 0, 238.00]
    room_width = REAL_ROOM_DATA['dimensions']['width_cm']   # 387cm
    room_depth = REAL_ROOM_DATA['dimensions']['depth_cm']    # 465cm
    
    # 상대적 위치 계산 (방 중심 기준)
    rel_x_pct = (bed_pos[0] / room_width) * 100      # X축 상대 위치 %
    rel_z_pct = (bed_pos[2] / room_depth) * 100      # Z축 상대 위치 %
    
    print(f"MongoDB 좌표 분석:")
    print(f"   방 크기: {room_width}cm × {room_depth}cm")
    print(f"   침대 위치: ({bed_pos[0]}, {bed_pos[2]})cm")
    print(f"   상대 위치: {rel_x_pct:.1f}% X, {rel_z_pct:.1f}% Z")
    
    # 예상 픽셀 위치 (512x512 기준)
    expected_pixel_x = int((bed_pos[0] / room_width) * 512)
    expected_pixel_z = int((bed_pos[2] / room_depth) * 512)
    print(f"   예상 픽셀: ({expected_pixel_x}, {expected_pixel_z})px")
    
    # 기존 생성기들과의 비교 (만약 데이터가 있다면)
    print(f"\n생성기별 정확도 비교:")
    
    generators_to_test = [
        ("DALL-E 3", "/generate-interior-dalle"),
        ("Stable Diffusion", "/generate-interior-sd"),
        ("Colab Inpainting", "/generate-interior-colab")
    ]
    
    results = {}
    
    for gen_name, endpoint in generators_to_test:
        try:
            print(f"   테스트 중: {gen_name}...")
            
            response = requests.post(
                f"http://localhost:8000{endpoint}",
                json={
                    "room_data": REAL_ROOM_DATA,
                    "style": "scandinavian",
                    "use_real_ai": False  # 빠른 테스트용 Mock 모드
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    accuracy = result.get('accuracy_score', 0.0)
                    results[gen_name] = accuracy
                    print(f"     OK {gen_name}: {accuracy*100:.1f}%")
                else:
                    print(f"     ERROR {gen_name}: 생성 실패")
            else:
                print(f"     WARNING {gen_name}: 서비스 불가 ({response.status_code})")
                
        except Exception as e:
            print(f"     ERROR {gen_name}: 오류 ({str(e)[:50]}...)")
    
    # 최종 결과
    print(f"\n최종 결과:")
    if results:
        best_generator = max(results.items(), key=lambda x: x[1])
        print(f"   최고 정확도: {best_generator[0]} - {best_generator[1]*100:.1f}%")
        
        colab_accuracy = results.get("Colab Inpainting", 0.0)
        if colab_accuracy >= 0.95:
            print(f"   목표 달성! Colab Inpainting {colab_accuracy*100:.1f}% >= 95%")
            return True, {"best": best_generator, "colab_achieved": True, "results": results}
        else:
            print(f"   WARNING 목표 미달성: Colab {colab_accuracy*100:.1f}% < 95%")
            return False, {"best": best_generator, "colab_achieved": False, "results": results}
    else:
        print("   ERROR 모든 테스트 실패")
        return False, {"error": "All tests failed"}

def main():
    """메인 테스트 실행"""
    print("Colab Inpainting 시스템 통합 테스트")
    print("=" * 60)
    
    test_results = {}
    
    # 1. 로컬 좌표 변환 테스트
    success, data = test_local_coordinate_conversion()
    test_results['coordinate_conversion'] = {'success': success, 'data': data}
    
    if not success:
        print("\nERROR 로컬 좌표 변환 실패로 테스트 중단")
        return
    
    # 2. API 서버 상태 확인
    success, data = test_api_server_status()
    test_results['api_server'] = {'success': success, 'data': data}
    
    if not success:
        print("\nERROR API 서버 연결 실패로 테스트 중단")
        return
    
    # 3. Colab 엔드포인트 테스트 (환경변수가 설정된 경우에만)
    colab_url = os.environ.get('COLAB_API_URL')
    if colab_url and colab_url != 'https://your-ngrok-url.ngrok.io':
        success, data = test_colab_endpoint()
        test_results['colab_endpoint'] = {'success': success, 'data': data}
    else:
        print("\nWARNING Colab 엔드포인트 테스트 건너뛰기 (환경변수 미설정)")
        test_results['colab_endpoint'] = {'success': False, 'data': {'skipped': True}}
    
    # 4. 정확도 검증
    success, data = test_accuracy_verification()
    test_results['accuracy_verification'] = {'success': success, 'data': data}
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "OK 성공" if result['success'] else "ERROR 실패"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    # 핵심 성취도 확인
    coord_success = test_results['coordinate_conversion']['success']
    api_success = test_results['api_server']['success']
    
    if coord_success and api_success:
        print(f"\n핵심 시스템:")
        print(f"   OK MongoDB → 픽셀 변환: 구현 완료")
        print(f"   OK API 연동 구조: 준비 완료")
        
        colab_data = test_results.get('colab_endpoint', {}).get('data', {})
        if colab_data.get('target_achieved'):
            print(f"   95%+ 정확도: 달성!")
        elif colab_data.get('skipped'):
            print(f"   WARNING 95%+ 정확도: Colab URL 설정 후 테스트 필요")
        else:
            print(f"   95%+ 정확도: 추가 튜닝 필요")
    
    print(f"\n다음 단계:")
    if not colab_url or colab_url == 'https://your-ngrok-url.ngrok.io':
        print(f"   1. Google Colab에서 Notebook 실행")
        print(f"   2. Ngrok URL을 COLAB_API_URL 환경변수로 설정")
        print(f"   3. 이 테스트 스크립트 재실행")
    else:
        print(f"   1. Colab에서 ComfyUI 모델 다운로드 완료")
        print(f"   2. Inpainting 워크플로우 튜닝")
        print(f"   3. 실제 프로젝트에서 사용")
    
    # 로그 파일 저장
    log_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n상세 결과: {log_file}")

if __name__ == "__main__":
    main()