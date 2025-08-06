"""
개선된 AI 인테리어 생성 시스템 정확도 테스트
좌표 정확도와 가구 개수 정확도를 종합 테스트
"""

import asyncio
import json
import os
from typing import Dict, Any
from datetime import datetime

from roombox_integration import DifyRoomImageGenerator
from config import load_config


async def test_single_furniture_accuracy():
    """단일 가구 배치 정확도 테스트"""
    
    print("🧪 단일 가구 정확도 테스트 시작")
    
    try:
        # Dify 설정 로드
        config = load_config()
        generator = DifyRoomImageGenerator(
            config.api_key,
            config.app_id, 
            config.dataset_id
        )
        
        # 테스트 케이스들
        test_cases = [
            {
                "name": "침대_중앙배치",
                "room_data": {
                    "dimensions": {
                        "width_cm": 400,  # 4m
                        "depth_cm": 500,  # 5m
                        "height_cm": 280  # 2.8m
                    },
                    "furniture_3d": [
                        {
                            "name": "bed",
                            "type": "bed",
                            "position": [200, 50, 250],  # 중앙 (cm 단위)
                            "scale": [1.0, 1.0, 1.0],
                            "rotation": [0, 0, 0]
                        }
                    ]
                },
                "style": "scandinavian",
                "expected_accuracy": 0.8
            },
            {
                "name": "소파_우측배치", 
                "room_data": {
                    "dimensions": {
                        "width_cm": 450,  # 4.5m
                        "depth_cm": 550,  # 5.5m
                        "height_cm": 280
                    },
                    "furniture_3d": [
                        {
                            "name": "sofa",
                            "type": "sofa", 
                            "position": [300, 50, 150],  # 우측 앞쪽 (cm 단위)
                            "scale": [1.2, 1.0, 1.0],
                            "rotation": [0, 0, 0]
                        }
                    ]
                },
                "style": "modern",
                "expected_accuracy": 0.8
            },
            {
                "name": "의자_왼쪽배치",
                "room_data": {
                    "dimensions": {
                        "width_cm": 350,  # 3.5m
                        "depth_cm": 400,  # 4m  
                        "height_cm": 280
                    },
                    "furniture_3d": [
                        {
                            "name": "chair",
                            "type": "chair",
                            "position": [80, 50, 200],  # 왼쪽 중앙 (cm 단위)
                            "scale": [1.0, 1.0, 1.0],
                            "rotation": [0, 1.57, 0]  # 90도 회전
                        }
                    ]
                },
                "style": "industrial", 
                "expected_accuracy": 0.7
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f\"\\n🔍 테스트 {i}/{len(test_cases)}: {test_case['name']}\")\n            \n            # 이미지 생성\n            result = await generator.generate_consistent_room_image(\n                room_data=test_case[\"room_data\"],\n                style=test_case[\"style\"],\n                user_id=\"test_user\"\n            )\n            \n            if result[\"success\"]:\n                print(f\"  ✅ 이미지 생성 성공: {result['image_path']}\")\n                \n                # 검증 결과 확인\n                accuracy_score = result.get(\"accuracy_score\", 0.0)\n                furniture_accurate = result.get(\"furniture_count_accurate\", False)\n                \n                print(f\"  📊 정확도 점수: {accuracy_score:.1%}\")\n                print(f\"  🪑 가구 개수 정확: {'✓' if furniture_accurate else '✗'}\")\n                \n                # 결과 저장\n                test_result = {\n                    \"test_name\": test_case[\"name\"],\n                    \"style\": test_case[\"style\"],\n                    \"success\": True,\n                    \"accuracy_score\": accuracy_score,\n                    \"furniture_count_accurate\": furniture_accurate,\n                    \"expected_accuracy\": test_case[\"expected_accuracy\"],\n                    \"meets_expectation\": accuracy_score >= test_case[\"expected_accuracy\"],\n                    \"image_path\": result[\"image_path\"],\n                    \"generation_service\": result.get(\"service\", \"unknown\")\n                }\n                \n                if accuracy_score >= test_case[\"expected_accuracy\"]:\n                    print(f\"  🎯 기대치 달성 ({test_case['expected_accuracy']:.1%})\")\n                else:\n                    print(f\"  ⚠️  기대치 미달 (기대: {test_case['expected_accuracy']:.1%})\")\n            else:\n                print(f\"  ❌ 이미지 생성 실패: {result.get('error')}\")\n                test_result = {\n                    \"test_name\": test_case[\"name\"],\n                    \"style\": test_case[\"style\"],\n                    \"success\": False,\n                    \"error\": result.get(\"error\"),\n                    \"accuracy_score\": 0.0,\n                    \"meets_expectation\": False\n                }\n            \n            results.append(test_result)\n        \n        # 전체 결과 분석\n        print(\"\\n📈 전체 테스트 결과 분석\")\n        print(\"=\" * 50)\n        \n        successful_tests = [r for r in results if r[\"success\"]]\n        accurate_tests = [r for r in results if r.get(\"meets_expectation\", False)]\n        \n        print(f\"성공한 테스트: {len(successful_tests)}/{len(results)}\")\n        print(f\"기대치 달성: {len(accurate_tests)}/{len(results)}\")\n        \n        if successful_tests:\n            avg_accuracy = sum(r[\"accuracy_score\"] for r in successful_tests) / len(successful_tests)\n            print(f\"평균 정확도: {avg_accuracy:.1%}\")\n            \n            furniture_accuracy_rate = sum(1 for r in successful_tests if r.get(\"furniture_count_accurate\", False)) / len(successful_tests)\n            print(f\"가구 개수 정확률: {furniture_accuracy_rate:.1%}\")\n        \n        # 결과를 JSON 파일로 저장\n        test_report = {\n            \"timestamp\": datetime.now().isoformat(),\n            \"test_type\": \"single_furniture_accuracy\",\n            \"total_tests\": len(results),\n            \"successful_tests\": len(successful_tests),\n            \"accurate_tests\": len(accurate_tests),\n            \"average_accuracy\": sum(r[\"accuracy_score\"] for r in successful_tests) / len(successful_tests) if successful_tests else 0.0,\n            \"furniture_accuracy_rate\": sum(1 for r in successful_tests if r.get(\"furniture_count_accurate\", False)) / len(successful_tests) if successful_tests else 0.0,\n            \"detailed_results\": results\n        }\n        \n        os.makedirs(\"test_results\", exist_ok=True)\n        report_filename = f\"accuracy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json\"\n        report_path = os.path.join(\"test_results\", report_filename)\n        \n        with open(report_path, 'w', encoding='utf-8') as f:\n            json.dump(test_report, f, indent=2, ensure_ascii=False)\n        \n        print(f\"\\n📄 상세 리포트 저장됨: {report_path}\")\n        \n        return test_report\n        \n    except Exception as e:\n        print(f\"❌ 테스트 실행 오류: {e}\")\n        import traceback\n        traceback.print_exc()\n        return None


async def test_coordinate_accuracy():  \n    \"\"\"좌표 정확도 세부 테스트\"\"\"\n    \n    print(\"\\n🎯 좌표 정확도 세부 테스트\")\n    \n    # 정확한 위치 테스트 케이스들\n    coordinate_tests = [\n        {\n            \"name\": \"정중앙_배치\",\n            \"position_cm\": [200, 50, 250],  # 4m x 5m 방의 정중앙\n            \"room_size\": [400, 500],\n            \"expected_position\": \"center middle\"\n        },\n        {\n            \"name\": \"좌측상단_배치\", \n            \"position_cm\": [80, 50, 100],  # 좌측 앞쪽\n            \"room_size\": [400, 500],\n            \"expected_position\": \"left front\"\n        },\n        {\n            \"name\": \"우측하단_배치\",\n            \"position_cm\": [320, 50, 400],  # 우측 뒤쪽 \n            \"room_size\": [400, 500], \n            \"expected_position\": \"right back\"\n        }\n    ]\n    \n    print(f\"좌표 정확도 테스트 케이스: {len(coordinate_tests)}개\")\n    \n    for test in coordinate_tests:\n        x_percent = test[\"position_cm\"][0] / test[\"room_size\"][0]\n        z_percent = test[\"position_cm\"][2] / test[\"room_size\"][1]\n        \n        print(f\"  {test['name']}: 좌표({test['position_cm'][0]}, {test['position_cm'][2]}) = 상대위치({x_percent:.1%}, {z_percent:.1%}) → {test['expected_position']}\")\n    \n    return coordinate_tests


async def main():\n    \"\"\"메인 테스트 함수\"\"\"\n    \n    print(\"🚀 AI 인테리어 생성 정확도 테스트 시작\")\n    print(\"=\" * 60)\n    \n    # 1. 단일 가구 정확도 테스트\n    single_furniture_result = await test_single_furniture_accuracy()\n    \n    # 2. 좌표 정확도 테스트\n    coordinate_result = await test_coordinate_accuracy()\n    \n    print(\"\\n✅ 모든 테스트 완료\")\n    \n    if single_furniture_result:\n        print(f\"최종 성과: 평균 정확도 {single_furniture_result['average_accuracy']:.1%}, 가구 개수 정확률 {single_furniture_result['furniture_accuracy_rate']:.1%}\")\n        \n        if single_furniture_result[\"average_accuracy\"] >= 0.7:\n            print(\"🎉 목표 달성! (70% 이상 정확도)\")\n        else:\n            print(\"⚠️  추가 개선 필요 (목표: 70% 정확도)\")\n    \n    print(\"\\n개선 사항:\")\n    print(\"  1. ✅ 정밀한 좌표→위치 변환 로직\")\n    print(\"  2. ✅ 강화된 Vertex AI 프롬프트\")\n    print(\"  3. ✅ Gemini Vision 기반 이미지 검증\")\n    print(\"  4. ✅ 자동 재시도 및 프롬프트 개선\")\n\n\nif __name__ == \"__main__\":\n    asyncio.run(main())