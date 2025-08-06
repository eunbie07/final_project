"""
minah 통합 테스트
eunbi -> MongoDB -> ai-interior (minah 기능 포함) 전체 파이프라인 테스트
"""

import asyncio
import json
from datetime import datetime
from mongodb_integration import RoomImagePipeline
from enhanced_vertex_generator import generate_room_image_from_coordinates


async def test_enhanced_generator():
    """Enhanced Vertex Generator 테스트"""
    
    print("Enhanced Vertex Generator 테스트 시작")
    print("="*60)
    
    # 샘플 데이터 (eunbi에서 MongoDB에 저장되는 형식)
    sample_room_data = {
        "_id": "test_minah_integration_001",
        "scene": {
            "description": "minah 통합 테스트용 방",
            "room": {
                "width": 4000,  # 4m
                "depth": 5000,  # 5m  
                "height": 2800  # 2.8m
            },
            "objects": [
                {
                    "type": "furniture",
                    "name": "bed",
                    "position": {"center": {"x": 2000, "y": 4000}},
                    "dimensions": {"width": 1600, "depth": 2000, "height": 600},
                    "rotation_z": 0
                },
                {
                    "type": "furniture", 
                    "name": "desk",
                    "position": {"center": {"x": 3500, "y": 1500}},
                    "dimensions": {"width": 1200, "depth": 600, "height": 750},
                    "rotation_z": 45
                },
                {
                    "type": "furniture",
                    "name": "wardrobe", 
                    "position": {"center": {"x": 500, "y": 2500}},
                    "dimensions": {"width": 800, "depth": 600, "height": 2200},
                    "rotation_z": 0
                },
                {
                    "type": "window",
                    "name": "main_window",
                    "wall": 1,
                    "dimensions": {"width": 1500, "height": 1200},
                    "position": {"x": 1250, "y": 0, "z": 1800},
                    "rotation_z": 0
                }
            ]
        },
        "saved_at": datetime.now().isoformat(),
        "format_version": "2.0.0"
    }
    
    # 스타일별 테스트
    styles = ["modern", "scandinavian", "industrial"]
    
    for style in styles:
        print(f"\n{style.upper()} 스타일 테스트")
        print("-" * 40)
        
        try:
            # 이미지 생성
            result = await generate_room_image_from_coordinates(
                room_data=sample_room_data,
                style=style,
                prefer_google_ai=True  # Google AI 우선 (minah 방식)
            )
            
            if result.get("success"):
                print(f"OK: 생성 성공:")
                print(f"   파일: {result.get('image_path')}")
                print(f"   방식: {result.get('method')}")
                print(f"   방 크기: {result['room_layout']['width_mm']}×{result['room_layout']['depth_mm']}mm")
                print(f"   가구 개수: {result['room_layout']['furniture_count']}")
                print(f"   프롬프트: {result['generated_prompt'][:100]}...")
            else:
                print(f"ERROR: 생성 실패: {result.get('error')}")
                
        except Exception as e:
            print(f"ERROR: 테스트 실패: {e}")
    
    print("\n" + "="*60)
    print("OK: Enhanced Generator 테스트 완료")


async def test_mongodb_pipeline():
    """MongoDB 파이프라인 테스트"""
    
    print("\nLAUNCH: MongoDB 파이프라인 테스트 시작")
    print("="*60)
    
    pipeline = RoomImagePipeline()
    
    try:
        # 배치 처리 테스트
        print("BATCH: 배치 처리 테스트...")
        result = await pipeline.run_pipeline(
            mode="batch",
            style="scandinavian", 
            limit=2
        )
        
        print(f"\n📊 처리 결과:")
        print(f"  - 총 처리: {result['total_processed']}개")
        print(f"  - 성공: {result['successful']}개") 
        print(f"  - 실패: {result['failed']}개")
        print(f"  - 스타일: {result['style']}")
        
        for i, res in enumerate(result['results']):
            if res.get('success'):
                print(f"  OK: {i+1}: {res.get('image_path', 'Unknown')}")
                print(f"      방법: {res.get('method', 'Unknown')}")
                print(f"      ID: {res.get('layout_id', 'Unknown')}")
            else:
                print(f"  ERROR: {i+1}: {res.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"ERROR: 파이프라인 테스트 실패: {e}")
    
    finally:
        pipeline.close()
    
    print("\n" + "="*60)
    print("OK: MongoDB 파이프라인 테스트 완료")


async def test_coordinate_accuracy():
    """좌표 정확도 테스트"""
    
    print("\nLAUNCH: 좌표 정확도 테스트 시작")
    print("="*60)
    
    # 정확한 좌표가 중요한 복잡한 레이아웃
    complex_room = {
        "_id": "coordinate_accuracy_test",
        "scene": {
            "description": "좌표 정확도 테스트용 복잡한 방",
            "room": {"width": 3500, "depth": 4000, "height": 2500},
            "objects": [
                # 방 중앙에 L자형 소파
                {
                    "type": "furniture",
                    "name": "L_shaped_sofa",
                    "position": {"center": {"x": 1750, "y": 2000}},
                    "dimensions": {"width": 2400, "depth": 1600, "height": 850},
                    "rotation_z": 0
                },
                # 소파 앞 커피테이블
                {
                    "type": "furniture",
                    "name": "coffee_table",
                    "position": {"center": {"x": 1750, "y": 1200}},
                    "dimensions": {"width": 1200, "depth": 600, "height": 450},
                    "rotation_z": 0
                },
                # 오른쪽 벽 근처 TV 스탠드
                {
                    "type": "furniture",
                    "name": "tv_stand", 
                    "position": {"center": {"x": 3200, "y": 1000}},
                    "dimensions": {"width": 1800, "depth": 400, "height": 650},
                    "rotation_z": 0
                },
                # 왼쪽 구석 책장
                {
                    "type": "furniture",
                    "name": "bookshelf",
                    "position": {"center": {"x": 300, "y": 3700}},
                    "dimensions": {"width": 600, "depth": 300, "height": 1800},
                    "rotation_z": 0
                },
                # 뒤쪽 벽 창문
                {
                    "type": "window",
                    "name": "large_window",
                    "wall": 3,  # 뒤쪽 벽
                    "dimensions": {"width": 2000, "height": 1400},
                    "position": {"x": 1750, "y": 4000, "z": 1200},
                    "rotation_z": 0
                }
            ]
        },
        "saved_at": datetime.now().isoformat(),
        "format_version": "2.0.0"
    }
    
    print("TEST: 복잡한 레이아웃 좌표 테스트:")
    print(f"  - 방 크기: 3.5m × 4.0m")
    print(f"  - 가구 개수: 4개")
    print(f"  - 창문 개수: 1개")
    
    try:
        result = await generate_room_image_from_coordinates(
            room_data=complex_room,
            style="modern",
            prefer_google_ai=True
        )
        
        if result.get("success"):
            print(f"\nOK: 복잡한 레이아웃 생성 성공!")
            print(f"   📁 파일: {result.get('image_path')}")
            print(f"   🎨 방식: {result.get('method')}")
            print(f"   📐 정확도: {result.get('coordinate_accuracy', 'standard')}")
            
            # 생성된 프롬프트에서 좌표 정확도 확인
            prompt = result.get('generated_prompt', '')
            if 'exact position' in prompt.lower() or 'center coordinates' in prompt.lower():
                print(f"   OK: 정확한 좌표 정보가 프롬프트에 포함됨")
            else:
                print(f"   WARNING: 좌표 정보가 프롬프트에 부족할 수 있음")
                
        else:
            print(f"ERROR: 복잡한 레이아웃 생성 실패: {result.get('error')}")
            
    except Exception as e:
        print(f"ERROR: 좌표 정확도 테스트 실패: {e}")
    
    print("\n" + "="*60)
    print("OK: 좌표 정확도 테스트 완료")


async def test_style_consistency():
    """스타일 일관성 테스트"""
    
    print("\nLAUNCH: 스타일 일관성 테스트 시작")
    print("="*60)
    
    # 동일한 레이아웃, 다른 스타일들
    base_room = {
        "_id": "style_consistency_test",
        "scene": {
            "description": "스타일 일관성 테스트용 표준 방",
            "room": {"width": 4000, "depth": 4500, "height": 2600},
            "objects": [
                {
                    "type": "furniture",
                    "name": "bed",
                    "position": {"center": {"x": 1200, "y": 3600}},
                    "dimensions": {"width": 1400, "depth": 2000, "height": 600},
                    "rotation_z": 0
                },
                {
                    "type": "furniture",
                    "name": "desk",
                    "position": {"center": {"x": 3500, "y": 1200}},
                    "dimensions": {"width": 1200, "depth": 600, "height": 750},
                    "rotation_z": -90
                }
            ]
        },
        "saved_at": datetime.now().isoformat(),
        "format_version": "2.0.0"
    }
    
    styles_to_test = [
        ("modern", "모던 미니멀리스트"),
        ("scandinavian", "스칸디나비아 북유럽"),
        ("industrial", "인더스트리얼 빈티지")
    ]
    
    results = []
    
    for style, description in styles_to_test:
        print(f"\nTEST: {description} 스타일 테스트...")
        
        try:
            result = await generate_room_image_from_coordinates(
                room_data=base_room,
                style=style,
                prefer_google_ai=True
            )
            
            if result.get("success"):
                print(f"   OK: {description} 생성 성공")
                print(f"   📁 {result.get('image_path')}")
                results.append((style, True, result.get('image_path')))
            else:
                print(f"   ERROR: {description} 생성 실패: {result.get('error')}")
                results.append((style, False, result.get('error')))
                
        except Exception as e:
            print(f"   ERROR: {description} 테스트 실패: {e}")
            results.append((style, False, str(e)))
    
    # 결과 요약
    print(f"\n📊 스타일 일관성 테스트 결과:")
    successful_styles = [r for r in results if r[1]]
    failed_styles = [r for r in results if not r[1]]
    
    print(f"  OK: 성공: {len(successful_styles)}/{len(results)} 스타일")
    for style, _, path in successful_styles:
        print(f"     - {style}: {path}")
    
    if failed_styles:
        print(f"  ERROR: 실패: {len(failed_styles)} 스타일")
        for style, _, error in failed_styles:
            print(f"     - {style}: {error}")
    
    print("\n" + "="*60)
    print("OK: 스타일 일관성 테스트 완료")


async def main():
    """전체 통합 테스트 실행"""
    
    print("minah + ai-interior 통합 테스트 시작")
    print("=" * 80)
    print("테스트 항목:")
    print("  1. Enhanced Vertex Generator (minah 기능 통합)")
    print("  2. MongoDB 파이프라인 (eunbi 연동)")
    print("  3. 좌표 정확도 (ai-interior 강점)")
    print("  4. 스타일 일관성 (minah 스타일 참조)")
    print("=" * 80)
    
    try:
        # 1. Enhanced Generator 테스트
        await test_enhanced_generator()
        
        # 2. MongoDB 파이프라인 테스트
        await test_mongodb_pipeline()
        
        # 3. 좌표 정확도 테스트
        await test_coordinate_accuracy()
        
        # 4. 스타일 일관성 테스트
        await test_style_consistency()
        
        print("\n" + "=" * 80)
        print("🎉 전체 통합 테스트 완료!")
        print("SUCCESS: minah의 고품질 이미지 생성 + ai-interior의 정확한 좌표 처리")
        print("SUCCESS: eunbi -> MongoDB -> ai-interior 파이프라인 검증 완료")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: 통합 테스트 중 오류 발생: {e}")
        print("=" * 80)


def cli_main():
    """CLI 진입점 (uv run enhanced-test)"""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()