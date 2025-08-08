"""
RoomBox.jsx와 Dify 통합 사용 예제 및 테스트
좌표 일관성 보장 및 실시간 동기화 시스템 데모
"""

import asyncio
import json
from roombox_integration import DifyRoomImageGenerator
from realtime_sync import RealtimeCoordinateSync

# 샘플 RoomBox.jsx 데이터
SAMPLE_ROOMBOX_DATA = {
    "scene": {
        "description": "A 3D scene definition for a room. All units are in millimeters.",
        "room": {
            "width": 4000,
            "depth": 5000, 
            "height": 2800
        },
        "objects": [
            {
                "type": "furniture",
                "name": "double_bed",
                "position": {
                    "x": 2000,  # RoomBox 3D 좌표 (중심점)
                    "y": 100,
                    "z": 4500
                },
                "dimensions": {
                    "width": 1600,
                    "depth": 2100,
                    "height": 1000
                },
                "rotation": [0, 3.14159, 0],  # Y축 180도 회전
                "material": "dark wood frame with grey linen bedding"
            },
            {
                "type": "furniture", 
                "name": "wardrobe",
                "position": {
                    "x": 300,
                    "y": 100,
                    "z": 2500
                },
                "dimensions": {
                    "width": 600,
                    "depth": 1800,
                    "height": 2200
                },
                "rotation": [0, 0, 0],
                "material": "white matte finish"
            },
            {
                "type": "furniture",
                "name": "desk",
                "position": {
                    "x": 3700,
                    "y": 75,
                    "z": 3100
                },
                "dimensions": {
                    "width": 600,
                    "depth": 1200,
                    "height": 750
                },
                "rotation": [0, -1.5708, 0],  # Y축 -90도 회전
                "material": "light oak wood with black metal legs"
            },
            {
                "type": "window",
                "name": "main_window",
                "position": {
                    "x": 4000,
                    "y": 1000,
                    "z": 2800
                },
                "dimensions": {
                    "width": 50,
                    "depth": 2500,
                    "height": 1500
                },
                "details": "covered with sheer white curtains"
            }
        ]
    }
}

# 좌표 변경 시뮬레이션 데이터
COORDINATE_CHANGES = [
    {
        "description": "침대를 오른쪽으로 이동",
        "changes": {
            "objects[0].position.x": 2500
        }
    },
    {
        "description": "책상 회전 변경",
        "changes": {
            "objects[2].rotation[1]": 0  # 0도 회전
        }
    },
    {
        "description": "옷장을 벽 쪽으로 이동",
        "changes": {
            "objects[1].position.x": 100
        }
    }
]


class IntegrationDemo:
    """통합 시스템 데모 클래스"""
    
    def __init__(self):
        # 환경 변수에서 API 키 로드 (실제 사용 시)
        self.dify_api_key = "your-dify-api-key"
        self.dify_app_id = "your-dify-app-id" 
        self.dify_dataset_id = "your-dify-dataset-id"
        
        # 시스템 초기화
        self.dify_generator = DifyRoomImageGenerator(
            self.dify_api_key, 
            self.dify_app_id, 
            self.dify_dataset_id
        )
        
        self.realtime_sync = RealtimeCoordinateSync(
            self.dify_api_key,
            self.dify_app_id, 
            self.dify_dataset_id
        )
    
    async def demo_coordinate_processing(self):
        """좌표 처리 시스템 데모"""
        
        print("=" * 60)
        print("🔧 RoomBox.jsx 좌표 처리 시스템 데모")
        print("=" * 60)
        
        # 1. 데이터 파싱 및 검증
        print("\n1️⃣ RoomBox.jsx 데이터 파싱 및 좌표 검증")
        print("-" * 40)
        
        room_layout = self.dify_generator.data_processor.parse_roombox_data(SAMPLE_ROOMBOX_DATA)
        
        print(f"✅ 방 크기: {room_layout.width_mm}×{room_layout.depth_mm}×{room_layout.height_mm}mm")
        print(f"✅ 면적: {room_layout.area_sqm:.1f}㎡")
        print(f"✅ 가구 개수: {len(room_layout.furniture)}개")
        
        # 각 가구 좌표 검증
        for furniture in room_layout.furniture:
            validation = self.dify_generator.data_processor.coordinate_validator.validate_furniture_position(
                furniture, room_layout
            )
            
            status = "✅" if validation["valid"] else "❌"
            print(f"{status} {furniture.name}: ({furniture.center_x}, {furniture.center_y}) - {furniture.width}×{furniture.depth}mm")
            
            if validation["warnings"]:
                print(f"   ⚠️ 경고: {', '.join(validation['warnings'])}")
            
            if not validation["valid"]:
                print(f"   ❌ 오류: {', '.join(validation['errors'])}")
    
    async def demo_consistent_prompt_generation(self):
        """일관성 있는 프롬프트 생성 데모"""
        
        print("\n2️⃣ 일관성 있는 프롬프트 생성")
        print("-" * 40)
        
        room_layout = self.dify_generator.data_processor.parse_roombox_data(SAMPLE_ROOMBOX_DATA)
        
        # 다양한 스타일로 프롬프트 생성
        styles = ["modern", "scandinavian", "industrial"]
        
        for style in styles:
            print(f"\n🎨 {style.upper()} 스타일 프롬프트:")
            print("-" * 30)
            
            prompt = self.dify_generator.style_generator.generate_consistent_prompt(
                room_layout, style
            )
            
            # 프롬프트 미리보기 (축약)
            preview = prompt[:300] + "..." if len(prompt) > 300 else prompt
            print(preview)
    
    async def demo_realtime_coordinate_sync(self):
        """실시간 좌표 동기화 데모"""
        
        print("\n3️⃣ 실시간 좌표 동기화 시뮬레이션")
        print("-" * 40)
        
        # 세션 시작
        session_id = "demo_session_001"
        result = await self.realtime_sync.start_session(session_id, SAMPLE_ROOMBOX_DATA)
        
        if result["success"]:
            print(f"✅ 세션 시작: {session_id}")
            print(f"   초기 해시: {result['layout_hash']}")
        else:
            print(f"❌ 세션 시작 실패: {result['error']}")
            return
        
        # 좌표 변경 시뮬레이션
        current_data = json.loads(json.dumps(SAMPLE_ROOMBOX_DATA))  # 깊은 복사
        
        for i, change in enumerate(COORDINATE_CHANGES):
            print(f"\n🔄 변경 {i+1}: {change['description']}")
            
            # 데이터 변경 적용
            for path, new_value in change["changes"].items():
                self._apply_data_change(current_data, path, new_value)
            
            # 좌표 동기화
            sync_result = await self.realtime_sync.update_coordinates(session_id, current_data)
            
            if sync_result["success"]:
                if sync_result["changed"]:
                    print(f"   ✅ 동기화 완료 - 새 해시: {sync_result['layout_hash']}")
                    
                    if sync_result["validation"] and not sync_result["validation"]["valid"]:
                        print(f"   ⚠️ 검증 실패: {sync_result['validation']['errors']}")
                else:
                    print("   ℹ️ 변경 없음")
            else:
                print(f"   ❌ 동기화 실패: {sync_result['error']}")
            
            await asyncio.sleep(1)  # 1초 대기
        
        # 세션 종료
        await self.realtime_sync.close_session(session_id)
        print(f"\n🔌 세션 종료: {session_id}")
    
    async def demo_image_generation(self):
        """이미지 생성 데모"""
        
        print("\n4️⃣ 일관성 있는 이미지 생성 데모")
        print("-" * 40)
        
        # 이미지 생성 (Mock)
        result = await self.dify_generator.generate_consistent_room_image(
            SAMPLE_ROOMBOX_DATA,
            style="scandinavian",
            user_id="demo_user"
        )
        
        if result["success"]:
            print(f"✅ 이미지 생성 성공!")
            print(f"   파일 경로: {result['image_path']}")
            print(f"   스타일: {result['style']}")
            print(f"   생성 방법: {result['method']}")
            print(f"   서비스: {result.get('service', 'unknown')}")
        else:
            print(f"❌ 이미지 생성 실패: {result['error']}")
        
        # 피드백 시뮬레이션
        if result["success"]:
            print(f"\n📝 피드백 시뮬레이션")
            
            feedback_result = await self.dify_generator.learn_from_feedback(
                SAMPLE_ROOMBOX_DATA,
                result["image_path"],
                user_rating=4.5,
                style="scandinavian",
                comments="아늑하고 깔끔한 느낌이 좋습니다!"
            )
            
            if feedback_result["learned"]:
                print(f"✅ 학습 완료: {feedback_result['message']}")
            else:
                print(f"ℹ️ 학습 안됨: {feedback_result['message']}")
    
    def _apply_data_change(self, data, path, new_value):
        """데이터 경로에 새 값 적용"""
        
        keys = path.split('.')
        current = data
        
        # 마지막 키까지 탐색
        for key in keys[:-1]:
            if '[' in key and ']' in key:
                # 배열 인덱스 처리
                array_key = key.split('[')[0]
                index = int(key.split('[')[1].split(']')[0])
                current = current[array_key][index]
            else:
                current = current[key]
        
        # 마지막 값 설정
        last_key = keys[-1]
        if '[' in last_key and ']' in last_key:
            array_key = last_key.split('[')[0]
            index = int(last_key.split('[')[1].split(']')[0])
            current[array_key][index] = new_value
        else:
            current[last_key] = new_value
    
    async def run_full_demo(self):
        """전체 데모 실행"""
        
        print("🚀 RoomBox.jsx ↔ Dify 통합 시스템 데모 시작")
        print("🔗 좌표 일관성 보장 및 실시간 동기화")
        
        try:
            await self.demo_coordinate_processing()
            await self.demo_consistent_prompt_generation()
            await self.demo_realtime_coordinate_sync()
            await self.demo_image_generation()
            
            print("\n" + "="*60)
            print("✅ 전체 데모 완료!")
            print("📊 성능 및 일관성 분석:")
            print("   - 좌표 검증: 자동 보정 기능 포함")
            print("   - 프롬프트 생성: 스타일별 일관성 보장")  
            print("   - 실시간 동기화: 변경 감지 및 자동 동기화")
            print("   - 이미지 생성: Dify RAG 최적화 적용")
            print("   - 피드백 학습: 고품질 결과 자동 학습")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ 데모 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """메인 실행 함수"""
    
    print("🎯 RoomBox.jsx & Dify 통합 시스템")
    print("   좌표 일관성 보장 및 실시간 동기화 데모")
    print("")
    
    demo = IntegrationDemo()
    await demo.run_full_demo()


def cli_main():
    """CLI 진입점 (uv run demo)"""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()