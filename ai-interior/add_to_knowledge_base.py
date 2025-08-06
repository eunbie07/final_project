"""
Dify 지식베이스에 성공 사례를 수동으로 추가하는 스크립트
"""

import asyncio
from dify_rag import DifyLayoutRAG
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Dify RAG 초기화
    dify_rag = DifyLayoutRAG(
        os.getenv("DIFY_API_KEY"),
        os.getenv("DIFY_APP_ID"),
        os.getenv("DIFY_DATASET_ID")
    )
    
    # 현재 테스트 중인 방 데이터 (중앙 침대 배치)
    sample_room_data = {
        'dimensions': {'width_cm': 391.3, 'depth_cm': 455.6, 'height_cm': 230}, 
        'furniture_3d': [
            {
                'name': 'Bed', 
                'type': 'bed', 
                'position': [196.08993750000002, 30, 218.45600000000002], 
                'rotation': [0, 0, 0]
            }
        ]
    }
    
    print("현재 방 데이터로 공간 임베딩 생성 중...")
    spatial_embedding = dify_rag.create_spatial_embedding(sample_room_data)
    print("생성된 공간 임베딩:")
    print("-" * 50)
    try:
        print(spatial_embedding.encode('utf-8').decode('utf-8'))
    except:
        print("공간 임베딩 생성 완료 (인코딩 이슈로 표시 생략)")
    print("-" * 50)
    
    # 성공 사례로 추가 (가상의 높은 점수)
    print("\n지식베이스에 성공 사례 추가 중...")
    success = dify_rag.add_successful_layout(
        room_data=sample_room_data,
        user_rating=4.5,  # 높은 평점
        image_path="vertex_industrial_20250807_030702.png"
    )
    
    if success:
        print("SUCCESS: 성공 사례가 지식베이스에 추가되었습니다!")
    else:
        print("FAILED: 지식베이스 추가 실패")
    
    # 추가 성공 사례들 (다양한 중앙 배치)
    additional_cases = [
        {
            'dimensions': {'width_cm': 400, 'depth_cm': 500, 'height_cm': 280},
            'furniture_3d': [{'name': 'Sofa', 'type': 'sofa', 'position': [200, 25, 250], 'rotation': [0, 0, 0]}]
        },
        {
            'dimensions': {'width_cm': 350, 'depth_cm': 420, 'height_cm': 260}, 
            'furniture_3d': [{'name': 'Desk', 'type': 'desk', 'position': [175, 35, 210], 'rotation': [0, 0, 0]}]
        }
    ]
    
    for i, case in enumerate(additional_cases, 1):
        print(f"\n추가 사례 {i} 추가 중...")
        success = dify_rag.add_successful_layout(
            room_data=case,
            user_rating=4.2,
            image_path=f"success_case_{i}.png"
        )
        if success:
            print(f"SUCCESS: 추가 사례 {i} 성공!")
        else:
            print(f"FAILED: 추가 사례 {i} 실패")

if __name__ == "__main__":
    asyncio.run(main())