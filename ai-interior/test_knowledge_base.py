"""
Knowledge Base 연결 테스트
"""

from dify_rag import DifyLayoutRAG
import os
from dotenv import load_dotenv

load_dotenv()

def test_knowledge_base():
    print("=== Dify Knowledge Base 연결 테스트 ===")
    
    # Dify RAG 초기화
    try:
        dify_rag = DifyLayoutRAG(
            os.getenv("DIFY_API_KEY"),
            os.getenv("DIFY_APP_ID"), 
            os.getenv("DIFY_DATASET_ID")
        )
        print("OK: DifyLayoutRAG 초기화 성공")
    except Exception as e:
        print(f"ERROR: DifyLayoutRAG 초기화 실패: {e}")
        return
    
    # 테스트 방 데이터
    test_room = {
        'dimensions': {'width_cm': 391.3, 'depth_cm': 455.6, 'height_cm': 230}, 
        'furniture_3d': [{'name': 'Bed', 'type': 'bed', 'position': [196.1, 30, 218.5], 'rotation': [0, 0, 0]}]
    }
    
    # 1. 공간 임베딩 테스트
    print("\n1. 공간 임베딩 생성 테스트...")
    try:
        spatial_embedding = dify_rag.create_spatial_embedding(test_room)
        print(f"OK: 공간 임베딩 생성 완료 ({len(spatial_embedding)} 문자)")
        print("생성된 임베딩 (처음 200자):")
        print(spatial_embedding[:200] + "...")
    except Exception as e:
        print(f"ERROR: 공간 임베딩 실패: {e}")
        return
    
    # 2. 유사 레이아웃 검색 테스트  
    print("\n2. Knowledge Base 검색 테스트...")
    try:
        similar_layouts = dify_rag.find_similar_layouts(test_room)
        if similar_layouts:
            print("OK: 유사 레이아웃 발견!")
            print(f"검색 결과: {similar_layouts}")
        else:
            print("INFO: 유사 레이아웃 없음 (Knowledge Base가 비어있거나 API 연결 실패)")
    except Exception as e:
        print(f"ERROR: Knowledge Base 검색 실패: {e}")
    
    # 3. 성공 사례 추가 테스트
    print("\n3. 성공 사례 추가 테스트...")
    try:
        success = dify_rag.add_successful_layout(
            room_data=test_room,
            user_rating=4.5,
            image_path="test_success_case.png"
        )
        if success:
            print("SUCCESS: 성공 사례가 Knowledge Base에 추가됨!")
        else:
            print("FAILED: Knowledge Base 추가 실패 (API 키 확인 필요)")
    except Exception as e:
        print(f"ERROR: 성공 사례 추가 실패: {e}")
    
    print("\n=== 테스트 완료 ===")
    print("만약 실패했다면:")
    print("1. Dify Knowledge Base → API 탭에서 Dataset API Key 복사")  
    print("2. .env 파일의 DIFY_API_KEY를 Dataset API Key로 교체")
    print("3. 다시 테스트 실행")

if __name__ == "__main__":
    test_knowledge_base()