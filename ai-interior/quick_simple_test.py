"""
간단한 통합 테스트
"""

import asyncio
import json
from datetime import datetime

def test_imports():
    """모든 모듈 import 테스트"""
    print("=== 모듈 Import 테스트 ===")
    
    try:
        from dify_rag import DifyLayoutRAG
        print("OK: dify_rag 모듈 import 성공")
    except Exception as e:
        print(f"ERROR: dify_rag import 실패: {e}")
    
    try:
        from roombox_integration import RoomBoxDataProcessor
        print("OK: roombox_integration 모듈 import 성공")
    except Exception as e:
        print(f"ERROR: roombox_integration import 실패: {e}")
    
    try:
        from enhanced_vertex_generator import EnhancedVertexGenerator
        print("OK: enhanced_vertex_generator 모듈 import 성공")
    except Exception as e:
        print(f"ERROR: enhanced_vertex_generator import 실패: {e}")
    
    try:
        from mongodb_integration import MongoDBRoomProcessor
        print("OK: mongodb_integration 모듈 import 성공")
    except Exception as e:
        print(f"ERROR: mongodb_integration import 실패: {e}")

def test_config():
    """환경 설정 테스트"""
    print("\n=== 환경 설정 테스트 ===")
    
    import os
    
    # Dify 설정 확인
    dify_api_key = os.getenv("DIFY_API_KEY")
    dify_app_id = os.getenv("DIFY_APP_ID") 
    dify_dataset_id = os.getenv("DIFY_DATASET_ID")
    
    print(f"DIFY_API_KEY: {'설정됨' if dify_api_key else '없음'}")
    print(f"DIFY_APP_ID: {'설정됨' if dify_app_id else '없음'}")
    print(f"DIFY_DATASET_ID: {'설정됨' if dify_dataset_id else '없음'}")
    
    # Google Cloud 설정 확인
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    google_api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {'설정됨' if google_creds else '없음'}")
    print(f"GOOGLE_AI_API_KEY: {'설정됨' if google_api_key else '없음'}")

async def main():
    """전체 테스트 실행"""
    print("간단한 통합 테스트 시작")
    print("=" * 50)
    
    # 1. Import 테스트
    test_imports()
    
    # 2. 환경 설정 테스트
    test_config()
    
    print("\n" + "=" * 50)
    print("간단한 통합 테스트 완료")

if __name__ == "__main__":
    asyncio.run(main())