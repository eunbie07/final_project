"""
기본 통합 테스트 (ASCII only)
"""

import asyncio
import os
from pathlib import Path

def test_imports():
    """모든 모듈 import 테스트"""
    print("=== Module Import Test ===")
    
    try:
        from dify_rag import DifyLayoutRAG
        print("OK: dify_rag module imported successfully")
    except Exception as e:
        print(f"ERROR: dify_rag import failed: {e}")
    
    try:
        from roombox_integration import RoomBoxDataProcessor
        print("OK: roombox_integration module imported successfully")
    except Exception as e:
        print(f"ERROR: roombox_integration import failed: {e}")
    
    try:
        from enhanced_vertex_generator import EnhancedVertexGenerator
        print("OK: enhanced_vertex_generator module imported successfully")
    except Exception as e:
        print(f"ERROR: enhanced_vertex_generator import failed: {e}")

def test_config():
    """환경 설정 테스트"""
    print("\n=== Environment Config Test ===")
    
    # .env 파일 로드
    from dotenv import load_dotenv
    load_dotenv()
    
    # Dify 설정 확인
    dify_api_key = os.getenv("DIFY_API_KEY")
    dify_app_id = os.getenv("DIFY_APP_ID") 
    dify_dataset_id = os.getenv("DIFY_DATASET_ID")
    
    print(f"DIFY_API_KEY: {'SET' if dify_api_key else 'NOT SET'}")
    print(f"DIFY_APP_ID: {'SET' if dify_app_id else 'NOT SET'}")
    print(f"DIFY_DATASET_ID: {'SET' if dify_dataset_id else 'NOT SET'}")
    
    # Google Cloud 설정 확인
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {'SET' if google_creds else 'NOT SET'}")

def test_image_folder():
    """이미지 폴더 확인"""
    print("\n=== Image Folder Test ===")
    
    image_dir = Path("image")
    
    if image_dir.exists():
        image_files = list(image_dir.glob("*.png"))
        print(f"OK: Image folder found with {len(image_files)} files")
        
        # 컨셉별 분류
        concepts = {}
        for img_file in image_files:
            filename_lower = img_file.name.lower()
            
            if "modern" in filename_lower or "minimalist" in filename_lower:
                concept = "Modern,Minimalist"
            elif "scandinavian" in filename_lower:
                concept = "Scandinavian"
            elif "industrial" in filename_lower:
                concept = "Industrial"
            elif "bohemian" in filename_lower or "natural" in filename_lower:
                concept = "Bohemian,Natural"
            elif "cozy" in filename_lower:
                concept = "Cozy"
            else:
                concept = "Unknown"
            
            if concept not in concepts:
                concepts[concept] = 0
            concepts[concept] += 1
        
        print("   Concept distribution:")
        for concept, count in concepts.items():
            print(f"     - {concept}: {count} images")
    else:
        print("ERROR: Image folder not found")

def test_dify_basic():
    """기본 Dify 연결 테스트"""
    print("\n=== Basic Dify Test ===")
    
    try:
        from dify_rag import DifyLayoutRAG
        
        dify_rag = DifyLayoutRAG(
            os.getenv("DIFY_API_KEY", ""),
            os.getenv("DIFY_APP_ID", ""),
            os.getenv("DIFY_DATASET_ID", "")
        )
        
        # 간단한 텍스트 임베딩 테스트
        test_data = {
            "scene": {
                "room": {"width": 4000, "depth": 5000, "height": 2800},
                "objects": [
                    {
                        "type": "furniture",
                        "name": "test_bed",
                        "position": {"center": {"x": 2000, "y": 2500}},
                        "dimensions": {"width": 1600, "depth": 2000, "height": 600}
                    }
                ]
            }
        }
        
        test_text = dify_rag.create_spatial_embedding(test_data)
        
        if test_text and len(test_text) > 50:
            print("OK: Dify text embedding generated successfully")
            print(f"   Generated text length: {len(test_text)} characters")
        else:
            print("ERROR: Dify text embedding generation failed")
            
    except Exception as e:
        print(f"ERROR: Dify connection test failed: {e}")

async def test_generator_basic():
    """기본 생성기 테스트"""
    print("\n=== Basic Generator Test ===")
    
    try:
        from enhanced_vertex_generator import EnhancedVertexGenerator
        
        generator = EnhancedVertexGenerator()
        
        # Mock 이미지 생성 테스트
        result = await generator._generate_mock_image("modern", "test_image.png")
        
        if result.get("success"):
            print("OK: Image generation system initialized successfully")
            print(f"   Test result method: {result['method']}")
        else:
            print("ERROR: Image generation system initialization failed")
            
    except Exception as e:
        print(f"ERROR: Generator test failed: {e}")

async def main():
    """전체 테스트 실행"""
    print("Basic Integration Test Started")
    print("=" * 50)
    
    # 1. Import 테스트
    test_imports()
    
    # 2. 환경 설정 테스트
    test_config()
    
    # 3. 이미지 폴더 테스트
    test_image_folder()
    
    # 4. Dify 기본 테스트
    test_dify_basic()
    
    # 5. 생성기 기본 테스트
    await test_generator_basic()
    
    print("\n" + "=" * 50)
    print("Basic Integration Test Completed")
    print("\nNext Steps:")
    print("1. Fix any ERROR items shown above")
    print("2. Run: uv run python image_embedding_processor.py")
    print("3. Run: uv run python mongodb_integration.py")

if __name__ == "__main__":
    asyncio.run(main())