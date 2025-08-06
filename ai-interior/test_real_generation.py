"""
실제 Vertex AI 이미지 생성 테스트 (단순 버전)
"""

import os
import asyncio
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

async def generate_single_image():
    """단순한 이미지 생성 테스트"""
    
    print("실제 Vertex AI 이미지 생성 테스트...")
    
    try:
        # 인증 설정
        key_path = os.path.join(os.path.dirname(__file__), "key.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        
        # Vertex AI 초기화
        vertexai.init(project="virtual-muse-466706-v2", location="us-central1")
        
        # 모델 로드
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        print("모델 로드 완료")
        
        # 간단한 프롬프트
        prompt = """
        A modern Korean interior room design:
        - Clean white walls and light wood flooring
        - A grey sofa in the center of the room  
        - A wooden coffee table in front of the sofa
        - Large windows with natural lighting
        - Minimalist and cozy atmosphere
        - Professional interior photography, 8K quality
        """
        
        print("이미지 생성 중... (30초 정도 소요)")
        
        # 비동기로 실행
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="block_some",
            person_generation="dont_allow"
        ))
        
        if response.images:
            # 이미지 저장
            os.makedirs("generated_images", exist_ok=True)
            filename = "vertex_test_real.png"
            filepath = os.path.join("generated_images", filename)
            
            response.images[0].save(location=filepath, include_generation_parameters=False)
            
            print(f"SUCCESS: 이미지 생성 완료!")
            print(f"파일 위치: {filepath}")
            print("이제 generated_images 폴더에서 이미지를 확인하세요!")
            
            return True
        else:
            print("FAILED: 이미지가 생성되지 않았습니다")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=== 실제 Vertex AI 이미지 생성 테스트 ===")
    success = asyncio.run(generate_single_image())
    
    if success:
        print("\n" + "="*50)
        print("완벽! Google Vertex AI로 실제 이미지 생성 성공!")
        print("이제 RoomBox 데이터로 일관성 있는 인테리어 이미지를 만들 수 있습니다!")
    else:
        print("\n이미지 생성 실패. 오류를 확인해주세요.")