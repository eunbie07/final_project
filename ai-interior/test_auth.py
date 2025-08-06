"""
Google Cloud 인증 테스트
"""

import os
import vertexai

def test_authentication():
    """Google Cloud 인증 테스트"""
    
    print("Google Cloud 인증 테스트...")
    
    # 키 파일 확인
    key_path = os.path.join(os.path.dirname(__file__), "key.json")
    print(f"키 파일 경로: {key_path}")
    print(f"키 파일 존재: {os.path.exists(key_path)}")
    
    if os.path.exists(key_path):
        # 환경 변수 설정
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        print("환경 변수 설정 완료")
        
        try:
            # Vertex AI 초기화 테스트
            vertexai.init(project="virtual-muse-466706-v2", location="us-central1")
            print("Vertex AI 초기화 성공!")
            
            # 모델 로드 테스트
            from vertexai.preview.vision_models import ImageGenerationModel
            model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            print("이미지 생성 모델 로드 성공!")
            
            return True
            
        except Exception as e:
            print(f"Vertex AI 초기화 실패: {e}")
            return False
    else:
        print("키 파일이 없습니다.")
        return False

if __name__ == "__main__":
    success = test_authentication()
    if success:
        print("\n모든 인증 테스트 통과! 이제 실제 이미지 생성이 가능합니다.")
    else:
        print("\n인증 실패. 키 파일을 확인해주세요.")