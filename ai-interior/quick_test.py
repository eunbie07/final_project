"""
빠른 연결 테스트
"""
import os
from config import load_config, print_config_status

def test_connection():
    """Dify 연결 테스트"""
    print("Dify 연결 테스트 시작...")
    
    try:
        # 설정 로드
        config = load_config()
        print_config_status(config)
        
        # API 키 확인
        if config.api_key and len(config.api_key) > 10:
            print("API 키 설정 확인됨")
        else:
            print("API 키가 올바르지 않습니다")
            return False
            
        if config.app_id:
            print("App ID 설정 확인됨")
        else:
            print("App ID가 설정되지 않았습니다")
            return False
            
        print("기본 설정 완료! 이제 실제 테스트를 실행할 수 있습니다.")
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print(".env 파일의 API 키와 App ID를 확인해주세요")
        return False

if __name__ == "__main__":
    test_connection()