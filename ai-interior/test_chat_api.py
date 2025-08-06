"""
Dify Chat API로 Knowledge Base 연결 테스트
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_chat_with_knowledge():
    api_key = os.getenv("DIFY_API_KEY")
    app_id = os.getenv("DIFY_APP_ID")
    
    print(f"API Key: {api_key}")
    print(f"App ID: {app_id}")
    
    # Chat API로 Knowledge Base 활용 테스트
    print("\n=== Chat API Knowledge Base 테스트 ===")
    
    test_query = "391cm x 455cm 방에서 침대를 중앙에 배치하는 방법을 알려줘"
    
    try:
        response = requests.post(
            "https://api.dify.ai/v1/chat-messages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": {},
                "query": test_query,
                "response_mode": "streaming",
                "user": "test_user"
            }
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("\nSUCCESS: Chat API 연결 성공!")
            response_text = response.text
            print(f"응답 길이: {len(response_text)} 문자")
            
            # 첫 500문자만 출력 (인코딩 문제 방지)
            try:
                print("\n응답 내용 (처음 500자):")
                print(response_text[:500])
            except:
                print("응답 내용 출력 실패 (인코딩 문제)")
            
            # Knowledge Base 활용 여부 확인
            response_lower = response_text.lower()
            if any(word in response_lower for word in ["knowledge", "database", "coordinates", "layout", "success", "case"]):
                print("\nKnowledge Base가 활용된 것 같습니다!")
            else:
                print("\nKnowledge Base 활용이 불분명합니다.")
                
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_chat_with_knowledge()