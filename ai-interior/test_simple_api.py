"""
Dify Dataset API 직접 테스트
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_dataset_api():
    api_key = os.getenv("DIFY_API_KEY")
    dataset_id = os.getenv("DIFY_DATASET_ID")
    
    print(f"API Key: {api_key}")
    print(f"Dataset ID: {dataset_id}")
    
    # 1. Dataset 목록 조회 테스트
    print("\n=== Dataset 목록 조회 ===")
    try:
        response = requests.get(
            "https://api.dify.ai/v1/datasets",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Dataset 개수: {len(data.get('data', []))}")
            for dataset in data.get('data', []):
                print(f"- {dataset.get('name')} ({dataset.get('id')})")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")
    
    # 2. 특정 Dataset의 Documents 조회
    print("\n=== Dataset Documents 조회 ===")
    try:
        response = requests.get(
            f"https://api.dify.ai/v1/datasets/{dataset_id}/documents",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Document 개수: {len(data.get('data', []))}")
            for doc in data.get('data', []):
                print(f"- {doc.get('name')} (ID: {doc.get('id')})")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_dataset_api()