import json

# JSON 키 파일의 경로
key_file_path = 'key.json'

try:
    with open(key_file_path, 'r', encoding='utf-8') as f:
        # JSON 파일 내용을 파이썬 딕셔너리로 로드
        service_account_info = json.load(f)

    print("JSON 키 파일을 성공적으로 로드했습니다.")
    # 필요한 정보에 접근 (예시)
    print(f"Project ID: {service_account_info.get('project_id')}")
    print(f"Private Key ID: {service_account_info.get('private_key_id')}")
    # print(f"Private Key: {service_account_info.get('private_key')}") # 보안상 출력하지 않는 것이 좋습니다.

    # 이제 service_account_info 딕셔너리를 사용하여 Google Cloud 클라이언트 라이브러리를 초기화할 수 있습니다.
    # 예:
    # from google.cloud import storage
    # from google.oauth2 import service_account
    # credentials = service_account.Credentials.from_service_account_info(service_account_info)
    # client = storage.Client(credentials=credentials, project=service_account_info.get('project_id'))

except FileNotFoundError:
    print(f"오류: 키 파일 '{key_file_path}'를 찾을 수 없습니다. 경로를 확인해주세요.")
except json.JSONDecodeError:
    print(f"오류: '{key_file_path}' 파일이 유효한 JSON 형식이 아닙니다.")
except Exception as e:
    print(f"예상치 못한 오류 발생: {e}")