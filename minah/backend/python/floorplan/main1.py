import json
import base64
import re
import os

json_path = os.path.join(os.getcwd(), 'floorplan_json', '84C.json')

# 저장할 폴더 및 파일 경로 지정
output_dir = os.path.join(os.getcwd(), 'floorplan_image') 
output_png = os.path.join(output_dir, '84C.png')

# 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# JSON 파일 열기 및 전체 텍스트 읽기
with open(json_path, 'r', encoding='utf-8') as f:
    json_text = f.read()

# "data":" 다음부터 마지막 따옴표(") 또는 작은따옴표(') 전까지 추출 (줄바꿈 포함)
match = re.search(r'"data"\s*:\s*["\']([^"\']+)["\']', json_text)
if not match:
    raise ValueError('"data":" 다음에 base64 문자열을 찾을 수 없습니다.')

base64_data = match.group(1)

# base64 디코딩 및 이미지 파일로 저장
with open(output_png, 'wb') as img_file:
    img_file.write(base64.b64decode(base64_data))

print(f'이미지 저장 완료: {output_png}')
