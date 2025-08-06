import json
import base64
import os

# 59A.json 파일 경로 지정
json_path = os.path.join(os.getcwd(), 'floorplan_json', '59A.json')

# 저장할 폴더 및 파일 경로 지정
output_dir = os.path.join(os.getcwd(), 'floorplan_image') 
output_png = os.path.join(output_dir, '59A.png')

# 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# JSON 파일 열기
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# base64 문자열 추출 (구조에 따라 경로 수정 필요)
if 'data' in data:
    base64_str = data['data']
elif 'image' in data and '@attributes' in data['image'] and 'xlink:href' in data['image']['@attributes']:
    base64_str = data['image']['@attributes']['xlink:href']
else:
    base64_str = None
    for value in data.values():
        if isinstance(value, str) and 'base64,' in value:
            base64_str = value
            break
    if base64_str is None:
        raise ValueError('Base64 문자열을 찾을 수 없습니다.')

# 'base64,' 이후 부분만 추출
if 'base64,' in base64_str:
    base64_data = base64_str.split('base64,')[1]
else:
    base64_data = base64_str

# 디코딩 및 PNG 파일로 저장
with open(output_png, 'wb') as img_file:
    img_file.write(base64.b64decode(base64_data))

print(f'이미지 저장 완료: {output_png}')
