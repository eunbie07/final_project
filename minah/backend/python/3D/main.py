import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# .env 파일에서 환경 변수(API 키)를 로드합니다.
load_dotenv()

# OpenAI 클라이언트 초기화
try:
    client = OpenAI()
except Exception as e:
    print(f"❌ OpenAI 클라이언트 초기화 중 오류 발생: {e}")
    if "api_key" in str(e).lower():
        print("'.env' 파일에 OPENAI_API_KEY가 올바르게 설정되었는지 확인하세요.")
    exit()

# JSON의 모든 정보를 최대한 정확하게 반영한 프롬프트
prompt_text = """
아래 json 코드를 바탕으로 한국의 방이미지를 제작해줘.



{

    "scene": {

      "description": "A 3D scene definition for a room. All units are in millimeters. The origin (0,0,0) is the front-left-bottom corner. X-axis is width, Y-axis is depth, Z-axis is height.",

      "view_point": {

        "camera_position": { "x": 2000, "y": -1000, "z": 1700 },

        "look_at": { "x": 2000, "y": 2500, "z": 1200 },

        "style": "cozy, modern, bright, photorealistic, wide-angle lens"

      },

      "room": {

        "dimensions": {

          "width": 4000,

          "depth": 5000,

          "height": 2800

        },

        "walls": {

          "material": "light beige painted plaster"

        },

        "floor": {

          "material": "light oak wood planks"

        }

      },

      "objects": [

        {

          "type": "door",

          "name": "entrance_door",

          "dimensions": { "width": 900, "depth": 50, "height": 2100 },

          "position": { "x": 500, "y": 0, "z": 0 },

          "rotation_z": 0,

          "material": "white painted wood"

        },

        {

          "type": "window",

          "name": "main_window",

          "dimensions": { "width": 50, "depth": 2500, "height": 1500 },

          "position": { "x": 4000, "y": 1250, "z": 1000 },

          "rotation_z": 0,

          "details": "A window on the right wall (at x=4000), starting 1000mm from the floor. Covered with sheer white curtains, lets in soft daylight."

        },

        {

          "type": "furniture",

          "name": "double_bed",

          "dimensions": { "width": 1600, "depth": 2100, "height": 1000 },

          "position": { "x": 1200, "y": 5000, "z": 0 },

          "rotation_z": 180,

          "material": "dark wood frame with grey linen bedding, placed against the back wall."

        },

        {

          "type": "furniture",

          "name": "wardrobe",

          "dimensions": { "width": 600, "depth": 1800, "height": 2200 },

          "position": { "x": 0, "y": 1600, "z": 0 },

          "rotation_z": 0,

          "material": "white matte finish, placed against the left wall."

        },

        {

          "type": "furniture",

          "name": "desk",

          "dimensions": { "width": 600, "depth": 1200, "height": 750 },

          "position": { "x": 4000, "y": 1850, "z": 0 },

          "rotation_z": 0,

          "material": "light oak wood with black metal legs, placed against the right wall under the window."

        }

      ]

    }

  }
"""

print("🎨 이미지 생성을 요청합니다. 잠시만 기다려주세요...")

try:
    # OpenAI DALL-E 3 모델을 사용하여 이미지 생성 요청
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt_text,
        n=1,
        size="1792x1024", # 와이드 앵글 렌즈 느낌을 살리기 위해 가로가 긴 사이즈 선택
        quality="hd" # 더 선명한 이미지를 위해 hd 품질 선택
    )

    # 생성된 이미지의 URL 추출
    image_url = response.data[0].url
    print(f"✅ 이미지 생성 완료! URL: {image_url}")

    # 이미지 다운로드 및 저장
    print("🖼️ 이미지를 파일로 저장하는 중입니다...")
    image_response = requests.get(image_url)
    
    # HTTP 요청이 성공했는지 확인
    if image_response.status_code == 200:
        # 파일 이름에 현재 시간을 넣어 중복 방지
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"korean_room_{timestamp}.png"
        
        # 이미지를 바이너리 쓰기 모드로 파일에 저장
        with open(file_name, "wb") as f:
            f.write(image_response.content)
        
        print(f"🎉 이미지가 '{file_name}' 이름으로 성공적으로 저장되었습니다!")
    else:
        print(f"❌ 이미지 다운로드 실패. 상태 코드: {image_response.status_code}")

except Exception as e:
    print(f"❌ 작업 중 오류가 발생했습니다: {e}")