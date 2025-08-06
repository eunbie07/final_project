# .env 파일 예시:
# OPENAI_API_KEY=sk-xxxxxxx
# AWS_ACCESS_KEY_ID=AKIAxxxxxx
# AWS_SECRET_ACCESS_KEY=xxxxxxxx
# S3_BUCKET=your-bucket-name

import os
import uuid
import json
import requests
import boto3
import pymysql
from dotenv import load_dotenv
from openai import OpenAI

# 1. 환경변수(.env) 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')

# 2. MySQL 접속 정보
MYSQL_CONF = {
    "host": "13.236.16.220",
    "port": 3306,
    "user": "root",
    "password": "1234",
    "database": "furniture_db"
}

def generate_furniture_image(prompt):
    client = OpenAI()
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    return response.data[0].url

def upload_image_to_s3(image_url, category, subcategory, obj_id, view):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    img_data = requests.get(image_url).content
    s3_key = f'furniture/{category}/{subcategory}/{obj_id}/{view}.jpg'
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=img_data, ContentType='image/jpeg')
    return f'https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}'

def save_furniture_metadata(metadata_dict):
    connection = pymysql.connect(
        host=MYSQL_CONF["host"], port=MYSQL_CONF["port"],
        user=MYSQL_CONF["user"], password=MYSQL_CONF["password"],
        db=MYSQL_CONF["database"], charset='utf8mb4'
    )
    try:
        with connection.cursor() as cursor:
            sql = """
            INSERT INTO furniture 
            (id, name, category, width, depth, height, form, image_url, restriction, tags, notes) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                metadata_dict['id'],
                metadata_dict.get('name', ''),
                metadata_dict.get('category', ''),
                metadata_dict.get('width', 0),
                metadata_dict.get('depth', 0),
                metadata_dict.get('height', 0),
                metadata_dict.get('form', ''),
                json.dumps(metadata_dict.get('image_url', {})),
                json.dumps(metadata_dict.get('restriction', [])),
                json.dumps(metadata_dict.get('tags', [])),
                metadata_dict.get('notes', '')
            ))
            connection.commit()
    finally:
        connection.close()

def create_and_save_furniture_multi_view(prompts, category, subcategory, metadata):
    obj_id = str(uuid.uuid4())
    urls = {}
    for view, prompt in prompts.items():
        image_url = generate_furniture_image(prompt)
        s3_url = upload_image_to_s3(image_url, category, subcategory, obj_id, view)
        urls[view] = s3_url
    metadata = metadata.copy()
    metadata['id'] = obj_id
    metadata['image_url'] = urls  # {'front':..., 'top':..., 'side':...}
    metadata['category'] = category
    save_furniture_metadata(metadata)
    return obj_id, urls

if __name__ == "__main__":
    # 일부 코드만 수정: 책상 10종 각기 다른 프롬프트
    desks = [
        {
            "name": "월넛 모던 직사각 책상",
            "prompts": {
                "front": "Modern walnut rectangular desk, metal legs, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, entire furniture piece fully visible, not cropped",
                "top":   "Modern walnut rectangular desk, metal legs, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object, desk fully visible, not cropped",
                "side":  "Modern walnut rectangular desk, metal legs, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object, entire desk visible"
            },
            "tags": ["월넛", "모던", "직사각", "책상"]
        },
        {
            "name": "화이트 미니멀 ㄷ자 책상",
            "prompts": {
                "front": "Minimalist white desk, U-shaped (ㄷ자), sleek metal frame, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, all of the desk shown",
                "top":   "Minimalist white desk, U-shaped, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object, desk fully visible",
                "side":  "Minimalist white desk, U-shaped, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object, desk not cropped"
            },
            "tags": ["화이트", "미니멀", "ㄷ자", "책상"]
        },
        {
            "name": "빈티지 원목 2인용 테이블책상",
            "prompts": {
                "front": "Vintage double wooden desk, for two people, drawers on both sides, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, whole desk fully visible",
                "top":   "Vintage double wooden desk, for two, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object, desk in full frame",
                "side":  "Vintage double wooden desk, for two, drawers, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object"
            },
            "tags": ["빈티지", "원목", "2인용", "책상"]
        },
        {
            "name": "블랙 인더스트리얼 컴퓨터책상",
            "prompts": {
                "front": "Industrial black computer desk, cable management hole, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, desk completely shown",
                "top":   "Industrial black computer desk, cable hole, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object",
                "side":  "Industrial black computer desk, cable management, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object"
            },
            "tags": ["블랙", "인더스트리얼", "컴퓨터책상", "책상"]
        },
        {
            "name": "라이트그레이 접이식 벽걸이 테이블",
            "prompts": {
                "front": "Light grey foldable wall-mounted desk, slim silhouette, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, full desk in image",
                "top":   "Light grey foldable wall desk, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object",
                "side":  "Light grey foldable wall desk, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object, slim profile visible"
            },
            "tags": ["라이트그레이", "접이식", "벽걸이", "책상"]
        },
        {
            "name": "네추럴 컬러 1200x600 기본형책상",
            "prompts": {
                "front": "Natural color standard desk, 1200x600mm, simple design, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, full desk shown",
                "top":   "Natural color simple desk, 1200x600mm, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object",
                "side":  "Natural color standard desk, 1200x600mm, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object"
            },
            "tags": ["네추럴", "심플", "1200x600", "책상"]
        },
        {
            "name": "북유럽 오픈선반 우드 데스크",
            "prompts": {
                "front": "Scandinavian wood desk with open shelves, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, full desk in view",
                "top":   "Scandinavian wood desk with shelves, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object, desk fully captured",
                "side":  "Scandinavian wood desk, open shelf, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object"
            },
            "tags": ["북유럽", "오픈선반", "우드", "책상"]
        },
        {
            "name": "스틸프레임 L자 컴퓨터데스크",
            "prompts": {
                "front": "Steel frame L-shaped computer desk, spacious corner desk, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, entire desk visible",
                "top":   "L-shaped steel frame computer desk, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object",
                "side":  "Steel frame L-shaped desk, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object"
            },
            "tags": ["L자", "스틸프레임", "코너", "책상"]
        },
        {
            "name": "글라스 상판 투명책상",
            "prompts": {
                "front": "Transparent glass top desk, minimalist frame, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, all of the desk shown",
                "top":   "Transparent glass desk, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object, desk in the frame",
                "side":  "Glass top desk, minimalist, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object"
            },
            "tags": ["글라스", "투명", "미니멀", "책상"]
        },
        {
            "name": "원형 북유럽 디자인 원목책상",
            "prompts": {
                "front": "Round Scandinavian wooden desk, natural tone, front view, white background, realistic, high quality, no text, no watermark, clear focus, single object, entire desk shown",
                "top":   "Round Scandinavian wood desk, top view, white background, realistic, high quality, no text, no watermark, clear focus, single object, desk fully visible",
                "side":  "Round Scandinavian wooden desk, side view, white background, realistic, high quality, no text, no watermark, clear focus, single object"
            },
            "tags": ["원형", "북유럽", "원목", "책상"]
        }
    ]

    category = "desk"
    subcategory = "standard_desk"

    for idx, desk in enumerate(desks, 1):
        metadata = {
            "name": desk["name"],
            "width": 1200 if idx != 6 else 1200,  # 필요시 사이즈 조정
            "depth": 600 if idx != 10 else 1200,  # 원형(지름) 예시 적용
            "height": 750,
            "form": "rectangle" if idx != 10 else "circle",
            "restriction": [],
            "tags": desk["tags"] + [f"유형{idx}"],
            "notes": f"OpenAI 3뷰 자동생성 책상 타입 {idx}"
        }
        obj_id, img_urls = create_and_save_furniture_multi_view(desk["prompts"], category, subcategory, metadata)
        print(f"[{idx}] 등록 완료: {obj_id}")
        print(f"S3 이미지(front): {img_urls['front']}")
        print(f"S3 이미지(top):   {img_urls['top']}")
        print(f"S3 이미지(side):  {img_urls['side']}\n")

