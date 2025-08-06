import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def generate_furniture_image_and_save(prompt, save_fname):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url

    # 이미지 저장 디렉토리 생성(없으면 자동 생성)
    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    img_data = requests.get(image_url).content
    with open(save_fname, "wb") as f:
        f.write(img_data)
    print(f"이미지 생성 완료: {save_fname}")
    return image_url

if __name__ == "__main__":
    prompt = "Scandinavian style dining table, natural oak, realistic, top view, clean studio lighting, minimalist background, high detail"
    save_filename = os.path.join("dummy", "2.jpg")
    image_url = generate_furniture_image_and_save(prompt, save_filename)
    print(f"이미지 원본 URL: {image_url}")
