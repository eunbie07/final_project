# 1. 필요한 라이브러리들을 불러옵니다.
import os
from google import genai
from google.genai.types import Image, StyleReferenceImage, StyleReferenceConfig, EditImageConfig
from PIL import Image as PIL_Image
import json

# ----------------------------------------------------------------------
# !!! 1. 기본 설정 (한 번만 설정하면 됩니다) !!!
# ----------------------------------------------------------------------
PROJECT_ID = "virtual-muse-466706-v2"
LOCATION = "us-central1"

# 스타일의 기준이 될 '대표 이미지' 4개의 GCS 경로
# 이 이미지들의 스타일을 따라 새로운 방이 그려집니다.
STYLE_REFERENCE_GCS_URIS = [
    "gs://my-room-finetune-bucket/image/7a1b2c3d4e5f6a7b8c9d0e1f.png",
    "gs://my-room-finetune-bucket/image/8b2c3d4e5f6a7b8c9d0e1f2a.png",
    "gs://my-room-finetune-bucket/image/9c3d4e5f6a7b8c9d0e1f2a3b.png"
]

# 결과 이미지를 저장할 폴더 이름
OUTPUT_DIR = "generated_rooms"
# ----------------------------------------------------------------------


# Vertex AI 클라이언트 초기화
print("Vertex AI 클라이언트를 초기화합니다...")
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
customization_model = "imagen-3.0-capability-001"
print("초기화 완료.")

# 스타일 참조 이미지 객체 미리 준비
style_reference_images = []
for i, gcs_uri in enumerate(STYLE_REFERENCE_GCS_URIS):
    style_image = Image(gcs_uri=gcs_uri)
    style_ref = StyleReferenceImage(
        reference_id=i + 1,
        reference_image=style_image,
        config=StyleReferenceConfig(style_description="a clean, modern Korean-style room with minimalist furniture"),
    )
    style_reference_images.append(style_ref)
print(f"{len(style_reference_images)}개의 스타일 참조 이미지 준비 완료.")


def json_to_prompt(json_data):
    """
    입력된 JSON 데이터를 분석하여 상세한 영어 프롬프트로 변환하는 함수.
    """
    scene = json_data[0]['scene']
    room = scene['room']
    objects = scene['objects']

    # 방 크기 설명 (mm -> m 변환)
    prompt = (f"A wide-angle side view of a room, meticulously arranged, reflecting a modern Korean residential sensibility. "
              f"The room's dimensions are {room['width']/1000:.1f}m in width, {room['depth']/1000:.1f}m in depth, and {room['height']/1000:.1f}m in height. "
              f"The floor has a linoleum texture, and the walls have white simplified wallpaper. ")

    # 창문, 문 등 구조물 설명
    structures_prompts = []
    for obj in objects:
        if obj['type'] in ['window', 'door']:
            pos = obj['position']
            dim = obj['dimensions']
            structures_prompts.append(
                f"On wall_{obj['wall']}, there is a {obj['name']} {dim['width']/1000:.1f}m wide and {dim['height']/1000:.1f}m high, "
                f"positioned at (x={pos['x']/1000:.2f}m, y={pos['y']/1000:.2f}m)."
            )
    if structures_prompts:
        prompt += " ".join(structures_prompts)

    # 가구 설명
    furniture_prompts = []
    for obj in objects:
        if obj['type'] == 'furniture':
            pos = obj['position']['center']
            dim = obj['dimensions']
            furniture_prompts.append(
                f"A rectangular {obj['name']} ({dim['width']/1000:.1f}m x {dim['depth']/1000:.1f}m x {dim['height']/1000:.1f}m) "
                f"is centered at (x={pos['x']/1000:.2f}m, y={pos['y']/1000:.2f}m) and rotated {obj['rotation_z']} degrees."
            )
    if furniture_prompts:
        prompt += " The furniture is precisely placed: " + " ".join(furniture_prompts)

    # 마무리 문구
    style_ids = "".join([f"[{i+1}]" for i in range(len(style_reference_images))])
    prompt += (f" The overall aesthetic is clean, uncluttered, and neat. "
               f"Render it in the realistic style of {style_ids}.")
    
    return prompt


def create_image_from_json(json_data, output_filename="new_room.png"):
    """
    JSON 데이터를 입력받아 최종 이미지를 생성하고 파일로 저장하는 메인 함수.
    """
    # 1단계: JSON을 프롬프트로 번역
    print("\n--- 1단계: JSON을 프롬프트로 번역 중 ---")
    prompt = json_to_prompt(json_data)
    print("생성된 프롬프트:", prompt[:150] + "...")

    # 2단계: 프롬프트와 스타일 참조로 이미지 생성
    print("\n--- 2단계: 이미지 생성 요청 중 ---")
    response = client.models.edit_image(
        model=customization_model,
        prompt=prompt,
        reference_images=style_reference_images,
        config=EditImageConfig(
            edit_mode="EDIT_MODE_DEFAULT",
            number_of_images=1,
            seed=1, # 일관된 결과를 위해 시드 고정
            safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
            person_generation="ALLOW_ADULT",
        ),
    )
    
    # 3단계: 결과 이미지 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_path = os.path.join(OUTPUT_DIR, output_filename)
    pil_image = response.generated_images[0].image._pil_image
    pil_image.save(full_path)
    print(f"\n성공! 생성된 이미지를 '{full_path}' 경로에 저장했습니다.")
    return full_path


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    # !!! 2. 사용 예시 !!!
    # 여기에 사용자가 웹에서 생성한 새로운 JSON 데이터를 넣으세요.
    # ----------------------------------------------------------------------
    new_user_json_string = """
    [
        {
            "_id": { "$oid": "new_user_room_001" },
            "scene": {
                "description": "A new 3.0m x 4.0m room.",
                "walls": {},
                "room": { "width": 3000, "depth": 4000, "height": 2400 },
                "objects": [
                    {
                        "type": "door", "name": "entry_door", "wall": 4, "dimensions": { "width": 900, "height": 2100 },
                        "position": { "x": 0, "y": 3550, "z": 0 }, "rotation_z": 0
                    },
                    {
                        "type": "furniture", "name": "bed", "shape": "rectangle",
                        "position": { "center": { "x": 1500, "y": 1000, "z": 0 } },
                        "dimensions": { "width": 1400, "depth": 2000, "height": 500 }, "rotation_z": 0
                    }
                ]
            }
        }
    ]
    """
    
    # JSON 문자열을 파이썬 객체로 변환
    new_room_data = json.loads(new_user_json_string)
    
    # 메인 함수 호출하여 이미지 생성
    create_image_from_json(new_room_data, output_filename="new_user_room_001.png")

