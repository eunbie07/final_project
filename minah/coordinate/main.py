import os
import json
from dotenv import load_dotenv
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image

# .env 파일 로드
load_dotenv()

# Vertex AI 프로젝트 설정
PROJECT_ID = "virtual-muse-466706-v2"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)


# ----------------- 여기가 완전히 새로워진 프롬프트 생성 로직입니다 -----------------

def describe_position(geometry, room_dims, threshold=300):
    """
    객체의 좌표와 방 크기를 기반으로 상대적 위치를 서술하는 헬퍼 함수.
    (threshold: 벽에 '근접'했다고 판단하는 거리 mm)
    """
    # 가구의 geometry 구조가 없을 경우를 대비
    if 'position' not in geometry:
        return "somewhere in the room"
        
    pos = geometry['position']
    x, y = pos['x'], pos['y']
    room_w, room_d = room_dims['width'], room_dims['depth']

    # JSON의 좌표계 기준 (오른쪽 아래가 0,0 / x: 왼쪽+, y: 위쪽+)
    is_near_right_wall = (x <= threshold)
    is_near_left_wall = (x >= room_w - threshold)
    is_near_bottom_wall = (y <= threshold)
    is_near_top_wall = (y >= room_d - threshold)

    # 1. 구석 위치 서술
    if is_near_right_wall and is_near_bottom_wall:
        return "in the bottom-right corner"
    if is_near_left_wall and is_near_bottom_wall:
        return "in the bottom-left corner"
    if is_near_right_wall and is_near_top_wall:
        return "in the top-right corner"
    if is_near_left_wall and is_near_top_wall:
        return "in the top-left corner"

    # 2. 벽 근처 위치 서술
    if is_near_right_wall:
        return "against the right wall"
    if is_near_left_wall:
        return "against the left wall"
    if is_near_bottom_wall:
        return "against the bottom wall"
    if is_near_top_wall:
        return "against the top wall"

    # 3. 그 외 중앙 영역 서술
    if (room_w * 0.3 < x < room_w * 0.7) and (room_d * 0.3 < y < room_d * 0.7):
        return "in the center of the room"
    
    return "in the room" # 기본값

def create_dynamic_prompt_from_json(json_data):
    """
    JSON 데이터를 기반으로 AI가 이해하기 쉬운 서술형 프롬프트를 동적으로 생성합니다.
    """
    scene = json_data["scene"]
    room = scene["room"]
    objects = scene["objects"]

    # 기본 스타일과 분위기 설정 (가장 중요)
    prompt_parts = [
        "A hyper-realistic 4K 3D render of a bright, clean, and modern Korean-style room.",
        "The room has simple off-white wallpaper and light oak wood flooring.",
        "The overall aesthetic is minimalist and cozy, with a focus on functional furniture and soft, natural lighting."
    ]

    # 객체 정보를 동적으로 서술
    for obj in objects:
        obj_type = obj.get("type", "object")
        material = obj.get("material", "a standard material")
        name = obj.get("name", "an object")

        desc = ""
        if obj_type == "door":
            desc = f"A {material} {name} is on wall number {obj.get('wall', 'unknown')}."
        
        elif obj_type == "window":
            details = obj.get("details", f"on wall number {obj.get('wall', 'unknown')}")
            desc = f"A large window, {details}, lets in bright, natural light."

        elif obj_type == "furniture":
            # 이전 대화에서 정의한 geometry 구조를 사용합니다.
            # 가구 좌표가 center를 기준으로 되어 있으므로, center 좌표를 사용합니다.
            geometry = obj.get("geometry", obj.get("position")) # 이전/새로운 구조 모두 호환
            
            # describe_position 함수는 pos['x'], pos['y']를 기대하므로, 
            # 'center' 키의 값을 'position'으로 간주하여 전달합니다.
            location_desc = describe_position({'position': geometry['center']}, room)
            shape = obj.get("shape", "rectangular") # shape 정보가 없으면 사각형으로 가정
            
            desc = f"A {shape} {name} made of {material} is placed {location_desc}."
        
        if desc:
            prompt_parts.append(desc)

    # 최종적인 구도와 퀄리티 요구사항 추가
    prompt_parts.extend([
        "The camera view is a wide-angle shot from a corner, showing the entire room layout clearly.",
        "No furniture is awkwardly cut off by the frame.",
        "The scene is rendered with ultra-realistic shadows and soft, diffused light from the window.",
        "Focus on realism, clean lines, and a sense of calm. No text or watermarks."
    ])
    
    return " ".join(prompt_parts)

# ----------------- 로직 개선 완료 -----------------


def generate_image_with_imagen(prompt: str, output_filename: str = "generated_image.png"):
    """
    Vertex AI Imagen 모델을 사용하여 이미지를 생성하고 파일로 저장합니다.
    """
    print("✅ Imagen 모델을 사용하여 이미지 생성을 시작합니다...")
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="16:9",
        negative_prompt="text, watermark, unrealistic, cartoon, 3d model, blurry, low quality, human, people"
    )
    if images:
        images[0].save(location=output_filename, include_generation_parameters=True)
        print(f"🎉 이미지가 '{output_filename}' 파일로 성공적으로 저장되었습니다.")
        return output_filename
    else:
        print("❌ 이미지 생성에 실패했습니다.")
        return None

# **이제 어떤 JSON을 넣어도 코드를 수정할 필요가 없습니다.**
# 처음 제공된 JSON 데이터로 테스트
json_input_data = {
    "_id": {
        "$oid": "6886da815ec321eed44e86ee"
    },
    "scene": {
        "description": "왼쪽 아래 꼭짓점(0,0,0)을 기준으로 하는 3.9m × 4.6m 방 공간.",
        "walls": {
            "wall_1": {
                "direction": "bottom",
                "start": [
                    0,
                    0
                ],
                "end": [
                    3928,
                    0
                ]
            },
            "wall_2": {
                "direction": "right",
                "start": [
                    3928,
                    0
                ],
                "end": [
                    3928,
                    4635
                ]
            },
            "wall_3": {
                "direction": "top",
                "start": [
                    3928,
                    4635
                ],
                "end": [
                    0,
                    4635
                ]
            },
            "wall_4": {
                "direction": "left",
                "start": [
                    0,
                    4635
                ],
                "end": [
                    0,
                    0
                ]
            }
        },
        "room": {
            "width": 3928,
            "depth": 4635,
            "height": 2300
        },
        "objects": [
            {
                "type": "window",
                "name": "main_window_1",
                "wall": 1,
                "dimensions": {
                    "width": 2000,
                    "depth": 50,
                    "height": 960
                },
                "position": {
                    "x": 1664,
                    "y": 0,
                    "z": 1610
                },
                "rotation_z": 0,
                "details": "wall_1 벽에 위치"
            },
            {
                "type": "furniture",
                "name": "책상",
                "material": "#98FB98",
                "shape": "rectangle",
                "position": {
                    "center": {
                        "x": 3250,
                        "y": 3250,
                        "z": 0
                    },
                    "corners": {
                        "bottom_left": {
                            "x": 2650,
                            "y": 2950
                        },
                        "top_right": {
                            "x": 3850,
                            "y": 3550
                        }
                    }
                },
                "dimensions": {
                    "width": 1200,
                    "depth": 600,
                    "height": 750
                },
                "rotation_z": 0
            },
            {
                "type": "furniture",
                "name": "침대",
                "material": "#FFB6C1",
                "shape": "rectangle",
                "position": {
                    "center": {
                        "x": 800,
                        "y": 1250,
                        "z": 0
                    },
                    "corners": {
                        "bottom_left": {
                            "x": 50,
                            "y": 250
                        },
                        "top_right": {
                            "x": 1550,
                            "y": 2250
                        }
                    }
                },
                "dimensions": {
                    "width": 1500,
                    "depth": 2000,
                    "height": 600
                },
                "rotation_z": 0
            }
        ]
    },
    "saved_at": "2025-07-28 11:03:45",
    "format_version": "2.0.0"
}

# **개선된 함수를 사용하여 프롬프트 생성**
generated_prompt = create_dynamic_prompt_from_json(json_input_data)
print("--- [동적으로 생성된 프롬프트] ---")
print(generated_prompt)
print("\n--- 이미지 생성 시도 ---")

# 이미지 생성 함수 호출
generate_image_with_imagen(generated_prompt, output_filename="5.png")

# # .env 파일을 읽기 위한 도구와 os 도구를 가져옵니다.
# from dotenv import load_dotenv
# import os

# # 나머지 필요한 도구들
# import vertexai
# from vertexai.generative_models import GenerativeModel

# # 코드 실행 시 가장 먼저 .env 파일을 읽어 키 파일의 위치를 인식합니다.
# load_dotenv()

# # --- JSON을 텍스트 프롬프트로 변환하는 함수 (변경 없음) ---
# def generate_prompt_from_json(data: dict) -> str:
#     scene = data.get("scene", {})
#     room = scene.get("room", {})
#     objects = scene.get("objects", [])
#     width_m, depth_m, height_m = room.get("width", 0)/1000.0, room.get("depth", 0)/1000.0, room.get("height", 0)/1000.0
#     prompt_parts = [ f"A wide-angle, side view of a realistic and clean room with a modern Korean residential aesthetic. The room is {width_m:.1f}m wide, {depth_m:.1f}m deep, and {height_m:.1f}m high. The floor has a light-colored linoleum texture and the walls are a simple, plain wallpaper." ]
#     for obj in objects:
#         obj_type, dims = obj.get("type"), obj.get("dimensions", {})
#         w, h = dims.get("width", 0)/1000.0, dims.get("height", 0)/1000.0
#         if obj_type == "window": prompt_parts.append(f"On a wall, there is a window {w:.1f}m wide and {h:.1f}m high.")
#         elif obj_type == "door": prompt_parts.append(f"On another wall, there is a door {w:.1f}m wide and {h:.1f}m high.")
#         elif obj_type == "furniture":
#             name, d, rotation = obj.get("name"), dims.get("depth", 0)/1000.0, obj.get("rotation_z", 0)
#             prompt_parts.append(f"There is a {name} that is {w:.1f}m wide and {d:.1f}m deep, rotated by {rotation} degrees. It must be complete with all necessary parts.")
#     prompt_parts.append("The lighting is natural with soft shadows. The overall aesthetic is neat and uncluttered. All objects are fully visible.")
#     return " ".join(prompt_parts)

# # --- 이미지를 생성하는 함수 (오류 수정됨) ---
# def generate_image_from_prompt(project_id: str, location: str, prompt: str, output_filename: str):
#     """최신 Vertex AI SDK를 사용하여 이미지를 생성하고 파일로 저장합니다."""
#     vertexai.init(project=project_id, location=location)
#     model = GenerativeModel("imagegeneration@006")
    
#     print("🎨 AI 화가가 최신 방식으로 그림을 그리고 있습니다...")
    
#     # ------------------------------------------------------------------- #
#     # [수정됨] number_of_images -> candidate_count 로 변경
#     response = model.generate_content(
#         [prompt],
#         generation_config={"candidate_count": 1}
#     )
#     # ------------------------------------------------------------------- #
    
#     image_data = response.candidates[0].content.parts[0].data
    
#     if image_data:
#         with open(output_filename, "wb") as f:
#             f.write(image_data)
#         print(f"✅ 그림 완성! '{os.path.abspath(output_filename)}' 파일로 저장했어요.")
#     else:
#         print(f"❌ API로부터 이미지 데이터를 받지 못했습니다.")

# # --- 메인 실행 부분 (변경 없음) ---
# if __name__ == "__main__":
#     MY_PROJECT_ID = "virtual-muse-466706-v2"
#     room_design_json = {
#       "scene": { "room": { "width": 3500, "depth": 4000, "height": 2350 },
#         "objects": [
#           { "type": "window", "dimensions": { "width": 1200, "height": 1400 }},
#           { "type": "door", "dimensions": { "width": 900, "height": 2100 }},
#           { "type": "furniture", "name": "bed", "dimensions": { "width": 1500, "depth": 2000 }, "rotation_z": 180 },
#           { "type": "furniture", "name": "desk", "dimensions": { "width": 1200, "depth": 600 }, "rotation_z": 45 } ] } }
    
#     try:
#         if MY_PROJECT_ID == "YOUR_GOOGLE_CLOUD_PROJECT_ID":
#             print("🛑 중요: 코드의 MY_PROJECT_ID 부분을 실제 값으로 수정해주세요!")
#         elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
#             print("🛑 중요: .env 파일이 없거나 파일 안에 경로가 비어있습니다!")
#         else:
#             print("1️⃣ 설계도를 AI가 이해할 수 있는 말로 번역합니다...")
#             text_prompt = generate_prompt_from_json(room_design_json)
#             print(f"   ▶ 번역된 내용: {text_prompt[:100]}...")
#             print("\n2️⃣ 번역된 내용을 AI 화가에게 전달합니다...")
#             generate_image_from_prompt(
#                 project_id=MY_PROJECT_ID,
#                 location="us-central1",
#                 prompt=text_prompt,
#                 output_filename="my_beautiful_room.png"
#             )
#     except Exception as e:
#         print(f"❌ 이런! 문제가 발생했어요: {e}")