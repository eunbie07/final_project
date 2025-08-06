import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# --- 기본 설정 ---
# .env 파일 로드
load_dotenv()

# Vertex AI 프로젝트 설정
PROJECT_ID = "virtual-muse-466706-v2"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# FastAPI 앱 생성
app = FastAPI(
    title="AI Room Image Generator API",
    description="JSON으로 방 구조를 보내면 이미지를 생성하여 반환합니다."
)


# --- Pydantic 모델 정의 (API 입력 데이터 검증용) ---
class Geometry(BaseModel):
    position: Dict[str, int]
    # 다른 기하학적 정보는 선택적으로 받음
    shape: str = 'rectangle'
    width: int = None
    depth: int = None
    height: int = None
    radius: int = None
    rotation_z: int = 0

class RoomObject(BaseModel):
    type: str
    name: str
    material: str
    wall: int = None
    details: str = None
    geometry: Geometry = None # 가구용
    dimensions: Dict[str, int] = None # 문/창문용

class Room(BaseModel):
    width: int
    depth: int
    height: int

class Wall(BaseModel):
    direction: str
    start: List[int]
    end: List[int]

class Scene(BaseModel):
    description: str
    walls: Dict[str, Wall]
    room: Room
    objects: List[RoomObject]

class SceneInput(BaseModel):
    """API가 받을 최종 입력 모델"""
    scene: Scene


# --- 이미지 생성 로직 (이전 코드와 동일) ---

def describe_position(geometry, room_dims, threshold=300):
    if not geometry or 'position' not in geometry:
        return "somewhere in the room"
    pos = geometry.position
    x, y = pos.get('x', 0), pos.get('y', 0)
    room_w, room_d = room_dims.width, room_dims.depth
    is_near_right_wall = (x <= threshold)
    is_near_left_wall = (x >= room_w - threshold)
    is_near_bottom_wall = (y <= threshold)
    is_near_top_wall = (y >= room_d - threshold)
    if is_near_right_wall and is_near_bottom_wall: return "in the bottom-right corner"
    if is_near_left_wall and is_near_bottom_wall: return "in the bottom-left corner"
    if is_near_right_wall and is_near_top_wall: return "in the top-right corner"
    if is_near_left_wall and is_near_top_wall: return "in the top-left corner"
    if is_near_right_wall: return "against the right wall"
    if is_near_left_wall: return "against the left wall"
    if is_near_bottom_wall: return "against the bottom wall"
    if is_near_top_wall: return "against the top wall"
    if (room_w * 0.3 < x < room_w * 0.7) and (room_d * 0.3 < y < room_d * 0.7):
        return "in the center of the room"
    return "in the room"

def create_dynamic_prompt_from_json(json_data):
    scene = json_data["scene"]
    room = scene["room"]
    objects = scene["objects"]
    prompt_parts = [
        "A hyper-realistic 4K 3D render of a bright, clean, and modern Korean-style room.",
        "The room has simple off-white wallpaper and light oak wood flooring.",
        "The overall aesthetic is minimalist and cozy, with a focus on functional furniture and soft, natural lighting."
    ]
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
            geometry = obj.get("geometry") or obj.get("dimensions")
            location_desc = describe_position(SceneInput(scene=json_data['scene']).scene.objects[0].geometry, SceneInput(scene=json_data['scene']).scene.room) # Pass pydantic models
            shape = geometry.get("shape", "rectangular") if geometry else "rectangular"
            desc = f"A {shape} {name} made of {material} is placed {location_desc}."
        if desc: prompt_parts.append(desc)
    prompt_parts.extend([
        "The camera view is a wide-angle shot from a corner, showing the entire room layout clearly.",
        "No furniture is awkwardly cut off by the frame.",
        "The scene is rendered with ultra-realistic shadows and soft, diffused light from the window.",
        "Focus on realism, clean lines, and a sense of calm. No text or watermarks."
    ])
    return " ".join(prompt_parts)

def generate_image_with_imagen(prompt: str, output_filename: str = "generated_image.png"):
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


# --- FastAPI 엔드포인트 정의 ---
@app.post("/generate-image/")
async def generate_room_image(data: SceneInput = Body(...)):
    """
    방 구조 JSON을 받아 이미지를 생성하고, 이미지 파일을 직접 반환합니다.
    """
    try:
        # Pydantic 모델을 다시 Python dict로 변환하여 함수에 전달
        json_data = data.model_dump()
        
        # 1. 프롬프트 생성
        prompt = create_dynamic_prompt_from_json(json_data)
        print("--- [생성된 프롬프트] ---")
        print(prompt)

        # 2. 이미지 생성
        output_filename = "temp_room_image.png"
        generated_file = generate_image_with_imagen(prompt, output_filename)

        if generated_file:
            # 3. 생성된 이미지 파일을 클라이언트에게 전송
            return FileResponse(path=generated_file, media_type='image/png', filename=generated_file)
        else:
            raise HTTPException(status_code=500, detail="이미지 생성에 실패했습니다.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {str(e)}")