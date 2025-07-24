from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import os
import uuid
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/analyze-room")
async def analyze_room(
    photo: UploadFile = File(...),
    ceiling: float = Form(...),       # 실제 층고(cm)
    ceiling_y: int = Form(...),       # 이미지 내 천장 y 좌표
    floor_y: int = Form(...),         # 이미지 내 바닥 y 좌표
):
    try:
        # 파일 저장
        ext = os.path.splitext(photo.filename)[-1]
        filename = f"{uuid.uuid4()}{ext}"
        image_path = os.path.join(UPLOAD_DIR, filename)

        photo.file.seek(0)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)

        # 이미지 열기
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        # 유효성 검증
        pixel_height = abs(floor_y - ceiling_y)
        if pixel_height == 0:
            return JSONResponse(status_code=400, content={"error": "천장과 바닥 y좌표가 동일합니다."})

        cm_per_pixel = ceiling / pixel_height  # 환산 비율
        room_width_cm = img_width * cm_per_pixel
        room_height_cm = img_height * cm_per_pixel

        return {
            "width_cm": round(room_width_cm),
            "height_cm": round(room_height_cm),
            "layout": []  # 감지된 객체 없음
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
