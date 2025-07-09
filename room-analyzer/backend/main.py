# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
from analyzer import analyze_room

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-room")
async def analyze(photo: UploadFile = File(...), ceiling: int = Form(230)):
    file_path = os.path.join(UPLOAD_DIR, photo.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(photo.file, f)

    bbox, w_cm, h_cm, layout = analyze_room(file_path, ceiling)
    if bbox:
        return {
            "width_cm": w_cm,
            "height_cm": h_cm,
            "layout": layout
        }
    return {"error": "방 분석 실패"}

@app.get("/static/{filename}")
async def get_image(filename: str):
    return FileResponse(f"static/{filename}")
