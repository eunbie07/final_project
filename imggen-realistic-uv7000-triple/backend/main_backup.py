import os, base64, subprocess, io
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from PIL import Image
from dotenv import load_dotenv
from prompt_templates import DEFAULT_PROMPT, NEGATIVE_PROMPT, DEFAULT_STRENGTH, DEFAULT_GUIDANCE

load_dotenv()
STABILITY_KEY = os.getenv("STABILITY_API_KEY")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

app = FastAPI(title="Realistic Room i2i (triple)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class I2IResponse(BaseModel):
    images: List[str]

@app.get("/healthz")
async def healthz():
    return {"ok": True}

def resize_image_for_sdxl(image_bytes: bytes) -> bytes:
    """이미지를 SDXL 호환 크기로 리사이즈"""
    # SDXL 지원 크기 목록
    sdxl_sizes = [
        (1024, 1024), (1152, 896), (1216, 832), (1344, 768),
        (1536, 640), (640, 1536), (768, 1344), (832, 1216), (896, 1152)
    ]
    
    # 이미지 로드
    image = Image.open(io.BytesIO(image_bytes))
    original_width, original_height = image.size
    original_ratio = original_width / original_height
    
    # 가장 가까운 비율의 SDXL 크기 찾기
    best_size = min(sdxl_sizes, key=lambda size: abs(size[0]/size[1] - original_ratio))
    
    # 리사이즈
    if (original_width, original_height) != best_size:
        print(f"이미지 리사이즈: {original_width}x{original_height} → {best_size[0]}x{best_size[1]}")
        image = image.resize(best_size, Image.Resampling.LANCZOS)
    
    # 바이트로 변환
    output = io.BytesIO()
    image.save(output, format='PNG', quality=95)
    return output.getvalue()

# --- Providers ---
async def call_stability_i2i(image_bytes: bytes, prompt: str, strength: float, guidance: float) -> str:
    if not STABILITY_KEY:
        raise HTTPException(500, "Stability API key missing.")
    
    print(f"[Stability] 처리 시작 - strength: {strength}, guidance: {guidance}")
    
    # SDXL 호환 크기로 리사이즈
    resized_image_bytes = resize_image_for_sdxl(image_bytes)
    print(f"[Stability] 이미지 리사이즈 완료 - {len(resized_image_bytes)} bytes")
    
    headers = {"Authorization": f"Bearer {STABILITY_KEY}", "Accept": "application/json"}
    files = {"init_image": ("input.png", resized_image_bytes, "image/png")}
    data = {
        "text_prompts[0][text]": prompt,
        "text_prompts[1][text]": NEGATIVE_PROMPT,
        "text_prompts[1][weight]": "-1",
        "image_strength": str(strength),
        "cfg_scale": str(guidance),
        "samples": "1",
        "steps": "30"
    }
    
    print(f"[Stability] API 호출 시작...")
    try:
        async with httpx.AsyncClient(timeout=300) as client:  # 타임아웃 증가
            r = await client.post(
                "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
                headers=headers, files=files, data=data
            )
            print(f"[Stability] API 응답 수신 - 상태코드: {r.status_code}")
            
            if r.status_code >= 400:
                print(f"[Stability] 오류 응답: {r.text}")
                raise HTTPException(r.status_code, r.text)
                
            j = r.json()
            if "artifacts" not in j or not j["artifacts"]:
                print(f"[Stability] 응답 구조 문제: {j}")
                raise HTTPException(500, "No artifacts in response")
                
            b64 = j["artifacts"][0]["base64"]
            print(f"[Stability] 성공 완료 - 이미지 크기: {len(b64)} chars")
            return f"data:image/png;base64,{b64}"
            
    except httpx.TimeoutException:
        print("[Stability] 타임아웃 발생")
        raise HTTPException(408, "Request timeout")
    except Exception as e:
        print(f"[Stability] 예외 발생: {str(e)}")
        raise HTTPException(500, f"Stability API error: {str(e)}")

async def call_replicate_sdxl_i2i(image_bytes: bytes, prompt: str, strength: float, guidance: float) -> List[str]:
    if not REPLICATE_TOKEN:
        raise HTTPException(500, "Replicate API token missing.")
    
    # SDXL 호환 크기로 리사이즈
    resized_image_bytes = resize_image_for_sdxl(image_bytes)
    img_b64 = base64.b64encode(resized_image_bytes).decode()
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}", "Content-Type": "application/json"}
    
    # 잘 알려진 SDXL 모델 사용
    payload = {
        "version": "7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        "input": {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "image": f"data:image/png;base64,{img_b64}",
            "strength": strength,
            "width": 1024, "height": 1024,
            "guidance_scale": guidance,
            "num_inference_steps": 25,
            "scheduler": "K_EULER"
        }
    }
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post("https://api.replicate.com/v1/predictions",
                              headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        pred = r.json()
        get_url = pred["urls"]["get"]
        while True:
            s = await client.get(get_url, headers=headers)
            if s.status_code >= 400:
                raise HTTPException(s.status_code, s.text)
            j = s.json()
            status = j.get("status")
            if status in ("succeeded", "failed", "canceled"):
                if status != "succeeded":
                    raise HTTPException(500, f"Replicate job {status}")
                return j["output"]

async def call_vertex_imagen3(image_bytes: bytes, prompt: str, strength: float, guidance: float) -> str:
    if not GCP_PROJECT_ID:
        raise HTTPException(500, "GCP_PROJECT_ID 환경변수가 설정되지 않았습니다.")
    
    # SDXL 호환 크기로 리사이즈 (Vertex AI도 크기 제한이 있을 수 있음)
    resized_image_bytes = resize_image_for_sdxl(image_bytes)
    
    # Google CLI 인증 사용 (Application Default Credentials)
    try:
        import google.auth
        from google.auth.transport.requests import Request
        
        print("[Vertex] Google CLI 인증 사용")
        
        # Application Default Credentials 사용
        credentials, project_id = google.auth.default(
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Access Token 생성
        credentials.refresh(Request())
        token = credentials.token
        
        print(f"[Vertex] 인증 성공 - 프로젝트: {project_id or GCP_PROJECT_ID}")
        
    except Exception as e:
        raise HTTPException(500, f"Vertex AI 인증 오류: {str(e)}. 'gcloud auth application-default login' 명령어를 실행하세요.")
    
    # Vertex AI Imagen Image-to-Image 편집 모델들 (최신순)
    model_names = [
        "imagen-3.0-edit-001",        # 최신 이미지 편집 전용 모델 (권장)
        "imagegeneration@006",        # Imagen 2 기반, 마스크 편집 지원
        "imagegeneration@002",        # Imagen 1 기반, 생성+편집
        "imagen-3.0-generate-001",    # 백업용
        "imagen-3.0-fast-generate-001" # 빠른 생성용
    ]
    
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    b64_img = base64.b64encode(resized_image_bytes).decode()
    
    last_error = None
    
    for model_name in model_names:
        try:
            endpoint = f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/publishers/google/models/{model_name}:predict"
            
            # Image-to-Image editing용 페이로드
            body = {
                "instances": [{
                    "prompt": prompt,
                    "image": {
                        "bytesBase64Encoded": b64_img
                    },
                    "editConfig": {
                        "editMode": "inpainting-insert",
                        "guidanceScale": guidance,
                        "outputImageType": "PNG"
                    }
                }],
                "parameters": {
                    "sampleCount": 1
                }
            }
            
            print(f"[Vertex] {model_name} 모델 시도 중...")
            
            async with httpx.AsyncClient(timeout=180) as client:
                r = await client.post(endpoint, headers=headers, json=body)
                
                if r.status_code < 400:
                    resp = r.json()
                    img_base64 = resp["predictions"][0]["bytesBase64Encoded"]
                    print(f"[Vertex] {model_name} 모델 성공!")
                    return f"data:image/png;base64,{img_base64}"
                else:
                    print(f"[Vertex] {model_name} 실패: {r.status_code}")
                    last_error = r.text
                    
        except Exception as e:
            print(f"[Vertex] {model_name} 예외: {str(e)}")
            last_error = str(e)
            continue
    
    # 모든 모델이 실패한 경우
    raise HTTPException(500, f"모든 Vertex AI 모델 시도 실패. 마지막 오류: {last_error}")

# Unified endpoint
@app.post("/api/realistic-room", response_model=I2IResponse)
async def realistic_room(
    image: UploadFile = File(...),
    provider: str = Form("stability"),  # stability | replicate | vertex (replicate requires credits)
    prompt: str = Form(DEFAULT_PROMPT),
    strength: float = Form(DEFAULT_STRENGTH),
    guidance: float = Form(DEFAULT_GUIDANCE),
):
    img_bytes = await image.read()
    if provider == "stability":
        out = await call_stability_i2i(img_bytes, prompt, strength, guidance)
        return {"images": [out]}
    elif provider == "replicate":
        urls = await call_replicate_sdxl_i2i(img_bytes, prompt, strength, guidance)
        return {"images": urls}
    elif provider == "vertex":
        out = await call_vertex_imagen3(img_bytes, prompt, strength, guidance)
        return {"images": [out]}
    else:
        raise HTTPException(400, "Unsupported provider")