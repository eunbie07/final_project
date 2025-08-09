import os, base64, subprocess, io, uuid, asyncio
from typing import List
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from PIL import Image
from dotenv import load_dotenv
from prompt_templates import (
    DEFAULT_PROMPT, STABILITY_PROMPT, REPLICATE_PROMPT, VERTEX_PROMPT, 
    NEGATIVE_PROMPT, DEFAULT_STRENGTH, DEFAULT_GUIDANCE,
    CAPTURE_TO_REAL_PROMPT, STYLE_PRESETS
)

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

def save_uploaded_image(image_bytes: bytes, original_filename: str) -> str:
    """업로드된 이미지를 저장"""
    # uploads 폴더 생성
    os.makedirs('uploads', exist_ok=True)
    
    # 고유한 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_id = str(uuid.uuid4())[:8]
    filename = f"{timestamp}_{file_id}_{original_filename}"
    filepath = os.path.join('uploads', filename)
    
    # 파일 저장
    with open(filepath, 'wb') as f:
        f.write(image_bytes)
    
    print(f"[Storage] 업로드 이미지 저장: {filename}")
    return filepath

def save_generated_image(image_data: str, provider: str, request_id: str) -> str:
    """생성된 이미지를 저장 (base64 data URL에서)"""
    # outputs 폴더 생성
    os.makedirs('outputs', exist_ok=True)
    
    try:
        # data:image/png;base64, 부분 제거
        if image_data.startswith('data:image'):
            base64_data = image_data.split(',')[1]
        else:
            base64_data = image_data
        
        # base64 디코딩
        image_bytes = base64.b64decode(base64_data)
        
        # 파일명 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{request_id}_{provider}.png"
        filepath = os.path.join('outputs', filename)
        
        # 파일 저장
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        print(f"[Storage] 생성 이미지 저장: {filename}")
        return filepath
        
    except Exception as e:
        print(f"[Storage] 이미지 저장 실패: {e}")
        return None

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
        "image_strength": str(strength if strength < 0.5 else 0.3),  # 레이아웃 유지 모드
        "cfg_scale": str(guidance),
        "samples": "1",
        "steps": "40"
    }
    
    print(f"[Stability] 레이아웃 유지 모드 (strength=0.3)")
    
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

async def call_replicate_depth2img(image_bytes: bytes, prompt: str, strength: float, guidance: float) -> List[str]:
    """ControlNet Depth2Img 모델 - 레이아웃 유지 특화"""
    if not REPLICATE_TOKEN:
        raise HTTPException(500, "Replicate API token missing.")
    
    # 이미지 리사이즈 (Depth2Img는 다양한 해상도 지원)
    resized_image_bytes = resize_image_for_sdxl(image_bytes)
    img_b64 = base64.b64encode(resized_image_bytes).decode()
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}", "Content-Type": "application/json"}
    
    # ControlNet Depth2Img 모델 (정확한 버전 ID)
    payload = {
        "version": "922c7bb67b87ec32cbc2fd11b1d5f94f0ba4f5519c4dbd02856376444127cc60",
        "input": {
            "image": f"data:image/png;base64,{img_b64}",
            "eta": 0,
            "scale": guidance,  # guidance scale (기본 9)
            "a_prompt": f"best quality, extremely detailed, photorealistic interior bedroom, {REPLICATE_PROMPT}",
            "n_prompt": f"{NEGATIVE_PROMPT}, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, 3d render, cgi, artificial",
            "ddim_steps": 20,  # 기본값 사용
            "num_samples": "1",
            "image_resolution": "1024",  # 문자열로 전달
            "detect_resolution": 512
        }
    }
    
    print(f"[Replicate] ControlNet Depth2Img 모델로 레이아웃 유지 처리 중...")
    
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
            await asyncio.sleep(1)

async def call_replicate_controlnet_structure(image_bytes: bytes, prompt: str, strength: float, guidance: float, structure: str = "depth") -> List[str]:
    """RossJillian ControlNet 모델 - 다양한 구조 보존"""
    if not REPLICATE_TOKEN:
        raise HTTPException(500, "Replicate API token missing.")
    
    resized_image_bytes = resize_image_for_sdxl(image_bytes)
    img_b64 = base64.b64encode(resized_image_bytes).decode()
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}", "Content-Type": "application/json"}
    
    # RossJillian ControlNet 모델
    payload = {
        "version": "795433b19458d0f4fa172a7ccf93178d2adb1cb8ab2ad6c8fdc33fdbcd49f477",
        "input": {
            "image": f"data:image/png;base64,{img_b64}",
            "prompt": f"photorealistic bedroom interior, {REPLICATE_PROMPT}",
            "negative_prompt": f"{NEGATIVE_PROMPT}, 3d render, cgi",
            "structure": structure,  # "canny", "depth", "hed", "mlsd", "normal", "openpose", "scribble", "seg"
            "scale": guidance,
            "steps": 40
        }
    }
    
    print(f"[Replicate] ControlNet Structure ({structure}) 모델로 처리 중...")
    
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
            await asyncio.sleep(1)

# 기존 SDXL 모델 유지 (백업용)
async def call_replicate_sdxl_i2i(image_bytes: bytes, prompt: str, strength: float, guidance: float) -> List[str]:
    if not REPLICATE_TOKEN:
        raise HTTPException(500, "Replicate API token missing.")
    
    # SDXL 호환 크기로 리사이즈
    resized_image_bytes = resize_image_for_sdxl(image_bytes)
    img_b64 = base64.b64encode(resized_image_bytes).decode()
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}", "Content-Type": "application/json"}
    
    # 검증된 실사 SDXL 모델 (기존 모델로 돌아가되 파라미터 조정)
    payload = {
        "version": "7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        "input": {
            "prompt": f"photorealistic, ultra realistic, professional photography, {REPLICATE_PROMPT}",
            "negative_prompt": f"{NEGATIVE_PROMPT}, cartoon, anime, illustration, painting, drawing, sketch, stylized, artistic",
            "image": f"data:image/png;base64,{img_b64}",
            "strength": strength if strength < 0.5 else 0.3,  # 레이아웃 유지 모드
            "width": 1024, "height": 1024,
            "guidance_scale": 15,  # 더 높게 (프롬프트 더 강하게)
            "num_inference_steps": 40,  # 적당한 스텝
            "scheduler": "DPMSolverMultistep"  # 더 나은 스케줄러
        }
    }
    
    print(f"[Replicate] SDXL 백업 모델로 처리 중...")
    
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
            await asyncio.sleep(1)

async def call_replicate_fooocus_inpaint(image_bytes: bytes, prompt: str, strength: float, guidance: float) -> List[str]:
    """Fooocus 인페인팅 모델 - 부분 수정 및 스타일 튜닝"""
    if not REPLICATE_TOKEN:
        raise HTTPException(500, "Replicate API token missing.")
    
    resized_image_bytes = resize_image_for_sdxl(image_bytes)
    img_b64 = base64.b64encode(resized_image_bytes).decode()
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}", "Content-Type": "application/json"}
    
    # Fooocus 인페인팅 모델
    payload = {
        "version": "7c662db7ec7f06095f494c152e3de69084249385e12c2224bdcb6178a650d7c8",
        "input": {
            "prompt": f"photorealistic bedroom interior, {REPLICATE_PROMPT}",
            "cn_type1": "ImagePrompt",
            "cn_type2": "ImagePrompt", 
            "cn_type3": "ImagePrompt",
            "cn_type4": "ImagePrompt",
            "sharpness": 2,
            "image_seed": -1,
            "uov_method": "Disabled",
            "adaptive_cfg": 7,
            "image_number": 1,
            "sampler_name": "dpmpp_2m_sde_gpu",
            "adm_scaler_end": 0.3,
            "guidance_scale": guidance,
            "inpaint_engine": "v2.6",
            "overwrite_step": -1,
            "refiner_switch": 0.5,
            "scheduler_name": "karras",
            "negative_prompt": f"{NEGATIVE_PROMPT}, 3d render, cgi, unrealistic",
            "overwrite_width": -1,
            "inpaint_strength": strength if strength < 0.7 else 0.5,
            "overwrite_height": -1,
            "overwrite_switch": -1,
            "style_selections": "Fooocus V2,Fooocus Enhance,Fooocus Sharp",
            "loras_custom_urls": "",
            "uov_upscale_value": 0,
            "use_default_loras": True,
            "adm_scaler_negative": 0.8,
            "adm_scaler_positive": 1.5,
            "canny_low_threshold": 64,
            "controlnet_softness": 0.25,
            "outpaint_selections": "",
            "canny_high_threshold": 128,
            "invert_mask_checkbox": False,
            "outpaint_distance_top": 0,
            "performance_selection": "Speed",
            "outpaint_distance_left": 0,
            "aspect_ratios_selection": "1152*896",
            "inpaint_erode_or_dilate": 0,
            "outpaint_distance_right": 0,
            "overwrite_vary_strength": -1,
            "inpaint_respective_field": 1,
            "outpaint_distance_bottom": 0,
            "skipping_cn_preprocessor": False,
            "debugging_cn_preprocessor": False,
            "inpaint_additional_prompt": "",
            "overwrite_upscale_strength": -1,
            "debugging_inpaint_preprocessor": False,
            "inpaint_disable_initial_latent": False,
            "mixing_image_prompt_and_inpaint": False,
            "mixing_image_prompt_and_vary_upscale": False
        }
    }
    
    print(f"[Replicate] Fooocus 인페인팅 모델로 처리 중...")
    
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
            await asyncio.sleep(1)

async def call_replicate_flux_ultra(image_bytes: bytes, prompt: str, strength: float, guidance: float) -> List[str]:
    """FLUX 1.1 Pro Ultra 모델 - 고해상도 포토리얼 업스케일"""
    if not REPLICATE_TOKEN:
        raise HTTPException(500, "Replicate API token missing.")
    
    resized_image_bytes = resize_image_for_sdxl(image_bytes)
    img_b64 = base64.b64encode(resized_image_bytes).decode()
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}", "Content-Type": "application/json"}
    
    # FLUX 1.1 Pro Ultra 모델
    payload = {
        "version": "c6e5086a542c99e7e523a83d3017654e8618fe64ef427c772a1def05bb599f0c",
        "input": {
            "prompt": f"photorealistic bedroom interior, {REPLICATE_PROMPT}",
            "image_prompt": f"data:image/png;base64,{img_b64}",
            "image_prompt_strength": strength if strength < 0.8 else 0.1,
            "aspect_ratio": "1:1",
            "safety_tolerance": 5,
            "seed": None,
            "raw": True,
            "output_format": "png"
        }
    }
    
    print(f"[Replicate] FLUX Pro Ultra 모델로 고해상도 처리 중...")
    
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
            await asyncio.sleep(1)

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
    
    # Vertex AI Imagen Image-to-Image 편집 모델들 (공식 문서 기준)
    model_names = [
        "imagegeneration@002"        # 안정적 백업 (Imagen 1 기반)
    ]
    
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    b64_img = base64.b64encode(resized_image_bytes).decode()
    
    last_error = None
    
    for model_name in model_names:
        try:
            endpoint = f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/publishers/google/models/{model_name}:predict"
            
            # 모델에 따라 다른 요청 본문 생성
            if model_name == "imagen-3.0-capability-001":
                # Imagen 3 Capability 모델 (context_image 사용)
                body = {
                    "instances": [{
                        "prompt": prompt,
                        "context_image": { "bytesBase64Encoded": b64_img }
                    }],
                    "parameters": {
                        "sampleCount": 1,
                        "guidanceScale": guidance,
                        "strength": strength if strength < 0.5 else 0.3  # 레이아웃 유지 모드
                    }
                }
            else:
                # 이전 세대 Imagen 모델 (image 사용)
                body = {
                    "instances": [{
                        "prompt": prompt,
                        "image": { "bytesBase64Encoded": b64_img }
                    }],
                    "parameters": {
                        "sampleCount": 1,
                        "guidanceScale": guidance,
                        "strength": strength if strength < 0.5 else 0.3  # 레이아웃 유지 모드
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
                    print(f"[Vertex] Full error response for {model_name}: {r.text}") # 추가된 라인
                    last_error = r.text
                    
        except Exception as e:
            print(f"[Vertex] {model_name} 예외: {str(e)}")
            last_error = str(e)
            continue
    
    # 모든 모델이 실패한 경우
    raise HTTPException(500, f"모든 Vertex AI 모델 시도 실패. 마지막 오류: {last_error}")

# 3D 캡처 전용 2단계 파이프라인
async def run_pipeline_3d_capture(
    img_bytes: bytes, style: str, strength: float, guidance: float
) -> str:
    # 1단계: ControlNet Depth로 레이아웃 유지 i2i
    style_text = STYLE_PRESETS.get(style, "")
    depth_prompt = (
        f"{CAPTURE_TO_REAL_PROMPT} "
        f"{DEFAULT_PROMPT} "
        f"If any CGI flat shading is detected, replace with realistic PBR textures. "
        f"Style: {style_text}".strip()
    )

    # ControlNet(rossjillian) 또는 depth2img(jagilley) 중 하나 선택
    try:
        urls = await call_replicate_controlnet_structure(
            img_bytes, prompt=depth_prompt, strength=strength,
            guidance=guidance, structure="depth"
        )
    except Exception:
        # 폴백: depth2img
        urls = await call_replicate_depth2img(
            img_bytes, prompt=depth_prompt, strength=strength, guidance=guidance
        )

    if not urls:
        raise HTTPException(500, "Stage1 실패: ControlNet 결과가 비어 있습니다.")
    stage1_url = urls[0]

    # 2단계: FLUX Ultra로 고해상도 디테일 강화 (이미지 프롬프트 비중 약간 올림)
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.get(stage1_url)
        if r.status_code != 200:
            raise HTTPException(502, f"Stage1 이미지 다운로드 실패: {r.status_code}")
        stage1_bytes = r.content

    # 실사 유지 + 과도한 재해석 방지
    flux_prompt = (
        "Enhance photorealism, preserve layout and furniture identity from the image, "
        "add subtle micro-textures on fabric and wood, clean edges, reduce CGI feel."
    )
    flux_urls = await call_replicate_flux_ultra(
        stage1_bytes, prompt=flux_prompt, strength=0.25, guidance=guidance
    )
    if not flux_urls:
        raise HTTPException(500, "Stage2 실패: FLUX 업스케일 실패")
    return flux_urls[0]

# Unified endpoint
@app.post("/api/realistic-room", response_model=I2IResponse)
async def realistic_room(
    image: UploadFile = File(...),
    provider: str = Form("stability"),  # stability | replicate | vertex 
    model: str = Form("default"),  # default | depth2img | controlnet | fooocus | flux_ultra | pipeline_3d_capture
    structure: str = Form("depth"),  # depth | canny | hed | mlsd | normal | openpose | scribble | seg (controlnet용)
    style: str = Form("scandinavian")  # scandinavian | modern | bohemian | japanese
):
    # 프롬프트와 파라미터를 코드에서 직접 설정
    prompt = DEFAULT_PROMPT
    strength = DEFAULT_STRENGTH
    guidance = DEFAULT_GUIDANCE
    # 요청 고유 ID 생성
    request_id = str(uuid.uuid4())[:8]
    
    # 업로드된 이미지 읽기
    img_bytes = await image.read()
    
    # 업로드된 이미지 저장
    uploaded_path = save_uploaded_image(img_bytes, image.filename or "upload.png")
    print(f"[Main] 요청 ID: {request_id}, 업로드 파일: {uploaded_path}")
    
    try:
        if provider == "replicate" and model == "pipeline_3d_capture":
            # 전용 파이프라인 실행
            final_url = await run_pipeline_3d_capture(img_bytes, style, strength, guidance)

            # 저장
            async with httpx.AsyncClient() as client:
                resp = await client.get(final_url)
                if resp.status_code == 200:
                    img_b64 = base64.b64encode(resp.content).decode()
                    save_generated_image(f"data:image/png;base64,{img_b64}",
                                         f"{provider}_{model}", request_id)
            return {"images": [final_url]}
        
        elif provider == "stability":
            out = await call_stability_i2i(img_bytes, STABILITY_PROMPT, strength, guidance)
            # 생성된 이미지 저장
            if out:
                save_generated_image(out, f"{provider}_{model}", request_id)
            return {"images": [out]}
        elif provider == "replicate":
            # 모델에 따라 다른 Replicate 함수 호출
            try:
                if model == "depth2img":
                    urls = await call_replicate_depth2img(img_bytes, REPLICATE_PROMPT, strength, guidance)
                elif model == "controlnet":
                    # 임시로 기본 SDXL 사용 (버전 ID 확인 필요)
                    print("[Warning] ControlNet Structure 모델 버전 확인 필요, 기본 SDXL 사용")
                    urls = await call_replicate_sdxl_i2i(img_bytes, REPLICATE_PROMPT, strength, guidance)
                elif model == "fooocus":
                    # 임시로 기본 SDXL 사용 (버전 ID 확인 필요)
                    print("[Warning] Fooocus 모델 버전 확인 필요, 기본 SDXL 사용")
                    urls = await call_replicate_sdxl_i2i(img_bytes, REPLICATE_PROMPT, strength, guidance)
                elif model == "flux_ultra":
                    # 임시로 기본 SDXL 사용 (버전 ID 확인 필요)
                    print("[Warning] FLUX Ultra 모델 버전 확인 필요, 기본 SDXL 사용")
                    urls = await call_replicate_sdxl_i2i(img_bytes, REPLICATE_PROMPT, strength, guidance)
                else:  # default
                    urls = await call_replicate_sdxl_i2i(img_bytes, REPLICATE_PROMPT, strength, guidance)
            except HTTPException as e:
                print(f"[Replicate] {model} 모델 실패, 기본 SDXL로 폴백")
                urls = await call_replicate_sdxl_i2i(img_bytes, REPLICATE_PROMPT, strength, guidance)
            
            # Replicate은 URL을 반환하므로 다운로드해서 저장
            if urls:
                for i, url in enumerate(urls):
                    try:
                        async with httpx.AsyncClient() as client:
                            resp = await client.get(url)
                            if resp.status_code == 200:
                                # URL 이미지를 base64로 변환하여 저장
                                img_b64 = base64.b64encode(resp.content).decode()
                                save_generated_image(f"data:image/png;base64,{img_b64}", f"{provider}_{model}_{i}", request_id)
                    except Exception as e:
                        print(f"[Storage] Replicate 이미지 저장 실패: {e}")
            return {"images": urls}
        elif provider == "vertex":
            out = await call_vertex_imagen3(img_bytes, VERTEX_PROMPT, strength, guidance)
            # 생성된 이미지 저장
            if out:
                save_generated_image(out, f"{provider}_{model}", request_id)
            return {"images": [out]}
        else:
            raise HTTPException(400, "Unsupported provider")
            
    except Exception as e:
        print(f"[Main] 처리 실패 - 요청 ID: {request_id}, 오류: {str(e)}")
        raise
