# Provider별 최적화된 프롬프트 (교체본)

STABILITY_PROMPT = (
    "Preserve the exact room layout and furniture positions from the input image. "
    "Convert any simple colored block into a realistic bed with white cotton sheets and fluffy pillows. "
    "Photorealistic interior photography look, DSLR quality, soft natural daylight from a window "
    "(4500–5500K), realistic shadows and depth of field. "
    "Materials must look real: painted drywall with subtle texture, genuine hardwood floor with visible wood grain, "
    "fabric fibers on bedding, clean edges, no plastic sheen. "
    "Calm, peaceful mood, balanced exposure and white balance."
)

REPLICATE_PROMPT = (
    "Keep the original layout and do not move objects. "
    "Transform the CGI look into a photorealistic interior photograph. "
    "Convert the colored block into a modern platform bed with white cotton bedding and fluffy pillows. "
    "Walls become real painted drywall with slight imperfections, floor becomes genuine hardwood with natural grain, "
    "soft natural window lighting, realistic shadows, DSLR architectural photography aesthetic, "
    "sharp but natural micro-textures on wood and fabric, correct perspective."
)

VERTEX_PROMPT = (
    "Preserve the exact layout and furniture positions from the input image (do not add or remove objects). "
    "Convert CGI surfaces into authentic materials: painted drywall, genuine hardwood floor with visible grain, "
    "white cotton bedding with soft fabric texture. "
    "Soft natural daylight from a window (4500–5500K), realistic shadows and depth of field, "
    "professional interior/real-estate photography style, balanced exposure, calm atmosphere, "
    "avoid any CGI or plastic look."
)

# 기본 프롬프트
DEFAULT_PROMPT = STABILITY_PROMPT

# 네거티브: Stability/Replicate에서 사용 (Vertex에는 쓰지 않음)
NEGATIVE_PROMPT = (
    "cgi, 3d render, plastic look, flat shading, unnaturally clean surfaces, perfect geometry, "
    "cartoon, anime, illustration, stylized, low poly, colored blocks, text, watermark, logo, "
    "overexposed, underexposed, harsh shadows, oversaturated, noise, artifacts, blurry, lowres, "
    "video game render, architectural visualization"
)

# 3D 캡처 실사화용: 레이아웃 고정 강조 + 재질·조명 구체화
CAPTURE_TO_REAL_PROMPT = (
    "Keep the exact room layout and furniture arrangement from the input image. "
    "Convert CGI look to realistic interior photography. Use physically plausible materials "
    "(wood grains, fabric fibers, metal reflections), correct perspective, soft natural light, "
    "balanced white balance, subtle shadows, no over-sharpening."
)

# 선택 스타일 프리셋
STYLE_PRESETS = {
    "scandinavian": "light wood, white walls, linen fabrics, minimal decor, soft daylight",
    "modern": "neutral palette, matte finishes, clean lines, low contrast lighting", 
    "bohemian": "warm tones, layered textiles, plants, rattan, cozy ambient light",
    "japanese": "natural wood, tatami-inspired textures, shoji-like diffusion, calm ambiance",
}

# 권장 기본값 (레이아웃 보존을 위해 낮은 strength)
DEFAULT_STRENGTH = 0.3
DEFAULT_GUIDANCE = 9   # Stability cfg_scale 7–10 / Replicate scale 8–12 권장
