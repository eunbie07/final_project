# Provider별 최적화된 프롬프트
STABILITY_PROMPT = (
    "A hyperrealistic, professional photograph of a modern bedroom interior, captured with a DSLR camera, natural depth of field, and authentic textures. "
    "The large pink block in the image MUST be transformed into a real, comfortable bed with white sheets and pillows. "
    "The entire scene should have soft, natural lighting from a window, creating realistic shadows. "
    "The floor is made of genuine hardwood. "
    "The overall mood is calm and peaceful. "
    "NO 3D render elements, NO simple colored blocks."
)

REPLICATE_PROMPT = (
    "Transform this 3D render into a photorealistic interior photograph: "
    "Convert the pink bed block into a real modern platform bed with white cotton bedding, fluffy pillows. "
    "White walls become real painted drywall with subtle texture and imperfections. "
    "Floor transforms to genuine hardwood with natural grain and realistic finish. "
    "Add natural window lighting with authentic shadows and depth of field. "
    "Professional architectural photography aesthetic, DSLR camera quality, "
    "realistic materials and textures throughout."
)

VERTEX_PROMPT = (
    "High-quality interior design photography: "
    "Transform pink bed block into realistic modern bedroom furniture with white bedding. "
    "Convert 3D rendered surfaces into authentic materials - painted walls, hardwood floors. "
    "Natural lighting from windows creating realistic shadows and ambiance. "
    "Professional real estate photography style, "
    "genuine textures and materials, calm peaceful atmosphere."
)

# 기본 프롬프트 (백워드 호환성)
DEFAULT_PROMPT = STABILITY_PROMPT
NEGATIVE_PROMPT = (
    "3d render, 3d rendering, CGI, computer graphics, digital art, artificial surfaces, synthetic materials, "
    "perfect geometry, clean surfaces, uniform lighting, plastic appearance, fake textures, "
    "cartoon, anime, illustration, low poly, geometric shapes, pink box, red box, colored blocks, "
    "sterile environment, too clean, unrealistic perfection, computer generated lighting, "
    "flat colors, uniform textures, artificial shadows, digital artifacts, "
    "blurry, distorted, deformed, low quality, watermark, oversaturated, "
    "game graphics, video game render, architectural visualization style, "
    "digital painting, digital illustration, stylized, unrealistic, flat lighting, over-processed, fake lighting"
)
DEFAULT_STRENGTH = 0.3
DEFAULT_GUIDANCE = 12
