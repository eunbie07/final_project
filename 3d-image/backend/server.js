const express = require("express");
const multer = require("multer");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const axios = require("axios");
const { PredictionServiceClient } = require("@google-cloud/aiplatform");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 9001;

// Google Cloud 설정 (실제 프로젝트 정보)
const PROJECT_ID = "virtual-muse-466706-v2";
const LOCATION = "us-central1";
const MODEL_ID = "imagegeneration@006"; // Imagen 모델

// Google Cloud 인증 설정
process.env.GOOGLE_APPLICATION_CREDENTIALS = path.join(
  __dirname,
  "service-account-key.json"
);

// Vertex AI 클라이언트 초기화
let predictionClient = null;
try {
  predictionClient = new PredictionServiceClient({
    apiEndpoint: `${LOCATION}-aiplatform.googleapis.com`,
  });
  console.log("✅ Vertex AI 클라이언트 초기화 성공");
  console.log(`📁 인증 파일: ${process.env.GOOGLE_APPLICATION_CREDENTIALS}`);
} catch (error) {
  console.log("⚠️ Vertex AI 클라이언트 초기화 실패 (모의 모드로 실행)");
  console.log("   - 오류:", error.message);
}

// 미들웨어 설정
app.use(
  cors({
    origin: ["http://localhost:9000", "http://127.0.0.1:9000"],
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With"],
  })
);
app.use(express.json());
app.use(express.static("public"));

// 상태 확인 라우트
app.get("/", (req, res) => {
  res.json({
    status: "running",
    message: "Vertex AI Interior Design API",
    endpoints: ["POST /api/change-furniture-style", "POST /api/photorealistic"],
    vertexAI: predictionClient ? "connected" : "mock mode",
  });
});

// 업로드 설정
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = "uploads";
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname)
    );
  },
});

const upload = multer({ storage: storage });

// Vertex AI 이미지 스타일 변경
app.post(
  "/api/change-furniture-style",
  upload.single("image"),
  async (req, res) => {
    try {
      if (!req.file) {
        return res
          .status(400)
          .json({ error: "이미지가 업로드되지 않았습니다." });
      }

      const style = req.body.style || "modern";
      const imagePath = req.file.path;

      // 이미지를 base64로 변환
      const imageBuffer = fs.readFileSync(imagePath);
      const base64Image = imageBuffer.toString("base64");

      // Vertex AI API 호출
      const result = await processImageWithVertexAI(base64Image, style);

      // 임시 파일 삭제
      fs.unlinkSync(imagePath);

      res.json(result);
    } catch (error) {
      console.error("Error:", error);
      res.status(500).json({ error: "이미지 처리 중 오류가 발생했습니다." });
    }
  }
);

// Vertex AI 실사화 변환
app.post("/api/photorealistic", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "이미지가 업로드되지 않았습니다." });
    }

    const imagePath = req.file.path;

    // 이미지를 base64로 변환
    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString("base64");

    // Vertex AI API 호출 (실사화)
    const result = await processImageWithVertexAI(
      base64Image,
      "photorealistic"
    );

    // 임시 파일 삭제
    fs.unlinkSync(imagePath);

    res.json(result);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "이미지 처리 중 오류가 발생했습니다." });
  }
});

// 실제 Vertex AI 이미지 처리 함수
async function processImageWithVertexAI(base64Image, style) {
  try {
    if (!predictionClient) {
      // Vertex AI 클라이언트가 없으면 모의 응답
      return await processImageWithMock(base64Image, style);
    }

    console.log(`🚀 Vertex AI 이미지 생성 시작: ${style} 스타일`);

    // 스타일별 프롬프트 설정
    const prompt = getPromptForStyle(style);

    console.log(`📝 프롬프트: ${prompt}`);

    // Vertex AI API 호출 - 이미지 생성 (이미지 입력 없이)
    const request = {
      endpoint: `projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}`,
      instances: [
        {
          prompt: prompt,
          parameters: {
            sampleCount: 1,
            aspectRatio: "1:1",
            safetyFilterLevel: "block_only_high",
            personGeneration: "dont_allow",
          },
        },
      ],
    };

    console.log(`🔗 API 엔드포인트: ${request.endpoint}`);

    const [response] = await predictionClient.predict(request);

    console.log(`📊 응답 구조:`, JSON.stringify(response, null, 2));

    if (response.predictions && response.predictions[0]) {
      const generatedImage = response.predictions[0].bytesBase64Encoded;

      return {
        success: true,
        imageUrl: `data:image/png;base64,${generatedImage}`,
        style: style,
        prompt: prompt,
        message: "Vertex AI 처리 완료",
        isRealAI: true,
      };
    } else {
      throw new Error("Vertex AI에서 이미지를 생성하지 못했습니다.");
    }
  } catch (error) {
    console.error("Vertex AI 오류:", error);
    console.log("🔄 모의 모드로 폴백...");
    return await processImageWithMock(base64Image, style);
  }
}

// 모의 이미지 처리 함수 (폴백용)
async function processImageWithMock(base64Image, style) {
  const prompt = getPromptForStyle(style);

  // 간단한 색상 변경을 시뮬레이션
  const styleColors = {
    modern: "#3498db", // 파란색
    scandinavian: "#e74c3c", // 빨간색
    industrial: "#2c3e50", // 어두운 회색
    luxury: "#f39c12", // 주황색
    minimalist: "#95a5a6", // 회색
    bohemian: "#9b59b6", // 보라색
    photorealistic: "#27ae60", // 초록색
  };

  const color = styleColors[style] || "#3498db";

  // 간단한 SVG 이미지 생성 (색상이 다른 사각형)
  const svgImage = `
    <svg width="400" height="400" xmlns="http://www.w3.org/2000/svg">
      <rect width="400" height="400" fill="${color}" opacity="0.8"/>
      <rect x="100" y="100" width="200" height="200" fill="white" opacity="0.9"/>
      <text x="200" y="220" text-anchor="middle" font-family="Arial" font-size="16" fill="black">
        ${style.toUpperCase()} STYLE
      </text>
      <text x="200" y="240" text-anchor="middle" font-family="Arial" font-size="12" fill="gray">
        AI Generated
      </text>
    </svg>
  `;

  // SVG를 base64로 인코딩
  const svgBase64 = Buffer.from(svgImage).toString("base64");

  return {
    success: true,
    imageUrl: `data:image/svg+xml;base64,${svgBase64}`,
    style: style,
    prompt: prompt,
    message: "Vertex AI 처리 완료 (모의 응답)",
    isRealAI: false,
  };
}

// 스타일별 프롬프트 생성
function getPromptForStyle(style) {
  const stylePrompts = {
    modern:
      "modern interior design, sleek furniture, minimalist style, clean lines, contemporary aesthetic, high-end materials",
    scandinavian:
      "scandinavian interior design, natural wood furniture, cozy atmosphere, light colors, hygge style, functional design",
    industrial:
      "industrial interior design, exposed brick walls, metal furniture, vintage elements, urban loft style, raw materials",
    luxury:
      "luxury interior design, premium furniture, elegant style, sophisticated decor, high-end finishes, opulent atmosphere",
    minimalist:
      "minimalist interior design, simple furniture, clean spaces, uncluttered design, essential elements only",
    bohemian:
      "bohemian interior design, eclectic furniture, artistic style, colorful decor, free-spirited atmosphere, vintage pieces",
    photorealistic:
      "photorealistic interior design, high quality, detailed furniture, natural lighting, realistic textures, professional photography style",
  };

  return stylePrompts[style] || stylePrompts.modern;
}

// 서버 시작
app.listen(PORT, () => {
  console.log(`🚀 Server is running on port ${PORT}`);
  console.log(`🎨 Vertex AI Interior Design API`);
  console.log(`- POST /api/change-furniture-style`);
  console.log(`- POST /api/photorealistic`);

  if (predictionClient) {
    console.log(`✅ Vertex AI 연결됨: ${PROJECT_ID} (${LOCATION})`);
    console.log(`🔑 인증 파일: ${process.env.GOOGLE_APPLICATION_CREDENTIALS}`);
  } else {
    console.log(`⚠️ 모의 모드로 실행 중 (Google Cloud 설정 필요)`);
  }
});
