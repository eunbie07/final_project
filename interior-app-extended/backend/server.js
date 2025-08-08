import express from "express";
import multer from "multer";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import axios from "axios";
import os from "os";
import path from "path";

dotenv.config();
const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });

// DALL·E 인페인팅 전용
app.post(
  "/api/generate-dalle-only",
  upload.fields([{ name: "image" }, { name: "mask" }]),
  async (req, res) => {
    const cleanup = () => {
      try {
        fs.unlinkSync(req.files?.image?.[0]?.path);
      } catch {}
      try {
        fs.unlinkSync(req.files?.mask?.[0]?.path);
      } catch {}
    };
    try {
      const prompt =
        req.body.prompt || "modern cozy bedroom, keep furniture layout";

      const formData = new FormData();
      formData.append("model", "gpt-image-1");
      formData.append("image", fs.createReadStream(req.files.image[0].path));
      formData.append("mask", fs.createReadStream(req.files.mask[0].path));
      formData.append("prompt", prompt);
      formData.append("size", "1024x1024");
      formData.append("n", "1");
      formData.append("response_format", "b64_json");

      const response = await axios.post(
        "https://api.openai.com/v1/images/edits",
        formData,
        {
          headers: {
            Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          },
        }
      );

      cleanup();
      if (!response.data.data || !response.data.data[0]?.b64_json) {
        return res
          .status(400)
          .json({ error: response.data.error || "OpenAI error" });
      }
      return res.json({ imageB64: response.data.data[0].b64_json });
    } catch (e) {
      cleanup();
      return res.status(500).json({ error: String(e) });
    }
  }
);

// Stability SDXL image-to-image 전용
app.post(
  "/api/generate-stability-only",
  upload.single("image"),
  async (req, res) => {
    const cleanup = () => {
      try {
        fs.unlinkSync(req.file?.path);
      } catch {}
    };
    try {
      const stylePrompt = req.body.prompt || "modern cozy bedroom";
      const strength = req.body.strength || "0.35";
      const stabilityHost =
        process.env.STABILITY_API_HOST || "https://api.stability.ai";
      const sdxlPath =
        process.env.STABILITY_SDXL_PATH ||
        "/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image";
      const url = `${stabilityHost}${sdxlPath}`;

      const formData = new FormData();
      formData.append("init_image", fs.createReadStream(req.file.path));
      formData.append("text_prompts[0][text]", stylePrompt);
      formData.append("text_prompts[0][weight]", "1");
      formData.append("cfg_scale", "7");
      formData.append("sampler", "K_DPM_2_ANCESTRAL");
      formData.append("samples", "1");
      formData.append("steps", "30");
      formData.append("image_strength", strength);

      const response = await axios.post(url, formData, {
        headers: {
          Authorization: `Bearer ${process.env.STABILITY_API_KEY}`,
        },
        responseType: "arraybuffer",
      });

      cleanup();
      const outB64 = Buffer.from(response.data).toString("base64");
      if (!outB64)
        return res
          .status(400)
          .json({ error: "No image returned from Stability" });
      return res.json({ imageB64: outB64 });
    } catch (e) {
      cleanup();
      return res.status(500).json({ error: String(e) });
    }
  }
);

// 2단계 파이프라인: Stability SDXL -> DALL·E 인페인팅
app.post(
  "/api/generate-two-step",
  upload.fields([{ name: "image" }, { name: "mask" }]),
  async (req, res) => {
    const cleanup = () => {
      try {
        fs.unlinkSync(req.files?.image?.[0]?.path);
      } catch {}
      try {
        fs.unlinkSync(req.files?.mask?.[0]?.path);
      } catch {}
      const tempFilePath = path.join(os.tmpdir(), "sdxl_out.png");
      try {
        fs.unlinkSync(tempFilePath);
      } catch {}
    };
    try {
      const stylePrompt =
        req.body.prompt || "modern cozy bedroom, keep furniture layout";
      const strength = req.body.strength || "0.35";
      const stabilityHost =
        process.env.STABILITY_API_HOST || "https://api.stability.ai";
      const sdxlPath =
        process.env.STABILITY_SDXL_PATH ||
        "/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image";
      const url = `${stabilityHost}${sdxlPath}`;

      const formData1 = new FormData();
      formData1.append(
        "init_image",
        fs.createReadStream(req.files.image[0].path)
      );
      formData1.append("text_prompts[0][text]", stylePrompt);
      formData1.append("text_prompts[0][weight]", "1");
      formData1.append("cfg_scale", "7");
      formData1.append("sampler", "K_DPM_2_ANCESTRAL");
      formData1.append("samples", "1");
      formData1.append("steps", "30");
      formData1.append("image_strength", strength);

      const response1 = await axios.post(url, formData1, {
        headers: {
          Authorization: `Bearer ${process.env.STABILITY_API_KEY}`,
        },
        responseType: "arraybuffer",
      });

      const sdxlImageBuffer = Buffer.from(response1.data);
      const tempFilePath = path.join(os.tmpdir(), "sdxl_out.png");
      fs.writeFileSync(tempFilePath, sdxlImageBuffer);

      const formData2 = new FormData();
      formData2.append("model", "gpt-image-1");
      formData2.append("image", fs.createReadStream(tempFilePath));
      formData2.append("mask", fs.createReadStream(req.files.mask[0].path));
      formData2.append(
        "prompt",
        stylePrompt +
          ", do not add any furniture, keep existing furniture only, do not move the bed"
      );
      formData2.append("size", "1024x1024");
      formData2.append("n", "1");
      formData2.append("response_format", "b64_json");

      const response2 = await axios.post(
        "https://api.openai.com/v1/images/edits",
        formData2,
        {
          headers: {
            Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          },
        }
      );

      cleanup();
      if (!response2.data.data || !response2.data.data[0]?.b64_json) {
        return res
          .status(400)
          .json({ error: response2.data.error || "OpenAI error" });
      }
      return res.json({ imageB64: response2.data.data[0].b64_json });
    } catch (e) {
      cleanup();
      return res.status(500).json({ error: String(e) });
    }
  }
);

const port = process.env.PORT || 7001;
app.listen(port, () => console.log(`Backend on http://localhost:${port}`));
