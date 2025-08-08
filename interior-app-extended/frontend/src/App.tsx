import React, { useRef, useState } from "react";
import Uploader from "./components/Uploader";
import MaskCanvas, { MaskCanvasHandle } from "./components/MaskCanvas";
import StyleSelect from "./components/StyleSelect";
import PresetButtons from "./components/PresetButtons";
import ModelSelect from "./components/ModelSelect";
import PromptTemplates from "./components/PromptTemplates";
import Gallery from "./components/Gallery";

export default function App() {
  const [imgEl, setImgEl] = useState<HTMLImageElement | undefined>();
  const [imgFile, setImgFile] = useState<File | undefined>();
  const maskBlobRef = useRef<Blob | null>(null);
  const maskRef = useRef<MaskCanvasHandle>(null);
  const [style, setStyle] = useState("modern cozy bedroom, warm neutral tones, soft daylight");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [brush, setBrush] = useState(28);
  const [noNewFurniture, setNoNewFurniture] = useState(true);
  const [strength, setStrength] = useState(0.35);
  const [model, setModel] = useState<"dalle" | "stability" | "two-step">("two-step");
  const [gallery, setGallery] = useState<string[]>([]);

  const onImage = (img: HTMLImageElement, file: File) => {
    setImgEl(img);
    setImgFile(file);
    setResult(null);
    maskRef.current?.clearMask();
  };

  const onMaskBlob = (b: Blob) => { maskBlobRef.current = b; };

  const buildPrompt = () => {
    let p = style + ", keep furniture layout, do not move the bed";
    if (noNewFurniture) {
      p += ", do not add any furniture, keep existing furniture only";
    }
    return p;
  };

  const onGenerate = async () => {
    if (!imgFile) { alert("이미지를 업로드하세요."); return; }
    setLoading(true);
    try {
      let endpoint = "/api/generate-two-step";
      const form = new FormData();
      if (model === "dalle") {
        if (!maskBlobRef.current) { alert("마스크를 칠하세요."); setLoading(false); return; }
        endpoint = "/api/generate-dalle-only";
        form.append("image", imgFile);
        form.append("mask", maskBlobRef.current, "mask.png");
        form.append("prompt", buildPrompt());
      } else if (model === "stability") {
        endpoint = "/api/generate-stability-only";
        form.append("image", imgFile);
        form.append("prompt", buildPrompt());
        form.append("strength", String(strength));
      } else {
        if (!maskBlobRef.current) { alert("마스크를 칠하세요."); setLoading(false); return; }
        endpoint = "/api/generate-two-step";
        form.append("image", imgFile);
        form.append("mask", maskBlobRef.current, "mask.png");
        form.append("prompt", buildPrompt());
        form.append("strength", String(strength));
      }

      const res = await fetch(endpoint, { method: "POST", body: form });
      const json = await res.json();
      if (json.error) throw new Error(json.error.message || json.error);
      const b64 = json.imageB64;
      const dataUrl = `data:image/png;base64,${b64}`;
      setResult(dataUrl);
      setGallery((g) => [dataUrl, ...g].slice(0, 12));
    } catch (e) {
      alert(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div>
        <h2>원본 업로드 및 마스크</h2>
        <Uploader onImage={onImage} />
        <ModelSelect value={model} onChange={setModel} />
        {model !== "stability" && (
          <>
            <p>바꿀 영역만 마우스로 칠하거나, 프리셋 버튼으로 빠르게 지정하세요. 침대 위치를 유지하려면 침대 위는 칠하지 마세요.</p>
            <PresetButtons onApply={(key)=>maskRef.current?.applyPreset(key)} />
            <MaskCanvas ref={maskRef} baseImage={imgEl} brushSize={brush} onMaskBlob={onMaskBlob} />
          </>
        )}

        <div className="controls">
          <label>브러시 크기</label>
          <input type="range" min={8} max={64} value={brush} onChange={(e)=>setBrush(parseInt(e.target.value))} />
          <label>
            <input
              type="checkbox"
              checked={noNewFurniture}
              onChange={(e) => setNoNewFurniture(e.target.checked)}
            />
            가구 추가 금지
          </label>
          {model !== "dalle" && (
            <>
              <label>구조 보존 강도(SDXL strength)</label>
              <input type="range" min={0.1} max={0.8} step={0.05} value={strength} onChange={(e)=>setStrength(parseFloat(e.target.value))} />
              <span>{strength}</span>
            </>
          )}
          <button onClick={()=>maskRef.current?.clearMask()}>마스크 모두 지우기</button>
        </div>

        <PromptTemplates onPick={(txt)=>setStyle(txt)} />
        <StyleSelect value={style} onChange={setStyle} />

        <div className="controls">
          <button onClick={onGenerate} disabled={loading}>
            {loading ? "생성 중..." : "이미지 생성"}
          </button>
        </div>
      </div>

      <div>
        <h2>결과</h2>
        {result ? <img className="result" src={result} /> : <div>아직 결과가 없습니다.</div>}
        <div style={{ marginTop: 12 }}>
          {result && (<a href={result} download="interior.png">다운로드</a>)}
        </div>
        <hr className="sep" />
        <h3>히스토리</h3>
        <Gallery items={gallery} />
      </div>
    </div>
  );
}
