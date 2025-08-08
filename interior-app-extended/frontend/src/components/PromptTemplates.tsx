import React from "react";

export default function PromptTemplates({ onPick }:{ onPick:(t:string)=>void }) {
  const items = [
    ["모던 코지", "modern cozy bedroom, warm neutral tones, soft daylight, matte textures, subtle wood accents"],
    ["북유럽", "scandinavian bedroom, light wood, white walls, linen bedding, natural daylight, airy"],
    ["호텔 스타일", "luxury hotel bedroom, elegant design, soft indirect lighting, premium textiles"],
    ["미니멀", "minimalist bedroom, clean lines, neutral palette, smooth surfaces"]
  ];
  return (
    <div className="controls">
      <span>프롬프트 템플릿:</span>
      {items.map(([label, txt])=>(
        <button key={label} onClick={()=>onPick(txt)}>{label}</button>
      ))}
    </div>
  );
}
