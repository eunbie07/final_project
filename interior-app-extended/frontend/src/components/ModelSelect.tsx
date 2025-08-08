import React from "react";

type Props = {
  value: "dalle" | "stability" | "two-step";
  onChange: (v: "dalle" | "stability" | "two-step") => void;
};

export default function ModelSelect({ value, onChange }: Props) {
  return (
    <div className="controls">
      <label>모델</label>
      <select value={value} onChange={(e)=>onChange(e.target.value as any)}>
        <option value="two-step">Stability SDXL → DALL·E 인페인팅</option>
        <option value="stability">Stability SDXL 단독</option>
        <option value="dalle">DALL·E 인페인팅 단독</option>
      </select>
    </div>
  );
}
