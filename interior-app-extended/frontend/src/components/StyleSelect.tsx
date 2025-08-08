import React from "react";

type Props = { value: string; onChange: (v: string) => void; };

export default function StyleSelect({ value, onChange }: Props) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="modern cozy bedroom, warm neutral tones, soft daylight">모던 코지</option>
      <option value="scandinavian bedroom, light wood floor, white walls, linen bedding">북유럽</option>
      <option value="luxury hotel bedroom, elegant, soft indirect lighting">호텔 스타일</option>
      <option value="minimalist bedroom, clean lines, matte textures">미니멀</option>
    </select>
  );
}
