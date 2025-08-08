import React from "react";
import type { PresetKey } from "./MaskCanvas";

export default function PresetButtons({ onApply }:{ onApply:(k:PresetKey)=>void }) {
  return (
    <div className="presetBar">
      <button onClick={()=>onApply("walls")}>벽 색만</button>
      <button onClick={()=>onApply("floor")}>바닥만</button>
      <button onClick={()=>onApply("bedding")}>침구만</button>
    </div>
  );
}
