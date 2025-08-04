import React from "react";

// 시점 프리셋
const ViewPresets = React.memo(function ViewPresets({
  onViewChange,
  roomSize,
}) {
  const [w, h, d] = roomSize;
  const presets = [
    {
      name: "조감도",
      position: [w / 2, h * 2, d / 2],
      target: [w / 2, 0, d / 2],
    },
    {
      name: "입구",
      position: [w / 2, h * 0.75, d + d * 0.1],
      target: [w / 2, h * 0.5, 0],
    },
    {
      name: "코너",
      position: [w + w * 0.1, h, d + d * 0.1],
      target: [0, 0, 0],
    },
    {
      name: "눈높이",
      position: [w / 2, 170, d * 0.8],
      target: [w / 2, 160, 0],
    },
  ];

  return (
    <div className="bg-surface/90 backdrop-blur p-2 rounded-lg border border-border">
      <h4 className="font-semibold text-xs mb-2 text-text-primary">시점 변경</h4>
      <div className="grid grid-cols-2 gap-1">
        {presets.map((preset) => (
          <button
            key={preset.name}
            onClick={() => onViewChange(preset)}
            className="flex flex-col items-center p-1 bg-background hover:bg-primary/10 rounded text-xs transition-colors text-text-secondary"
          >
            <span>{preset.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
});

export default ViewPresets;