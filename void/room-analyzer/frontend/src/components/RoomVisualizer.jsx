import React from "react";

const RoomVisualizer = ({ layout, roomWidth, roomHeight }) => {
  if (!layout || layout.length === 0) {
    return <p>시각화할 가구가 없습니다.</p>;
  }

  const scale = 2; // 확대 배율 (캔버스에서 cm → px 변환 비율)
  const canvasWidth = roomWidth * scale;
  const canvasHeight = roomHeight * scale;

  return (
    <div style={{ border: "1px solid #ccc", marginTop: "20px" }}>
      <svg
        width={canvasWidth}
        height={canvasHeight}
        style={{ background: "#f8f8f8" }}
      >
        {/* 가구 사각형 */}
        {layout.map((obj, index) => (
          <g key={index}>
            <rect
              x={obj.x * scale}
              y={obj.y * scale}
              width={obj.w * scale}
              height={obj.h * scale}
              fill="lightblue"
              stroke="#333"
              strokeWidth="1"
            />
            <text
              x={obj.x * scale + 5}
              y={obj.y * scale + 20}
              fontSize="12"
              fill="#000"
            >
              {obj.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default RoomVisualizer;
