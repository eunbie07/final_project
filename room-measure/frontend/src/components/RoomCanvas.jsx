// frontend/src/components/RoomCanvas.jsx
import React from "react";

const RoomCanvas = ({ x, y }) => {
  const maxCanvasWidth = 400; // 기준 너비(px)
  const aspectRatio = x / y;

  let canvasWidth = maxCanvasWidth;
  let canvasHeight = maxCanvasWidth / aspectRatio;

  // 세로가 더 긴 경우에는 기준을 반대로 설정
  if (aspectRatio < 1) {
    canvasHeight = maxCanvasWidth;
    canvasWidth = maxCanvasWidth * aspectRatio;
  }

  return (
    <div className="mt-6">
      <h2 className="font-bold mb-2">2D 평면도</h2>
      <svg
        width={canvasWidth + 100}
        height={canvasHeight + 80}
        style={{ border: "1px solid #ccc" }}
      >
        <rect
          x={50}
          y={30}
          width={canvasWidth}
          height={canvasHeight}
          fill="#d1e8ff"
          stroke="#2c82c9"
          strokeWidth="2"
        />
        <text
          x={50 + canvasWidth / 2}
          y={20}
          textAnchor="middle"
          fontSize="14"
        >
          가로 {x}cm
        </text>
        <text
          x={55 + canvasWidth}
          y={30 + canvasHeight / 2}
          textAnchor="start"
          fontSize="14"
          transform={`rotate(90, ${55 + canvasWidth}, ${30 + canvasHeight / 2})`}
        >
          세로 {y}cm
        </text>
      </svg>
    </div>
  );
};

export default RoomCanvas;
