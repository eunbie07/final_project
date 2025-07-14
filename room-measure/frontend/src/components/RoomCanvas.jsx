// frontend/src/components/RoomCanvas.jsx
import React from "react";

const RoomCanvas = ({ x, y }) => {
  // 유효성 검사 및 기본값 설정
  const validX = isNaN(x) || x <= 0 ? 400 : x;
  const validY = isNaN(y) || y <= 0 ? 300 : y;
  
  console.log("🎨 RoomCanvas received:", { x, y, validX, validY });
  
  const maxCanvasWidth = 400; // 기준 너비(px)
  const aspectRatio = validX / validY;

  let canvasWidth = maxCanvasWidth;
  let canvasHeight = maxCanvasWidth / aspectRatio;

  // 세로가 더 긴 경우에는 기준을 반대로 설정
  if (aspectRatio < 1) {
    canvasHeight = maxCanvasWidth;
    canvasWidth = maxCanvasWidth * aspectRatio;
  }

  // 최소/최대 크기 제한
  canvasWidth = Math.max(100, Math.min(canvasWidth, 500));
  canvasHeight = Math.max(100, Math.min(canvasHeight, 500));

  return (
    <div className="mt-6">
      <h2 className="font-bold mb-2">📐 2D 평면도</h2>
      
      {/* 원본 데이터가 유효하지 않을 때 경고 표시 */}
      {(isNaN(x) || isNaN(y) || x <= 0 || y <= 0) && (
        <div className="mb-2 p-2 bg-yellow-100 border border-yellow-300 rounded text-sm">
          ⚠️ 측정 데이터에 문제가 있어 기본값으로 표시됩니다.
        </div>
      )}
      
      <svg
        width={canvasWidth + 100}
        height={canvasHeight + 80}
        style={{ border: "1px solid #ccc", backgroundColor: "#f9f9f9" }}
      >
        {/* 방 평면도 */}
        <rect
          x={50}
          y={30}
          width={canvasWidth}
          height={canvasHeight}
          fill="#d1e8ff"
          stroke="#2c82c9"
          strokeWidth="2"
        />
        
        {/* 가로 길이 표시 */}
        <text
          x={50 + canvasWidth / 2}
          y={20}
          textAnchor="middle"
          fontSize="14"
          fontWeight="bold"
          fill="#2c82c9"
        >
          가로 {validX.toFixed(1)}cm
        </text>
        
        {/* 세로 길이 표시 */}
        <text
          x={55 + canvasWidth}
          y={30 + canvasHeight / 2}
          textAnchor="start"
          fontSize="14"
          fontWeight="bold"
          fill="#2c82c9"
          transform={`rotate(90, ${55 + canvasWidth}, ${30 + canvasHeight / 2})`}
        >
          세로 {validY.toFixed(1)}cm
        </text>
        
        {/* 면적 표시 */}
        <text
          x={50 + canvasWidth / 2}
          y={30 + canvasHeight / 2}
          textAnchor="middle"
          fontSize="12"
          fill="#666"
        >
          {((validX * validY) / 10000).toFixed(1)}㎡
        </text>
        
        {/* 평수 표시 */}
        <text
          x={50 + canvasWidth / 2}
          y={30 + canvasHeight / 2 + 16}
          textAnchor="middle"
          fontSize="12"
          fill="#666"
        >
          ({(((validX * validY) / 10000) / 3.3058).toFixed(1)}평)
        </text>
      </svg>
      
      {/* 추가 정보 */}
      <div className="mt-2 text-xs text-gray-500">
        비율: {aspectRatio.toFixed(2)}:1 | 캔버스 크기: {canvasWidth.toFixed(0)}×{canvasHeight.toFixed(0)}px
      </div>
    </div>
  );
};

export default RoomCanvas;