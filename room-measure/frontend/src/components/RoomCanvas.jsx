// frontend/src/components/RoomCanvas.jsx
import React from "react";

const RoomCanvas = ({ x, y }) => {
  // 유효성 검사 및 기본값 설정
  const validX = isNaN(x) || x <= 0 ? 400 : x; // 가로 (width)
  const validY = isNaN(y) || y <= 0 ? 300 : y; // 세로 (depth)

  console.log("🏠 RoomCanvas 입력값:", { x: validX, y: validY });
  console.log("🏠 실제 비율:", (validX / validY).toFixed(2));

  // 최대 캔버스 크기 설정
  const maxCanvasSize = 400;

  // 실제 비율 계산
  const aspectRatio = validX / validY;

  // 캔버스 크기 계산 (비율 유지)
  let canvasWidth, canvasHeight;

  if (aspectRatio >= 1) {
    // 가로가 더 긴 경우
    canvasWidth = maxCanvasSize;
    canvasHeight = maxCanvasSize / aspectRatio;
  } else {
    // 세로가 더 긴 경우
    canvasHeight = maxCanvasSize;
    canvasWidth = maxCanvasSize * aspectRatio;
  }

  // 최소/최대 크기 제한
  canvasWidth = Math.max(200, Math.min(canvasWidth, 500));
  canvasHeight = Math.max(150, Math.min(canvasHeight, 400));

  console.log("🖼️ 캔버스 크기:", {
    width: canvasWidth.toFixed(0),
    height: canvasHeight.toFixed(0),
    ratio: (canvasWidth / canvasHeight).toFixed(2),
  });

  // 치수 텍스트 스타일
  const dimFont = {
    fontSize: 16,
    fontWeight: 700,
    fill: "#1F2937",
    fontFamily: "inherit",
  };

  return (
    <div className="flex flex-col items-center justify-center bg-white py-8 rounded-lg shadow-sm border border-gray-200">
      {/* Room layout 제목 */}
      <div className="mb-4 text-xl font-bold text-gray-800 text-center">
        📐 Room Layout
      </div>

      {/* 실제 크기 정보 */}
      <div className="mb-4 text-sm text-gray-600 text-center">
        실제 크기: {validX.toFixed(0)} × {validY.toFixed(0)} cm | 비율:{" "}
        {aspectRatio.toFixed(2)}:1 |{((validX * validY) / 10000).toFixed(1)}㎡
      </div>

      <svg
        width={canvasWidth + 100}
        height={canvasHeight + 80}
        style={{ background: "#fff", display: "block" }}
      >
        {/* 방 평면도 윤곽 */}
        <rect
          x={50}
          y={30}
          width={canvasWidth}
          height={canvasHeight}
          fill="#f8fafc"
          stroke="#1F2937"
          strokeWidth={3}
        />

        {/* 내부 그리드 (선택사항) */}
        <defs>
          <pattern
            id="roomGrid"
            width="20"
            height="20"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 20 0 L 0 0 0 20"
              fill="none"
              stroke="#e2e8f0"
              strokeWidth="0.5"
            />
          </pattern>
        </defs>
        <rect
          x={50}
          y={30}
          width={canvasWidth}
          height={canvasHeight}
          fill="url(#roomGrid)"
        />

        {/* 가로 치수 (하단) */}
        <text
          x={50 + canvasWidth / 2}
          y={canvasHeight + 60}
          textAnchor="middle"
          style={dimFont}
        >
          {validX.toFixed(0)} cm
        </text>

        {/* 세로 치수 (좌측) */}
        <text
          x={25}
          y={30 + canvasHeight / 2}
          textAnchor="middle"
          style={dimFont}
          transform={`rotate(-90, 25, ${30 + canvasHeight / 2})`}
        >
          {validY.toFixed(0)} cm
        </text>

        {/* 가로 치수선 */}
        <line
          x1={50}
          y1={canvasHeight + 45}
          x2={50 + canvasWidth}
          y2={canvasHeight + 45}
          stroke="#374151"
          strokeWidth={2}
        />
        <line
          x1={50}
          y1={canvasHeight + 40}
          x2={50}
          y2={canvasHeight + 50}
          stroke="#374151"
          strokeWidth={2}
        />
        <line
          x1={50 + canvasWidth}
          y1={canvasHeight + 40}
          x2={50 + canvasWidth}
          y2={canvasHeight + 50}
          stroke="#374151"
          strokeWidth={2}
        />

        {/* 세로 치수선 */}
        <line
          x1={35}
          y1={30}
          x2={35}
          y2={30 + canvasHeight}
          stroke="#374151"
          strokeWidth={2}
        />
        <line
          x1={30}
          y1={30}
          x2={40}
          y2={30}
          stroke="#374151"
          strokeWidth={2}
        />
        <line
          x1={30}
          y1={30 + canvasHeight}
          x2={40}
          y2={30 + canvasHeight}
          stroke="#374151"
          strokeWidth={2}
        />

        {/* 방향 표시 */}
        <text
          x={50 + canvasWidth / 2}
          y={30 + canvasHeight / 2}
          textAnchor="middle"
          style={{ fontSize: 12, fill: "#6B7280", fontWeight: 500 }}
        >
          Width × Depth
        </text>
        <text
          x={50 + canvasWidth / 2}
          y={30 + canvasHeight / 2 + 15}
          textAnchor="middle"
          style={{ fontSize: 12, fill: "#6B7280", fontWeight: 500 }}
        >
          {validX} × {validY}
        </text>
      </svg>

      {/* 비율 정보 */}
      <div className="mt-4 text-xs text-gray-500 text-center max-w-md">
        💡 <strong>비율 정보:</strong>
        {aspectRatio >= 1.5
          ? " 가로가 매우 긴 방"
          : aspectRatio >= 1.2
          ? " 가로가 긴 방"
          : aspectRatio >= 0.8
          ? " 정사각형에 가까운 방"
          : " 세로가 긴 방"}
        <br />
        <span className="text-xs">
          (가로 {validX}cm ÷ 세로 {validY}cm = {aspectRatio.toFixed(2)})
        </span>
      </div>
    </div>
  );
};

export default RoomCanvas;
