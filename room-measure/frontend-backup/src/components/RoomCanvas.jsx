// frontend/src/components/RoomCanvas.jsx
import React from "react";

const RoomCanvas = ({ x, y }) => {
  // 유효성 검사 및 기본값 설정
  const validX = isNaN(x) || x <= 0 ? 400 : x; // 가로 (width)
  const validY = isNaN(y) || y <= 0 ? 300 : y; // 세로 (depth)

  console.log("RoomCanvas 입력값:", { x: validX, y: validY });
  console.log("실제 비율:", (validX / validY).toFixed(2));

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

  console.log("캔버스 크기:", {
    width: canvasWidth.toFixed(0),
    height: canvasHeight.toFixed(0),
    ratio: (canvasWidth / canvasHeight).toFixed(2),
  });

  // 치수 텍스트 스타일
  const dimFont = {
    fontSize: 16,
    fontWeight: 700,
    fill: "#1E293B",
    fontFamily: "inherit",
  };

  return (
    <div className="flex flex-col items-center justify-center bg-surface py-8 rounded-lg shadow-sm border border-border">
      {/* Room layout 제목 */}
      <div className="mb-4 text-xl font-bold text-text-primary text-center">
        <strong>Room Layout</strong>
      </div>

      {/* 실제 크기 정보 */}
      <div className="mb-4 text-sm text-text-secondary text-center">
        실제 크기: {validX.toFixed(0)} × {validY.toFixed(0)} cm | 비율:{" "}
        {aspectRatio.toFixed(2)}:1 |{((validX * validY) / 10000).toFixed(1)}㎡
      </div>

      <svg
        width={canvasWidth + 100}
        height={canvasHeight + 80}
        style={{ background: "#fff", display: "block" }}
        viewBox={`0 0 ${canvasWidth + 100} ${canvasHeight + 80}`}
      >
        {/* Y축 뒤집기 - 왼쪽 아래가 (0,0)이 되도록 변환 */}
        <g transform={`scale(1, -1) translate(0, -${canvasHeight + 80})`}>
          {/* 방 평면도 윤곽 */}
          <rect
            x={50}
            y={50}
            width={canvasWidth}
            height={canvasHeight}
            fill="#F1F5F9"
            stroke="#334155"
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
                stroke="#E2E8F0"
                strokeWidth="0.5"
              />
            </pattern>
          </defs>
          <rect
            x={50}
            y={50}
            width={canvasWidth}
            height={canvasHeight}
            fill="url(#roomGrid)"
          />

          {/* 좌표계 표시 (원점 표시) */}
          <circle
            cx={50}
            cy={50}
            r={3}
            fill="#EF4444"
            stroke="white"
            strokeWidth={1}
          />

          {/* 방향 표시 (뒤집힌 상태에서는 텍스트도 뒤집어야 함) */}
          <g transform={`scale(1, -1) translate(0, -${50 + canvasHeight / 2})`}>
            <text
              x={50 + canvasWidth / 2}
              y={0}
              textAnchor="middle"
              style={{ fontSize: 12, fill: "#475569", fontWeight: 500 }}
            >
              Width × Depth
            </text>
            <text
              x={50 + canvasWidth / 2}
              y={15}
              textAnchor="middle"
              style={{ fontSize: 12, fill: "#475569", fontWeight: 500 }}
            >
              {validX} × {validY}
            </text>
          </g>
        </g>

        {/* 치수선과 텍스트는 정상 방향으로 유지 */}
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
          stroke="#334155"
          strokeWidth={2}
        />
        <line
          x1={50}
          y1={canvasHeight + 40}
          x2={50}
          y2={canvasHeight + 50}
          stroke="#334155"
          strokeWidth={2}
        />
        <line
          x1={50 + canvasWidth}
          y1={canvasHeight + 40}
          x2={50 + canvasWidth}
          y2={canvasHeight + 50}
          stroke="#334155"
          strokeWidth={2}
        />

        {/* 세로 치수선 */}
        <line
          x1={35}
          y1={30}
          x2={35}
          y2={30 + canvasHeight}
          stroke="#334155"
          strokeWidth={2}
        />
        <line
          x1={30}
          y1={30}
          x2={40}
          y2={30}
          stroke="#334155"
          strokeWidth={2}
        />
        <line
          x1={30}
          y1={30 + canvasHeight}
          x2={40}
          y2={30 + canvasHeight}
          stroke="#334155"
          strokeWidth={2}
        />

        {/* 원점 라벨 */}
        <text
          x={50}
          y={canvasHeight + 75}
          textAnchor="middle"
          style={{ fontSize: 10, fill: "#EF4444", fontWeight: 600 }}
        >
          (0,0)
        </text>
      </svg>

      {/* 비율 정보 */}
      <div className="mt-4 text-xs text-text-secondary text-center max-w-md">
        <strong>비율 정보:</strong>
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
