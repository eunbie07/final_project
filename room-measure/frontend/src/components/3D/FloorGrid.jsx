import React from "react";
import { Line } from "@react-three/drei";

// 바닥 그리드와 거리 표시
const FloorGrid = React.memo(function FloorGrid({ roomSize, visible = true }) {
  if (!visible) return null;

  const [roomWidth, roomHeight, roomDepth] = roomSize;
  const gridSize = 100; // 1m
  const lines = [];

  for (let x = 0; x <= roomWidth; x += gridSize) {
    lines.push(
      <Line
        key={`major-v-${x}`}
        points={[
          [x, 0.3, 0],
          [x, 0.3, roomDepth],
        ]}
        color="#E2E8F0"
        lineWidth={1}
      />
    );
  }

  for (let z = 0; z <= roomDepth; z += gridSize) {
    lines.push(
      <Line
        key={`major-h-${z}`}
        points={[
          [0, 0.3, z],
          [roomWidth, 0.3, z],
        ]}
        color="#E2E8F0"
        lineWidth={1}
      />
    );
  }

  return <group>{lines}</group>;
});

export default FloorGrid;