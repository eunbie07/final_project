// src/components/LayoutRoomVisualizer2D.jsx
import React from 'react';

const ROOM_SCALE = 1; // 캔버스 배율 (1 = 1px per cm)
const REAL_BED_WIDTH_CM = 200; // 실제 침대 가로 길이 기준

const LayoutRoomVisualizer2D = ({ layout }) => {
  if (!layout || layout.length === 0) return null;

  // 기준 객체: 침대
  const bed = layout.find(obj => obj.label === 'bed');
  if (!bed) return <p>침대를 기준으로 크기를 환산할 수 없습니다.</p>;

  const cmPerRatio = REAL_BED_WIDTH_CM / bed.w;

  // 전체 감지 범위 계산
  const minX = Math.min(...layout.map(obj => obj.x));
  const minY = Math.min(...layout.map(obj => obj.y));
  const maxX = Math.max(...layout.map(obj => obj.x + obj.w));
  const maxY = Math.max(...layout.map(obj => obj.y + obj.h));

  // 방 전체 크기 계산
  const roomWidthRatio = maxX - minX;
  const roomHeightRatio = maxY - minY;
  const roomWidthCm = roomWidthRatio * cmPerRatio;
  const roomHeightCm = roomHeightRatio * cmPerRatio;

  const canvasWidth = roomWidthCm * ROOM_SCALE;
  const canvasHeight = roomHeightCm * ROOM_SCALE;

  return (
    <div>
      <h4>📏 실제 크기 기반 평면도 (단위: cm):</h4>
      <p>🧱 방 전체 크기: <strong>{Math.round(roomWidthCm)}cm × {Math.round(roomHeightCm)}cm</strong></p>
      <svg width={canvasWidth} height={canvasHeight} style={{ border: '1px solid #ccc' }}>
        {/* 방 배경 */}
        <rect x={0} y={0} width={canvasWidth} height={canvasHeight} fill="#fff" />

        {/* 감지된 가구 시각화 */}
        {layout.map((obj, idx) => {
          const x = (obj.x - minX) * cmPerRatio * ROOM_SCALE;
          const y = (obj.y - minY) * cmPerRatio * ROOM_SCALE;
          const w = obj.w * cmPerRatio * ROOM_SCALE;
          const h = obj.h * cmPerRatio * ROOM_SCALE;

          return (
            <g key={idx}>
              <rect
                x={x}
                y={y}
                width={w}
                height={h}
                fill="#a2d2ff"
                stroke="#333"
              />
              <text x={x + 5} y={y + 15} fontSize="10" fill="black">
                {obj.label} ({Math.round(obj.w * cmPerRatio)}×{Math.round(obj.h * cmPerRatio)}cm)
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

export default LayoutRoomVisualizer2D;
