import React from 'react';

const ROOM_SCALE = 1;

const LayoutRoomVisualizer2D = ({ layout }) => {
  if (!layout || layout.length === 0) return null;

  const bed = layout.find(obj => obj.label === 'bed');
  if (!bed) return <p>bed 감지 실패</p>;

  // ✅ layout의 단위가 cm라고 가정
  const minX = Math.min(...layout.map(obj => obj.x));
  const minY = Math.min(...layout.map(obj => obj.y));
  const maxX = Math.max(...layout.map(obj => obj.x + obj.w));
  const maxY = Math.max(...layout.map(obj => obj.y + obj.h));

  const roomWidthCm = maxX - minX;
  const roomHeightCm = maxY - minY;

  const canvasWidth = roomWidthCm * ROOM_SCALE;
  const canvasHeight = roomHeightCm * ROOM_SCALE;

  return (
    <div>
      <h4>📏 실제 크기 기반 평면도 (단위: cm):</h4>
      <p><b>방 전체 크기:</b> {Math.round(roomWidthCm)}cm × {Math.round(roomHeightCm)}cm</p>

      <svg
        viewBox={`0 0 ${canvasWidth} ${canvasHeight + 30}`}
        style={{ width: '100%', height: 'auto', border: '1px solid #ccc' }}
        preserveAspectRatio="xMidYMid meet"
      >
        <rect x={0} y={0} width={canvasWidth} height={canvasHeight} fill="#fefefe" />
        <text x={10} y={20} fontSize="14" fill="#333">
          Room Size: {Math.round(roomWidthCm)}cm × {Math.round(roomHeightCm)}cm
        </text>

        {layout.map((obj, idx) => {
          const realX = (obj.x - minX) * ROOM_SCALE;
          const realY = (obj.y - minY) * ROOM_SCALE;
          const realW = obj.w * ROOM_SCALE;
          const realH = obj.h * ROOM_SCALE;

          return (
            <g key={idx}>
              <rect
                x={realX}
                y={realY}
                width={realW}
                height={realH}
                fill="#a2d2ff"
                stroke="#333"
              />
              <text x={realX + 5} y={realY + 15} fontSize="10" fill="black">
                {obj.label} ({Math.round(obj.w)}×{Math.round(obj.h)}cm)
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

export default LayoutRoomVisualizer2D;
