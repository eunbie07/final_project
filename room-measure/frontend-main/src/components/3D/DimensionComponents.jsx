import React, { useMemo } from "react";
import { Line, Text } from "@react-three/drei";
import * as THREE from "three";

// 치수선 관련 컴포넌트들
const DIMENSION_COLOR = "#FFFFFF";

export const DimensionArrow = React.memo(function DimensionArrow({ start, end }) {
  const points = useMemo(() => {
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const direction = new THREE.Vector3()
      .subVectors(endVec, startVec)
      .normalize();
    const arrowSize = 10;
    const perpendicular = new THREE.Vector3(-direction.z, 0, direction.x);

    return {
      main: [start, end],
      startArrow1: [
        start,
        [
          start[0] +
            direction.x * arrowSize +
            perpendicular.x * arrowSize * 0.5,
          start[1],
          start[2] +
            direction.z * arrowSize +
            perpendicular.z * arrowSize * 0.5,
        ],
      ],
      startArrow2: [
        start,
        [
          start[0] +
            direction.x * arrowSize -
            perpendicular.x * arrowSize * 0.5,
          start[1],
          start[2] +
            direction.z * arrowSize -
            perpendicular.z * arrowSize * 0.5,
        ],
      ],
      endArrow1: [
        end,
        [
          end[0] - direction.x * arrowSize + perpendicular.x * arrowSize * 0.5,
          end[1],
          end[2] - direction.z * arrowSize + perpendicular.z * arrowSize * 0.5,
        ],
      ],
      endArrow2: [
        end,
        [
          end[0] - direction.x * arrowSize - perpendicular.x * arrowSize * 0.5,
          end[1],
          end[2] - direction.z * arrowSize - perpendicular.z * arrowSize * 0.5,
        ],
      ],
    };
  }, [start, end]);

  return (
    <>
      <Line points={points.main} color={DIMENSION_COLOR} lineWidth={2} />
      <Line points={points.startArrow1} color={DIMENSION_COLOR} lineWidth={2} />
      <Line points={points.startArrow2} color={DIMENSION_COLOR} lineWidth={2} />
      <Line points={points.endArrow1} color={DIMENSION_COLOR} lineWidth={2} />
      <Line points={points.endArrow2} color={DIMENSION_COLOR} lineWidth={2} />
    </>
  );
});

export const DimensionLabel = React.memo(function DimensionLabel({
  position,
  text,
  rotation = [0, 0, 0],
}) {
  return (
    <group position={position} rotation={rotation}>
      <Text
        position={[0, 0, 0]}
        fontSize={12}
        color={DIMENSION_COLOR}
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {text}
      </Text>
    </group>
  );
});

// 가구 치수 컴포넌트
export const FurnitureDimensions = React.memo(function FurnitureDimensions({
  position,
  size,
  rotation = [0, 0, 0],
  selected,
}) {
  if (!selected || !size) return null;

  const [originalWidth, height, originalDepth] = size;
  const [x, y, z] = position;
  const offset = 5;

  // 회전 각도에 따라 실제 width와 depth 계산
  const rotationY = rotation[1];
  const normalizedRotation = Math.abs(rotationY % (Math.PI * 2));
  const is90or270 = (normalizedRotation > Math.PI/4 && normalizedRotation < 3*Math.PI/4) || 
                    (normalizedRotation > 5*Math.PI/4 && normalizedRotation < 7*Math.PI/4);
  
  // 90도 또는 270도 회전 시 width와 depth를 바꿈
  const width = is90or270 ? originalDepth : originalWidth;
  const depth = is90or270 ? originalWidth : originalDepth;

  const baseY = y - height / 2; // 최하단 높이(y = 0에 붙음)

  return (
    <group>
      {/* 가로 치수선 (앞쪽에) */}
      <DimensionArrow
        start={[x - width / 2, baseY, z + depth / 2 + offset]}
        end={[x + width / 2, baseY, z + depth / 2 + offset]}
      />
      <DimensionLabel
        position={[x, baseY + 8, z + depth / 2 + offset]}
        text={`${width.toFixed(0)}cm`}
      />

      {/* 세로 치수선 (오른쪽에) */}
      <DimensionArrow
        start={[x + width / 2 + offset, baseY, z - depth / 2]}
        end={[x + width / 2 + offset, baseY, z + depth / 2]}
      />
      <DimensionLabel
        position={[x + width / 2 + offset + 8, baseY + 8, z]}
        text={`${depth.toFixed(0)}cm`}
        rotation={[0, Math.PI / 2, 0]}
      />

      {/* 높이 치수선 (위쪽으로) */}
      <DimensionArrow
        start={[x + width / 2 + offset, baseY, z + depth / 2 + offset]}
        end={[x + width / 2 + offset, baseY + height, z + depth / 2 + offset]}
      />
      <DimensionLabel
        position={[
          x + width / 2 + offset + 8,
          baseY + height / 2,
          z + depth / 2 + offset,
        ]}
        text={`${height.toFixed(0)}cm`}
      />
    </group>
  );
});

// 실시간 거리 측정 도구
export const DistanceMeasurer = React.memo(function DistanceMeasurer({
  point1,
  point2,
  visible = false,
}) {
  if (!visible || !point1) return null;

  // 거리 계산 (point2가 없으면 0)
  const distance = point2 
    ? Math.sqrt(
        Math.pow(point2[0] - point1[0], 2) + Math.pow(point2[2] - point1[2], 2)
      ).toFixed(0)
    : null;

  // 중점 계산 (point2가 없으면 point1 위에 표시)
  const midpoint = point2 
    ? [
        (point1[0] + point2[0]) / 2,
        Math.max(point1[1], point2[1]) + 10,
        (point1[2] + point2[2]) / 2,
      ]
    : [point1[0], point1[1] + 15, point1[2]];

  return (
    <group>
      {/* 첫 번째 측정 포인트 시각화 */}
      <mesh position={point1}>
        <sphereGeometry args={[5, 16, 16]} />
        <meshBasicMaterial color="#007bff" />
      </mesh>
      
      {/* 두 번째 측정 포인트 시각화 (있을 때만) */}
      {point2 && (
        <mesh position={point2}>
          <sphereGeometry args={[5, 16, 16]} />
          <meshBasicMaterial color="#007bff" />
        </mesh>
      )}
      
      {/* 연결선 (두 번째 포인트가 있을 때만) */}
      {point2 && (
        <Line points={[point1, point2]} color="#334155" lineWidth={3} />
      )}
      
      {/* 거리 텍스트 또는 대기 메시지 */}
      <Text
        position={midpoint}
        fontSize={12}
        color="#334155"
        anchorX="center"
        fontWeight="bold"
        backgroundColor="#FFFFFF"
        backgroundOpacity={0.8}
        padding={2}
      >
        {distance ? `${distance}cm` : "두 번째 점을 클릭하세요"}
      </Text>
    </group>
  );
});