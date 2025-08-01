import React from "react";
import { ContactShadows, Text } from "@react-three/drei";
import * as THREE from "three";

// 창문 컴포넌트 (사실적인 디자인)
export const Window3D = React.memo(function Window3D({
  position,
  size,
  wallPosition,
  roomSize,
  rotation = [0, 0, 0],
}) {
  const [width, height, depth] = size;
  const frameThickness = 4; // 프레임을 더 얇게
  const glassThickness = 0.5;

  return (
    <group position={position} rotation={rotation}>
      {/* 외부 창틀 (벽에 박힌 부분) */}
      <mesh castShadow receiveShadow position={[0, 0, 0]}>
        <boxGeometry
          args={[
            width + frameThickness,
            height + frameThickness,
            frameThickness,
          ]}
        />
        <meshStandardMaterial 
          color="#FFFFFF" 
          roughness={0.2} 
          metalness={0.1}
        />
      </mesh>

      {/* 하늘 배경 (창문 구멍을 통해 보이는 배경) */}
      <mesh position={[0, 0, frameThickness / 2 + 1]}>
        <planeGeometry args={[width * 0.8, height * 0.8]} />
        <meshBasicMaterial 
          color="#87CEEB" // 스카이블루
        />
      </mesh>
      
      {/* 구름 효과 */}
      <mesh position={[-width * 0.1, height * 0.1, frameThickness / 2 + 1.5]}>
        <planeGeometry args={[width * 0.15, height * 0.08]} />
        <meshBasicMaterial 
          color="#FFFFFF" 
          transparent={true}
          opacity={0.8}
        />
      </mesh>
      <mesh position={[width * 0.08, height * 0.05, frameThickness / 2 + 1.5]}>
        <planeGeometry args={[width * 0.12, height * 0.06]} />
        <meshBasicMaterial 
          color="#FFFFFF" 
          transparent={true}
          opacity={0.6}
        />
      </mesh>

      {/* 유리창 - 투명하게 */}
      <mesh position={[0, 0, frameThickness / 2]} receiveShadow>
        <boxGeometry args={[width * 0.85, height * 0.85, 1]} />
        <meshPhysicalMaterial
          color="#E8F4FD"
          transparent={true}
          opacity={0.1}
          roughness={0.02}
          metalness={0.0}
          transmission={0.9}
          thickness={0.1}
          ior={1.52}
          clearcoat={0.8}
          clearcoatRoughness={0.1}
        />
      </mesh>

      {/* 창문 프레임 (가로, 세로 분할) */}
      <group position={[0, 0, frameThickness / 2]}>
        {/* 세로 중앙 프레임 */}
        <mesh castShadow receiveShadow>
          <boxGeometry args={[3, height * 0.8, 2]} />
          <meshStandardMaterial 
            color="#FFFFFF" 
            roughness={0.2} 
            metalness={0.1}
          />
        </mesh>
        {/* 가로 중앙 프레임 */}
        <mesh castShadow receiveShadow>
          <boxGeometry args={[width * 0.8, 3, 2]} />
          <meshStandardMaterial 
            color="#FFFFFF" 
            roughness={0.2} 
            metalness={0.1}
          />
        </mesh>
      </group>

      {/* 창문 손잡이 */}
      <mesh castShadow receiveShadow position={[width * 0.3, 0, frameThickness]}>
        <cylinderGeometry args={[1.5, 1.5, 3, 8]} />
        <meshStandardMaterial 
          color="#D1D5DB" 
          roughness={0.1} 
          metalness={0.8}
        />
      </mesh>
      
      {/* 그림자 */}
      <ContactShadows
        position={[0, 0, -frameThickness - 2]}
        opacity={0.3}
        scale={Math.max(width, height) * 1.3}
        blur={2}
        far={20}
      />
    </group>
  );
});

// 벽에 창문을 배치하는 컴포넌트 (개선된 위치 계산)
export const WindowsOnWalls = React.memo(function WindowsOnWalls({
  windows,
  roomSize,
}) {
  const [roomWidth, roomHeight, roomDepth] = roomSize;

  if (!windows || windows.length === 0) return null;

  return (
    <group>
      {windows.map((window, index) => {
        // 백엔드에서 전달된 실제 창문 크기 사용 (cm 단위로 변환)
        const windowWidth3D = (window.width_meters || 1.2) * 100;
        const windowHeight3D = (window.height_meters || 1.5) * 100;

        let position = [0, 0, 0];
        let rotation = [0, 0, 0];
        const wallThickness = 5; // 벽 두께

        // 위치 계산 개선 - 백분율 기반에서 실제 좌표 기반으로
        let x_pos, y_pos, z_pos;
        
        // 창문 높이 계산 (사용자 y_position 슬라이더 값 반영)
        const userYPosition = window.y_position !== undefined ? window.y_position : 0.8; // 기본값 80%
        const calculatedYPos = userYPosition * roomHeight; // 사용자 설정 높이
        
        switch (window.wall_position) {
          case "front":
            // 앞벽: Z 최대값
            x_pos = window.x_position ? window.x_position * roomWidth : roomWidth / 2;
            y_pos = calculatedYPos;
            z_pos = roomDepth - wallThickness;
            rotation = [0, 0, 0];
            break;
          case "back":
            // 뒷벽: Z=0
            x_pos = window.x_position ? window.x_position * roomWidth : roomWidth / 2;
            y_pos = calculatedYPos;
            z_pos = wallThickness;
            rotation = [0, 0, 0];
            break;
          case "left":
            // 왼쪽 벽: X=0
            x_pos = wallThickness;
            y_pos = calculatedYPos;
            z_pos = window.x_position ? window.x_position * roomDepth : roomDepth / 2;
            rotation = [0, Math.PI / 2, 0];
            break;
          case "right":
            // 오른쪽 벽: X 최대값
            x_pos = roomWidth - wallThickness;
            y_pos = calculatedYPos;
            z_pos = window.x_position ? window.x_position * roomDepth : roomDepth / 2;
            rotation = [0, -Math.PI / 2, 0];
            break;
          default:
            // 기본값: 뒷벽 중앙
            x_pos = roomWidth / 2;
            y_pos = calculatedYPos;
            z_pos = wallThickness;
            rotation = [0, Math.PI, 0];
        }
        
        position = [x_pos, y_pos, z_pos];

        // 범위 제한 (더 현실적인 범위로)
        const margin = 20; // 20cm 여백

        // X 좌표 제한 - 벽별 처리
        if (
          window.wall_position === "front" ||
          window.wall_position === "back"
        ) {
          position[0] = Math.max(
            windowWidth3D / 2 + margin,
            Math.min(roomWidth - windowWidth3D / 2 - margin, position[0])
          );
        }

        // Y 좌표 제한 (높이) - 창문이 바닥 아래로 가지 않도록
        const minHeight = windowHeight3D / 2 + 30; // 바닥에서 최소 30cm 위
        const maxHeight = roomHeight - windowHeight3D / 2 - margin; // 천장에서 여백
        position[1] = Math.max(minHeight, Math.min(maxHeight, position[1]));

        // Z 좌표 제한 - 벽별 처리
        if (
          window.wall_position === "left" ||
          window.wall_position === "right"
        ) {
          position[2] = Math.max(
            windowWidth3D / 2 + margin,
            Math.min(roomDepth - windowWidth3D / 2 - margin, position[2])
          );
        }

        // 사용자가 수직 위치를 설정하지 않은 경우에만 기본 높이 적용
        if (window.y_position === undefined && window.wall_position === "back") {
          const targetHeight = roomHeight * 0.8; // 높이의 80% 위치 (기본값)
          position[1] = Math.max(position[1], targetHeight);
        }

        return (
          <group key={`window-group-${index}`}>
            <Window3D
              key={`window-${index}`}
              position={position}
              size={[windowWidth3D, windowHeight3D, 10]}
              wallPosition={window.wall_position}
              roomSize={roomSize}
              rotation={rotation}
            />
            {/* 창문 정보 텍스트 */}
            <Text
              position={[
                position[0],
                position[1] - windowHeight3D / 2 - 15,
                position[2],
              ]}
              fontSize={8}
              color="#475569"
              anchorX="center"
              anchorY="middle"
            >
              {`${(windowWidth3D / 100).toFixed(1)}m × ${(
                windowHeight3D / 100
              ).toFixed(1)}m`}
            </Text>
          </group>
        );
      })}
    </group>
  );
});