/**
 * 좌표 변환 유틸리티 함수들
 */

// 로컬 좌표 변환 (백엔드 없이도 작동)
export const convertCoordinatesLocally = (id, newPosition, size, roomSize) => {
  const [x3d, y3d, z3d] = newPosition;
  const [roomWidth, roomHeight, roomDepth] = roomSize;
  
  // 3D 중심 좌표 → 2D 왼쪽아래 좌표로 변환
  let x_2d = x3d - size[0] / 2;
  let z_2d = z3d - size[2] / 2;
  
  // 경계 검사 - 방 경계 내로 제한
  const furniture_width = size[0];
  const furniture_depth = size[2];
  
  x_2d = Math.max(0, Math.min(x_2d, roomWidth - furniture_width));
  z_2d = Math.max(0, Math.min(z_2d, roomDepth - furniture_depth));
  
  return { x: x_2d, z: z_2d };
};

// 2D -> 3D 좌표 변환 (가구 배치용)
export const convertTo3DCoordinates = (item, furnitureSize) => {
  // 2D -> 3D 좌표 변환
  const x3d = item.x;
  const z3d = item.z;

  return [
    x3d + furnitureSize[0] / 2,
    furnitureSize[1] / 2,
    z3d + furnitureSize[2] / 2,
  ];
};

// 3D 중심 좌표를 2D 왼쪽 상단 모서리로 변환
export const convertTo2DCoordinates = (position3D, size) => {
  const [x3D, y3D, z3D] = position3D;
  
  // 3D 중심점을 2D 왼쪽 상단 모서리로 변환
  const x2D = x3D - size[0] / 2;
  const z2D = z3D - size[2] / 2;
  
  return { x: x2D, z: z2D };
};