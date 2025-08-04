import { FURNITURE_PRESETS } from '../constants/furniture.js';

/**
 * 데이터 변환 유틸리티 함수들
 */

// 창문 데이터를 저장 형식으로 변환하는 헬퍼 함수
export const convertWindowsToSaveFormat = (detectedWindows, roomDimensions) => {
  const { wMm, dMm, hMm } = roomDimensions;
  
  return detectedWindows.map((window, index) => {
    const widthMm = Math.round((window.width_meters || 1.2) * 1000);
    const heightMm = Math.round((window.height_meters || 1.5) * 1000);
    const userYPosition = window.y_position !== undefined ? window.y_position : 0.8;
    
    let wallNum, xMm, yMm, zMm;
    
    switch (window.wall_position) {
      case "front":
        wallNum = 3;
        xMm = Math.round((window.x_position || 0.5) * wMm);
        yMm = dMm;
        zMm = Math.round(userYPosition * hMm);
        break;
      case "back":
        wallNum = 1;
        xMm = Math.round((window.x_position || 0.5) * wMm);
        yMm = 0;
        zMm = Math.round(userYPosition * hMm);
        break;
      case "left":
        wallNum = 4;
        xMm = 0;
        yMm = Math.round((window.x_position || 0.5) * dMm);
        zMm = Math.round(userYPosition * hMm);
        break;
      case "right":
        wallNum = 2;
        xMm = wMm;
        yMm = Math.round((window.x_position || 0.5) * dMm);
        zMm = Math.round(userYPosition * hMm);
        break;
      default:
        wallNum = 1;
        xMm = Math.round((window.x_position || 0.5) * wMm);
        yMm = 0;
        zMm = Math.round(userYPosition * hMm);
    }
    
    return {
      type: "window",
      name: `main_window_${index + 1}`,
      wall: wallNum,
      dimensions: { width: widthMm, depth: 50, height: heightMm },
      position: { x: xMm, y: yMm, z: zMm },
      rotation_z: 0,
      details: `wall_${wallNum} 벽에 위치`
    };
  });
};

// 가구 데이터를 저장 형식으로 변환하는 헬퍼 함수
export const convertFurnitureToSaveFormat = (furniture) => {
  return furniture.map((item) => {
    const presetData = FURNITURE_PRESETS[item.type];
    const furnitureSize = presetData ? presetData.size : [100, 60, 100];
    
    const centerXMm = Math.round(item.position[0] * 10);
    const centerYMm = Math.round(item.position[2] * 10);
    
    const widthMm = Math.round(furnitureSize[0] * 10);
    const depthMm = Math.round(furnitureSize[2] * 10);
    const heightMm = Math.round(furnitureSize[1] * 10);
    
    const bottomLeftX = centerXMm - widthMm / 2;
    const bottomLeftY = centerYMm - depthMm / 2;
    const topRightX = centerXMm + widthMm / 2;
    const topRightY = centerYMm + depthMm / 2;
    
    const rotation = Array.isArray(item.rotation) ? item.rotation[1] || 0 : item.rotation || 0;
    const rotationZ = rotation;
    
    return {
      type: "furniture",
      name: presetData ? presetData.name.toLowerCase().replace(/\s+/g, '_') : "furniture",
      shape: "rectangle",
      position: {
        center: { x: centerXMm, y: centerYMm, z: 0 },
        corners: {
          bottom_left: { x: Math.round(bottomLeftX), y: Math.round(bottomLeftY) },
          top_right: { x: Math.round(topRightX), y: Math.round(topRightY) }
        }
      },
      dimensions: {
        width: widthMm,
        depth: depthMm, 
        height: heightMm
      },
      rotation_z: Math.round(rotationZ)
    };
  });
};

// 방 레이아웃 데이터 생성 헬퍼 함수
export const createRoomLayoutData = (w, d, h, furniture, detectedWindows) => {
  const wMm = Math.round(w * 10);
  const dMm = Math.round(d * 10);
  const hMm = Math.round(h * 10);
  
  const roomDimensions = { wMm, dMm, hMm };
  
  return {
    scene: {
      description: `왼쪽 아래 꼭짓점(0,0,0)을 기준으로 하는 ${(w/100).toFixed(1)}m × ${(d/100).toFixed(1)}m 방 공간.`,
      walls: {
        wall_1: { direction: "bottom", start: [0, 0], end: [wMm, 0] },
        wall_2: { direction: "right", start: [wMm, 0], end: [wMm, dMm] },
        wall_3: { direction: "top", start: [wMm, dMm], end: [0, dMm] },
        wall_4: { direction: "left", start: [0, dMm], end: [0, 0] }
      },
      room: {
        width: wMm,
        depth: dMm,
        height: hMm
      },
      objects: [
        ...convertWindowsToSaveFormat(detectedWindows, roomDimensions),
        ...convertFurnitureToSaveFormat(furniture)
      ]
    }
  };
};