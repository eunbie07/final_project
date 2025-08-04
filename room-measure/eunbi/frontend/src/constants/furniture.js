// 🪑 FurniturePlacement ↔ RoomBox 가구 ID 매핑 (1:1 매핑)
export const FURNITURE_ID_MAPPING = {
  bed: "bed",
  desk: "desk",
  chair: "chair",
  sofa: "sofa",
  table: "table",
  wardrobe: "wardrobe",
};

// 가구 프리셋 정의 (필수 가구만, cm 단위)
export const FURNITURE_PRESETS = {
  // 침실
  bed: {
    name: "Bed",
    size: [150, 60, 200], // width, height, depth (cm)
    color: "#FFB6C1",
    icon: "faBed",
    category: "bedroom",
    model3D: {
      path: "/bed.glb",
      meshNames: ["bed", "Bed", "mesh", "Mesh", "Object"], // 가능한 메시 이름들
      scale: [6, 3.5, 3.5], // 크기를 5배로 확대
      rotation: [-Math.PI / 2, 0, 0], // X축 -90도 회전
      offset: [0, -60, 0], // Y축 오프셋을 가구 높이의 절반으로 조정
    },
  },
  // 사무용
  desk: {
    name: "Desk",
    size: [120, 75, 60],
    color: "#98FB98",
    icon: "faTable",
    category: "office",
    model3D: {
      path: "/desk.glb",
      meshNames: ["*"], // 전체 파일 로드 (메시 이름 무관)
      scale: [2, 2.5, 2],
      rotation: [0, -Math.PI / 2, 0],
      offset: [-25, -350, 655], // X축으로 50cm 이동, Y축 중립
    },
  },
  chair: {
    name: "Chair",
    size: [50, 85, 50],
    color: "#90EE90",
    icon: "faChair",
    category: "office",
    model3D: {
      path: "/chair.glb",
      meshNames: ["*"],
      scale: [1, 1, 1],
      rotation: [0, 0, 0],
      offset: [0, -85, 0],
    },
  },
  // 거실
  sofa: {
    name: "Sofa",
    size: [180, 85, 80],
    color: "#87CEEB",
    icon: "faCouch",
    category: "living",
    model3D: {
      path: "/sofa.glb",
      meshNames: ["*"],
      scale: [1, 1, 1],
      rotation: [0, 0, 0],
      offset: [0, -85, 0],
    },
  },
  table: {
    name: "Table",
    size: [100, 40, 50],
    color: "#B0E0E6",
    icon: "faTable",
    category: "living",
    model3D: {
      path: "/table.glb",
      meshNames: ["*"],
      scale: [1, 1, 1],
      rotation: [0, 0, 0],
      offset: [0, -40, 0],
    },
  },
  // 수납
  wardrobe: {
    name: "Wardrobe",
    size: [120, 200, 60],
    color: "#DDA0DD",
    icon: "faBox",
    category: "storage",
    model3D: {
      path: "/wardrobe.glb",
      meshNames: ["*"],
      scale: [0.9, 1.3, 1.4],
      rotation: [0, 0, 0],
      offset: [-200, -230, -260],
    },
  },
};
