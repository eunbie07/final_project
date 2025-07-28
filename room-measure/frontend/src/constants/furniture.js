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
    name: "침대",
    size: [150, 60, 200], // width, height, depth (cm)
    color: "#FFB6C1",
    icon: "faBed",
    category: "bedroom",
  },
  // 사무용
  desk: {
    name: "책상",
    size: [120, 75, 60],
    color: "#98FB98",
    icon: "faTable",
    category: "office",
  },
  chair: {
    name: "의자",
    size: [50, 85, 50],
    color: "#90EE90",
    icon: "faChair",
    category: "office",
  },
  // 거실
  sofa: {
    name: "소파",
    size: [180, 85, 80],
    color: "#87CEEB",
    icon: "faCouch",
    category: "living",
  },
  table: {
    name: "테이블",
    size: [100, 45, 50],
    color: "#B0E0E6",
    icon: "faTable",
    category: "living",
  },
  // 수납
  wardrobe: {
    name: "옷장",
    size: [80, 200, 60],
    color: "#DDA0DD",
    icon: "faBox",
    category: "storage",
  },
};