// 🪑 FurniturePlacement ↔ RoomBox 가구 ID 매핑 (1:1 매핑)
export const FURNITURE_ID_MAPPING = {
  single_bed: "single_bed",
  double_bed: "double_bed",
  queen_bed: "queen_bed",
  king_bed: "king_bed",
  desk: "desk",
  chair: "chair",
  sofa_2: "sofa_2",
  sofa_3: "sofa_3",
  coffee_table: "coffee_table",
  tv_stand: "tv_stand",
  wardrobe: "wardrobe",
  bookshelf: "bookshelf",
  dresser: "dresser",
};

// 가구 프리셋 정의 (FurniturePlacement와 통합, cm 단위)
export const FURNITURE_PRESETS = {
  // 침실 가구
  single_bed: {
    name: "싱글 베드",
    size: [100, 60, 200], // width, height, depth (cm)
    color: "#FFB6C1",
    icon: "🛏️",
    category: "bedroom",
  },
  double_bed: {
    name: "더블 베드",
    size: [150, 60, 200],
    color: "#FFD1DC",
    icon: "🛏️",
    category: "bedroom",
  },
  queen_bed: {
    name: "퀸 베드",
    size: [160, 60, 200],
    color: "#FFC0CB",
    icon: "🛏️",
    category: "bedroom",
  },
  king_bed: {
    name: "킹 베드",
    size: [180, 60, 200],
    color: "#FFB7C5",
    icon: "🛏️",
    category: "bedroom",
  },
  // 책상/의자
  desk: {
    name: "책상",
    size: [120, 75, 60],
    color: "#98FB98",
    icon: "🪑",
    category: "office",
  },
  chair: {
    name: "의자",
    size: [50, 85, 50],
    color: "#90EE90",
    icon: "🪑",
    category: "office",
  },
  // 거실 가구
  sofa_2: {
    name: "2인 소파",
    size: [140, 85, 80],
    color: "#87CEEB",
    icon: "🛋️",
    category: "living",
  },
  sofa_3: {
    name: "3인 소파",
    size: [180, 85, 80],
    color: "#ADD8E6",
    icon: "🛋️",
    category: "living",
  },
  coffee_table: {
    name: "커피 테이블",
    size: [100, 45, 50],
    color: "#B0E0E6",
    icon: "🪑",
    category: "living",
  },
  tv_stand: {
    name: "TV 스탠드",
    size: [120, 50, 40],
    color: "#E0FFFF",
    icon: "📺",
    category: "living",
  },
  // 수납 가구
  wardrobe: {
    name: "옷장",
    size: [80, 200, 60],
    color: "#DDA0DD",
    icon: "🚪",
    category: "storage",
  },
  bookshelf: {
    name: "책장",
    size: [80, 180, 30],
    color: "#D8BFD8",
    icon: "📚",
    category: "storage",
  },
  dresser: {
    name: "화장대",
    size: [100, 75, 45],
    color: "#E6E6FA",
    icon: "💄",
    category: "storage",
  },
};