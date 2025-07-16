// frontend/src/components/FurniturePlacement.jsx
import React, { useState, useRef } from "react";

const FURNITURE_CATALOG = [
  // 침실 가구
  {
    id: "single_bed",
    name: "싱글 베드",
    width: 100,
    height: 200,
    category: "bedroom",
    color: "#8B4513",
    icon: "🛏️",
  },
  {
    id: "double_bed",
    name: "더블 베드",
    width: 150,
    height: 200,
    category: "bedroom",
    color: "#8B4513",
    icon: "🛏️",
  },
  {
    id: "queen_bed",
    name: "퀸 베드",
    width: 160,
    height: 200,
    category: "bedroom",
    color: "#8B4513",
    icon: "🛏️",
  },
  {
    id: "king_bed",
    name: "킹 베드",
    width: 180,
    height: 200,
    category: "bedroom",
    color: "#8B4513",
    icon: "🛏️",
  },

  // 책상/의자
  {
    id: "desk",
    name: "책상",
    width: 120,
    height: 60,
    category: "office",
    color: "#654321",
    icon: "🪑",
  },
  {
    id: "chair",
    name: "의자",
    width: 50,
    height: 50,
    category: "office",
    color: "#333333",
    icon: "🪑",
  },

  // 거실 가구
  {
    id: "sofa_2",
    name: "2인 소파",
    width: 140,
    height: 80,
    category: "living",
    color: "#4A5568",
    icon: "🛋️",
  },
  {
    id: "sofa_3",
    name: "3인 소파",
    width: 180,
    height: 80,
    category: "living",
    color: "#4A5568",
    icon: "🛋️",
  },
  {
    id: "coffee_table",
    name: "커피 테이블",
    width: 100,
    height: 50,
    category: "living",
    color: "#8B7355",
    icon: "🪑",
  },
  {
    id: "tv_stand",
    name: "TV 스탠드",
    width: 120,
    height: 40,
    category: "living",
    color: "#2D3748",
    icon: "📺",
  },

  // 수납 가구
  {
    id: "wardrobe",
    name: "옷장",
    width: 80,
    height: 60,
    category: "storage",
    color: "#744C9E",
    icon: "🚪",
  },
  {
    id: "bookshelf",
    name: "책장",
    width: 80,
    height: 30,
    category: "storage",
    color: "#744C9E",
    icon: "📚",
  },
  {
    id: "dresser",
    name: "화장대",
    width: 100,
    height: 45,
    category: "storage",
    color: "#E53E3E",
    icon: "💄",
  },
];

// 문과 창문 카탈로그 추가
const DOOR_WINDOW_CATALOG = [
  {
    id: "door_single",
    name: "일반 문",
    width: 80,
    height: 10, // 벽 두께
    category: "door",
    color: "#8B4513",
    icon: "🚪",
  },
  {
    id: "door_double",
    name: "이중 문",
    width: 140,
    height: 10,
    category: "door",
    color: "#8B4513",
    icon: "🚪",
  },
  {
    id: "door_sliding",
    name: "슬라이딩 문",
    width: 150,
    height: 10,
    category: "door",
    color: "#654321",
    icon: "🚪",
  },
  {
    id: "window_small",
    name: "작은 창문",
    width: 60,
    height: 10,
    category: "window",
    color: "#3B82F6",
    icon: "🪟",
  },
  {
    id: "window_medium",
    name: "중간 창문",
    width: 100,
    height: 10,
    category: "window",
    color: "#3B82F6",
    icon: "🪟",
  },
  {
    id: "window_large",
    name: "큰 창문",
    width: 150,
    height: 10,
    category: "window",
    color: "#3B82F6",
    icon: "🪟",
  },
  {
    id: "window_bay",
    name: "베이 창문",
    width: 200,
    height: 10,
    category: "window",
    color: "#1E40AF",
    icon: "🪟",
  },
];

const CATEGORIES = [
  { id: "all", name: "전체", icon: "🏠" },
  { id: "bedroom", name: "침실", icon: "🛏️" },
  { id: "living", name: "거실", icon: "🛋️" },
  { id: "office", name: "사무", icon: "🪑" },
  { id: "storage", name: "수납", icon: "📦" },
  { id: "door", name: "문", icon: "🚪" },
  { id: "window", name: "창문", icon: "🪟" },
];

const FurnitureItem = ({ furniture, onDragStart }) => (
  <div
    className="p-3 border-2 border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm rounded-lg cursor-grab active:cursor-grabbing transition-all"
    draggable
    onDragStart={(e) => onDragStart(e, furniture)}
  >
    <div className="text-center">
      <div className="text-2xl mb-1">{furniture.icon}</div>
      <div className="text-sm font-medium text-gray-700">{furniture.name}</div>
      <div className="text-xs text-gray-500">
        {furniture.width} × {furniture.height} cm
      </div>
    </div>
  </div>
);

const FurniturePlacement = ({ roomWidth, roomHeight }) => {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [placedFurniture, setPlacedFurniture] = useState([]);
  const [placedDoorWindows, setPlacedDoorWindows] = useState([]); // 문/창문 별도 관리
  const [selectedFurnitureIndex, setSelectedFurnitureIndex] = useState(null);
  const [selectedDoorWindowIndex, setSelectedDoorWindowIndex] = useState(null);
  const [draggedFurniture, setDraggedFurniture] = useState(null);
  const [wallPlacementMode, setWallPlacementMode] = useState(false); // 벽 배치 모드
  const canvasRef = useRef(null);

  // 유효성 검사
  const validRoomWidth = isNaN(roomWidth) || roomWidth <= 0 ? 400 : roomWidth;
  const validRoomHeight =
    isNaN(roomHeight) || roomHeight <= 0 ? 300 : roomHeight;

  // 카테고리별 아이템 필터링
  const filteredItems = [
    ...FURNITURE_CATALOG.filter(
      (item) => selectedCategory === "all" || item.category === selectedCategory
    ),
    ...DOOR_WINDOW_CATALOG.filter(
      (item) => selectedCategory === "all" || item.category === selectedCategory
    ),
  ];

  // 벽에 스냅시키는 함수 (문/창문용)
  const snapToWall = (x, y, itemWidth, itemHeight) => {
    const SNAP_THRESHOLD = 30; // 30cm 이내면 벽에 스냅

    // 각 벽까지의 거리 계산
    const distanceToTop = y;
    const distanceToBottom = validRoomHeight - y;
    const distanceToLeft = x;
    const distanceToRight = validRoomWidth - x;

    // 가장 가까운 벽 찾기
    const minDistance = Math.min(
      distanceToTop,
      distanceToBottom,
      distanceToLeft,
      distanceToRight
    );

    if (minDistance > SNAP_THRESHOLD) {
      return null; // 스냅하지 않음
    }

    let snappedX = x;
    let snappedY = y;
    let wall = "";

    if (minDistance === distanceToTop) {
      // 상단 벽에 스냅
      snappedY = 0;
      snappedX = Math.max(0, Math.min(x, validRoomWidth - itemWidth));
      wall = "top";
    } else if (minDistance === distanceToBottom) {
      // 하단 벽에 스냅
      snappedY = validRoomHeight - itemHeight;
      snappedX = Math.max(0, Math.min(x, validRoomWidth - itemWidth));
      wall = "bottom";
    } else if (minDistance === distanceToLeft) {
      // 좌측 벽에 스냅
      snappedX = 0;
      snappedY = Math.max(0, Math.min(y, validRoomHeight - itemHeight));
      wall = "left";
    } else if (minDistance === distanceToRight) {
      // 우측 벽에 스냅
      snappedX = validRoomWidth - itemWidth;
      snappedY = Math.max(0, Math.min(y, validRoomHeight - itemHeight));
      wall = "right";
    }

    return {
      x: snappedX,
      y: snappedY,
      wall: wall,
      rotation: wall === "left" || wall === "right" ? 90 : 0,
    };
  };

  // 드래그 시작
  const handleDragStart = (e, item) => {
    setDraggedFurniture(item);
    // 문/창문인 경우 벽 배치 모드 활성화
    setWallPlacementMode(
      item.category === "door" || item.category === "window"
    );
    e.dataTransfer.effectAllowed = "copy";
  };

  // 캔버스에 드롭
  const handleDrop = (e) => {
    e.preventDefault();
    if (!draggedFurniture) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // 동적 SVG 크기 계산
    const aspectRatio = validRoomWidth / validRoomHeight;
    const maxSize = 300;

    let svgWidth, svgHeight;
    if (aspectRatio >= 1) {
      svgWidth = maxSize;
      svgHeight = maxSize / aspectRatio;
    } else {
      svgHeight = maxSize;
      svgWidth = maxSize * aspectRatio;
    }

    // SVG 좌표계로 변환 (마진 20px 고려)
    const svgX = (x / rect.width) * (svgWidth + 40) - 20;
    const svgY = (y / rect.height) * (svgHeight + 40) - 20;

    // SVG 좌표를 실제 방 크기(cm)로 변환
    let realX = (svgX / svgWidth) * validRoomWidth;
    let realY = (svgY / svgHeight) * validRoomHeight;

    const isDoorOrWindow =
      draggedFurniture.category === "door" ||
      draggedFurniture.category === "window";
    let finalRotation = 0;
    let wallInfo = null;

    if (isDoorOrWindow) {
      // 문/창문은 벽에 스냅
      const snapResult = snapToWall(
        realX,
        realY,
        draggedFurniture.width,
        draggedFurniture.height
      );

      if (!snapResult) {
        alert("문과 창문은 벽 근처(30cm 이내)에 배치해야 합니다!");
        setDraggedFurniture(null);
        setWallPlacementMode(false);
        return;
      }

      realX = snapResult.x;
      realY = snapResult.y;
      finalRotation = snapResult.rotation;
      wallInfo = snapResult.wall;

      console.log(`🚪 벽 스냅: ${draggedFurniture.name} → ${wallInfo} 벽`);
    } else {
      // 일반 가구는 기존 로직
      const maxX = validRoomWidth - draggedFurniture.width;
      const maxY = validRoomHeight - draggedFurniture.height;

      if (realX < 0 || realY < 0 || realX > maxX || realY > maxY) {
        alert(`가구 위치가 경계를 벗어납니다!`);
        setDraggedFurniture(null);
        setWallPlacementMode(false);
        return;
      }
    }

    // 충돌 체크 (문/창문은 다른 문/창문과만, 가구는 가구끼리만)
    const MARGIN = 5;
    let hasCollision = false;

    if (isDoorOrWindow) {
      // 문/창문끼리 충돌 체크
      hasCollision = placedDoorWindows.some((existing) => {
        const actualWidth =
          finalRotation % 180 === 0
            ? draggedFurniture.width
            : draggedFurniture.height;
        const actualHeight =
          finalRotation % 180 === 0
            ? draggedFurniture.height
            : draggedFurniture.width;

        const existingRotation = existing.rotation || 0;
        const existingActualWidth =
          existingRotation % 180 === 0 ? existing.width : existing.height;
        const existingActualHeight =
          existingRotation % 180 === 0 ? existing.height : existing.width;

        return !(
          realX + actualWidth + MARGIN <= existing.x ||
          realX >= existing.x + existingActualWidth + MARGIN ||
          realY + actualHeight + MARGIN <= existing.y ||
          realY >= existing.y + existingActualHeight + MARGIN
        );
      });
    } else {
      // 가구끼리 충돌 체크
      hasCollision = placedFurniture.some((existing) => {
        const existingRotation = existing.rotation || 0;
        const existingActualWidth =
          existingRotation % 180 === 0 ? existing.width : existing.height;
        const existingActualHeight =
          existingRotation % 180 === 0 ? existing.height : existing.width;

        return !(
          realX + draggedFurniture.width + MARGIN <= existing.x ||
          realX >= existing.x + existingActualWidth + MARGIN ||
          realY + draggedFurniture.height + MARGIN <= existing.y ||
          realY >= existing.y + existingActualHeight + MARGIN
        );
      });
    }

    if (hasCollision) {
      alert(
        isDoorOrWindow
          ? "다른 문/창문과 겹칩니다!"
          : "다른 가구와 너무 가깝습니다!"
      );
      setDraggedFurniture(null);
      setWallPlacementMode(false);
      return;
    }

    // 아이템 배치
    const newItem = {
      ...draggedFurniture,
      x: realX,
      y: realY,
      rotation: finalRotation,
      wall: wallInfo, // 문/창문의 경우 어느 벽인지 저장
      id: `${draggedFurniture.id}_${Date.now()}`,
    };

    if (isDoorOrWindow) {
      setPlacedDoorWindows([...placedDoorWindows, newItem]);
    } else {
      setPlacedFurniture([...placedFurniture, newItem]);
    }

    setDraggedFurniture(null);
    setWallPlacementMode(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  // 가구 선택
  const handleSelectFurniture = (index) => {
    setSelectedFurnitureIndex(selectedFurnitureIndex === index ? null : index);
    setSelectedDoorWindowIndex(null); // 문/창문 선택 해제
  };

  // 가구 회전
  const handleRotateFurniture = (index) => {
    const furniture = placedFurniture[index];
    const currentRotation = furniture.rotation || 0;
    const newRotation = (currentRotation + 90) % 360;

    // 회전 후 크기 계산 (90도, 270도 회전 시 가로세로 바뀜)
    const newActualWidth =
      newRotation % 180 === 0 ? furniture.width : furniture.height;
    const newActualHeight =
      newRotation % 180 === 0 ? furniture.height : furniture.width;

    // 회전 후 방 경계 체크
    const maxX = validRoomWidth - newActualWidth;
    const maxY = validRoomHeight - newActualHeight;

    if (furniture.x > maxX || furniture.y > maxY) {
      alert("회전하면 방 범위를 벗어납니다! 위치를 조정해주세요.");
      return;
    }

    // 다른 가구와 충돌 체크 (회전된 크기로)
    const MARGIN = 5;
    const otherFurniture = placedFurniture.filter((_, i) => i !== index);

    const hasCollision = otherFurniture.some((existing) => {
      const newLeft = furniture.x - MARGIN;
      const newRight = furniture.x + newActualWidth + MARGIN;
      const newTop = furniture.y - MARGIN;
      const newBottom = furniture.y + newActualHeight + MARGIN;

      const existingActualWidth =
        (existing.rotation || 0) % 180 === 0 ? existing.width : existing.height;
      const existingActualHeight =
        (existing.rotation || 0) % 180 === 0 ? existing.height : existing.width;

      const existingLeft = existing.x - MARGIN;
      const existingRight = existing.x + existingActualWidth + MARGIN;
      const existingTop = existing.y - MARGIN;
      const existingBottom = existing.y + existingActualHeight + MARGIN;

      return !(
        newRight <= existingLeft ||
        newLeft >= existingRight ||
        newBottom <= existingTop ||
        newTop >= existingBottom
      );
    });

    if (hasCollision) {
      alert("회전하면 다른 가구와 겹칩니다! 위치를 조정해주세요.");
      return;
    }

    // 회전 적용
    const newPlaced = [...placedFurniture];
    newPlaced[index] = {
      ...furniture,
      rotation: newRotation,
    };
    setPlacedFurniture(newPlaced);

    console.log(`🔄 가구 회전: ${furniture.name} → ${newRotation}°`);
  };

  // 가구 이동
  const handleMoveFurniture = (index, newX, newY) => {
    // 경계 체크 (회전 고려)
    const furniture = placedFurniture[index];
    const rotation = furniture.rotation || 0;
    const actualWidth =
      rotation % 180 === 0 ? furniture.width : furniture.height;
    const actualHeight =
      rotation % 180 === 0 ? furniture.height : furniture.width;

    const maxX = validRoomWidth - actualWidth;
    const maxY = validRoomHeight - actualHeight;

    // 경계 내로 제한
    const clampedX = Math.max(0, Math.min(newX, maxX));
    const clampedY = Math.max(0, Math.min(newY, maxY));

    // 다른 가구와 충돌 체크 (현재 이동 중인 가구 제외)
    const MARGIN = 5;
    const otherFurniture = placedFurniture.filter((_, i) => i !== index);

    const hasCollision = otherFurniture.some((existing) => {
      const newLeft = clampedX - MARGIN;
      const newRight = clampedX + actualWidth + MARGIN;
      const newTop = clampedY - MARGIN;
      const newBottom = clampedY + actualHeight + MARGIN;

      const existingRotation = existing.rotation || 0;
      const existingActualWidth =
        existingRotation % 180 === 0 ? existing.width : existing.height;
      const existingActualHeight =
        existingRotation % 180 === 0 ? existing.height : existing.width;

      const existingLeft = existing.x - MARGIN;
      const existingRight = existing.x + existingActualWidth + MARGIN;
      const existingTop = existing.y - MARGIN;
      const existingBottom = existing.y + existingActualHeight + MARGIN;

      return !(
        newRight <= existingLeft ||
        newLeft >= existingRight ||
        newBottom <= existingTop ||
        newTop >= existingBottom
      );
    });

    // 충돌하지 않으면 이동
    if (!hasCollision) {
      const newPlaced = [...placedFurniture];
      newPlaced[index] = {
        ...furniture,
        x: clampedX,
        y: clampedY,
      };
      setPlacedFurniture(newPlaced);
    }
  };

  // 가구 삭제
  const handleDeleteFurniture = (index) => {
    const newPlaced = [...placedFurniture];
    newPlaced.splice(index, 1);
    setPlacedFurniture(newPlaced);
    setSelectedFurnitureIndex(null);
  };

  // 전체 초기화
  const handleClearAll = () => {
    const totalItems = placedFurniture.length + placedDoorWindows.length;
    if (totalItems === 0) return;
    if (confirm("모든 가구와 문/창문을 삭제하시겠습니까?")) {
      setPlacedFurniture([]);
      setPlacedDoorWindows([]);
      setSelectedFurnitureIndex(null);
      setSelectedDoorWindowIndex(null);
    }
  };

  // 공간 활용률 계산 (가구만)
  const calculateSpaceUtilization = () => {
    const totalFurnitureArea = placedFurniture.reduce((sum, furniture) => {
      const rotation = furniture.rotation || 0;
      const actualWidth =
        rotation % 180 === 0 ? furniture.width : furniture.height;
      const actualHeight =
        rotation % 180 === 0 ? furniture.height : furniture.width;
      return sum + actualWidth * actualHeight;
    }, 0);
    const roomArea = validRoomWidth * validRoomHeight;
    return ((totalFurnitureArea / roomArea) * 100).toFixed(1);
  };

  return (
    <div className="mt-8 p-6 bg-white rounded-xl shadow-lg border">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
          🏠 가구 배치 시뮬레이션
        </h2>
        <div className="flex gap-2">
          <button
            onClick={handleClearAll}
            className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg font-medium transition-colors"
            disabled={
              placedFurniture.length === 0 && placedDoorWindows.length === 0
            }
          >
            🗑️ 전체 삭제
          </button>
        </div>
      </div>

      {/* 통계 정보 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-sm text-blue-600 font-medium">방 크기</div>
          <div className="text-lg font-bold text-blue-800">
            {validRoomWidth.toFixed(1)} × {validRoomHeight.toFixed(1)} cm
          </div>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-sm text-green-600 font-medium">배치된 가구</div>
          <div className="text-lg font-bold text-green-800">
            {placedFurniture.length} 개
          </div>
        </div>
        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="text-sm text-orange-600 font-medium">문/창문</div>
          <div className="text-lg font-bold text-orange-800">
            {placedDoorWindows.length} 개
          </div>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-sm text-purple-600 font-medium">공간 활용률</div>
          <div className="text-lg font-bold text-purple-800">
            {calculateSpaceUtilization()}%
          </div>
        </div>
      </div>

      {/* 디버깅 정보 */}
      <div className="mb-4 p-3 bg-gray-100 rounded-lg text-sm">
        <div className="font-medium text-gray-700 mb-1">🔍 디버깅 정보</div>
        <div className="text-gray-600 space-y-1">
          <div>
            입력된 방 크기: {roomWidth} × {roomHeight}
          </div>
          <div>
            유효한 방 크기: {validRoomWidth} × {validRoomHeight}
          </div>
          <div>SVG 크기: 300 × 200 (viewBox)</div>
          <div>
            변환 비율: {(validRoomWidth / 300).toFixed(3)} ×{" "}
            {(validRoomHeight / 200).toFixed(3)}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 가구 카탈로그 */}
        <div className="lg:col-span-1">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            🛋️ 가구 선택
          </h3>

          {/* 카테고리 탭 */}
          <div className="flex flex-wrap gap-2 mb-4">
            {CATEGORIES.map((category) => (
              <button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedCategory === category.id
                    ? "bg-blue-500 text-white"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                }`}
              >
                {category.icon} {category.name}
              </button>
            ))}
          </div>

          {/* 가구 목록 */}
          <div className="grid grid-cols-2 gap-3 max-h-96 overflow-y-auto">
            {filteredItems.map((item) => (
              <FurnitureItem
                key={item.id}
                furniture={item}
                onDragStart={handleDragStart}
              />
            ))}
          </div>

          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-sm text-yellow-800">
              💡 <strong>사용법:</strong>
              <br />
              • 가구: 자유롭게 배치
              <br />
              • 문/창문: 벽 근처에 드래그하면 자동으로 벽에 붙음
              <br />• 드래그해서 오른쪽 방 평면도에 놓아보세요!
            </p>
          </div>
        </div>

        {/* 방 평면도 */}
        <div className="lg:col-span-2">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            📐 방 평면도
          </h3>

          {/* SVG 크기 동적 계산 */}
          {(() => {
            const aspectRatio = validRoomWidth / validRoomHeight;
            const maxSize = 300;

            let svgWidth, svgHeight;
            if (aspectRatio >= 1) {
              // 가로가 더 긴 경우
              svgWidth = maxSize;
              svgHeight = maxSize / aspectRatio;
            } else {
              // 세로가 더 긴 경우
              svgHeight = maxSize;
              svgWidth = maxSize * aspectRatio;
            }

            console.log("🏠 SVG 비율 계산:", {
              roomSize: `${validRoomWidth} × ${validRoomHeight}`,
              aspectRatio: aspectRatio.toFixed(3),
              svgSize: `${svgWidth.toFixed(0)} × ${svgHeight.toFixed(0)}`,
            });

            return (
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 bg-gray-50">
                {/* 실제 비율 정보 표시 */}
                <div className="mb-2 text-sm text-gray-600 text-center">
                  실제 비율: {validRoomWidth} × {validRoomHeight} cm (
                  {aspectRatio.toFixed(2)}:1)
                </div>

                <div className="flex justify-center">
                  <svg
                    ref={canvasRef}
                    width={svgWidth + 40}
                    height={svgHeight + 40}
                    viewBox={`0 0 ${svgWidth + 40} ${svgHeight + 40}`}
                    className="border border-gray-400 bg-white rounded-lg cursor-crosshair"
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                  >
                    {/* 방 윤곽 */}
                    <rect
                      x="20"
                      y="20"
                      width={svgWidth}
                      height={svgHeight}
                      fill="#f8f9fa"
                      stroke="#343a40"
                      strokeWidth="2"
                    />

                    {/* 그리드 */}
                    <defs>
                      <pattern
                        id="grid"
                        width="20"
                        height="20"
                        patternUnits="userSpaceOnUse"
                      >
                        <path
                          d="M 20 0 L 0 0 0 20"
                          fill="none"
                          stroke="#e9ecef"
                          strokeWidth="1"
                        />
                      </pattern>
                    </defs>
                    <rect
                      x="20"
                      y="20"
                      width={svgWidth}
                      height={svgHeight}
                      fill="url(#grid)"
                    />

                    {/* 배치된 가구들 */}
                    {placedFurniture.map((furniture, index) => {
                      // 새로운 스케일링 적용
                      const scaleX = svgWidth / validRoomWidth;
                      const scaleY = svgHeight / validRoomHeight;

                      const rotation = furniture.rotation || 0;
                      const actualWidth =
                        rotation % 180 === 0
                          ? furniture.width
                          : furniture.height;
                      const actualHeight =
                        rotation % 180 === 0
                          ? furniture.height
                          : furniture.width;

                      const scaledWidth = actualWidth * scaleX;
                      const scaledHeight = actualHeight * scaleY;
                      const scaledX = 20 + furniture.x * scaleX;
                      const scaledY = 20 + furniture.y * scaleY;

                      return (
                        <g key={furniture.id}>
                          {/* 여유 공간 표시 (선택된 가구만) */}
                          {selectedFurnitureIndex === index && (
                            <rect
                              x={scaledX - 5 * scaleX}
                              y={scaledY - 5 * scaleY}
                              width={scaledWidth + 10 * scaleX}
                              height={scaledHeight + 10 * scaleY}
                              fill="none"
                              stroke="#FbbF24"
                              strokeWidth="1"
                              strokeDasharray="3,3"
                              opacity="0.7"
                            />
                          )}

                          {/* 가구 본체 */}
                          <rect
                            x={scaledX}
                            y={scaledY}
                            width={scaledWidth}
                            height={scaledHeight}
                            fill={furniture.color}
                            stroke={
                              selectedFurnitureIndex === index
                                ? "#3B82F6"
                                : "#374151"
                            }
                            strokeWidth={
                              selectedFurnitureIndex === index ? 3 : 1
                            }
                            className={`cursor-pointer hover:opacity-80 transition-opacity ${
                              selectedFurnitureIndex === index
                                ? "cursor-move"
                                : ""
                            }`}
                            onMouseDown={(e) => {
                              e.preventDefault();
                              if (selectedFurnitureIndex !== index) {
                                handleSelectFurniture(index);
                                return;
                              }

                              const svg = e.currentTarget.closest("svg");
                              const rect = svg.getBoundingClientRect();

                              const startMouseX = e.clientX;
                              const startMouseY = e.clientY;
                              const startFurnitureX = furniture.x;
                              const startFurnitureY = furniture.y;

                              const handleMouseMove = (moveEvent) => {
                                const deltaX = moveEvent.clientX - startMouseX;
                                const deltaY = moveEvent.clientY - startMouseY;

                                const realDeltaX =
                                  (deltaX / rect.width) * validRoomWidth;
                                const realDeltaY =
                                  (deltaY / rect.height) * validRoomHeight;

                                const newX = startFurnitureX + realDeltaX;
                                const newY = startFurnitureY + realDeltaY;

                                handleMoveFurniture(index, newX, newY);
                              };

                              const handleMouseUp = () => {
                                document.removeEventListener(
                                  "mousemove",
                                  handleMouseMove
                                );
                                document.removeEventListener(
                                  "mouseup",
                                  handleMouseUp
                                );
                              };

                              document.addEventListener(
                                "mousemove",
                                handleMouseMove
                              );
                              document.addEventListener(
                                "mouseup",
                                handleMouseUp
                              );
                            }}
                          />

                          {/* 가구 아이콘 */}
                          <text
                            x={scaledX + scaledWidth / 2}
                            y={scaledY + scaledHeight / 2}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fontSize="12"
                            fill="white"
                            className="pointer-events-none select-none"
                          >
                            {furniture.icon}
                          </text>

                          {/* 회전 표시 */}
                          {rotation !== 0 && (
                            <text
                              x={scaledX + scaledWidth / 2}
                              y={scaledY + scaledHeight / 2 + 15}
                              textAnchor="middle"
                              dominantBaseline="middle"
                              fontSize="10"
                              fill="white"
                              className="pointer-events-none select-none"
                            >
                              {rotation}°
                            </text>
                          )}

                          {/* 가구 치수 표시 (선택된 가구만) */}
                          {selectedFurnitureIndex === index && (
                            <g>
                              {/* 가로 치수 */}
                              <text
                                x={scaledX + scaledWidth / 2}
                                y={scaledY - 5}
                                textAnchor="middle"
                                fontSize="10"
                                fill="#4B5563"
                                className="pointer-events-none select-none"
                              >
                                {actualWidth}cm
                              </text>

                              {/* 세로 치수 */}
                              <text
                                x={scaledX - 5}
                                y={scaledY + scaledHeight / 2}
                                textAnchor="middle"
                                fontSize="10"
                                fill="#4B5563"
                                className="pointer-events-none select-none"
                                transform={`rotate(-90, ${scaledX - 5}, ${
                                  scaledY + scaledHeight / 2
                                })`}
                              >
                                {actualHeight}cm
                              </text>

                              {/* 가구 이름 */}
                              <text
                                x={scaledX + scaledWidth / 2}
                                y={scaledY + scaledHeight + 15}
                                textAnchor="middle"
                                fontSize="10"
                                fill="#1F2937"
                                className="pointer-events-none select-none font-medium"
                              >
                                {furniture.name}{" "}
                                {rotation !== 0 && `(${rotation}°)`}
                              </text>
                            </g>
                          )}

                          {/* 컨트롤 버튼들 (선택된 가구만) */}
                          {selectedFurnitureIndex === index && (
                            <g>
                              {/* 회전 버튼 */}
                              <circle
                                cx={scaledX + scaledWidth - 8}
                                cy={scaledY + scaledHeight - 8}
                                r="8"
                                fill="#10B981"
                                className="cursor-pointer"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleRotateFurniture(index);
                                }}
                              />
                              <text
                                x={scaledX + scaledWidth - 8}
                                y={scaledY + scaledHeight - 8}
                                textAnchor="middle"
                                dominantBaseline="middle"
                                fontSize="10"
                                fill="white"
                                className="pointer-events-none select-none"
                              >
                                ↻
                              </text>

                              {/* 삭제 버튼 */}
                              <circle
                                cx={scaledX + scaledWidth - 8}
                                cy={scaledY + 8}
                                r="8"
                                fill="#EF4444"
                                className="cursor-pointer"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteFurniture(index);
                                }}
                              />
                              <text
                                x={scaledX + scaledWidth - 8}
                                y={scaledY + 8}
                                textAnchor="middle"
                                dominantBaseline="middle"
                                fontSize="10"
                                fill="white"
                                className="pointer-events-none select-none"
                              >
                                ×
                              </text>
                            </g>
                          )}
                        </g>
                      );
                    })}

                    {/* 배치된 문/창문들 */}
                    {placedDoorWindows.map((item, index) => {
                      const scaleX = svgWidth / validRoomWidth;
                      const scaleY = svgHeight / validRoomHeight;

                      const rotation = item.rotation || 0;
                      const actualWidth =
                        rotation % 180 === 0 ? item.width : item.height;
                      const actualHeight =
                        rotation % 180 === 0 ? item.height : item.width;

                      const scaledWidth = actualWidth * scaleX;
                      const scaledHeight = actualHeight * scaleY;
                      const scaledX = 20 + item.x * scaleX;
                      const scaledY = 20 + item.y * scaleY;

                      const isSelected = selectedDoorWindowIndex === index;

                      return (
                        <g key={item.id}>
                          {/* 문/창문 본체 */}
                          <rect
                            x={scaledX}
                            y={scaledY}
                            width={scaledWidth}
                            height={scaledHeight}
                            fill={item.color}
                            stroke={isSelected ? "#F59E0B" : "#6B7280"}
                            strokeWidth={isSelected ? 3 : 2}
                            className="cursor-pointer hover:opacity-80 transition-opacity"
                            onClick={() =>
                              setSelectedDoorWindowIndex(
                                isSelected ? null : index
                              )
                            }
                          />

                          {/* 문/창문 아이콘 */}
                          <text
                            x={scaledX + scaledWidth / 2}
                            y={scaledY + scaledHeight / 2}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fontSize="12"
                            fill="white"
                            className="pointer-events-none select-none"
                          >
                            {item.icon}
                          </text>

                          {/* 벽 정보 표시 (선택된 경우) */}
                          {isSelected && item.wall && (
                            <text
                              x={scaledX + scaledWidth / 2}
                              y={scaledY + scaledHeight + 15}
                              textAnchor="middle"
                              fontSize="10"
                              fill="#F59E0B"
                              className="pointer-events-none select-none font-medium"
                            >
                              {item.name} ({item.wall} 벽)
                            </text>
                          )}

                          {/* 삭제 버튼 (선택된 경우) */}
                          {isSelected && (
                            <g>
                              <circle
                                cx={scaledX + scaledWidth - 8}
                                cy={scaledY + 8}
                                r="8"
                                fill="#EF4444"
                                className="cursor-pointer"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const newPlaced = [...placedDoorWindows];
                                  newPlaced.splice(index, 1);
                                  setPlacedDoorWindows(newPlaced);
                                  setSelectedDoorWindowIndex(null);
                                }}
                              />
                              <text
                                x={scaledX + scaledWidth - 8}
                                y={scaledY + 8}
                                textAnchor="middle"
                                dominantBaseline="middle"
                                fontSize="10"
                                fill="white"
                                className="pointer-events-none select-none"
                              >
                                ×
                              </text>
                            </g>
                          )}
                        </g>
                      );
                    })}

                    {/* 크기 표시 */}
                    <text
                      x={20 + svgWidth / 2}
                      y={svgHeight + 35}
                      textAnchor="middle"
                      fontSize="12"
                      fill="#666"
                    >
                      {validRoomWidth.toFixed(0)} cm
                    </text>
                    <text
                      x="10"
                      y={20 + svgHeight / 2}
                      textAnchor="middle"
                      fontSize="12"
                      fill="#666"
                      transform={`rotate(-90, 10, ${20 + svgHeight / 2})`}
                    >
                      {validRoomHeight.toFixed(0)} cm
                    </text>
                  </svg>
                </div>
              </div>
            );
          })()}

          {/* 드롭 영역 안내 */}
          <div className="mt-2 text-center">
            <p className="text-sm text-gray-600">
              💡 위 회색 영역에 가구를 드래그해서 놓으세요
            </p>
          </div>

          {/* 선택된 항목 정보 */}
          {selectedFurnitureIndex !== null && (
            <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <h4 className="font-medium text-blue-800 mb-2">
                선택된 가구: {placedFurniture[selectedFurnitureIndex].name}
              </h4>
              <div className="text-sm text-blue-700 space-y-1">
                <div>
                  크기:{" "}
                  {(placedFurniture[selectedFurnitureIndex].rotation || 0) %
                    180 ===
                  0
                    ? placedFurniture[selectedFurnitureIndex].width
                    : placedFurniture[selectedFurnitureIndex].height}{" "}
                  ×{" "}
                  {(placedFurniture[selectedFurnitureIndex].rotation || 0) %
                    180 ===
                  0
                    ? placedFurniture[selectedFurnitureIndex].height
                    : placedFurniture[selectedFurnitureIndex].width}{" "}
                  cm
                </div>
                <div>
                  위치: ({Math.round(placedFurniture[selectedFurnitureIndex].x)}
                  , {Math.round(placedFurniture[selectedFurnitureIndex].y)}) cm
                </div>
                <div>
                  회전: {placedFurniture[selectedFurnitureIndex].rotation || 0}°
                </div>
              </div>
              <div className="mt-2 flex gap-2">
                <button
                  onClick={() => handleRotateFurniture(selectedFurnitureIndex)}
                  className="px-3 py-1 bg-green-500 hover:bg-green-600 text-white text-xs rounded font-medium"
                >
                  🔄 90° 회전
                </button>
              </div>
            </div>
          )}

          {selectedDoorWindowIndex !== null && (
            <div className="mt-4 p-4 bg-orange-50 border border-orange-200 rounded-lg">
              <h4 className="font-medium text-orange-800 mb-2">
                선택된{" "}
                {placedDoorWindows[selectedDoorWindowIndex].category === "door"
                  ? "문"
                  : "창문"}
                : {placedDoorWindows[selectedDoorWindowIndex].name}
              </h4>
              <div className="text-sm text-orange-700 space-y-1">
                <div>
                  크기: {placedDoorWindows[selectedDoorWindowIndex].width} ×{" "}
                  {placedDoorWindows[selectedDoorWindowIndex].height} cm
                </div>
                <div>
                  위치: {placedDoorWindows[selectedDoorWindowIndex].wall} 벽
                </div>
                <div>
                  좌표: (
                  {Math.round(placedDoorWindows[selectedDoorWindowIndex].x)},{" "}
                  {Math.round(placedDoorWindows[selectedDoorWindowIndex].y)}) cm
                </div>
              </div>
            </div>
          )}

          {/* 배치 가이드 */}
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-700 mb-2">📋 사용 가이드</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>
                • <strong>가구 배치:</strong> 자유롭게 드래그해서 배치
              </li>
              <li>
                • <strong>문/창문:</strong> 벽 근처(30cm 이내)로 드래그하면
                자동으로 벽에 붙음
              </li>
              <li>
                • <strong>이동:</strong> 배치된 항목을 클릭 선택 후 드래그로
                이동
              </li>
              <li>
                • <strong>회전:</strong> 가구는 90도씩 회전 가능
              </li>
              <li>
                • <strong>삭제:</strong> 선택된 항목의 빨간 × 버튼 클릭
              </li>
              <li>
                • <strong>자동 스냅:</strong> 문/창문은 가장 가까운 벽에 자동
                정렬
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FurniturePlacement;
