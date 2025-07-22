// frontend/src/components/FurniturePlacement.jsx
import React, { useState, useRef, useCallback, useMemo } from "react";

const FURNITURE_CATALOG = [
  // 침실 가구
  {
    id: "single_bed",
    name: "싱글 베드",
    width: 100,
    height: 200,
    category: "bedroom",
    color: "#FFB6C1",
    icon: "🛏️",
  },
  {
    id: "double_bed",
    name: "더블 베드",
    width: 150,
    height: 200,
    category: "bedroom",
    color: "#FFD1DC",
    icon: "🛏️",
  },
  {
    id: "queen_bed",
    name: "퀸 베드",
    width: 160,
    height: 200,
    category: "bedroom",
    color: "#FFC0CB",
    icon: "🛏️",
  },
  {
    id: "king_bed",
    name: "킹 베드",
    width: 180,
    height: 200,
    category: "bedroom",
    color: "#FFB7C5",
    icon: "🛏️",
  },
  // 책상/의자
  {
    id: "desk",
    name: "책상",
    width: 120,
    height: 60,
    category: "office",
    color: "#98FB98",
    icon: "🪑",
  },
  {
    id: "chair",
    name: "의자",
    width: 50,
    height: 50,
    category: "office",
    color: "#90EE90",
    icon: "🪑",
  },
  // 거실 가구
  {
    id: "sofa_2",
    name: "2인 소파",
    width: 140,
    height: 80,
    category: "living",
    color: "#87CEEB",
    icon: "🛋️",
  },
  {
    id: "sofa_3",
    name: "3인 소파",
    width: 180,
    height: 80,
    category: "living",
    color: "#ADD8E6",
    icon: "🛋️",
  },
  {
    id: "coffee_table",
    name: "커피 테이블",
    width: 100,
    height: 50,
    category: "living",
    color: "#B0E0E6",
    icon: "🪑",
  },
  {
    id: "tv_stand",
    name: "TV 스탠드",
    width: 120,
    height: 40,
    category: "living",
    color: "#E0FFFF",
    icon: "📺",
  },
  // 수납 가구
  {
    id: "wardrobe",
    name: "옷장",
    width: 80,
    height: 60,
    category: "storage",
    color: "#DDA0DD",
    icon: "🚪",
  },
  {
    id: "bookshelf",
    name: "책장",
    width: 80,
    height: 30,
    category: "storage",
    color: "#D8BFD8",
    icon: "📚",
  },
  {
    id: "dresser",
    name: "화장대",
    width: 100,
    height: 45,
    category: "storage",
    color: "#E6E6FA",
    icon: "💄",
  },
];

const CATEGORIES = [
  { id: "all", name: "전체", icon: "" },
  { id: "bedroom", name: "침실", icon: "" },
  { id: "living", name: "거실", icon: "" },
  { id: "office", name: "사무", icon: "" },
  { id: "storage", name: "수납", icon: "" },
];

const FurnitureItem = ({ furniture, onDragStart }) => (
  <div
    className="p-3 border-2 border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm rounded-lg cursor-grab active:cursor-grabbing transition-all"
    draggable
    onDragStart={(e) => {
      e.dataTransfer.effectAllowed = "copy";
      onDragStart(e, furniture);
    }}
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
  const [selectedFurnitureIndex, setSelectedFurnitureIndex] = useState(null);
  const [draggedFurniture, setDraggedFurniture] = useState(null);
  const [isDraggingPlaced, setIsDraggingPlaced] = useState(false);
  const canvasRef = useRef(null);

  // 유효성 검사
  const validRoomWidth = isNaN(roomWidth) || roomWidth <= 0 ? 400 : roomWidth;
  const validRoomHeight =
    isNaN(roomHeight) || roomHeight <= 0 ? 300 : roomHeight;

  // SVG 크기 계산
  const svgDimensions = useMemo(() => {
    const aspectRatio = validRoomWidth / validRoomHeight;
    const maxSize = 500;

    let svgWidth, svgHeight;
    if (aspectRatio >= 1) {
      svgWidth = maxSize;
      svgHeight = maxSize / aspectRatio;
    } else {
      svgHeight = maxSize;
      svgWidth = maxSize * aspectRatio;
    }

    return { svgWidth, svgHeight };
  }, [validRoomWidth, validRoomHeight]);

  // 카테고리별 아이템 필터링
  const filteredItems = FURNITURE_CATALOG.filter(
    (item) => selectedCategory === "all" || item.category === selectedCategory
  );

  // SVG 좌표를 실제 방 좌표로 변환 (왼쪽 아래가 0,0)
  const convertToRealCoordinates = useCallback(
    (clientX, clientY) => {
      const rect = canvasRef.current.getBoundingClientRect();
      const svgX = clientX - rect.left - 20;
      const svgY = clientY - rect.top - 20;

      const scaleX = validRoomWidth / svgDimensions.svgWidth;
      const scaleY = validRoomHeight / svgDimensions.svgHeight;

      // Y좌표를 뒤집어서 왼쪽 아래가 (0,0)이 되도록 변환
      return {
        x: svgX * scaleX,
        y: (svgDimensions.svgHeight - svgY) * scaleY,
      };
    },
    [validRoomWidth, validRoomHeight, svgDimensions]
  );

  // 충돌 체크 함수
  const checkCollision = useCallback(
    (x, y, width, height, excludeIndex = -1, rotation = 0) => {
      const actualWidth = rotation % 180 === 0 ? width : height;
      const actualHeight = rotation % 180 === 0 ? height : width;

      for (let i = 0; i < placedFurniture.length; i++) {
        if (i === excludeIndex) continue;

        const item = placedFurniture[i];
        const itemRotation = item.rotation || 0;
        const itemActualWidth =
          itemRotation % 180 === 0 ? item.width : item.height;
        const itemActualHeight =
          itemRotation % 180 === 0 ? item.height : item.width;

        if (
          x < item.x + itemActualWidth &&
          x + actualWidth > item.x &&
          y < item.y + itemActualHeight &&
          y + actualHeight > item.y
        ) {
          return true;
        }
      }

      return false;
    },
    [placedFurniture]
  );

  // 드래그 시작
  const handleDragStart = useCallback((e, item) => {
    setDraggedFurniture(item);
    e.dataTransfer.setData("application/json", JSON.stringify(item));
  }, []);

  // 드래그 오버
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  // 드롭
  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();

      if (!draggedFurniture || isDraggingPlaced) return;

      const coords = convertToRealCoordinates(e.clientX, e.clientY);

      // 경계 체크
      const maxX = validRoomWidth - draggedFurniture.width;
      const maxY = validRoomHeight - draggedFurniture.height;

      coords.x = Math.max(0, Math.min(coords.x, maxX));
      coords.y = Math.max(0, Math.min(coords.y, maxY));

      // 일반 가구 처리
      if (
        !checkCollision(
          coords.x,
          coords.y,
          draggedFurniture.width,
          draggedFurniture.height
        )
      ) {
        const newItem = {
          ...draggedFurniture,
          x: coords.x,
          y: coords.y,
          rotation: 0,
          id: `${draggedFurniture.id}_${Date.now()}`,
        };
        setPlacedFurniture((prev) => [...prev, newItem]);
      } else {
        alert("이 위치에는 다른 가구가 있습니다!");
      }

      setDraggedFurniture(null);
    },
    [
      draggedFurniture,
      isDraggingPlaced,
      convertToRealCoordinates,
      validRoomWidth,
      validRoomHeight,
      checkCollision,
    ]
  );

  // 가구 선택
  const handleSelectFurniture = useCallback(
    (index) => {
      setSelectedFurnitureIndex(
        selectedFurnitureIndex === index ? null : index
      );
    },
    [selectedFurnitureIndex]
  );

  // 가구 회전
  const handleRotateFurniture = useCallback(
    (index) => {
      const furniture = placedFurniture[index];
      const currentRotation = furniture.rotation || 0;
      const newRotation = (currentRotation + 90) % 360;

      const newActualWidth =
        newRotation % 180 === 0 ? furniture.width : furniture.height;
      const newActualHeight =
        newRotation % 180 === 0 ? furniture.height : furniture.width;

      // 회전 후 경계 체크
      const maxX = validRoomWidth - newActualWidth;
      const maxY = validRoomHeight - newActualHeight;

      if (furniture.x > maxX || furniture.y > maxY) {
        alert("회전하면 방 범위를 벗어납니다!");
        return;
      }

      // 충돌 체크
      if (
        checkCollision(
          furniture.x,
          furniture.y,
          furniture.width,
          furniture.height,
          index,
          newRotation
        )
      ) {
        alert("회전하면 다른 가구와 겹칩니다!");
        return;
      }

      const newPlaced = [...placedFurniture];
      newPlaced[index] = {
        ...furniture,
        rotation: newRotation,
      };
      setPlacedFurniture(newPlaced);
    },
    [placedFurniture, validRoomWidth, validRoomHeight, checkCollision]
  );

  // 가구 이동
  const handleMoveFurniture = useCallback(
    (index, newX, newY) => {
      const furniture = placedFurniture[index];
      const rotation = furniture.rotation || 0;
      const actualWidth =
        rotation % 180 === 0 ? furniture.width : furniture.height;
      const actualHeight =
        rotation % 180 === 0 ? furniture.height : furniture.width;

      // 경계 체크
      const maxX = validRoomWidth - actualWidth;
      const maxY = validRoomHeight - actualHeight;

      const clampedX = Math.max(0, Math.min(newX, maxX));
      const clampedY = Math.max(0, Math.min(newY, maxY));

      // 충돌 체크
      if (
        !checkCollision(
          clampedX,
          clampedY,
          furniture.width,
          furniture.height,
          index,
          rotation
        )
      ) {
        const newPlaced = [...placedFurniture];
        newPlaced[index] = {
          ...furniture,
          x: clampedX,
          y: clampedY,
        };
        setPlacedFurniture(newPlaced);
      }
    },
    [placedFurniture, validRoomWidth, validRoomHeight, checkCollision]
  );

  // 가구 삭제
  const handleDeleteFurniture = useCallback(
    (index) => {
      const newPlaced = [...placedFurniture];
      newPlaced.splice(index, 1);
      setPlacedFurniture(newPlaced);
      setSelectedFurnitureIndex(null);
    },
    [placedFurniture]
  );

  // 전체 초기화
  const handleClearAll = useCallback(() => {
    if (placedFurniture.length === 0) return;
    if (confirm("모든 가구를 삭제하시겠습니까?")) {
      setPlacedFurniture([]);
      setSelectedFurnitureIndex(null);
    }
  }, [placedFurniture.length]);

  // JSON 저장 기능
  const handleSaveAsJson = useCallback(() => {
    if (placedFurniture.length === 0) {
      alert("저장할 가구가 없습니다!");
      return;
    }

    // JSON 데이터 구성
    const saveData = {
      roomInfo: {
        width: validRoomWidth,
        height: validRoomHeight,
        area: ((validRoomWidth * validRoomHeight) / 10000).toFixed(1) + "㎡",
        aspectRatio: (validRoomWidth / validRoomHeight).toFixed(2),
      },
      furniture: placedFurniture.map((furniture) => {
        const rotation = furniture.rotation || 0;
        const actualWidth =
          rotation % 180 === 0 ? furniture.width : furniture.height;
        const actualHeight =
          rotation % 180 === 0 ? furniture.height : furniture.width;

        return {
          id: furniture.id,
          name: furniture.name,
          category: furniture.category,
          originalSize: {
            width: furniture.width,
            height: furniture.height,
          },
          currentSize: {
            width: actualWidth,
            height: actualHeight,
          },
          position: {
            leftBottom: {
              x: Math.round(furniture.x),
              y: Math.round(furniture.y),
            },
            rightTop: {
              x: Math.round(furniture.x + actualWidth),
              y: Math.round(furniture.y + actualHeight),
            },
          },
          rotation: rotation,
          color: furniture.color,
          icon: furniture.icon,
        };
      }),
      statistics: {
        furnitureCount: placedFurniture.length,
        spaceUtilization: calculateSpaceUtilization + "%",
        totalFurnitureArea:
          placedFurniture.reduce((sum, furniture) => {
            const rotation = furniture.rotation || 0;
            const actualWidth =
              rotation % 180 === 0 ? furniture.width : furniture.height;
            const actualHeight =
              rotation % 180 === 0 ? furniture.height : furniture.width;
            return sum + actualWidth * actualHeight;
          }, 0) + " cm²",
      },
      metadata: {
        exportDate: new Date().toISOString(),
        coordinateSystem: "leftBottom_origin", // 왼쪽 아래가 (0,0)
        unit: "cm",
      },
    };

    // JSON 파일 다운로드
    const jsonString = JSON.stringify(saveData, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const currentDate = new Date().toISOString().split("T")[0]; // YYYY-MM-DD
    const filename = `furniture_layout_${validRoomWidth}x${validRoomHeight}_${currentDate}.json`;

    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);

    console.log("JSON 저장 완료:", saveData);
  }, [
    placedFurniture,
    validRoomWidth,
    validRoomHeight,
    calculateSpaceUtilization,
  ]);

  // 공간 활용률 계산
  const calculateSpaceUtilization = useMemo(() => {
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
  }, [placedFurniture, validRoomWidth, validRoomHeight]);

  return (
    <div className="mt-8 p-6 bg-white rounded-xl shadow-lg border">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
          <strong>가구 배치 시뮬레이션</strong>
        </h2>
        <div className="flex gap-2">
          <button
            onClick={handleSaveAsJson}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={placedFurniture.length === 0}
          >
            <strong>JSON 저장</strong>
          </button>
          <button
            onClick={handleClearAll}
            className="px-4 py-2 bg-rose-500 hover:bg-rose-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={placedFurniture.length === 0}
          >
            <strong>전체 삭제</strong>
          </button>
        </div>
      </div>

      {/* 통계 정보 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-rose-50 p-4 rounded-lg">
          <div className="text-sm text-rose-600 font-medium">
            {" "}
            <strong>방 크기</strong>
          </div>
          <div className="text-lg font-bold text-rose-800">
            {validRoomWidth.toFixed(1)} × {validRoomHeight.toFixed(1)} cm
          </div>
        </div>
        <div className="bg-pink-50 p-4 rounded-lg">
          <div className="text-sm text-pink-600 font-medium">
            <strong>배치된 가구</strong>
          </div>
          <div className="text-lg font-bold text-pink-800">
            {placedFurniture.length} 개
          </div>
        </div>
        <div className="bg-rose-100 p-4 rounded-lg">
          <div className="text-sm text-rose-700 font-medium">
            <strong>공간 활용률</strong>
          </div>
          <div className="text-lg font-bold text-rose-800">
            {calculateSpaceUtilization}%
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 가구 카탈로그 */}
        <div className="lg:col-span-1">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            <strong>가구 선택</strong>
          </h3>

          {/* 카테고리 탭 */}
          <div className="flex flex-wrap gap-2 mb-4">
            {CATEGORIES.map((category) => (
              <button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedCategory === category.id
                    ? "bg-pink-500 text-white"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                }`}
              >
                {category.name}
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

          <div className="mt-4 p-3 bg-pink-50 border border-pink-200 rounded-lg">
            <p className="text-sm text-pink-800">
              <strong>사용법:</strong>
              <br />
              • 가구를 드래그해서 방에 배치
              <br />
              • 배치된 항목 클릭 후 드래그로 이동
              <br />• 녹색 버튼으로 회전, 빨간 버튼으로 삭제
            </p>
          </div>
        </div>

        {/* 방 평면도 */}
        <div className="lg:col-span-2">
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            <strong>방 평면도</strong>
          </h3>

          <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 bg-gray-50">
            <div className="mb-2 text-sm text-gray-600 text-center">
              실제 비율: {validRoomWidth} × {validRoomHeight} cm (
              {(validRoomWidth / validRoomHeight).toFixed(2)}:1)
            </div>

            <div className="flex justify-center">
              <svg
                ref={canvasRef}
                width={svgDimensions.svgWidth + 40}
                height={svgDimensions.svgHeight + 40}
                className="border border-gray-400 bg-white rounded-lg cursor-crosshair"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                {/* 방 윤곽 */}
                <rect
                  x="20"
                  y="20"
                  width={svgDimensions.svgWidth}
                  height={svgDimensions.svgHeight}
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
                  width={svgDimensions.svgWidth}
                  height={svgDimensions.svgHeight}
                  fill="url(#grid)"
                />

                {/* 원점 표시 (0,0) - 왼쪽 아래 */}
                <circle
                  cx="20"
                  cy={20 + svgDimensions.svgHeight}
                  r="3"
                  fill="#ef4444"
                  stroke="#ffffff"
                  strokeWidth="1"
                />

                {/* 배치된 가구들 */}
                {placedFurniture.map((furniture, index) => {
                  const scaleX = svgDimensions.svgWidth / validRoomWidth;
                  const scaleY = svgDimensions.svgHeight / validRoomHeight;

                  const rotation = furniture.rotation || 0;
                  const actualWidth =
                    rotation % 180 === 0 ? furniture.width : furniture.height;
                  const actualHeight =
                    rotation % 180 === 0 ? furniture.height : furniture.width;

                  const scaledWidth = actualWidth * scaleX;
                  const scaledHeight = actualHeight * scaleY;
                  const scaledX = 20 + furniture.x * scaleX;
                  // Y좌표를 뒤집어서 왼쪽 아래 기준으로 계산
                  const scaledY =
                    20 +
                    (validRoomHeight - furniture.y - actualHeight) * scaleY;

                  return (
                    <g key={furniture.id}>
                      {/* 선택된 가구 하이라이트 */}
                      {selectedFurnitureIndex === index && (
                        <rect
                          x={scaledX - 5}
                          y={scaledY - 5}
                          width={scaledWidth + 10}
                          height={scaledHeight + 10}
                          fill="none"
                          stroke="#FbbF24"
                          strokeWidth="2"
                          strokeDasharray="5,5"
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
                        strokeWidth={selectedFurnitureIndex === index ? 3 : 1}
                        className="cursor-pointer hover:opacity-80 transition-opacity"
                        onMouseDown={(e) => {
                          e.preventDefault();
                          handleSelectFurniture(index);

                          if (selectedFurnitureIndex === index) {
                            setIsDraggingPlaced(true);

                            const startCoords = convertToRealCoordinates(
                              e.clientX,
                              e.clientY
                            );
                            const offsetX = startCoords.x - furniture.x;
                            const offsetY = startCoords.y - furniture.y;

                            const handleMouseMove = (moveEvent) => {
                              const currentCoords = convertToRealCoordinates(
                                moveEvent.clientX,
                                moveEvent.clientY
                              );
                              handleMoveFurniture(
                                index,
                                currentCoords.x - offsetX,
                                currentCoords.y - offsetY
                              );
                            };

                            const handleMouseUp = () => {
                              setIsDraggingPlaced(false);
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
                            document.addEventListener("mouseup", handleMouseUp);
                          }
                        }}
                      />

                      {/* 가구 아이콘 */}
                      <text
                        x={scaledX + scaledWidth / 2}
                        y={scaledY + scaledHeight / 2}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize="16"
                        fill="white"
                        className="pointer-events-none select-none"
                        style={{ userSelect: "none" }}
                      >
                        {furniture.icon}
                      </text>

                      {/* 선택된 가구 정보 표시 */}
                      {selectedFurnitureIndex === index && (
                        <g>
                          {/* 가구 이름 */}
                          <text
                            x={scaledX + scaledWidth / 2}
                            y={scaledY - 25}
                            textAnchor="middle"
                            fontSize="12"
                            fill="#1F2937"
                            className="pointer-events-none select-none font-medium"
                          >
                            {furniture.name}
                            {rotation !== 0 && ` (${rotation}°)`}
                          </text>

                          {/* 가로 치수 */}
                          <text
                            x={scaledX + scaledWidth / 2}
                            y={scaledY - 10}
                            textAnchor="middle"
                            fontSize="11"
                            fill="#4B5563"
                            className="pointer-events-none select-none"
                          >
                            {actualWidth}cm
                          </text>

                          {/* 세로 치수 */}
                          <text
                            x={scaledX - 10}
                            y={scaledY + scaledHeight / 2}
                            textAnchor="middle"
                            fontSize="11"
                            fill="#4B5563"
                            className="pointer-events-none select-none"
                            transform={`rotate(-90, ${scaledX - 10}, ${
                              scaledY + scaledHeight / 2
                            })`}
                          >
                            {actualHeight}cm
                          </text>

                          {/* 회전 버튼 */}
                          <circle
                            cx={scaledX + scaledWidth - 10}
                            cy={scaledY + scaledHeight - 10}
                            r="10"
                            fill="#10B981"
                            className="cursor-pointer hover:fill-green-600"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRotateFurniture(index);
                            }}
                          />
                          <text
                            x={scaledX + scaledWidth - 10}
                            y={scaledY + scaledHeight - 10}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fontSize="14"
                            fill="white"
                            className="pointer-events-none select-none"
                          >
                            ↻
                          </text>

                          {/* 삭제 버튼 */}
                          <circle
                            cx={scaledX + scaledWidth - 10}
                            cy={scaledY + 10}
                            r="10"
                            fill="#EF4444"
                            className="cursor-pointer hover:fill-red-600"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteFurniture(index);
                            }}
                          />
                          <text
                            x={scaledX + scaledWidth - 10}
                            y={scaledY + 10}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fontSize="16"
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
                  x={20 + svgDimensions.svgWidth / 2}
                  y={15}
                  textAnchor="middle"
                  fontSize="12"
                  fill="#666"
                >
                  {validRoomWidth.toFixed(0)} cm
                </text>
                <text
                  x="10"
                  y={20 + svgDimensions.svgHeight / 2}
                  textAnchor="middle"
                  fontSize="12"
                  fill="#666"
                  transform={`rotate(-90, 10, ${
                    20 + svgDimensions.svgHeight / 2
                  })`}
                >
                  {validRoomHeight.toFixed(0)} cm
                </text>

                {/* 원점 라벨 */}
                <text
                  x="20"
                  y={20 + svgDimensions.svgHeight + 15}
                  textAnchor="middle"
                  style={{ fontSize: 10, fill: "#ef4444", fontWeight: 600 }}
                >
                  (0,0)
                </text>
              </svg>
            </div>
          </div>

          {/* 드롭 영역 안내 */}
          {draggedFurniture && (
            <div className="mt-2 text-center">
              <p className="text-sm text-pink-600 font-medium animate-pulse">
                위 회색 영역에 {draggedFurniture.name}을(를) 드래그해서 놓으세요
              </p>
            </div>
          )}

          {/* 선택된 가구 정보 */}
          {selectedFurnitureIndex !== null &&
            placedFurniture[selectedFurnitureIndex] && (
              <div className="mt-4 p-4 bg-pink-50 border border-pink-200 rounded-lg">
                <h4 className="font-medium text-pink-800 mb-2">
                  선택된 가구: {placedFurniture[selectedFurnitureIndex].name}
                </h4>
                <div className="text-sm text-pink-700 space-y-1">
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
                    {(() => {
                      const furniture = placedFurniture[selectedFurnitureIndex];
                      const rotation = furniture.rotation || 0;
                      const actualWidth =
                        rotation % 180 === 0
                          ? furniture.width
                          : furniture.height;
                      const actualHeight =
                        rotation % 180 === 0
                          ? furniture.height
                          : furniture.width;

                      const leftBottomX = Math.round(furniture.x);
                      const leftBottomY = Math.round(furniture.y);
                      const rightTopX = Math.round(furniture.x + actualWidth);
                      const rightTopY = Math.round(furniture.y + actualHeight);

                      return (
                        <>
                          위치 (왼쪽아래): ({leftBottomX}, {leftBottomY}) cm
                          <br />
                          위치 (오른쪽위): ({rightTopX}, {rightTopY}) cm
                        </>
                      );
                    })()}
                  </div>
                  <div>
                    회전:{" "}
                    {placedFurniture[selectedFurnitureIndex].rotation || 0}°
                  </div>
                </div>
                <div className="mt-3 text-xs text-pink-600">
                  가구를 드래그하여 이동하거나, 녹색 버튼으로 회전할 수 있습니다
                </div>
              </div>
            )}

          {/* 사용 가이드 */}
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-700 mb-2">
              <strong>사용 가이드</strong>
            </h4>
            <div className="text-sm text-gray-600">
              <strong className="text-gray-700">가구 배치</strong>
              <ul className="mt-1 space-y-1 ml-4">
                <li>• 왼쪽 목록에서 드래그하여 배치</li>
                <li>• 클릭으로 선택, 드래그로 이동</li>
                <li>• 녹색 버튼으로 90° 회전</li>
                <li>• 빨간 버튼으로 삭제</li>
              </ul>
            </div>
            <div className="mt-3 text-xs text-gray-500">
              가구가 겹치거나 방 밖으로 나가지 않도록 자동으로 제한됩니다
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FurniturePlacement;
