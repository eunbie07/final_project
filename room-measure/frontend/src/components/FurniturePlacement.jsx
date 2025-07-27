// frontend/src/components/FurniturePlacement.jsx
import React, { useState, useRef, useCallback, useMemo, useEffect } from "react";

const FURNITURE_CATALOG = [
  // 침실 가구
  {
    id: "single_bed",
    name: "싱글 베드",
    width: 100,
    depth: 200,
    category: "bedroom",
    color: "#FFB6C1",
    icon: "🛏️",
  },
  {
    id: "double_bed",
    name: "더블 베드",
    width: 150,
    depth: 200,
    category: "bedroom",
    color: "#FFD1DC",
    icon: "🛏️",
  },
  {
    id: "queen_bed",
    name: "퀸 베드",
    width: 160,
    depth: 200,
    category: "bedroom",
    color: "#FFC0CB",
    icon: "🛏️",
  },
  {
    id: "king_bed",
    name: "킹 베드",
    width: 180,
    depth: 200,
    category: "bedroom",
    color: "#FFB7C5",
    icon: "🛏️",
  },
  // 책상/의자
  {
    id: "desk",
    name: "책상",
    width: 120,
    depth: 60,
    category: "office",
    color: "#98FB98",
    icon: "🪑",
  },
  {
    id: "chair",
    name: "의자",
    width: 50,
    depth: 50,
    category: "office",
    color: "#90EE90",
    icon: "🪑",
  },
  // 거실 가구
  {
    id: "sofa_2",
    name: "2인 소파",
    width: 140,
    depth: 80,
    category: "living",
    color: "#87CEEB",
    icon: "🛋️",
  },
  {
    id: "sofa_3",
    name: "3인 소파",
    width: 180,
    depth: 80,
    category: "living",
    color: "#ADD8E6",
    icon: "🛋️",
  },
  {
    id: "coffee_table",
    name: "커피 테이블",
    width: 100,
    depth: 50,
    category: "living",
    color: "#B0E0E6",
    icon: "🪑",
  },
  {
    id: "tv_stand",
    name: "TV 스탠드",
    width: 120,
    depth: 40,
    category: "living",
    color: "#E0FFFF",
    icon: "📺",
  },
  // 수납 가구
  {
    id: "wardrobe",
    name: "옷장",
    width: 80,
    depth: 60,
    category: "storage",
    color: "#DDA0DD",
    icon: "🚪",
  },
  {
    id: "bookshelf",
    name: "책장",
    width: 80,
    depth: 30,
    category: "storage",
    color: "#D8BFD8",
    icon: "📚",
  },
  {
    id: "dresser",
    name: "화장대",
    width: 100,
    depth: 45,
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
        {furniture.width} × {furniture.depth} cm
      </div>
    </div>
  </div>
);

const FurniturePlacement = ({
  roomWidth, // Width(X축) - 가로, cm 단위
  roomDepth, // Depth(Z축) - 세로, cm 단위 (3D와 일치)
  roomHeight = 240, // 방 높이, cm 단위
  placedFurniture = [],
  onFurnitureChange,
  detectedWindows = [], // 창문 정보
}) => {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [selectedFurnitureIndex, setSelectedFurnitureIndex] = useState(null);
  const [draggedFurniture, setDraggedFurniture] = useState(null);
  const [isDraggingPlaced, setIsDraggingPlaced] = useState(false);
  const canvasRef = useRef(null);
  
  // 드래그 미리보기 상태
  const [dragPreview, setDragPreview] = useState(null);
  const [previewCollision, setPreviewCollision] = useState(false);
  
  // 실행취소/다시실행 상태
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [isUndoRedoing, setIsUndoRedoing] = useState(false); // 실행취소/다시실행 중인지 확인
  
  // 복사/붙여넣기 상태
  const [copiedFurniture, setCopiedFurniture] = useState(null);
  
  // 커스텀 가구 상태
  const [customFurnitureName, setCustomFurnitureName] = useState("");
  const [customFurnitureSize, setCustomFurnitureSize] = useState({
    width: "",
    depth: "",
    height: ""
  });

  // 유효성 검사 - Width(X), Depth(Y) 단위: cm (먼저 선언)
  const validRoomWidth = isNaN(roomWidth) || roomWidth <= 0 ? 400 : roomWidth;
  const validRoomDepth = isNaN(roomDepth) || roomDepth <= 0 ? 300 : roomDepth;

  // 충돌 체크 함수 (먼저 선언)
  const checkCollision = useCallback(
    (x, z, width, depth, excludeIndex = -1, rotation = 0) => {
      const actualWidth = rotation % 180 === 0 ? width : depth;
      const actualDepth = rotation % 180 === 0 ? depth : width;

      for (let i = 0; i < placedFurniture.length; i++) {
        if (i === excludeIndex) continue;

        const item = placedFurniture[i];
        const itemRotation = item.rotation || 0;
        const itemActualWidth =
          itemRotation % 180 === 0 ? item.width : item.depth;
        const itemActualDepth =
          itemRotation % 180 === 0 ? item.depth : item.width;

        if (
          x < item.x + itemActualWidth &&
          x + actualWidth > item.x &&
          z < item.z + itemActualDepth &&
          z + actualDepth > item.z
        ) {
          return true;
        }
      }

      return false;
    },
    [placedFurniture]
  );

  // placedFurniture 변경 감지 및 히스토리 추가
  useEffect(() => {
    console.log('📐 FurniturePlacement - placedFurniture 업데이트됨:', placedFurniture);
    // 실행취소/다시실행 중이 아닐 때만 히스토리에 추가
    if (!isUndoRedoing) {
      addToHistory(placedFurniture);
    }
  }, [placedFurniture, isUndoRedoing]);

  // 히스토리 관리 함수들
  const addToHistory = useCallback((newState) => {
    setHistoryIndex(currentIndex => {
      setHistory(prev => {
        const newHistory = prev.slice(0, currentIndex + 1);
        newHistory.push(JSON.parse(JSON.stringify(newState))); // 깊은 복사
        console.log('📚 히스토리 추가:', { currentIndex, newLength: newHistory.length });
        
        if (newHistory.length > 50) { // 최대 50개까지 저장
          newHistory.shift();
          return newHistory;
        }
        return newHistory;
      });
      
      return Math.min(currentIndex + 1, 49);
    });
  }, []);

  const undo = useCallback(() => {
    console.log('🔄 Undo 시도:', { historyIndex, historyLength: history.length });
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      const prevState = history[newIndex];
      console.log('🔄 Undo 실행:', { newIndex, prevState });
      
      setIsUndoRedoing(true);
      setHistoryIndex(newIndex);
      onFurnitureChange(prevState);
      
      // 다음 렌더링 사이클에서 플래그 해제
      setTimeout(() => setIsUndoRedoing(false), 0);
    }
  }, [history, historyIndex, onFurnitureChange]);

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setIsUndoRedoing(true);
      setHistoryIndex(newIndex);
      const nextState = history[newIndex];
      onFurnitureChange(nextState);
      
      // 다음 렌더링 사이클에서 플래그 해제
      setTimeout(() => setIsUndoRedoing(false), 0);
    }
  }, [history, historyIndex, onFurnitureChange]);

  // 복사/붙여넣기 함수들
  const copySeletedFurniture = useCallback(() => {
    if (selectedFurnitureIndex !== null && placedFurniture[selectedFurnitureIndex]) {
      const furniture = placedFurniture[selectedFurnitureIndex];
      setCopiedFurniture(JSON.parse(JSON.stringify(furniture))); // 깊은 복사
    }
  }, [selectedFurnitureIndex, placedFurniture]);

  const pasteFurniture = useCallback(() => {
    if (!copiedFurniture) return;

    // 빈 공간을 찾는 로직
    const furnitureWidth = copiedFurniture.width;
    const furnitureDepth = copiedFurniture.depth;
    const stepSize = 25; // 25cm씩 이동하며 탐색
    
    // 여러 오프셋을 시도해보기
    const offsets = [
      { x: 20, z: 20 },   // 기본 오프셋
      { x: 40, z: 20 },   // 더 오른쪽
      { x: 20, z: 40 },   // 더 아래쪽
      { x: 60, z: 20 },   // 훨씬 오른쪽
      { x: 20, z: 60 },   // 훨씬 아래쪽
      { x: 40, z: 40 },   // 대각선
      { x: 80, z: 20 },   // 멀리 오른쪽
      { x: 20, z: 80 },   // 멀리 아래쪽
    ];
    
    // 각 오프셋을 시도
    for (const offset of offsets) {
      let newX = copiedFurniture.x + offset.x;
      let newZ = copiedFurniture.z + offset.z;
      
      // 방 경계 확인
      if (newX + furnitureWidth <= validRoomWidth && 
          newZ + furnitureDepth <= validRoomDepth) {
        
        // 충돌 확인
        if (!checkCollision(newX, newZ, furnitureWidth, furnitureDepth, -1, copiedFurniture.rotation || 0)) {
          const newFurniture = {
            ...copiedFurniture,
            id: `${copiedFurniture.id.split('_')[0]}_${Date.now()}`,
            x: newX,
            z: newZ,
          };
          
          onFurnitureChange([...placedFurniture, newFurniture]);
          setSelectedFurnitureIndex(placedFurniture.length);
          return; // 성공하면 즉시 종료
        }
      }
    }
    
    // 기본 오프셋으로도 안되면 격자 탐색
    for (let x = 0; x <= validRoomWidth - furnitureWidth; x += stepSize) {
      for (let z = 0; z <= validRoomDepth - furnitureDepth; z += stepSize) {
        if (!checkCollision(x, z, furnitureWidth, furnitureDepth, -1, copiedFurniture.rotation || 0)) {
          const newFurniture = {
            ...copiedFurniture,
            id: `${copiedFurniture.id.split('_')[0]}_${Date.now()}`,
            x: x,
            z: z,
          };
          
          onFurnitureChange([...placedFurniture, newFurniture]);
          setSelectedFurnitureIndex(placedFurniture.length);
          return; // 성공하면 즉시 종료
        }
      }
    }
    
    // 모든 시도가 실패하면 알림
    alert("방에 붙여넣을 빈 공간이 없습니다!");
  }, [copiedFurniture, validRoomWidth, validRoomDepth, checkCollision, placedFurniture, onFurnitureChange, setSelectedFurnitureIndex]);

  // 가구 삭제 함수
  const handleDeleteFurniture = useCallback(
    (index) => {
      const newPlaced = [...placedFurniture];
      newPlaced.splice(index, 1);
      onFurnitureChange(newPlaced);
      setSelectedFurnitureIndex(null);
    },
    [placedFurniture, onFurnitureChange, setSelectedFurnitureIndex]
  );

  // 키보드 단축키
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z' && !e.shiftKey) {
          e.preventDefault();
          undo();
        } else if ((e.key === 'y') || (e.key === 'z' && e.shiftKey)) {
          e.preventDefault();
          redo();
        } else if (e.key === 'c' && selectedFurnitureIndex !== null) {
          e.preventDefault();
          copySeletedFurniture();
        } else if (e.key === 'v' && copiedFurniture) {
          e.preventDefault();
          pasteFurniture();
        }
      }
      if (e.key === 'Delete' && selectedFurnitureIndex !== null) {
        e.preventDefault();
        handleDeleteFurniture(selectedFurnitureIndex);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [undo, redo, selectedFurnitureIndex, copiedFurniture, copySeletedFurniture, pasteFurniture, handleDeleteFurniture]);

  // 템플릿 저장/불러오기 함수들
  const saveTemplate = useCallback((templateName) => {
    if (!templateName || placedFurniture.length === 0) {
      alert("템플릿 이름을 입력하고 배치된 가구가 있어야 합니다.");
      return;
    }

    const templateData = {
      name: templateName,
      furniture: JSON.parse(JSON.stringify(placedFurniture)),
      roomSize: { width: validRoomWidth, depth: validRoomDepth },
      createdAt: new Date().toISOString(),
    };

    const savedTemplates = JSON.parse(localStorage.getItem('furnitureTemplates') || '[]');
    const existingIndex = savedTemplates.findIndex(t => t.name === templateName);
    
    if (existingIndex >= 0) {
      if (confirm(`"${templateName}" 템플릿이 이미 존재합니다. 덮어쓰시겠습니까?`)) {
        savedTemplates[existingIndex] = templateData;
      } else {
        return;
      }
    } else {
      savedTemplates.push(templateData);
    }

    localStorage.setItem('furnitureTemplates', JSON.stringify(savedTemplates));
    alert(`"${templateName}" 템플릿이 저장되었습니다.`);
  }, [placedFurniture, validRoomWidth, validRoomDepth]);

  const loadTemplate = useCallback((templateName) => {
    const savedTemplates = JSON.parse(localStorage.getItem('furnitureTemplates') || '[]');
    const template = savedTemplates.find(t => t.name === templateName);
    
    if (!template) {
      alert("템플릿을 찾을 수 없습니다.");
      return;
    }

    if (placedFurniture.length > 0) {
      if (!confirm("현재 배치된 가구들이 모두 삭제됩니다. 계속하시겠습니까?")) {
        return;
      }
    }

    onFurnitureChange(template.furniture);
    setSelectedFurnitureIndex(null);
    alert(`"${templateName}" 템플릿이 불러와졌습니다.`);
  }, [placedFurniture.length, onFurnitureChange]);

  const getSavedTemplates = useCallback(() => {
    return JSON.parse(localStorage.getItem('furnitureTemplates') || '[]');
  }, []);

  const deleteTemplate = useCallback((templateName) => {
    const savedTemplates = JSON.parse(localStorage.getItem('furnitureTemplates') || '[]');
    const filteredTemplates = savedTemplates.filter(t => t.name !== templateName);
    
    localStorage.setItem('furnitureTemplates', JSON.stringify(filteredTemplates));
    alert(`"${templateName}" 템플릿이 삭제되었습니다.`);
  }, []);

  // SVG 크기 계산
  const svgDimensions = useMemo(() => {
    const aspectRatio = validRoomWidth / validRoomDepth;
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
  }, [validRoomWidth, validRoomDepth]);

  // 카테고리별 아이템 필터링
  const filteredItems = FURNITURE_CATALOG.filter(
    (item) => selectedCategory === "all" || item.category === selectedCategory
  );

  // 클라이언트 좌표를 실제 가구 배치 좌표로 변환
  const convertToRealCoordinates = useCallback(
    (clientX, clientY) => {
      const rect = canvasRef.current.getBoundingClientRect();
      const svgX = clientX - rect.left - 20;
      const svgY = clientY - rect.top - 20;

      const scaleX = validRoomWidth / svgDimensions.svgWidth;
      const scaleZ = validRoomDepth / svgDimensions.svgHeight;

      // SVG 좌표를 방 좌표로 변환
      // SVG x → 실제 x (가로축)
      // SVG y → 실제 y (세로축) 하지만 z 필드에 저장
      // 이는 나중에 3D에서 z축으로 사용됨
      return {
        x: svgX * scaleX, // 2D x 좌표 (3D에서도 x축)
        z: svgY * scaleZ, // 2D y 좌표 (3D에서 z축으로 변환됨)
      };
    },
    [validRoomWidth, validRoomDepth, svgDimensions]
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
    
    if (draggedFurniture && !isDraggingPlaced) {
      const coords = convertToRealCoordinates(e.clientX, e.clientY);
      
      // 경계 체크
      const maxX = validRoomWidth - draggedFurniture.width;
      const maxZ = validRoomDepth - draggedFurniture.depth;
      
      coords.x = Math.max(0, Math.min(coords.x, maxX));
      coords.z = Math.max(0, Math.min(coords.z, maxZ));
      
      // 충돌 체크
      const hasCollision = checkCollision(
        coords.x,
        coords.z,
        draggedFurniture.width,
        draggedFurniture.depth
      );
      
      setDragPreview({
        x: coords.x,
        z: coords.z,
        furniture: draggedFurniture
      });
      setPreviewCollision(hasCollision);
    }
  }, [draggedFurniture, isDraggingPlaced, convertToRealCoordinates, validRoomWidth, validRoomDepth, checkCollision]);

  // 드롭
  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();

      if (!draggedFurniture || isDraggingPlaced) return;

      const coords = convertToRealCoordinates(e.clientX, e.clientY);

      // 경계 체크
      const maxX = validRoomWidth - draggedFurniture.width;
      const maxZ = validRoomDepth - draggedFurniture.depth;

      coords.x = Math.max(0, Math.min(coords.x, maxX));
      coords.z = Math.max(0, Math.min(coords.z, maxZ));

      // 일반 가구 처리
      if (
        !checkCollision(
          coords.x,
          coords.z,
          draggedFurniture.width,
          draggedFurniture.depth
        )
      ) {
        const newItem = {
          ...draggedFurniture,
          x: coords.x,
          z: coords.z,
          rotation: 0,
          id: `${draggedFurniture.id}_${Date.now()}`,
        };
        onFurnitureChange([...placedFurniture, newItem]);
      } else {
        alert("이 위치에는 다른 가구가 있습니다!");
      }

      setDraggedFurniture(null);
      setDragPreview(null);
      setPreviewCollision(false);
    },
    [
      draggedFurniture,
      isDraggingPlaced,
      convertToRealCoordinates,
      validRoomWidth,
      validRoomDepth,
      checkCollision,
      placedFurniture,
      onFurnitureChange,
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
        newRotation % 180 === 0 ? furniture.width : furniture.depth;
      const newActualDepth =
        newRotation % 180 === 0 ? furniture.depth : furniture.width;

      // 회전 후 경계 체크
      const maxX = validRoomWidth - newActualWidth;
      const maxZ = validRoomDepth - newActualDepth;

      if (furniture.x > maxX || furniture.z > maxZ) {
        alert("회전하면 방 범위를 벗어납니다!");
        return;
      }

      // 충돌 체크
      if (
        checkCollision(
          furniture.x,
          furniture.z,
          furniture.width,
          furniture.depth,
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
      onFurnitureChange(newPlaced);
    },
    [placedFurniture, validRoomWidth, validRoomDepth, checkCollision]
  );

  // 가구 이동
  const handleMoveFurniture = useCallback(
    (index, newX, newZ) => {
      const furniture = placedFurniture[index];
      const rotation = furniture.rotation || 0;
      const actualWidth =
        rotation % 180 === 0 ? furniture.width : furniture.depth;
      const actualDepth =
        rotation % 180 === 0 ? furniture.depth : furniture.width;

      // 경계 체크
      const maxX = validRoomWidth - actualWidth;
      const maxZ = validRoomDepth - actualDepth;

      const clampedX = Math.max(0, Math.min(newX, maxX));
      const clampedZ = Math.max(0, Math.min(newZ, maxZ));

      // 충돌 체크
      if (
        !checkCollision(
          clampedX,
          clampedZ,
          furniture.width,
          furniture.depth,
          index,
          rotation
        )
      ) {
        const newPlaced = [...placedFurniture];
        newPlaced[index] = {
          ...furniture,
          x: clampedX,
          z: clampedZ,
        };
        onFurnitureChange(newPlaced);
      }
    },
    [placedFurniture, validRoomWidth, validRoomDepth, checkCollision]
  );

  // 가구 삭제

  // 전체 초기화
  const handleClearAll = useCallback(() => {
    if (placedFurniture.length === 0) return;
    if (confirm("모든 가구를 삭제하시겠습니까?")) {
      onFurnitureChange([]);
      setSelectedFurnitureIndex(null);
    }
  }, [placedFurniture.length, onFurnitureChange]);

  // 커스텀 가구 추가
  const handleAddCustomFurniture = useCallback(() => {
    if (!customFurnitureName || !customFurnitureSize.width || !customFurnitureSize.depth || !customFurnitureSize.height) {
      alert("가구 이름과 크기(폭, 깊이, 높이)를 모두 입력해주세요.");
      return;
    }

    const width = parseInt(customFurnitureSize.width);
    const depth = parseInt(customFurnitureSize.depth);
    const height = parseInt(customFurnitureSize.height);

    if (width < 10 || width > 500 || depth < 10 || depth > 500) {
      alert("가구 폭과 깊이는 10cm ~ 500cm 사이여야 합니다.");
      return;
    }

    if (height < 10 || height > 300) {
      alert("가구 높이는 10cm ~ 300cm 사이여야 합니다.");
      return;
    }

    // 방 중앙에 배치
    const x = Math.max(0, (validRoomWidth - width) / 2);
    const z = Math.max(0, (validRoomDepth - depth) / 2);

    // 충돌 체크
    if (!checkCollision(x, z, width, depth, -1, 0)) {
      const newFurniture = {
        id: `custom_${Date.now()}`,
        name: customFurnitureName,
        width: width,
        depth: depth,
        height: height,
        x: x,
        z: z,
        rotation: 0,
        category: "custom",
        color: "#DDA0DD",
        icon: "📦",
        isCustom: true
      };

      onFurnitureChange([...placedFurniture, newFurniture]);
      
      // 입력 필드 초기화
      setCustomFurnitureName("");
      setCustomFurnitureSize({ width: "", depth: "", height: "" });
    } else {
      alert("해당 위치에 가구를 배치할 수 없습니다. (충돌 발생)");
    }
  }, [customFurnitureName, customFurnitureSize, validRoomWidth, validRoomDepth, checkCollision, placedFurniture, onFurnitureChange]);

  // 공간 활용률 계산
  const calculateSpaceUtilization = useMemo(() => {
    const totalFurnitureArea = placedFurniture.reduce((sum, furniture) => {
      const rotation = furniture.rotation || 0;
      const actualWidth =
        rotation % 180 === 0 ? furniture.width : furniture.depth;
      const actualDepth =
        rotation % 180 === 0 ? furniture.depth : furniture.width;
      return sum + actualWidth * actualDepth;
    }, 0);
    const roomArea = validRoomWidth * validRoomDepth;
    return ((totalFurnitureArea / roomArea) * 100).toFixed(1);
  }, [placedFurniture, validRoomWidth, validRoomDepth]);

  // JSON 저장 기능
  const handleSaveAsJson = useCallback(() => {
    if (placedFurniture.length === 0 && detectedWindows.length === 0) {
      alert("저장할 가구나 창문이 없습니다!");
      return;
    }

    // JSON 데이터 구성
    const saveData = {
      roomInfo: {
        width: validRoomWidth,
        depth: validRoomDepth,
        height: roomHeight,
        area: ((validRoomWidth * validRoomDepth) / 10000).toFixed(1) + "㎡",
        aspectRatio: (validRoomWidth / validRoomDepth).toFixed(2),
      },
      furniture: placedFurniture.map((furniture) => {
        const rotation = furniture.rotation || 0;
        const actualWidth =
          rotation % 180 === 0 ? furniture.width : furniture.depth;
        const actualDepth =
          rotation % 180 === 0 ? furniture.depth : furniture.width;

        return {
          id: furniture.id,
          name: furniture.name,
          category: furniture.category,
          originalSize: {
            width: furniture.width,
            depth: furniture.depth,
          },
          currentSize: {
            width: actualWidth,
            depth: actualDepth,
          },
          position: {
            leftBottom: {
              x: Math.round(furniture.x),
              z: Math.round(furniture.z),
            },
            rightTop: {
              x: Math.round(furniture.x + actualWidth),
              z: Math.round(furniture.z + actualDepth),
            },
          },
          rotation: rotation,
          color: furniture.color,
          icon: furniture.icon,
        };
      }),
      windows: detectedWindows.map((window, index) => {
        const widthCm = Math.round((window.width_meters || 1.2) * 100);
        const heightCm = Math.round((window.height_meters || 1.5) * 100);
        
        // 3D 좌표 계산
        const userYPosition = window.y_position !== undefined ? window.y_position : 0.8;
        const calculatedYPos = userYPosition * roomHeight;
        let x_3d, y_3d, z_3d;
        
        switch (window.wall_position) {
          case "front":
            x_3d = (window.x_position || 0.5) * validRoomWidth;
            y_3d = calculatedYPos;
            z_3d = validRoomDepth;
            break;
          case "back":
            x_3d = (window.x_position || 0.5) * validRoomWidth;
            y_3d = calculatedYPos;
            z_3d = 0;
            break;
          case "left":
            x_3d = 0;
            y_3d = calculatedYPos;
            z_3d = (window.x_position || 0.5) * validRoomDepth;
            break;
          case "right":
            x_3d = validRoomWidth;
            y_3d = calculatedYPos;
            z_3d = (window.x_position || 0.5) * validRoomDepth;
            break;
          default:
            x_3d = validRoomWidth / 2;
            y_3d = calculatedYPos;
            z_3d = 0;
        }
        
        return {
          id: `window_${index + 1}`,
          wall_position: window.wall_position || "back",
          size: {
            width_cm: widthCm,
            height_cm: heightCm,
            width_meters: (window.width_meters || 1.2).toFixed(2),
            height_meters: (window.height_meters || 1.5).toFixed(2),
          },
          position: {
            relative: {
              x_position: (window.x_position || 0.5).toFixed(3),
              y_position: (window.y_position || 0.8).toFixed(3),
            },
            absolute_3d: {
              x: Math.round(x_3d),
              y: Math.round(y_3d),
              z: Math.round(z_3d),
            },
            wall_coordinates: {
              horizontal_percent: Math.round((window.x_position || 0.5) * 100),
              vertical_percent: Math.round((window.y_position || 0.8) * 100),
            }
          },
          confidence: window.confidence || 1.0,
        };
      }),
      statistics: {
        furnitureCount: placedFurniture.length,
        windowCount: detectedWindows.length,
        spaceUtilization: calculateSpaceUtilization + "%",
        totalFurnitureArea:
          placedFurniture.reduce((sum, furniture) => {
            const rotation = furniture.rotation || 0;
            const actualWidth =
              rotation % 180 === 0 ? furniture.width : furniture.depth;
            const actualDepth =
              rotation % 180 === 0 ? furniture.depth : furniture.width;
            return sum + actualWidth * actualDepth;
          }, 0) + " cm²",
        totalWindowArea:
          detectedWindows.reduce((sum, window) => {
            const widthCm = (window.width_meters || 1.2) * 100;
            const heightCm = (window.height_meters || 1.5) * 100;
            return sum + widthCm * heightCm;
          }, 0).toFixed(0) + " cm²",
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
    const filename = `room_layout_${validRoomWidth}x${validRoomDepth}x${roomHeight}_${currentDate}.json`;

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
    validRoomDepth,
    calculateSpaceUtilization,
  ]);

  return (
    <div className="mt-8 p-6 bg-white rounded-xl shadow-lg border">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
          <strong>가구 배치 시뮬레이션</strong>
        </h2>
        <div className="flex flex-wrap gap-2">
          {/* 실행취소/다시실행 */}
          <div className="flex gap-1">
            <button
              onClick={undo}
              className="px-3 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={historyIndex <= 0}
              title="실행취소 (Ctrl+Z)"
            >
              ↶
            </button>
            <button
              onClick={redo}
              className="px-3 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={historyIndex >= history.length - 1}
              title="다시실행 (Ctrl+Y)"
            >
              ↷
            </button>
          </div>

          {/* 복사/붙여넣기 */}
          <div className="flex gap-1">
            <button
              onClick={copySeletedFurniture}
              className="px-3 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={selectedFurnitureIndex === null}
              title="복사 (Ctrl+C)"
            >
              📋
            </button>
            <button
              onClick={pasteFurniture}
              className="px-3 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!copiedFurniture}
              title="붙여넣기 (Ctrl+V)"
            >
              📌
            </button>
          </div>

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
            {validRoomWidth.toFixed(1)} × {validRoomDepth.toFixed(1)} cm
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

          {/* 커스텀 가구 추가 */}
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="text-sm font-semibold mb-3 text-blue-800">커스텀 가구 추가</h4>
            <div className="space-y-2">
              <input
                type="text"
                placeholder="가구 이름"
                value={customFurnitureName}
                onChange={(e) => setCustomFurnitureName(e.target.value)}
                className="w-full px-2 py-1 text-sm border rounded"
              />
              <div className="space-y-2">
                <div className="flex gap-2">
                  <input
                    type="number"
                    placeholder="폭(cm)"
                    value={customFurnitureSize.width}
                    onChange={(e) => setCustomFurnitureSize(prev => ({...prev, width: parseInt(e.target.value) || 0}))}
                    className="flex-1 px-2 py-1 text-sm border rounded"
                    min="10"
                    max="500"
                  />
                  <input
                    type="number"
                    placeholder="깊이(cm)"
                    value={customFurnitureSize.depth}
                    onChange={(e) => setCustomFurnitureSize(prev => ({...prev, depth: parseInt(e.target.value) || 0}))}
                    className="flex-1 px-2 py-1 text-sm border rounded"
                    min="10"
                    max="500"
                  />
                </div>
                <input
                  type="number"
                  placeholder="높이(cm)"
                  value={customFurnitureSize.height}
                  onChange={(e) => setCustomFurnitureSize(prev => ({...prev, height: parseInt(e.target.value) || 0}))}
                  className="w-full px-2 py-1 text-sm border rounded"
                  min="10"
                  max="300"
                />
              </div>
              <button
                onClick={handleAddCustomFurniture}
                disabled={!customFurnitureName || !customFurnitureSize.width || !customFurnitureSize.depth || !customFurnitureSize.height}
                className="w-full px-3 py-2 bg-blue-500 text-white rounded text-sm font-medium hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                커스텀 가구 추가
              </button>
            </div>
          </div>

          {/* 템플릿 관리 */}
          <div className="mt-4 p-3 bg-purple-50 border border-purple-200 rounded-lg">
            <h4 className="text-sm font-semibold mb-3 text-purple-800">템플릿 관리</h4>
            <div className="space-y-2">
              <div className="flex gap-1">
                <input
                  type="text"
                  placeholder="템플릿 이름"
                  id="templateName"
                  className="flex-1 px-2 py-1 text-sm border rounded"
                />
                <button
                  onClick={() => {
                    const templateName = document.getElementById('templateName').value.trim();
                    if (templateName) {
                      saveTemplate(templateName);
                      document.getElementById('templateName').value = '';
                    }
                  }}
                  disabled={placedFurniture.length === 0}
                  className="px-3 py-1 bg-purple-500 text-white rounded text-sm font-medium hover:bg-purple-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
                >
                  저장
                </button>
              </div>
              
              <div className="max-h-32 overflow-y-auto">
                {getSavedTemplates().map((template, index) => (
                  <div key={index} className="flex items-center justify-between py-1 px-2 bg-white rounded text-sm">
                    <span className="truncate flex-1">{template.name}</span>
                    <div className="flex gap-1 ml-2">
                      <button
                        onClick={() => loadTemplate(template.name)}
                        className="px-2 py-1 bg-blue-500 text-white rounded text-xs hover:bg-blue-600"
                      >
                        불러오기
                      </button>
                      <button
                        onClick={() => {
                          if (confirm(`"${template.name}" 템플릿을 삭제하시겠습니까?`)) {
                            deleteTemplate(template.name);
                            // 강제 리렌더링을 위해 상태 업데이트
                            setSelectedFurnitureIndex(selectedFurnitureIndex);
                          }
                        }}
                        className="px-2 py-1 bg-red-500 text-white rounded text-xs hover:bg-red-600"
                      >
                        삭제
                      </button>
                    </div>
                  </div>
                ))}
                {getSavedTemplates().length === 0 && (
                  <div className="text-xs text-gray-500 text-center py-2">저장된 템플릿이 없습니다</div>
                )}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-pink-50 border border-pink-200 rounded-lg">
            <p className="text-sm text-pink-800">
              <strong>사용법:</strong>
              <br />
              • 가구를 드래그해서 방에 배치 (미리보기 제공)
              <br />
              • 배치된 항목 클릭 후 드래그로 이동
              <br />
              • 녹색 버튼으로 회전, 빨간 버튼으로 삭제
              <br />
              <strong>단축키:</strong>
              <br />
              • Ctrl+Z: 실행취소, Ctrl+Y: 다시실행
              <br />
              • Ctrl+C: 복사, Ctrl+V: 붙여넣기
              <br />
              • Delete: 선택된 가구 삭제
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
              실제 비율: {validRoomWidth} × {validRoomDepth} cm (
              {(validRoomWidth / validRoomDepth).toFixed(2)}:1) - 좌표: 왼쪽 위
              (0,0)
            </div>

            <div className="flex justify-center">
              <svg
                ref={canvasRef}
                width={svgDimensions.svgWidth + 40}
                height={svgDimensions.svgHeight + 40}
                className="border border-gray-400 bg-white rounded-lg cursor-crosshair"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onDragLeave={(e) => {
                  // SVG 영역을 완전히 벗어날 때만 미리보기 제거
                  const rect = e.currentTarget.getBoundingClientRect();
                  const x = e.clientX;
                  const y = e.clientY;
                  
                  if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
                    setDragPreview(null);
                    setPreviewCollision(false);
                  }
                }}
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

                {/* 원점 표시 (0,0) - 왼쪽 위 */}
                <circle
                  cx="20"
                  cy="20"
                  r="3"
                  fill="#ef4444"
                  stroke="#ffffff"
                  strokeWidth="1"
                />

                {/* 창문들 표시 */}
                {detectedWindows.map((window, index) => {
                  const windowWidth = (window.width_meters || 1.2) * 100; // m -> cm
                  const windowHeight = (window.height_meters || 1.5) * 100; // m -> cm
                  const wallThickness = 8; // 벽 두께 (SVG 단위)
                  
                  let windowX, windowY, windowW, windowH;
                  
                  switch (window.wall_position) {
                    case "front": // 앞벽 (아래쪽)
                      windowX = 20 + (window.x_position || 0.5) * svgDimensions.svgWidth - (windowWidth * svgDimensions.svgWidth / validRoomWidth) / 2;
                      windowY = 20 + svgDimensions.svgHeight - wallThickness;
                      windowW = windowWidth * svgDimensions.svgWidth / validRoomWidth;
                      windowH = wallThickness;
                      break;
                    case "back": // 뒷벽 (위쪽)
                      windowX = 20 + (window.x_position || 0.5) * svgDimensions.svgWidth - (windowWidth * svgDimensions.svgWidth / validRoomWidth) / 2;
                      windowY = 20;
                      windowW = windowWidth * svgDimensions.svgWidth / validRoomWidth;
                      windowH = wallThickness;
                      break;
                    case "left": // 왼쪽 벽
                      windowX = 20;
                      windowY = 20 + (window.x_position || 0.5) * svgDimensions.svgHeight - (windowWidth * svgDimensions.svgHeight / validRoomDepth) / 2;
                      windowW = wallThickness;
                      windowH = windowWidth * svgDimensions.svgHeight / validRoomDepth;
                      break;
                    case "right": // 오른쪽 벽
                      windowX = 20 + svgDimensions.svgWidth - wallThickness;
                      windowY = 20 + (window.x_position || 0.5) * svgDimensions.svgHeight - (windowWidth * svgDimensions.svgHeight / validRoomDepth) / 2;
                      windowW = wallThickness;
                      windowH = windowWidth * svgDimensions.svgHeight / validRoomDepth;
                      break;
                    default:
                      return null;
                  }
                  
                  return (
                    <g key={`window-${index}`}>
                      {/* 창문 배경 */}
                      <rect
                        x={windowX}
                        y={windowY}
                        width={windowW}
                        height={windowH}
                        fill="#87CEEB"
                        stroke="#4682B4"
                        strokeWidth="1"
                        opacity="0.8"
                      />
                      {/* 창문 텍스트 */}
                      <text
                        x={windowX + windowW / 2}
                        y={windowY + windowH / 2}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize="8"
                        fill="#2F4F4F"
                        className="pointer-events-none select-none"
                      >
                        창문
                      </text>
                    </g>
                  );
                })}

                {/* 드래그 미리보기 */}
                {dragPreview && (
                  (() => {
                    const scaleX = svgDimensions.svgWidth / validRoomWidth;
                    const scaleZ = svgDimensions.svgHeight / validRoomDepth;
                    const scaledWidth = dragPreview.furniture.width * scaleX;
                    const scaledDepth = dragPreview.furniture.depth * scaleZ;
                    const scaledX = 20 + dragPreview.x * scaleX;
                    const scaledY = 20 + dragPreview.z * scaleZ;

                    return (
                      <g key="drag-preview">
                        {/* 미리보기 가구 */}
                        <rect
                          x={scaledX}
                          y={scaledY}
                          width={scaledWidth}
                          height={scaledDepth}
                          fill={previewCollision ? "#ff9999" : dragPreview.furniture.color}
                          stroke={previewCollision ? "#ff0000" : "#666666"}
                          strokeWidth="2"
                          opacity="0.6"
                          strokeDasharray="5,5"
                          className="pointer-events-none"
                        />
                        {/* 미리보기 아이콘 */}
                        <text
                          x={scaledX + scaledWidth / 2}
                          y={scaledY + scaledDepth / 2}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize="16"
                          fill={previewCollision ? "#ff0000" : "#666666"}
                          opacity="0.8"
                          className="pointer-events-none select-none"
                        >
                          {dragPreview.furniture.icon}
                        </text>
                      </g>
                    );
                  })()
                )}

                {/* 배치된 가구들 */}
                {placedFurniture.map((furniture, index) => {
                  const scaleX = svgDimensions.svgWidth / validRoomWidth;
                  const scaleZ = svgDimensions.svgHeight / validRoomDepth;

                  const rotation = furniture.rotation || 0;
                  const actualWidth =
                    rotation % 180 === 0 ? furniture.width : furniture.depth;
                  const actualDepth =
                    rotation % 180 === 0 ? furniture.depth : furniture.width;

                  const scaledWidth = actualWidth * scaleX;
                  const scaledDepth = actualDepth * scaleZ;
                  const scaledX = 20 + furniture.x * scaleX;
                  // SVG는 위쪽이 0이므로 Z좌표를 그대로 사용 (뒤집지 않음)
                  const scaledY = 20 + furniture.z * scaleZ;

                  return (
                    <g key={furniture.id}>
                      {/* 선택된 가구 하이라이트 */}
                      {selectedFurnitureIndex === index && (
                        <rect
                          x={scaledX - 5}
                          y={scaledY - 5}
                          width={scaledWidth + 10}
                          height={scaledDepth + 10}
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
                        height={scaledDepth}
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
                            const offsetZ = startCoords.z - furniture.z;

                            const handleMouseMove = (moveEvent) => {
                              const currentCoords = convertToRealCoordinates(
                                moveEvent.clientX,
                                moveEvent.clientY
                              );
                              handleMoveFurniture(
                                index,
                                currentCoords.x - offsetX,
                                currentCoords.z - offsetZ
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
                        y={scaledY + scaledDepth / 2}
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
                            y={scaledY + scaledDepth / 2}
                            textAnchor="middle"
                            fontSize="11"
                            fill="#4B5563"
                            className="pointer-events-none select-none"
                            transform={`rotate(-90, ${scaledX - 10}, ${
                              scaledY + scaledDepth / 2
                            })`}
                          >
                            {actualDepth}cm
                          </text>

                          {/* 회전 버튼 */}
                          <circle
                            cx={scaledX + scaledWidth - 10}
                            cy={scaledY + scaledDepth - 10}
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
                            y={scaledY + scaledDepth - 10}
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
                  {validRoomDepth.toFixed(0)} cm
                </text>

                {/* 원점 라벨 */}
                <text
                  x="20"
                  y="15"
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
                      : placedFurniture[selectedFurnitureIndex].depth}{" "}
                    ×{" "}
                    {(placedFurniture[selectedFurnitureIndex].rotation || 0) %
                      180 ===
                    0
                      ? placedFurniture[selectedFurnitureIndex].depth
                      : placedFurniture[selectedFurnitureIndex].width}
                    {placedFurniture[selectedFurnitureIndex].height && (
                      <> × {placedFurniture[selectedFurnitureIndex].height}</>
                    )}{" "}
                    cm
                  </div>
                  <div>
                    {(() => {
                      const furniture = placedFurniture[selectedFurnitureIndex];
                      const rotation = furniture.rotation || 0;
                      const actualWidth =
                        rotation % 180 === 0
                          ? furniture.width
                          : furniture.depth;
                      const actualDepth =
                        rotation % 180 === 0
                          ? furniture.depth
                          : furniture.width;

                      const leftBottomX = Math.round(furniture.x);
                      const leftBottomZ = Math.round(furniture.z);
                      const rightTopX = Math.round(furniture.x + actualWidth);
                      const rightTopZ = Math.round(furniture.z + actualDepth);

                      return (
                        <>
                          위치 (왼쪽아래): ({leftBottomX}, {leftBottomZ}) cm
                          <br />
                          위치 (오른쪽위): ({rightTopX}, {rightTopZ}) cm
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
