// frontend/src/components/FurniturePlacement.jsx
import React, { useState, useRef, useCallback, useMemo, useEffect } from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faBed,
  faChair,
  faCouch,
  faTable,
  faTv,
  faBook,
  faBox,
} from '@fortawesome/free-solid-svg-icons';

// Import furniture presets from constants
import { FURNITURE_PRESETS } from '../constants/furniture';

// Convert furniture presets to catalog format
const FURNITURE_CATALOG = Object.entries(FURNITURE_PRESETS).map(([id, preset]) => ({
  id: id,
  name: preset.name,
  width: preset.size[0], // width from size array
  depth: preset.size[2], // depth from size array (index 2)
  category: preset.category,
  icon: preset.icon === "faBed" ? faBed :
        preset.icon === "faChair" ? faChair :
        preset.icon === "faCouch" ? faCouch :
        preset.icon === "faTable" ? faTable :
        preset.icon === "faTv" ? faTv :
        preset.icon === "faBook" ? faBook :
        preset.icon === "faBox" ? faBox :
        faBox, // default icon
}));

const CATEGORIES = [
  { id: "all", name: "All", icon: "" },
  { id: "bedroom", name: "Bedroom", icon: "" },
  { id: "living", name: "Living", icon: "" },
  { id: "office", name: "Office", icon: "" },
  { id: "storage", name: "Storage", icon: "" },
];

const FurnitureItem = ({ furniture, onDragStart }) => (
  <div
    className="relative p-4 border-2 border-border bg-surface hover:border-primary hover:shadow-lg rounded-lg cursor-grab active:cursor-grabbing transition-all duration-200 flex flex-col items-center justify-center min-h-[100px]"
    draggable
    onDragStart={(e) => {
      e.dataTransfer.effectAllowed = "copy";
      onDragStart(e, furniture);
    }}
  >
    <div className="text-4xl mb-2 text-primary">
      <FontAwesomeIcon icon={furniture.icon} />
    </div>
    <div className="text-sm font-medium text-text-primary text-center">{furniture.name}</div>
    <div className="text-xs text-text-secondary text-center mt-1">
      {furniture.width} × {furniture.depth} cm
    </div>
  </div>
);

const FurniturePlacement = ({ roomWidth, roomDepth, placedFurniture, onFurnitureChange, detectedWindows = [], roomHeight = 250 }) => {
  // 유효성 검사 - Width(X), Depth(Y) 단위: cm (먼저 선언)
  const validRoomWidth = isNaN(roomWidth) || roomWidth <= 0 ? 400 : roomWidth;
  const validRoomDepth = isNaN(roomDepth) || roomDepth <= 0 ? 300 : roomDepth;

  // State 변수들
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [selectedFurnitureIndex, setSelectedFurnitureIndex] = useState(null);
  const [draggedFurniture, setDraggedFurniture] = useState(null);
  const [isDraggingPlaced, setIsDraggingPlaced] = useState(false);
  const [dragPreview, setDragPreview] = useState(null);
  const [previewCollision, setPreviewCollision] = useState(false);
  const [copiedFurniture, setCopiedFurniture] = useState(null);
  const [customFurnitureName, setCustomFurnitureName] = useState("");
  const [customFurnitureSize, setCustomFurnitureSize] = useState({
    width: "",
    depth: "",
    height: ""
  });
  
  // Refs
  const canvasRef = useRef(null);
  const isUndoRedoing = useRef(false);

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

  // addToHistory 함수 먼저 선언
  const addToHistory = useCallback((newState) => {
    setHistoryIndex(currentIndex => {
      setHistory(prev => {
        const newHistory = prev.slice(0, currentIndex + 1);
        newHistory.push(JSON.parse(JSON.stringify(newState))); // 깊은 복사
        return newHistory.slice(-50); // 최대 50개 히스토리 유지
      });
      return Math.min(currentIndex + 1, 49);
    });
  }, []);

  // placedFurniture 변경 감지 및 히스토리 추가
  useEffect(() => {
    console.log('📐 FurniturePlacement - placedFurniture 업데이트됨:', placedFurniture);
    // 실행취소/다시실행 중이 아닐 때만 히스토리에 추가
    if (!isUndoRedoing.current) {
      addToHistory(placedFurniture);
    }
  }, [placedFurniture, addToHistory]);

  // 히스토리 관리 함수들

  const undo = useCallback(() => {
    console.log('🔄 Undo 시도:', { historyIndex, historyLength: history.length });
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      const prevState = history[newIndex];
      console.log('🔄 Undo 실행:', { newIndex, prevState });
      
      isUndoRedoing.current = true;
      setHistoryIndex(newIndex);
      onFurnitureChange(prevState);
      
      // 다음 렌더링 사이클에서 플래그 해제
      setTimeout(() => { isUndoRedoing.current = false; }, 0);
    }
  }, [history, historyIndex, onFurnitureChange]);

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      isUndoRedoing.current = true;
      setHistoryIndex(newIndex);
      const nextState = history[newIndex];
      onFurnitureChange(nextState);
      
      // 다음 렌더링 사이클에서 플래그 해제
      setTimeout(() => { isUndoRedoing.current = false; }, 0);
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

      // 회전 후 경계 체크 및 자동 위치 조정
      const maxX = validRoomWidth - newActualWidth;
      const maxZ = validRoomDepth - newActualDepth;

      let newX = furniture.x;
      let newZ = furniture.z;

      // 범위를 벗어나면 안전한 위치로 이동
      if (newX > maxX) newX = Math.max(0, maxX);
      if (newZ > maxZ) newZ = Math.max(0, maxZ);

      // 충돌 체크 (새로운 위치와 회전에서)
      if (
        checkCollision(
          newX,
          newZ,
          furniture.width,
          furniture.depth,
          index,
          newRotation
        )
      ) {
        // 충돌이 발생하면 빈 공간을 찾아서 이동
        let foundPosition = false;
        const stepSize = 25; // 25cm씩 이동하며 탐색

        // 가까운 위치부터 탐색
        for (let radius = 0; radius <= Math.max(validRoomWidth, validRoomDepth) && !foundPosition; radius += stepSize) {
          // 원래 위치 주변을 나선형으로 탐색
          for (let angle = 0; angle < 360 && !foundPosition; angle += 45) {
            const testX = Math.max(0, Math.min(newX + radius * Math.cos(angle * Math.PI / 180), maxX));
            const testZ = Math.max(0, Math.min(newZ + radius * Math.sin(angle * Math.PI / 180), maxZ));
            
            if (!checkCollision(testX, testZ, furniture.width, furniture.depth, index, newRotation)) {
              newX = testX;
              newZ = testZ;
              foundPosition = true;
            }
          }
        }

        if (!foundPosition) {
          alert("회전할 수 있는 빈 공간이 없습니다!");
          return;
        }
      }

      const newPlaced = [...placedFurniture];
      newPlaced[index] = {
        ...furniture,
        x: newX,
        z: newZ,
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
        category: "storage", // Use existing category instead of "custom"
        color: "#DDA0DD",
        icon: faBox,
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


  return (
    <div className="mt-8 p-6 bg-surface rounded-xl shadow-lg border border-border">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-text-primary flex items-center gap-2">
          가구 배치 시뮬레이셔
        </h2>
        <div className="flex flex-wrap gap-2">
          {/* 실행취소/다시실행 */}
          <div className="flex gap-1">
            <button
              onClick={undo}
              className="px-3 py-2 bg-secondary hover:bg-primary text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={historyIndex <= 0}
              title="실행취소 (Ctrl+Z)"
            >
              ↶
            </button>
            <button
              onClick={redo}
              className="px-3 py-2 bg-secondary hover:bg-primary text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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
              className="px-3 py-2 bg-secondary hover:bg-primary text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={selectedFurnitureIndex === null}
              title="복사 (Ctrl+C)"
            >
              📋
            </button>
            <button
              onClick={pasteFurniture}
              className="px-3 py-2 bg-secondary hover:bg-primary text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!copiedFurniture}
              title="붙여넣기 (Ctrl+V)"
            >
              📌
            </button>
          </div>

          <button
            onClick={() => {
              // 데이터 저장 후 AI Design 페이지로 이동
              if (placedFurniture.length > 0) {
                const saveData = {
                  roomInfo: {
                    width: validRoomWidth,
                    depth: validRoomDepth,
                    height: roomHeight,
                    area: ((validRoomWidth * validRoomDepth) / 10000).toFixed(1) + "㎡",
                  },
                  furniture: placedFurniture,
                  statistics: {
                    furnitureCount: placedFurniture.length,
                    spaceUtilization: calculateSpaceUtilization + "%",
                  }
                };
                localStorage.setItem('roomPlannerData', JSON.stringify(saveData));
                window.location.href = '/ai-design';
              } else {
                alert("AI 디자인을 생성하려면 먼저 가구를 배치해주세요!");
              }
            }}
            className="px-4 py-2 bg-primary hover:bg-secondary text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={placedFurniture.length === 0}
          >
            <strong>AI 디자인 생성</strong>
          </button>
          <button
            onClick={handleClearAll}
            className="px-4 py-2 bg-danger hover:bg-danger-dark text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={placedFurniture.length === 0}
          >
            <strong>전체 삭제</strong>
          </button>
        </div>
      </div>

      {/* 통계 정보 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-background p-4 rounded-lg border border-border">
          <div className="text-sm text-text-secondary font-medium">
            방 크기
          </div>
          <div className="text-lg font-bold text-text-primary">
            {validRoomWidth.toFixed(1)} × {validRoomDepth.toFixed(1)} cm
          </div>
        </div>
        <div className="bg-background p-4 rounded-lg border border-border">
          <div className="text-sm text-text-secondary font-medium">
            배치된 가구
          </div>
          <div className="text-lg font-bold text-text-primary">
            {placedFurniture.length} 개
          </div>
        </div>
        <div className="bg-background p-4 rounded-lg border border-border">
          <div className="text-sm text-text-secondary font-medium">
            공간 활용률
          </div>
          <div className="text-lg font-bold text-text-primary">
            {calculateSpaceUtilization}%
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 가구 카탈로그 */}
        <div className="lg:col-span-1">
          <h3 className="text-2xl font-bold mb-4 text-text-primary">
            가구 선택
          </h3>

          {/* 카테고리 탭 */}
          <div className={`flex flex-wrap gap-2 mb-4 ${draggedFurniture && !isDraggingPlaced ? 'opacity-50 pointer-events-none' : ''}`}>
            {CATEGORIES.map((category) => (
              <button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedCategory === category.id
                    ? "bg-primary text-white"
                    : "bg-background text-text-secondary hover:bg-border"
                }`}
              >
                {category.name}
              </button>
            ))}
          </div>

          {/* 가구 목록 */}
          <div className={`grid grid-cols-2 gap-3 max-h-96 overflow-y-auto ${draggedFurniture && !isDraggingPlaced ? 'opacity-50' : ''}`}>
            {filteredItems.map((item) => (
              <FurnitureItem
                key={item.id}
                furniture={item}
                onDragStart={handleDragStart}
              />
            ))}
          </div>

          {/* 커스텀 가구 추가 */}
          <div className="mt-4 p-3 bg-background border border-border rounded-lg">
            <h4 className="text-lg font-semibold mb-3 text-text-primary">커스텀 가구 추가</h4>
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
                className="w-full px-3 py-2 bg-primary text-white rounded text-sm font-medium hover:bg-secondary disabled:bg-border disabled:cursor-not-allowed"
              >
                커스텀 가구 추가
              </button>
            </div>
          </div>

          {/* 템플릿 관리 */}
          <div className="mt-4 p-3 bg-background border border-border rounded-lg">
            <h4 className="text-lg font-semibold mb-3 text-text-primary">템플릿 관리</h4>
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
                  className="px-3 py-1 bg-primary text-white rounded text-sm font-medium hover:bg-secondary disabled:bg-border disabled:cursor-not-allowed"
                >
                  저장
                </button>
              </div>
              
              <div className="max-h-32 overflow-y-auto">
                {getSavedTemplates().map((template, index) => (
                  <div key={index} className="flex items-center justify-between py-1 px-2 bg-surface rounded text-sm">
                    <span className="truncate flex-1">{template.name}</span>
                    <div className="flex gap-1 ml-2">
                      <button
                        onClick={() => loadTemplate(template.name)}
                        className="px-2 py-1 bg-primary text-white rounded text-xs hover:bg-secondary"
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
                        className="px-2 py-1 bg-danger text-white rounded text-xs hover:bg-danger-dark"
                      >
                        삭제
                      </button>
                    </div>
                  </div>
                ))}
                {getSavedTemplates().length === 0 && (
                  <div className="text-xs text-text-secondary text-center py-2">저장된 템플릿이 없습니다</div>
                )}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-background border border-border rounded-lg">
            <p className="text-sm text-text-secondary">
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
          <h3 className="text-2xl font-bold mb-4 text-text-primary">
            <strong>방 평면도</strong>
          </h3>

          <div className="border-2 border-dashed border-border rounded-lg p-4 bg-surface">
            <div className="mb-2 text-sm text-text-secondary text-center">
              실제 비율: {validRoomWidth} × {validRoomDepth} cm (
              {(validRoomWidth / validRoomDepth).toFixed(2)}:1) - 좌표: 왼쪽 위
              (0,0)
            </div>

            <div className="flex justify-center">
              <svg
                ref={canvasRef}
                width={svgDimensions.svgWidth + 40}
                height={svgDimensions.svgHeight + 40}
                className="border border-border bg-surface rounded-lg cursor-crosshair"
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
                  fill="none"
                  stroke="#4A4A4A"
                  strokeWidth="3"
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
                      stroke="#D3D3D3" strokeWidth="1.5"
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
                  fill="var(--danger)"
                  stroke="var(--text-primary)"
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
                        fill="var(--window-fill)"
                        stroke="var(--window-stroke)"
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
                        fill="var(--text-primary)"
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
                          fill={previewCollision ? "var(--danger)" : "var(--surface)"}
                          stroke={previewCollision ? "var(--danger-dark)" : "var(--text-secondary)"}
                          strokeWidth="2"
                          opacity="0.6"
                          strokeDasharray="5,5"
                          className="pointer-events-none"
                        />
                        {/* 미리보기 아이콘 */}
                        <foreignObject
                          x={scaledX + scaledWidth / 2 - 8}
                          y={scaledY + scaledDepth / 2 - 8}
                          width="16"
                          height="16"
                          className="pointer-events-none"
                        >
                          <FontAwesomeIcon 
                            icon={dragPreview.furniture.icon} 
                            style={{ 
                              color: previewCollision ? "var(--danger)" : "var(--text-secondary)",
                              opacity: 0.8,
                              fontSize: "16px"
                            }} 
                          />
                        </foreignObject>
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
                          stroke="var(--accent)"
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
                        fill="#B0C4DE"
                        stroke={
                          selectedFurnitureIndex === index
                            ? "none"
                            : "var(--text-secondary)"
                        }
                        strokeWidth={selectedFurnitureIndex === index ? 0 : 1}
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

                      {/* 가구 텍스트 정보 */}
                      <text
                        x={scaledX + scaledWidth / 2}
                        y={scaledY + scaledDepth / 2 - 8}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize="12"
                        fill="white"
                        className="pointer-events-none select-none font-bold"
                      >
                        {furniture.name}
                      </text>
                      <text
                        x={scaledX + scaledWidth / 2}
                        y={scaledY + scaledDepth / 2 + 8}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize="10"
                        fill="white"
                        className="pointer-events-none select-none"
                      >
                        {actualWidth}x{actualDepth}cm
                      </text>

                      {/* 가로 치수 (가구 상단) */}
                      {selectedFurnitureIndex === index && (
                        <text
                          x={scaledX + scaledWidth / 2}
                          y={scaledY - 5}
                          textAnchor="middle"
                          fontSize="9"
                          fill="var(--text-primary)"
                          className="pointer-events-none select-none"
                        >
                          {actualWidth}cm
                        </text>
                      )}

                      {/* 세로 치수 (가구 왼쪽) */}
                      {selectedFurnitureIndex === index && (
                        <text
                          x={scaledX - 10}
                          y={scaledY + scaledDepth / 2}
                          textAnchor="middle"
                          fontSize="9"
                          fill="var(--text-primary)"
                          className="pointer-events-none select-none"
                          transform={`rotate(-90, ${scaledX - 10}, ${
                            scaledY + scaledDepth / 2
                          })`}
                        >
                          {actualDepth}cm
                        </text>
                      )}

                      {/* 선택된 가구 정보 표시 - 드래그 중이 아닐 때만 버튼 표시 */}
                      {selectedFurnitureIndex === index && !draggedFurniture && (
                        <g>
                          {/* 회전 버튼 */}
                          <circle
                            cx={scaledX + scaledWidth - 10}
                            cy={scaledY + scaledDepth - 10}
                            r="10"
                            fill="var(--accent)"
                            className="cursor-pointer hover:fill-accent-dark"
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
                            fill="var(--danger)"
                            className="cursor-pointer hover:fill-danger-dark"
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
                  fill="var(--text-secondary)"
                >
                  {validRoomWidth.toFixed(0)} cm
                </text>
                <text
                  x="10"
                  y={20 + svgDimensions.svgHeight / 2}
                  textAnchor="middle"
                  fontSize="12"
                  fill="var(--text-secondary)"
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
                  style={{ fontSize: 10, fill: "var(--danger)", fontWeight: 600 }}
                >
                  (0,0)
                </text>
              </svg>
            </div>
          </div>

          {/* 드롭 영역 안내 */}
          {draggedFurniture && (
            <div className="mt-2 text-center">
              <p className="text-sm text-text-secondary font-medium animate-pulse">
                위 회색 영역에 {draggedFurniture.name}을(를) 드래그해서 놓으세요
              </p>
            </div>
          )}

          {/* 선택된 가구 정보 */}
          {selectedFurnitureIndex !== null &&
            placedFurniture[selectedFurnitureIndex] && (
              <div className="mt-4 p-4 bg-background border border-border rounded-lg">
                <h4 className="font-medium text-text-primary mb-2">
                  선택된 가구: {placedFurniture[selectedFurnitureIndex].name}
                </h4>
                <div className="text-sm text-text-secondary space-y-1">
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
                <div className="mt-3 text-xs text-text-secondary">
                  가구를 드래그하여 이동하거나, 녹색 버튼으로 회전할 수 있습니다
                </div>
              </div>
            )}

          {/* 사용 가이드 */}
          <div className="mt-4 p-4 bg-background rounded-lg">
            <h4 className="font-medium text-text-primary mb-2">
              <strong>사용 가이드</strong>
            </h4>
            <div className="text-sm text-text-secondary">
              <strong className="text-text-primary">가구 배치</strong>
              <ul className="mt-1 space-y-1 ml-4">
                <li>• 왼쪽 목록에서 드래그하여 배치</li>
                <li>• 클릭으로 선택, 드래그로 이동</li>
                <li>• 녹색 버튼으로 90° 회전</li>
                <li>• 빨간 버튼으로 삭제</li>
              </ul>
            </div>
            <div className="mt-3 text-xs text-text-secondary">
              가구가 겹치거나 방 밖으로 나가지 않도록 자동으로 제한됩니다
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FurniturePlacement;
