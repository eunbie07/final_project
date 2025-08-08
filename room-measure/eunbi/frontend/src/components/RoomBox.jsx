import React, {
  useState,
  useMemo,
  Suspense,
  useRef,
  useCallback,
  useEffect,
} from "react";
import { useNavigate } from "react-router-dom";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  Text as DreiText,
  Line,
  Environment,
  ContactShadows,
} from "@react-three/drei";
import * as THREE from "three";

// 분리된 유틸리티들
import CollisionDetector from "../utils/CollisionDetector";
import PositionSnapper from "../utils/PositionSnapper";

// 분리된 UI 컴포넌트들
import ViewPresets from "./UI/ViewPresets";
import SpaceUtilization from "./UI/SpaceUtilization";
import ToastContainer from "./UI/Toast";
import { LoadingButton } from "./UI/LoadingSpinner";
import PlacementGuide from "./UI/PlacementGuide";

// 분리된 3D 컴포넌트들
import EnhancedLighting from "./3D/EnhancedLighting";
import FloorGrid from "./3D/FloorGrid";
import {
  DimensionArrow,
  DimensionLabel,
  DistanceMeasurer,
} from "./3D/DimensionComponents";
import { ValidPlacementArea } from "./3D/CollisionComponents";
import { WindowsOnWalls } from "./3D/WindowComponents";
import { DraggableFurnitureWithCollision } from "./3D/DraggableFurniture";
import { DraggableHuman } from "./3D/DraggableHuman";

// 분리된 훅들
import { useRoomState } from "../hooks/useRoomState";
import { useToast } from "../hooks/useToast";

// 분리된 상수들
import {
  FURNITURE_PRESETS,
  FURNITURE_ID_MAPPING,
} from "../constants/furniture";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faBed,
  faChair,
  faCouch,
  faTable,
  faTv,
  faBook,
  faBox,
} from "@fortawesome/free-solid-svg-icons";

const iconMapping = {
  faBed: faBed,
  faChair: faChair,
  faCouch: faCouch,
  faTable: faTable,
  faTv: faTv,
  faBook: faBook,
  faBox: faBox,
};

// 분리된 유틸리티들
import { convertCoordinatesLocally } from "../utils/coordinateConversion";
import { createRoomLayoutData } from "../utils/dataConversion";
import { saveRoomLayoutToMongoDB, detectWindowsInImage } from "../utils/api";

// GLB 추출 스크립트 임포트 (임시)
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";

// 스냅 그리드 컴포넌트
const SnapGrid = React.memo(function SnapGrid({
  roomSize,
  gridSize = 50,
  visible = false,
}) {
  if (!visible) return null;

  const [roomWidth, roomHeight, roomDepth] = roomSize;
  const lines = [];

  for (let x = 0; x <= roomWidth; x += gridSize) {
    lines.push(
      <Line
        key={`vertical-${x}`}
        points={[
          [x, 0.2, 0],
          [x, 0.2, roomDepth],
        ]}
        color="#e0e0e0"
        lineWidth={1}
      />
    );
  }

  for (let z = 0; z <= roomDepth; z += gridSize) {
    lines.push(
      <Line
        key={`horizontal-${z}`}
        points={[
          [0, 0.2, z],
          [roomWidth, 0.2, z],
        ]}
        color="#e0e0e0"
        lineWidth={1}
      />
    );
  }

  return <group>{lines}</group>;
});

// 벽 컴포넌트
const Wall = React.memo(function Wall({
  width,
  height,
  position,
  rotation,
  isWindow = false,
  color = "#f8f6f0",
  texture = null,
  roughness = 0.9,
  metalness = 0.02,
}) {
  return (
    <mesh position={position} rotation={rotation} castShadow receiveShadow>
      <planeGeometry args={[width, height]} />
      <meshPhysicalMaterial
        color={isWindow ? "#E8F4FD" : color}
        map={texture}
        roughness={isWindow ? 0.1 : roughness}
        metalness={isWindow ? 0.02 : metalness}
        clearcoat={isWindow ? 0.8 : metalness > 0.1 ? 0.6 : 0.1}
        clearcoatRoughness={isWindow ? 0.1 : roughness}
        opacity={isWindow ? 0.3 : 1}
        transparent={isWindow}
        side={THREE.DoubleSide}
        normalScale={[0.5, 0.5]}
        envMapIntensity={0.7}
        reflectivity={metalness > 0.1 ? 0.8 : 0.2}
      />
    </mesh>
  );
});

// 메인 컴포넌트
export default function RoomBox({
  width = 400,
  height = 230,
  depth = 400,
  isFullscreen = false,
  uploadedImageFile = null,
  placedFurniture = [],
  onFurnitureChange = null,
}) {
  const w = width;
  const h = height;
  const d = depth;
  const navigate = useNavigate();

  // 커스텀 훅으로 상태 관리
  const roomState = useRoomState([w, h, d]);

  // Toast 알림 시스템
  const { toasts, removeToast, showSuccess, showError, showWarning, showInfo } =
    useToast();

  const {
    furniture,
    setFurniture,
    selectedFurniture,
    setSelectedFurniture,
    showSnapGrid,
    setShowSnapGrid,
    showFloorGrid,
    setShowFloorGrid,
    enableSnap,
    setEnableSnap,
    showCollisions,
    setShowCollisions,
    showWindows,
    setShowWindows,
    showHuman,
    setShowHuman,
    placementMode,
    setPlacementMode,
    activeView,
    setActiveView,
    measurementMode,
    setMeasurementMode,
    isDragging,
    setIsDragging,
    collisionAlert,
    setCollisionAlert,
    measurePoints,
    setMeasurePoints,
    humanPosition,
    setHumanPosition,
    detectedWindows,
    setDetectedWindows,
    isDetectingWindows,
    setIsDetectingWindows,
    isSaving,
    setIsSaving,
  } = roomState;

  // AI 인테리어 생성 핸들러
  const handleAIInteriorGenerate = useCallback(async () => {
    try {
      showInfo("방 데이터를 저장하고 있습니다...");

      // 현재 방 데이터 준비
      const roomData = {
        dimensions: {
          width_cm: w,
          height_cm: h,
          depth_cm: d,
        },
        area_sqm: (w * d) / 10000,
        volume_cum: (w * h * d) / 1000000,
        furniture_3d: furniture.map((f) => ({
          name: f.name || f.type,
          type: f.type,
          position: f.position,
          scale: f.scale,
          rotation: f.rotation,
        })),
        created_at: new Date().toISOString(),
      };

      // MongoDB에 방 데이터 저장
      const saveData = {
        scene: {
          description: `AI 인테리어 생성을 위한 ${w / 10}cm × ${
            d / 10
          }cm 방 공간`,
          room: {
            width: w,
            depth: d,
            height: h,
          },
          objects: furniture.map((f) => ({
            name: f.name || f.type,
            type: f.type,
            position: f.position,
            scale: f.scale,
            rotation: f.rotation,
          })),
        },
        area_sqm: (w * d) / 10000,
        volume_cum: (w * h * d) / 1000000,
        created_at: new Date().toISOString(),
      };

      // MongoDB 저장
      const { saveRoomLayoutToMongoDB } = await import("../utils/api");
      const saveResult = await saveRoomLayoutToMongoDB(saveData);

      console.log("MongoDB 저장 성공:", saveResult);

      // localStorage에도 저장 (백업용)
      localStorage.setItem("currentRoomData", JSON.stringify(roomData));
      localStorage.setItem("mongoRoomId", saveResult.layout_id);

      showSuccess("방 데이터 저장 완료!");

      // AI 인테리어 페이지로 네비게이션 (MongoDB ID 포함)
      navigate("/ai-interior", {
        state: {
          roomData,
          mongoId: saveResult.layout_id,
        },
      });

      showInfo("AI 인테리어 디자이너로 이동합니다...");
    } catch (error) {
      console.error("방 데이터 저장 실패:", error);
      showError("방 데이터 저장에 실패했습니다. 다시 시도해주세요.");
    }
  }, [w, h, d, furniture, navigate, showInfo, showSuccess, showError]);

  // 3D 캡처 관련 상태
  const [capturedScreenshot, setCapturedScreenshot] = useState(null);
  const canvasRef = useRef();

  // 캡처된 이미지를 파일로 다운로드하는 함수
  const downloadCapturedImage = (dataURL, screenshotData) => {
    try {
      // 파일명 생성 (타임스탬프 + 선택된 가구 정보)
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const selectedFurnitureName =
        screenshotData.selectedFurniture?.name || "no-selection";
      const filename = `3d-capture-${selectedFurnitureName}-${timestamp}.png`;

      // Blob 생성
      const byteCharacters = atob(dataURL.split(",")[1]);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: "image/png" });

      // 다운로드 링크 생성
      const downloadLink = document.createElement("a");
      downloadLink.href = URL.createObjectURL(blob);
      downloadLink.download = filename;

      // 자동 다운로드 실행
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);

      // URL 정리
      URL.revokeObjectURL(downloadLink.href);

      console.log(`📸 3D 캡처 이미지 저장 완료: ${filename}`);
      return true;
    } catch (error) {
      console.error("이미지 저장 실패:", error);
      return false;
    }
  };

  // 3D 화면 캡처 핸들러
  const handle3DCapture = useCallback(() => {
    try {
      // Canvas 요소 찾기 (React Three Fiber의 Canvas)
      const canvasElement = document.querySelector("canvas");

      if (!canvasElement) {
        showError("3D 캔버스를 찾을 수 없습니다");
        return;
      }

      showInfo("3D 화면을 캡처하고 있습니다...");

      // 렌더링이 완료될 때까지 잠시 대기
      setTimeout(() => {
        try {
          // Canvas에서 이미지 데이터 추출 (고품질)
          const dataURL = canvasElement.toDataURL("image/png", 1.0);

          // 이미지가 검은색인지 확인
          const img = new Image();
          img.onload = () => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            const imageData = ctx.getImageData(
              0,
              0,
              canvas.width,
              canvas.height
            );
            const data = imageData.data;

            // 검은색 픽셀 비율 확인
            let blackPixels = 0;
            for (let i = 0; i < data.length; i += 4) {
              if (data[i] === 0 && data[i + 1] === 0 && data[i + 2] === 0) {
                blackPixels++;
              }
            }

            const blackRatio = blackPixels / (data.length / 4);

            if (blackRatio > 0.8) {
              showWarning("캡처된 이미지가 검은색입니다. 다시 시도해주세요.");
              return;
            }

            // 캡처된 스크린샷 데이터 생성
            const screenshotData = {
              imageData: dataURL,
              timestamp: new Date().toISOString(),
              furniture: furniture.map((f) => ({
                id: f.id,
                type: f.type,
                name: f.name,
                position: f.position,
                size: f.size,
                selected: f.id === selectedFurniture,
              })),
              selectedFurniture: selectedFurniture
                ? furniture.find((f) => f.id === selectedFurniture)
                : null,
              roomSize: [w, h, d],
              canvasSize: {
                width: canvasElement.width,
                height: canvasElement.height,
              },
            };

            setCapturedScreenshot(screenshotData);

            // 로컬 스토리지에 캡처 데이터 저장
            localStorage.setItem(
              "capturedScreenshot",
              JSON.stringify(screenshotData)
            );

            // 캡처된 이미지를 파일로 자동 저장
            const success = downloadCapturedImage(dataURL, screenshotData);

            showSuccess(
              `3D 화면 캡처 완료! ${furniture.length}개 가구 감지됨${
                success ? " (이미지 저장됨)" : ""
              }`
            );

            // 스타일 변경 패널로 데이터 전달
            if (selectedFurniture) {
              showInfo(
                `선택된 가구: ${
                  screenshotData.selectedFurniture?.name || "없음"
                } - AI 인테리어 페이지에서 스타일을 변경할 수 있습니다`
              );
            } else {
              showWarning("가구를 먼저 선택한 후 스타일을 변경할 수 있습니다");
            }
          };

          img.src = dataURL;
        } catch (error) {
          console.error("캡처 처리 실패:", error);
          showError("3D 화면 캡처에 실패했습니다");
        }
      }, 200); // 200ms 대기로 렌더링 완료 보장
    } catch (error) {
      console.error("3D 캡처 실패:", error);
      showError("3D 화면 캡처에 실패했습니다");
    }
  }, [
    furniture,
    selectedFurniture,
    w,
    h,
    d,
    showInfo,
    showSuccess,
    showError,
    showWarning,
  ]);

  // 3D 모델 사용 상태
  const [use3DModels, setUse3DModels] = useState(false);

  // 벽면 스타일 상태
  const [wallSettings, setWallSettings] = useState({
    color: "#f8f6f0",
    roughness: 0.9,
    metalness: 0.02,
    textureType: "none", // "none", "brick", "wood", "concrete", "wallpaper"
  });

  // 바닥 스타일 상태
  const [floorSettings, setFloorSettings] = useState({
    color: "#e8dcc0",
    roughness: 0.85,
    metalness: 0.05,
  });

  // 벽 텍스처 프리셋
  const wallPresets = {
    white: {
      color: "#f8f6f0",
      roughness: 0.9,
      metalness: 0.02,
      name: "화이트",
    },
    beige: {
      color: "#f5f5dc",
      roughness: 0.85,
      metalness: 0.01,
      name: "베이지",
    },
    gray: { color: "#d3d3d3", roughness: 0.8, metalness: 0.05, name: "회색" },
    blue: { color: "#e6f3ff", roughness: 0.9, metalness: 0.02, name: "파란색" },
    green: {
      color: "#f0fff0",
      roughness: 0.9,
      metalness: 0.02,
      name: "연두색",
    },
    pink: { color: "#ffeef5", roughness: 0.85, metalness: 0.01, name: "핑크" },
    yellow: {
      color: "#fffacd",
      roughness: 0.9,
      metalness: 0.02,
      name: "노란색",
    },
    brick: { color: "#cd853f", roughness: 0.95, metalness: 0.0, name: "벽돌" },
    wood: { color: "#deb887", roughness: 0.8, metalness: 0.0, name: "나무" },
    concrete: {
      color: "#a9a9a9",
      roughness: 0.95,
      metalness: 0.1,
      name: "콘크리트",
    },
  };

  // 바닥 텍스처 프리셋 (극단적 차이로 수정)
  const floorPresets = {
    wood_light: {
      color: "#e8dcc0",
      roughness: 0.9,
      metalness: 0.0,
      name: "밝은 목재",
    },
    wood_dark: {
      color: "#8b4513",
      roughness: 0.85,
      metalness: 0.0,
      name: "진한 목재",
    },
    tile_white: {
      color: "#f8f8ff",
      roughness: 0.05,
      metalness: 0.8,
      name: "화이트 타일",
    },
    tile_gray: {
      color: "#d3d3d3",
      roughness: 0.1,
      metalness: 0.7,
      name: "그레이 타일",
    },
    marble: {
      color: "#f0f0f0",
      roughness: 0.02,
      metalness: 0.9,
      name: "대리석",
    },
    concrete: {
      color: "#a9a9a9",
      roughness: 0.98,
      metalness: 0.0,
      name: "콘크리트",
    },
    carpet_beige: {
      color: "#f5deb3",
      roughness: 0.99,
      metalness: 0.0,
      name: "베이지 카펫",
    },
    carpet_gray: {
      color: "#808080",
      roughness: 0.99,
      metalness: 0.0,
      name: "그레이 카펫",
    },
    linoleum: {
      color: "#dda0dd",
      roughness: 0.2,
      metalness: 0.4,
      name: "리놀륨",
    },
    mirror: { color: "#c0c0c0", roughness: 0.0, metalness: 1.0, name: "거울" },
  };

  // GLB 메시 추출 함수 (임시)
  const extractMeshesToSeparateFiles = async () => {
    const loader = new GLTFLoader();
    const exporter = new GLTFExporter();

    try {
      console.log("GLB 파일 로딩 중...");
      const gltf = await new Promise((resolve, reject) => {
        loader.load(
          "/low_poly_furnitures_full_bundle.glb",
          resolve,
          undefined,
          reject
        );
      });

      console.log("=== 메시 추출 시작 ===");
      const meshes = [];

      gltf.scene.traverse((child) => {
        if (child.isMesh) {
          meshes.push({
            name: child.name,
            mesh: child.clone(),
          });
        }
      });

      console.log(`총 ${meshes.length}개의 메시를 찾았습니다.`);

      for (let i = 0; i < meshes.length; i++) {
        const { name, mesh } = meshes[i];

        try {
          const scene = new THREE.Scene();
          mesh.position.set(0, 0, 0);
          scene.add(mesh);

          const result = await new Promise((resolve, reject) => {
            exporter.parse(
              scene,
              (gltf) => resolve(gltf),
              { binary: true },
              (error) => reject(error)
            );
          });

          const blob = new Blob([result], { type: "application/octet-stream" });
          const url = URL.createObjectURL(blob);

          const link = document.createElement("a");
          link.href = url;
          link.download = `${name}.glb`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);

          URL.revokeObjectURL(url);
          console.log(`✅ ${name}.glb 파일 다운로드 완료`);

          await new Promise((resolve) => setTimeout(resolve, 500));
        } catch (error) {
          console.error(`❌ ${name} 메시 내보내기 실패:`, error);
        }
      }

      console.log("=== 모든 메시 추출 완료 ===");
    } catch (error) {
      console.error("GLB 파일 처리 실패:", error);
    }
  };

  // 브라우저 콘솔에서 사용할 수 있도록 등록
  useEffect(() => {
    window.extractMeshes = extractMeshesToSeparateFiles;
    console.log("🚀 GLB 추출 도구 준비 완료!");
    console.log("콘솔에서 extractMeshes() 를 실행하세요.");
  }, []);

  // 방 계산 최적화
  const roomSize = useMemo(() => [w, h, d], [w, h, d]);
  const roomArea = useMemo(() => (w * d) / 10000, [w, d]);

  const controlsRef = useRef();
  const isUpdatingFromDragRef = useRef(false);

  // 가구 변환 최적화
  const convertedFurniture = useMemo(() => {
    if (!placedFurniture || placedFurniture.length === 0) {
      return [];
    }

    return placedFurniture.map((item) => {
      let furnitureSize, mappedType, presetData;

      if (item.isCustom) {
        furnitureSize = [item.width, item.height || 60, item.depth];
        mappedType = "custom";
        presetData = {
          name: item.name,
          color: item.color || "#DDA0DD",
          size: furnitureSize,
          icon: "faBox", // Default icon for custom furniture
        };
      } else {
        const baseId = item.id
          ? item.id.split("_").slice(0, -1).join("_")
          : "desk";
        mappedType = FURNITURE_ID_MAPPING[baseId] || "desk";
        presetData = FURNITURE_PRESETS[mappedType];
        furnitureSize = presetData.size;
      }

      const x3d = item.x;
      const z3d = item.z;

      return {
        id: item.id,
        type: mappedType,
        name: presetData.name,
        color: presetData.color,
        size: furnitureSize,
        position: [
          x3d + furnitureSize[0] / 2,
          furnitureSize[1] / 2,
          z3d + furnitureSize[2] / 2,
        ],
        rotation: [0, (item.rotation || 0) * (Math.PI / 180), 0],
        original2D: item,
        isCustom: item.isCustom || false,
      };
    });
  }, [placedFurniture]);

  useEffect(() => {
    if (isDragging || isUpdatingFromDragRef.current) {
      return;
    }
    setFurniture(convertedFurniture);
  }, [convertedFurniture, isDragging]);

  useEffect(() => {
    setHumanPosition([w / 2, 0, d / 2]);
  }, [w, d]);

  const handleViewChange = useCallback(
    (preset) => {
      if (controlsRef.current) {
        controlsRef.current.object.position.set(...preset.position);
        controlsRef.current.target.set(...preset.target);
        controlsRef.current.update();
        setActiveView(preset.name);
      }
    },
    [setActiveView]
  );

  const handlePlaceFurniture = useCallback(
    (position) => {
      if (!placementMode) return;

      const preset = FURNITURE_PRESETS[placementMode];
      if (!preset) return;

      const newFurniture = {
        id: `${placementMode}_${Date.now()}`,
        type: placementMode,
        position: position,
        rotation: [0, 0, 0],
        size: preset.size,
        name: preset.name,
        color: preset.color,
      };

      const adjustedPosition = CollisionDetector.adjustToValidPosition(
        position,
        newFurniture.size,
        roomSize,
        furniture,
        newFurniture.id,
        FURNITURE_PRESETS
      );

      const furnitureCollisions = CollisionDetector.checkFurnitureCollisions(
        furniture,
        newFurniture.id,
        adjustedPosition,
        FURNITURE_PRESETS
      );

      if (furnitureCollisions.length === 0) {
        newFurniture.position = adjustedPosition;
        setFurniture((prev) => [...prev, newFurniture]);
        setSelectedFurniture(newFurniture.id);
        setPlacementMode(null);

        // 2D와 동기화
        if (typeof onFurnitureChange === "function") {
          const { position: pos3D, size } = newFurniture;
          const [x3D, y3D, z3D] = pos3D;

          const x2D = x3D - size[0] / 2;
          const z2D = z3D - size[2] / 2;

          const newItem2D = {
            id: newFurniture.id,
            x: x2D,
            z: z2D,
            width: size[0],
            depth: size[2],
            rotation: 0,
            type: newFurniture.type,
            name: newFurniture.name,
            color: newFurniture.color,
          };

          onFurnitureChange((prev) => [...prev, newItem2D]);
        }
      }
    },
    [placementMode, furniture, roomSize, onFurnitureChange]
  );

  const handleFloorClick = useCallback(
    (event) => {
      event.stopPropagation();
      if (placementMode) {
        const position = [
          event.point.x,
          FURNITURE_PRESETS[placementMode].size[1] / 2,
          event.point.z,
        ];
        handlePlaceFurniture(position);
      } else if (measurementMode) {
        const clickPoint = [event.point.x, event.point.y, event.point.z];
        setMeasurePoints((prev) =>
          prev[0] ? [prev[0], clickPoint] : [clickPoint, null]
        );
      }
    },
    [placementMode, handlePlaceFurniture, measurementMode]
  );

  const handleMoveFurniture = useCallback((id, newPosition) => {
    setFurniture((prev) => {
      const updated = prev.map((f) => {
        if (f.id === id) {
          return { ...f, position: newPosition };
        }
        return f;
      });
      return updated;
    });
  }, []);

  const updatePlacedFurniturePositionOnDragEnd = useCallback(
    (id, newPosition, newRotation) => {
      if (typeof onFurnitureChange === "function") {
        const furnitureItem = furniture.find((f) => f.id === id);

        if (furnitureItem) {
          const size = furnitureItem.size;
          isUpdatingFromDragRef.current = true;
          const converted2D = convertCoordinatesLocally(
            id,
            newPosition,
            size,
            roomSize
          );

          // 3D 회전을 2D 회전으로 변환 (Y축 회전만 사용)
          const rotation2D = newRotation
            ? Math.round((newRotation[1] * 180) / Math.PI)
            : 0;

          onFurnitureChange((prev) => {
            const updated = prev.map((item) => {
              if (item.id === id) {
                const newItem = {
                  ...item,
                  x: converted2D.x,
                  z: converted2D.z,
                  rotation: rotation2D,
                };
                console.log("3D → 2D 업데이트:", {
                  id,
                  "3D position": newPosition,
                  "2D position": { x: converted2D.x, z: converted2D.z },
                  "3D rotation": newRotation,
                  "2D rotation": rotation2D,
                });
                return newItem;
              }
              return item;
            });
            return updated;
          });

          setTimeout(() => {
            isUpdatingFromDragRef.current = false;
          }, 100);
        }
      }
    },
    [furniture, onFurnitureChange, convertCoordinatesLocally, roomSize]
  );

  const handleDetectWindows = useCallback(async () => {
    if (!uploadedImageFile) {
      return;
    }
    setIsDetectingWindows(true);
    try {
      const roomDimensions = {
        width_cm: w,
        height_cm: h,
        depth_cm: d,
        area_sqm: (w * d) / 10000,
        wall_height_cm: h,
        scale_factor: 1,
      };

      const wallInfo = {
        front_wall: { width: w, height: h, position: "front" },
        back_wall: { width: w, height: h, position: "back" },
        left_wall: { width: d, height: h, position: "left" },
        right_wall: { width: d, height: h, position: "right" },
      };

      const result = await detectWindowsInImage(
        uploadedImageFile,
        wallInfo,
        roomDimensions
      );

      if (result.windows && result.windows.length > 0) {
        const validatedWindows = result.windows.map((window, index) => {
          const minWidth = 60;
          const maxWidth = Math.min(200, w * 0.8);
          const minHeight = 80;
          const maxHeight = Math.min(180, h * 0.8);

          let adjustedWindow = { ...window };

          if (window.width_meters) {
            const widthCm = window.width_meters * 100;
            adjustedWindow.width_meters =
              Math.max(minWidth, Math.min(maxWidth, widthCm)) / 100;
          } else {
            adjustedWindow.width_meters = 1.2;
          }

          if (window.height_meters) {
            const heightCm = window.height_meters * 100;
            adjustedWindow.height_meters =
              Math.max(minHeight, Math.min(maxHeight, heightCm)) / 100;
          } else {
            adjustedWindow.height_meters = 1.5;
          }

          if (
            !window.wall_position ||
            !["front", "back", "left", "right"].includes(window.wall_position)
          ) {
            adjustedWindow.wall_position = "back";
          }

          return adjustedWindow;
        });

        setDetectedWindows(validatedWindows);
        setShowWindows(true);
        showSuccess(`${validatedWindows.length}개의 창문을 감지했습니다`);
      } else {
        showWarning("창문을 감지하지 못했습니다");
      }
    } catch (error) {
      showError(`창문 감지 중 오류가 발생했습니다: ${error.message}`);
    } finally {
      setIsDetectingWindows(false);
    }
  }, [uploadedImageFile, w, h, d, showSuccess, showWarning, showError]);

  const handleRotateFurniture = useCallback(
    (id) => {
      setFurniture((prev) =>
        prev.map((f) => {
          if (f.id === id) {
            const newRotation = [0, f.rotation[1] + Math.PI / 2, 0];
            const updatedFurniture = { ...f, rotation: newRotation };

            // 회전 후 2D 좌표도 업데이트
            if (updatePlacedFurniturePositionOnDragEnd) {
              updatePlacedFurniturePositionOnDragEnd(
                id,
                f.position,
                newRotation
              );
            }

            return updatedFurniture;
          }
          return f;
        })
      );
    },
    [updatePlacedFurniturePositionOnDragEnd]
  );

  const handleDeleteFurniture = useCallback(
    (id) => {
      setFurniture((prev) => prev.filter((f) => f.id !== id));
      if (selectedFurniture === id) {
        setSelectedFurniture(null);
      }

      if (typeof onFurnitureChange === "function") {
        onFurnitureChange((prev) => prev.filter((item) => item.id !== id));
      }
    },
    [selectedFurniture, onFurnitureChange]
  );

  const handleAddFurniture = useCallback((type) => {
    setPlacementMode(type);
  }, []);

  const selectedFurnitureData = useMemo(
    () => furniture.find((f) => f.id === selectedFurniture),
    [furniture, selectedFurniture]
  );

  useEffect(() => {
    const handleKeyPress = (event) => {
      if (
        event.target.tagName === "INPUT" ||
        event.target.tagName === "TEXTAREA" ||
        event.target.contentEditable === "true"
      ) {
        return;
      }

      const key = event.key.toLowerCase();

      if ((key === "delete" || key === "backspace") && selectedFurniture) {
        event.preventDefault();
        handleDeleteFurniture(selectedFurniture);
      }

      if (key === "r" && selectedFurniture) {
        event.preventDefault();
        handleRotateFurniture(selectedFurniture);
      }

      if (key === "escape") {
        event.preventDefault();
        if (placementMode) {
          setPlacementMode(null);
        } else if (selectedFurniture) {
          setSelectedFurniture(null);
        }
      }

      if (key === "m") {
        event.preventDefault();
        setMeasurementMode(!measurementMode);
        setMeasurePoints([null, null]);
      }

      if (key === "g") {
        event.preventDefault();
        setShowFloorGrid(!showFloorGrid);
      }

      if (key === "s") {
        event.preventDefault();
        setEnableSnap(!enableSnap);
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [
    selectedFurniture,
    handleDeleteFurniture,
    handleRotateFurniture,
    placementMode,
    setPlacementMode,
    setSelectedFurniture,
    measurementMode,
    setMeasurementMode,
    setMeasurePoints,
    showFloorGrid,
    setShowFloorGrid,
    enableSnap,
    setEnableSnap,
  ]);

  return (
    <div
      className={`room-3d-viewer relative w-full bg-background overflow-hidden shadow-lg transition-all duration-300 ${
        isFullscreen
          ? "h-screen rounded-none"
          : "h-[700px] rounded-xl border border-border"
      }`}
    >
      {/* 키보드 단축키 도움말 */}
      <div className="absolute top-4 right-4 z-20">
        <div className="relative group">
          <button
            className="p-2 rounded-lg backdrop-blur-sm bg-surface/20 hover:bg-surface/30 transition-all duration-200 shadow-lg"
            title="키보드 단축키 도움말"
            aria-label="키보드 단축키 도움말"
          >
            <svg
              className="w-5 h-5 text-text-secondary"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </button>

          <div className="absolute top-full right-0 mt-2 w-64 p-4 bg-surface/95 backdrop-blur-lg rounded-xl shadow-xl border border-border/50 opacity-0 group-hover:opacity-100 transition-all duration-200 pointer-events-none group-hover:pointer-events-auto z-50">
            <h4 className="font-bold text-sm text-text-primary mb-3">
              키보드 단축키
            </h4>
            <div className="space-y-2 text-xs text-text-secondary">
              <div className="flex justify-between">
                <span className="font-mono bg-background px-2 py-1 rounded">
                  Del/Backspace
                </span>
                <span>가구 삭제</span>
              </div>
              <div className="flex justify-between">
                <span className="font-mono bg-slate-100 px-2 py-1 rounded">
                  R
                </span>
                <span>가구 회전</span>
              </div>
              <div className="flex justify-between">
                <span className="font-mono bg-slate-100 px-2 py-1 rounded">
                  Esc
                </span>
                <span>취소/선택 해제</span>
              </div>
              <div className="flex justify-between">
                <span className="font-mono bg-slate-100 px-2 py-1 rounded">
                  M
                </span>
                <span>거리 측정</span>
              </div>
              <div className="flex justify-between">
                <span className="font-mono bg-slate-100 px-2 py-1 rounded">
                  G
                </span>
                <span>그리드 토글</span>
              </div>
              <div className="flex justify-between">
                <span className="font-mono bg-slate-100 px-2 py-1 rounded">
                  S
                </span>
                <span>스냅 토글</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 왼쪽 UI 패널들 - 주요 기능 */}
      <div className="absolute top-4 left-4 z-10 space-y-4 w-80 max-w-sm">
        {/* 가구 추가 패널 */}
        <div className="backdrop-blur-lg p-4 rounded-xl shadow-xl bg-surface/90 border border-border/50 hover:bg-surface/95 transition-all duration-200">
          <div className="flex items-center gap-2 mb-3">
            <div className="p-1.5 rounded-lg bg-primary/10">
              <svg
                className="w-4 h-4 text-primary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
            </div>
            <h3 className="text-base font-bold text-text-primary">
              {placementMode
                ? `${FURNITURE_PRESETS[placementMode].name} 배치 중`
                : "가구 추가"}
            </h3>
          </div>
          {placementMode ? (
            <button
              onClick={() => setPlacementMode(null)}
              className="w-full px-4 py-2.5 bg-danger text-white rounded-lg text-sm font-medium hover:bg-danger-dark transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-[1.02]"
            >
              <span className="flex items-center justify-center gap-2">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
                배치 취소
              </span>
            </button>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(FURNITURE_PRESETS).map(([type, preset]) => (
                <button
                  key={type}
                  onClick={() => handleAddFurniture(type)}
                  className="group flex items-center gap-2 px-3 py-2.5 bg-surface text-text-secondary border border-border rounded-lg hover:bg-background transition-all duration-200 text-sm font-medium shadow-sm hover:shadow-md transform hover:scale-[1.02]"
                >
                  <FontAwesomeIcon
                    icon={iconMapping[preset.icon]}
                    className="text-lg group-hover:scale-110 transition-transform duration-200 text-text-secondary"
                  />
                  <span className="text-text-secondary group-hover:text-primary">
                    {preset.name}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* 거리 측정 패널 */}
        <div className="backdrop-blur-lg p-4 rounded-xl shadow-lg bg-surface/85 border border-border/50 hover:bg-surface/90 transition-all duration-200">
          <div className="flex items-center gap-2 mb-3">
            <div className="p-1.5 rounded-lg bg-primary/10">
              <svg
                className="w-4 h-4 text-primary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 21l3-3 9.5-9.5a2.12 2.12 0 000-3l-1-1a2.12 2.12 0 00-3 0L6 14l-3 3v4h4z"
                />
              </svg>
            </div>
            <h4 className="font-bold text-sm text-text-primary">거리 측정</h4>
          </div>
          <div className="space-y-3">
            <button
              onClick={() => {
                setMeasurementMode(!measurementMode);
                setMeasurePoints([null, null]);
              }}
              className={`w-full px-4 py-3 rounded-lg text-sm font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-[1.02] ${
                measurementMode
                  ? "bg-accent text-white hover:bg-primary"
                  : "bg-white text-primary border border-primary hover:bg-background"
              }`}
            >
              <span className="flex items-center justify-center gap-2">
                {measurementMode ? (
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                ) : (
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 21l3-3 9.5-9.5a2.12 2.12 0 000-3l-1-1a2.12 2.12 0 00-3 0L6 14l-3 3v4h4z"
                    />
                  </svg>
                )}
                {measurementMode ? "측정 종료" : "거리 측정"}
              </span>
            </button>
            {measurementMode && (
              <div className="p-3 bg-background rounded-lg border border-border">
                <p className="text-xs text-text-secondary font-medium">
                  측정 방법:
                </p>
                <p className="text-xs text-text-secondary mt-1">
                  바닥을 클릭하여 두 점 사이의 거리를 측정하세요
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 왼쪽 하단 - 벽면 스타일 패널 */}
      <div className="absolute bottom-4 left-4 z-10 w-80 max-w-sm space-y-4">
        {/* 벽면 스타일 패널 */}
        <div className="backdrop-blur-lg p-4 rounded-xl shadow-lg bg-surface/85 border border-border/50 hover:bg-surface/90 transition-all duration-200">
          <div className="flex items-center gap-2 mb-3">
            <div className="p-1.5 rounded-lg bg-primary/10">
              <svg
                className="w-4 h-4 text-primary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 21a4.5 4.5 0 01-4.5-4.5V5a2 2 0 012-2h14L21 10v6.5a4.5 4.5 0 01-4.5 4.5"
                />
              </svg>
            </div>
            <h4 className="font-bold text-sm text-text-primary">벽면 스타일</h4>
          </div>
          <div className="space-y-3">
            {/* 색상 프리셋 선택 */}
            <div>
              <p className="text-xs text-text-secondary mb-2 font-medium">
                벽 색상
              </p>
              <div className="grid grid-cols-5 gap-1">
                {Object.entries(wallPresets).map(([key, preset]) => (
                  <button
                    key={key}
                    onClick={() =>
                      setWallSettings({
                        ...wallSettings,
                        color: preset.color,
                        roughness: preset.roughness,
                        metalness: preset.metalness,
                      })
                    }
                    className={`w-8 h-8 rounded-lg border-2 transition-all duration-200 hover:scale-110 ${
                      wallSettings.color === preset.color
                        ? "border-primary shadow-lg"
                        : "border-white/30 hover:border-white/60"
                    }`}
                    style={{ backgroundColor: preset.color }}
                    title={preset.name}
                  />
                ))}
              </div>
            </div>

            {/* 커스텀 색상 선택 */}
            <div>
              <p className="text-xs text-text-secondary mb-2 font-medium">
                커스텀 색상
              </p>
              <input
                type="color"
                value={wallSettings.color}
                onChange={(e) =>
                  setWallSettings({ ...wallSettings, color: e.target.value })
                }
                className="w-full h-8 rounded-lg border border-border bg-background cursor-pointer"
              />
            </div>

            {/* 재질 조절 */}
            <div>
              <p className="text-xs text-text-secondary mb-2 font-medium">
                거칠기 ({wallSettings.roughness.toFixed(1)})
              </p>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={wallSettings.roughness}
                onChange={(e) =>
                  setWallSettings({
                    ...wallSettings,
                    roughness: parseFloat(e.target.value),
                  })
                }
                className="w-full h-2 bg-border rounded-lg appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${
                    wallSettings.roughness * 100
                  }%, #e5e7eb ${wallSettings.roughness * 100}%, #e5e7eb 100%)`,
                }}
              />
              <div className="flex justify-between text-xs text-text-secondary mt-1">
                <span>매끄러움</span>
                <span>거침</span>
              </div>
            </div>

            {/* 금속성 조절 */}
            <div>
              <p className="text-xs text-text-secondary mb-2 font-medium">
                금속성 ({wallSettings.metalness.toFixed(2)})
              </p>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.01"
                value={wallSettings.metalness}
                onChange={(e) =>
                  setWallSettings({
                    ...wallSettings,
                    metalness: parseFloat(e.target.value),
                  })
                }
                className="w-full h-2 bg-border rounded-lg appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${
                    wallSettings.metalness * 200
                  }%, #e5e7eb ${wallSettings.metalness * 200}%, #e5e7eb 100%)`,
                }}
              />
              <div className="flex justify-between text-xs text-text-secondary mt-1">
                <span>매트</span>
                <span>금속</span>
              </div>
            </div>
          </div>
        </div>

        {/* 바닥 스타일 패널 */}
        <div className="backdrop-blur-lg p-4 rounded-xl shadow-lg bg-surface/85 border border-border/50 hover:bg-surface/90 transition-all duration-200">
          <div className="flex items-center gap-2 mb-3">
            <div className="p-1.5 rounded-lg bg-primary/10">
              <svg
                className="w-4 h-4 text-primary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"
                />
              </svg>
            </div>
            <h4 className="font-bold text-sm text-text-primary">바닥 스타일</h4>
          </div>
          <div className="space-y-3">
            {/* 바닥 색상 프리셋 */}
            <div>
              <p className="text-xs text-text-secondary mb-2 font-medium">
                바닥 재질
              </p>
              <div className="grid grid-cols-5 gap-1">
                {Object.entries(floorPresets).map(([key, preset]) => (
                  <button
                    key={key}
                    onClick={() =>
                      setFloorSettings({
                        color: preset.color,
                        roughness: preset.roughness,
                        metalness: preset.metalness,
                      })
                    }
                    className={`w-8 h-8 rounded-lg border-2 transition-all duration-200 hover:scale-110 ${
                      floorSettings.color === preset.color
                        ? "border-primary shadow-lg"
                        : "border-white/30 hover:border-white/60"
                    }`}
                    style={{ backgroundColor: preset.color }}
                    title={preset.name}
                  />
                ))}
              </div>
            </div>

            {/* 커스텀 바닥 색상 */}
            <div>
              <p className="text-xs text-text-secondary mb-2 font-medium">
                커스텀 색상
              </p>
              <input
                type="color"
                value={floorSettings.color}
                onChange={(e) =>
                  setFloorSettings({ ...floorSettings, color: e.target.value })
                }
                className="w-full h-8 rounded-lg border border-border bg-background cursor-pointer"
              />
            </div>

            {/* 바닥 거칠기 */}
            <div>
              <p className="text-xs text-text-secondary mb-2 font-medium">
                바닥 거칠기 ({floorSettings.roughness.toFixed(1)})
              </p>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={floorSettings.roughness}
                onChange={(e) =>
                  setFloorSettings({
                    ...floorSettings,
                    roughness: parseFloat(e.target.value),
                  })
                }
                className="w-full h-2 bg-border rounded-lg appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${
                    floorSettings.roughness * 100
                  }%, #e5e7eb ${floorSettings.roughness * 100}%, #e5e7eb 100%)`,
                }}
              />
              <div className="flex justify-between text-xs text-text-secondary mt-1">
                <span>매끄러움</span>
                <span>거침</span>
              </div>
            </div>
          </div>
        </div>

        {/* 시각 옵션 패널 */}
        <div className="backdrop-blur-lg p-4 rounded-xl shadow-lg bg-surface/85 border border-border/50 hover:bg-surface/90 transition-all duration-200">
          <div className="flex items-center gap-2 mb-3">
            <div className="p-1.5 rounded-lg bg-primary/10">
              <svg
                className="w-4 h-4 text-primary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
            </div>
            <h4 className="font-bold text-sm text-text-primary">시각 옵션</h4>
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <label className="flex items-center gap-2 group cursor-pointer">
              <input
                type="checkbox"
                checked={enableSnap}
                onChange={(e) => setEnableSnap(e.target.checked)}
                className="w-3 h-3 text-primary bg-background border-border rounded focus:ring-primary focus:ring-1"
              />
              <span className="text-text-primary group-hover:text-primary transition-colors duration-200">
                스냅 기능
              </span>
            </label>
            <label className="flex items-center gap-2 group cursor-pointer">
              <input
                type="checkbox"
                checked={showSnapGrid}
                onChange={(e) => setShowSnapGrid(e.target.checked)}
                className="w-3 h-3 text-primary bg-background border-border rounded focus:ring-primary focus:ring-1"
              />
              <span className="text-text-primary group-hover:text-primary transition-colors duration-200">
                스냅 그리드
              </span>
            </label>
            <label className="flex items-center gap-2 group cursor-pointer">
              <input
                type="checkbox"
                checked={showFloorGrid}
                onChange={(e) => setShowFloorGrid(e.target.checked)}
                className="w-3 h-3 text-primary bg-background border-border rounded focus:ring-primary focus:ring-1"
              />
              <span className="text-text-primary group-hover:text-primary transition-colors duration-200">
                바닥 그리드
              </span>
            </label>
            <label className="flex items-center gap-2 group cursor-pointer">
              <input
                type="checkbox"
                checked={showCollisions}
                onChange={(e) => setShowCollisions(e.target.checked)}
                className="w-3 h-3 text-primary bg-background border-border rounded focus:ring-primary focus:ring-1"
              />
              <span className="text-text-primary group-hover:text-primary transition-colors duration-200">
                충돌 표시
              </span>
            </label>
            <label className="flex items-center gap-2 group cursor-pointer">
              <input
                type="checkbox"
                checked={showWindows}
                onChange={(e) => setShowWindows(e.target.checked)}
                className="w-3 h-3 text-primary bg-background border-border rounded focus:ring-primary focus:ring-1"
              />
              <span className="text-text-primary group-hover:text-primary transition-colors duration-200">
                창문 표시
              </span>
            </label>
            <label className="flex items-center gap-2 group cursor-pointer">
              <input
                type="checkbox"
                checked={use3DModels}
                onChange={(e) => setUse3DModels(e.target.checked)}
                className="w-3 h-3 text-primary bg-background border-border rounded focus:ring-primary focus:ring-1"
              />
              <span className="text-text-primary group-hover:text-primary transition-colors duration-200">
                3D 모델
              </span>
            </label>
            <label className="flex items-center gap-2 group cursor-pointer">
              <input
                type="checkbox"
                checked={showHuman}
                onChange={(e) => setShowHuman(e.target.checked)}
                className="w-3 h-3 text-primary bg-background border-border rounded focus:ring-primary focus:ring-1"
              />
              <span className="text-text-primary group-hover:text-primary transition-colors duration-200">
                사람 표시
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* 오른쪽 UI 패널들 */}
      <div className="absolute top-4 right-4 z-10 space-y-4 w-80 max-w-sm">
        {/* 데이터 저장 패널 */}
        <div className="backdrop-blur-lg p-4 rounded-xl shadow-lg bg-surface/85 border border-border/50 hover:bg-surface/90 transition-all duration-200">
          <div className="flex items-center gap-2 mb-3">
            <div className="p-1.5 rounded-lg bg-primary/10">
              <svg
                className="w-4 h-4 text-primary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <h4 className="font-bold text-sm text-text-primary">데이터 저장</h4>
          </div>
          <div className="space-y-3">
            <LoadingButton
              onClick={async () => {
                if (furniture.length === 0 && detectedWindows.length === 0) {
                  return;
                }

                try {
                  setIsSaving(true);
                  const saveData = createRoomLayoutData(
                    w,
                    d,
                    h,
                    furniture,
                    detectedWindows
                  );
                  await saveRoomLayoutToMongoDB(saveData);
                  showSuccess(
                    `저장 완료! 가구 ${furniture.length}개, 창문 ${detectedWindows.length}개`
                  );
                } catch (error) {
                  showError(`MongoDB 저장 실패: ${error.message}`);
                } finally {
                  setIsSaving(false);
                }
              }}
              loading={isSaving}
              loadingText="저장 중..."
              disabled={furniture.length === 0 && detectedWindows.length === 0}
              className="w-full px-4 py-3 bg-white text-primary border border-primary rounded-lg text-sm font-semibold hover:bg-background disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-200"
            >
              <span className="flex items-center justify-center gap-2">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
                Save Layout
              </span>
            </LoadingButton>
            <div className="flex items-center justify-center gap-3 text-xs text-text-secondary bg-background px-3 py-2 rounded-lg">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-primary rounded-full"></div>
                <span>가구 {furniture.length}개</span>
              </div>
              <div className="w-px h-3 bg-border"></div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-primary rounded-full"></div>
                <span>창문 {detectedWindows.length}개</span>
              </div>
            </div>
          </div>
        </div>

        {/* 창문 감지 패널 */}
        {uploadedImageFile && (
          <div className="backdrop-blur-lg p-4 rounded-xl shadow-lg bg-surface/85 border border-border/50 hover:bg-surface/90 transition-all duration-200">
            <div className="flex items-center gap-2 mb-3">
              <div className="p-1.5 rounded-lg bg-primary/10">
                <svg
                  className="w-4 h-4 text-primary"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z"
                  />
                </svg>
              </div>
              <h4 className="font-bold text-sm text-text-primary">창문 감지</h4>
            </div>
            <div className="space-y-3">
              <LoadingButton
                onClick={handleDetectWindows}
                loading={isDetectingWindows}
                loadingText="창문 감지 중..."
                className="w-full px-4 py-3 bg-white text-primary border border-primary rounded-lg text-sm font-semibold hover:bg-background disabled:opacity-50 shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-200"
              >
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                  AI 창문 감지
                </span>
              </LoadingButton>
            </div>
          </div>
        )}

        {/* 창문 조정 패널 */}
        {detectedWindows.length > 0 && (
          <div className="backdrop-blur-lg p-4 rounded-xl shadow-lg bg-surface/85 border border-border/50 hover:bg-surface/90 transition-all duration-200">
            <div className="flex items-center gap-2 mb-3">
              <div className="p-1.5 rounded-lg bg-primary/10">
                <svg
                  className="w-4 h-4 text-primary"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4"
                  />
                </svg>
              </div>
              <h4 className="font-bold text-sm text-text-primary">창문 조정</h4>
            </div>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {detectedWindows.map((window, index) => (
                <div
                  key={index}
                  className="bg-background p-3 rounded-lg space-y-2"
                >
                  <div className="flex justify-between items-center">
                    <div className="text-xs font-medium text-text-secondary">
                      창문 {index + 1}:{" "}
                      {Math.round((window.width_meters || 1.2) * 100)}×
                      {Math.round((window.height_meters || 1.5) * 100)}cm (
                      {window.wall_position}벽)
                    </div>
                    <button
                      onClick={() => {
                        const newWindows = detectedWindows.filter(
                          (_, i) => i !== index
                        );
                        setDetectedWindows(newWindows);
                      }}
                      className="text-danger hover:text-danger-dark p-1"
                      title="창문 삭제"
                    >
                      <svg
                        className="w-3 h-3"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                      </svg>
                    </button>
                  </div>

                  {/* 수평 위치 조정 */}
                  <div className="flex gap-1 items-center">
                    <span className="text-xs w-12 text-text-secondary">
                      수평:
                    </span>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={window.x_position || 0.5}
                      onChange={(e) => {
                        const newWindows = [...detectedWindows];
                        newWindows[index] = {
                          ...window,
                          x_position: parseFloat(e.target.value),
                        };
                        setDetectedWindows(newWindows);
                      }}
                      className="flex-1 h-1"
                    />
                    <span className="text-xs w-8">
                      {Math.round((window.x_position || 0.5) * 100)}%
                    </span>
                  </div>

                  {/* 수직 위치 조정 */}
                  <div className="flex gap-1 items-center">
                    <span className="text-xs w-12 text-text-secondary">
                      수직:
                    </span>
                    <input
                      type="range"
                      min="0.3"
                      max="1.0"
                      step="0.05"
                      value={window.y_position || 0.8}
                      onChange={(e) => {
                        const newWindows = [...detectedWindows];
                        newWindows[index] = {
                          ...window,
                          y_position: parseFloat(e.target.value),
                        };
                        setDetectedWindows(newWindows);
                      }}
                      className="flex-1 h-1"
                    />
                    <span className="text-xs w-8">
                      {Math.round((window.y_position || 0.8) * 100)}%
                    </span>
                  </div>

                  {/* 창문 너비 조정 */}
                  <div className="flex gap-1 items-center">
                    <span className="text-xs w-12 text-text-secondary">
                      너비:
                    </span>
                    <input
                      type="range"
                      min="60"
                      max="400"
                      value={Math.round((window.width_meters || 1.2) * 100)}
                      onChange={(e) => {
                        const newWindows = [...detectedWindows];
                        newWindows[index] = {
                          ...window,
                          width_meters: parseInt(e.target.value) / 100,
                        };
                        setDetectedWindows(newWindows);
                      }}
                      className="flex-1 h-1"
                    />
                    <span className="text-xs w-12">
                      {Math.round((window.width_meters || 1.2) * 100)}cm
                    </span>
                  </div>

                  {/* 창문 높이 조정 */}
                  <div className="flex gap-1 items-center">
                    <span className="text-xs w-12 text-text-secondary">
                      높이:
                    </span>
                    <input
                      type="range"
                      min="60"
                      max="300"
                      value={Math.round((window.height_meters || 1.5) * 100)}
                      onChange={(e) => {
                        const newWindows = [...detectedWindows];
                        newWindows[index] = {
                          ...window,
                          height_meters: parseInt(e.target.value) / 100,
                        };
                        setDetectedWindows(newWindows);
                      }}
                      className="flex-1 h-1"
                    />
                    <span className="text-xs w-12">
                      {Math.round((window.height_meters || 1.5) * 100)}cm
                    </span>
                  </div>
                </div>
              ))}

              {/* 새 창문 추가 버튼 */}
              <button
                onClick={() => {
                  const newWindow = {
                    wall_position: "back",
                    x_position: 0.5,
                    y_position: 0.8,
                    width_meters: 1.2,
                    height_meters: 1.5,
                    confidence: 1.0,
                  };
                  setDetectedWindows([...detectedWindows, newWindow]);
                }}
                className="w-full px-3 py-2 bg-white text-primary border border-primary rounded-lg text-xs font-medium hover:bg-background transition-all duration-200 shadow-md hover:shadow-lg transform hover:scale-[1.02]"
              >
                <span className="flex items-center justify-center gap-1">
                  <svg
                    className="w-3 h-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                    />
                  </svg>
                  새 창문 추가
                </span>
              </button>
            </div>
          </div>
        )}

        {/* 뷰 컨트롤 */}
        <ViewPresets onViewChange={handleViewChange} roomSize={roomSize} />

        {/* 공간 분석 */}
        <SpaceUtilization
          furniture={furniture}
          roomArea={roomArea}
          furniturePresets={FURNITURE_PRESETS}
        />
      </div>

      {/* 3D Canvas */}
      <div className="relative w-full h-full">
        <Canvas
          camera={{
            position: [w, h, d],
            fov: 50,
            near: 1,
            far: Math.max(w, h, d) * 5,
          }}
          shadows="soft"
          gl={{
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true,
            powerPreference: "high-performance",
            failIfMajorPerformanceCaveat: false,
            toneMapping: THREE.ACESFilmicToneMapping,
            toneMappingExposure: 1.2,
            outputColorSpace: THREE.SRGBColorSpace,
            shadowMapType: THREE.PCFSoftShadowMap,
            physicallyCorrectLights: true,
          }}
        >
          <Suspense fallback={null}>
            <EnhancedLighting roomSize={roomSize} />
            <Environment
              preset="studio"
              background={false}
              environmentIntensity={0.8}
              environmentRotation={[0, Math.PI / 4, 0]}
            />

            {/* 바닥 */}
            <mesh
              position={[w / 2, 0, d / 2]}
              rotation={[-Math.PI / 2, 0, 0]}
              receiveShadow
              onClick={handleFloorClick}
            >
              <planeGeometry args={[w, d]} />
              <meshPhysicalMaterial
                color={placementMode ? "#E1F5FE" : floorSettings.color}
                roughness={floorSettings.roughness}
                metalness={floorSettings.metalness}
                clearcoat={floorSettings.metalness > 0.1 ? 0.8 : 0.15}
                clearcoatRoughness={floorSettings.roughness}
                normalScale={[1.0, 1.0]}
                envMapIntensity={1.0}
                reflectivity={floorSettings.metalness > 0.1 ? 0.9 : 0.1}
                transmission={0}
                thickness={0.1}
              />
            </mesh>

            {/* 왼쪽 벽 */}
            <Wall
              width={d}
              height={h}
              position={[0, h / 2, d / 2]}
              rotation={[0, Math.PI / 2, 0]}
              color={wallSettings.color}
              roughness={wallSettings.roughness}
              metalness={wallSettings.metalness}
            />

            {/* 뒤쪽 벽 */}
            <Wall
              width={w}
              height={h}
              position={[w / 2, h / 2, 0]}
              rotation={[0, 0, 0]}
              color={wallSettings.color}
              roughness={wallSettings.roughness}
              metalness={wallSettings.metalness}
            />

            <SnapGrid roomSize={roomSize} visible={showSnapGrid} />
            <FloorGrid roomSize={roomSize} visible={showFloorGrid} />

            {placementMode && (
              <ValidPlacementArea
                roomSize={roomSize}
                furniture={furniture}
                furniturePresets={FURNITURE_PRESETS}
                selectedFurnitureSize={FURNITURE_PRESETS[placementMode].size}
              />
            )}

            {furniture.map((f) => (
              <DraggableFurnitureWithCollision
                key={f.id}
                {...f}
                onMove={handleMoveFurniture}
                onSelect={setSelectedFurniture}
                selected={selectedFurniture === f.id}
                furniture={furniture}
                furniturePresets={FURNITURE_PRESETS}
                roomSize={roomSize}
                enableSnap={enableSnap}
                showCollisions={showCollisions}
                onDragStateChange={setIsDragging}
                customFurnitureData={f.isCustom ? f : null}
                updatePlacedFurniturePosition={
                  updatePlacedFurniturePositionOnDragEnd
                }
                use3DModels={use3DModels}
              />
            ))}

            {showWindows && detectedWindows.length > 0 && (
              <WindowsOnWalls windows={detectedWindows} roomSize={roomSize} />
            )}

            {showHuman && (
              <DraggableHuman
                height={170}
                position={humanPosition}
                onPositionChange={setHumanPosition}
                roomSize={roomSize}
                onDragStateChange={setIsDragging}
              />
            )}

            <DistanceMeasurer
              point1={measurePoints[0]}
              point2={measurePoints[1]}
              visible={measurementMode && !!measurePoints[0]}
            />

            <group position={[0, 0.1, 0]}>
              <DimensionArrow start={[0, 0, d + 10]} end={[w, 0, d + 10]} />
              <DimensionLabel
                position={[w / 2, 0, d + 15]}
                text={`${(w / 100).toFixed(1)}m`}
              />
              <DimensionArrow start={[w + 10, 0, 0]} end={[w + 10, 0, d]} />
              <DimensionLabel
                position={[w + 15, 0, d / 2]}
                text={`${(d / 100).toFixed(1)}m`}
                rotation={[0, Math.PI / 2, 0]}
              />
              <DimensionArrow start={[w + 10, 0, 0]} end={[w + 10, h, 0]} />
              <DimensionLabel
                position={[w + 15, h / 2, 0]}
                text={`${(h / 100).toFixed(1)}m`}
              />
            </group>

            <OrbitControls
              ref={controlsRef}
              enablePan={!isDragging}
              enableZoom={!isDragging}
              enableRotate={!isDragging}
              minDistance={50}
              maxDistance={Math.max(w, h, d) * 2}
              maxPolarAngle={Math.PI / 2.1}
              target={[w / 2, h / 3, d / 2]}
              enableDamping
              dampingFactor={0.1}
            />
          </Suspense>
        </Canvas>
      </div>

      {/* 선택된 가구 정보 */}
      {selectedFurnitureData && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-10 backdrop-blur-lg p-4 rounded-xl shadow-xl bg-white/90 border border-white/50 max-w-sm">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 rounded-lg bg-window-fill">
              <svg
                className="w-5 h-5 text-primary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                />
              </svg>
            </div>
            <div>
              <p className="text-base font-bold text-slate-800">
                {selectedFurnitureData.name}
              </p>
              <p className="text-xs text-slate-500">선택된 가구</p>
            </div>
          </div>
          {(() => {
            const [x3d, y3d, z3d] = selectedFurnitureData.position;
            const size = selectedFurnitureData.size || [100, 100, 100];
            const [width, height, depth] = size;

            const leftBottomX = Math.round(x3d - width / 2);
            const leftBottomZ = Math.round(z3d - depth / 2);
            const rightTopX = Math.round(x3d + width / 2);
            const rightTopZ = Math.round(z3d + depth / 2);

            return (
              <div className="text-xs text-slate-600 mb-3 space-y-2 bg-slate-50 p-3 rounded-lg border">
                <div className="flex items-center gap-2">
                  <svg
                    className="w-3 h-3 text-slate-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
                    />
                  </svg>
                  <span className="font-medium">크기:</span> {width} × {depth} ×{" "}
                  {height} cm
                </div>
                <div className="flex items-center gap-2">
                  <svg
                    className="w-3 h-3 text-slate-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                  </svg>
                  <span className="font-medium">왼쪽아래:</span> ({leftBottomX},{" "}
                  {leftBottomZ}) cm
                </div>
                <div className="flex items-center gap-2">
                  <svg
                    className="w-3 h-3 text-slate-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                  </svg>
                  <span className="font-medium">오른쪽위:</span> ({rightTopX},{" "}
                  {rightTopZ}) cm
                </div>
                <div className="flex items-center gap-2">
                  <svg
                    className="w-3 h-3 text-slate-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 3v1m0 12v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
                    />
                  </svg>
                  <span className="font-medium">중심 좌표:</span> (
                  {Math.round(x3d)}, {Math.round(z3d)}) cm
                </div>
                {selectedFurnitureData.rotation &&
                  selectedFurnitureData.rotation[1] !== 0 && (
                    <div className="flex items-center gap-2">
                      <svg
                        className="w-3 h-3 text-slate-500"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                        />
                      </svg>
                      <span className="font-medium">회전:</span>{" "}
                      {Math.round(
                        (selectedFurnitureData.rotation[1] * 180) / Math.PI
                      )}
                      °
                    </div>
                  )}
              </div>
            );
          })()}
          <div className="flex gap-2">
            <button
              onClick={() => handleRotateFurniture(selectedFurniture)}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2.5 bg-white text-primary border border-primary rounded-lg hover:bg-window-fill text-sm font-medium transition-all duration-200 shadow-md hover:shadow-lg transform hover:scale-[1.02]"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              회전
            </button>
            <button
              onClick={() => handleDeleteFurniture(selectedFurniture)}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2.5 bg-white text-red-600 border border-red-600 rounded-lg hover:bg-red-50 text-sm font-medium transition-all duration-200 shadow-md hover:shadow-lg transform hover:scale-[1.02]"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
              삭제
            </button>
          </div>
        </div>
      )}

      {/* 배치 가이드 */}
      {placementMode && (
        <div className="absolute inset-0 z-30 pointer-events-none">
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="animate-pulse bg-accent/20 rounded-full p-8">
              <div className="bg-primary/30 rounded-full p-6">
                <div className="bg-secondary/40 rounded-full p-4 flex items-center justify-center">
                  <svg
                    className="w-8 h-8 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                    />
                  </svg>
                </div>
              </div>
            </div>
          </div>
          <div className="absolute top-4 left-1/2 transform -translate-x-1/2">
            <div className="bg-primary text-white px-4 py-2 rounded-lg shadow-lg animate-bounce">
              바닥을 클릭하여 {FURNITURE_PRESETS[placementMode]?.name}를
              배치하세요
            </div>
          </div>
        </div>
      )}

      <PlacementGuide
        placementMode={placementMode}
        onClose={() => setPlacementMode(null)}
      />

      {/* Toast 알림 */}
      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {/* AI 인테리어 생성 및 3D 캡처 버튼들 - 우측 하단 */}
      <div className="absolute bottom-4 right-4 z-20 space-y-3">
        {/* 3D 화면 캡처 버튼 */}
        <button
          onClick={handle3DCapture}
          disabled={furniture.length === 0}
          className="group flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="p-2 bg-white/20 rounded-lg group-hover:bg-white/30 transition-colors">
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
              />
            </svg>
          </div>
          <div className="text-left">
            <div className="font-bold text-sm">3D 화면 캡처</div>
            <div className="text-xs text-white/80">가구 스타일 변경용</div>
          </div>
        </button>

        {/* 캡처된 이미지 미리보기 */}
        {capturedScreenshot && (
          <div className="bg-white rounded-xl shadow-lg p-3 max-w-[200px]">
            <div className="text-xs font-medium text-gray-600 mb-2">
              📸 캡처된 이미지
            </div>
            <img
              src={capturedScreenshot.imageData}
              alt="캡처된 3D 화면"
              className="w-full h-auto rounded-lg border border-gray-200"
            />
            <div className="text-xs text-gray-500 mt-2">
              {capturedScreenshot.selectedFurniture?.name || "가구 미선택"}
              <br />
              <span className="text-green-600">저장됨 ✓</span>
            </div>
          </div>
        )}

        {/* AI 인테리어 생성 버튼 */}
        <button
          onClick={handleAIInteriorGenerate}
          className="group flex items-center gap-3 px-6 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105"
        >
          <div className="p-2 bg-white/20 rounded-lg group-hover:bg-white/30 transition-colors">
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
          </div>
          <div className="text-left">
            <div className="font-bold text-lg">AI 인테리어 생성</div>
            <div className="text-xs text-white/80">
              방 크기 기반 맞춤 디자인
            </div>
          </div>
          <svg
            className="w-5 h-5 group-hover:translate-x-1 transition-transform"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
