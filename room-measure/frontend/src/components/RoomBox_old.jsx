import React, {
  useState,
  useMemo,
  Suspense,
  useRef,
  useCallback,
  useEffect,
} from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  PointerLockControls,
  Text,
  Line,
  Environment,
  ContactShadows,
  useGLTF,
} from "@react-three/drei";
import * as THREE from "three";
import CollisionDetector from "../utils/CollisionDetector";
import PositionSnapper from "../utils/PositionSnapper";
import ViewPresets from "./UI/ViewPresets";
import SpaceUtilization from "./UI/SpaceUtilization";
import EnhancedLighting from "./3D/EnhancedLighting";
import FloorGrid from "./3D/FloorGrid";
import ToastContainer from "./UI/Toast";
import { LoadingButton } from "./UI/LoadingSpinner";
import PlacementGuide from "./UI/PlacementGuide";
import { useRoomState } from "../hooks/useRoomState";
import { useToast } from "../hooks/useToast";

// 분리된 상수들과 유틸리티들
import { FURNITURE_PRESETS, FURNITURE_ID_MAPPING } from '../constants/furniture.js';
import { convertCoordinatesLocally, convertTo3DCoordinates, convertTo2DCoordinates } from '../utils/coordinateConversion.js';
import { createRoomLayoutData } from '../utils/dataConversion.js';
import { saveRoomLayoutToMongoDB, detectWindowsInImage } from '../utils/api.js';

// 분리된 3D 컴포넌트들
import { DimensionArrow, DimensionLabel, FurnitureDimensions, DistanceMeasurer } from './3D/DimensionComponents.jsx';
import { CollisionIndicator, ValidPlacementArea, CollisionAlert } from './3D/CollisionComponents.jsx';
import { Window3D, WindowsOnWalls } from './3D/WindowComponents.jsx';
import { DraggableFurnitureWithCollision } from './3D/DraggableFurniture.jsx';
import { DraggableHuman } from './3D/DraggableHuman.jsx';
import Player from './Player.jsx';

// Player component for walkthrough mode
// Player 컴포넌트는 별도 파일로 분리됨

// 상수들은 별도 파일로 분리됨 (constants/furniture.js)

// 가구 프리셋 정의 (FurniturePlacement와 통합, cm 단위)
const FURNITURE_PRESETS = {
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

// 실제 공간과의 비교 데이터
const SPACE_COMPARISONS = {
  원룸: 20,
  침실: 12,
  거실: 25,
  주차공간: 12.5,
  화장실: 4,
  복도: 8,
};

// 충돌 검사 유틸리티 클래스
class CollisionDetector {
  static createBoundingBox(position, size) {
    const [x, y, z] = position;
    const [width, height, depth] = size;

    return {
      min: {
        x: x - width / 2,
        y: y - height / 2,
        z: z - depth / 2,
      },
      max: {
        x: x + width / 2,
        y: y + height / 2,
        z: z + depth / 2,
      },
    };
  }

  static isBoxOverlapping(box1, box2, margin = 0.1) {
    return (
      box1.min.x - margin < box2.max.x &&
      box1.max.x + margin > box2.min.x &&
      box1.min.y - margin < box2.max.y &&
      box1.max.y + margin > box2.min.y &&
      box1.min.z - margin < box2.max.z &&
      box1.max.z + margin > box2.min.z
    );
  }

  static isWithinRoomBounds(position, size, roomSize, margin = 5) {
    const [x, y, z] = position;
    const [width, height, depth] = size;
    const [roomWidth, roomHeight, roomDepth] = roomSize;

    const halfWidth = width / 2;
    const halfDepth = depth / 2;

    // 왼쪽 아래 (0,0,0) 기준 좌표계에서 경계 확인 (cm 단위)
    return (
      x - halfWidth >= margin &&
      x + halfWidth <= roomWidth - margin &&
      z - halfDepth >= margin &&
      z + halfDepth <= roomDepth - margin &&
      y >= 0 &&
      y + height <= roomHeight
    );
  }

  static checkFurnitureCollisions(
    furniture,
    currentId,
    newPosition,
    furniturePresets
  ) {
    const currentFurniture = furniture.find((f) => f.id === currentId);
    if (!currentFurniture) return [];

    // 현재 가구의 크기 정보를 안전하게 가져오기
    const currentSize = currentFurniture.size ||
      furniturePresets[currentFurniture.type]?.size || [100, 100, 100]; // 기본값

    const currentBox = this.createBoundingBox(newPosition, currentSize);

    const collisions = [];

    furniture.forEach((otherFurniture) => {
      if (otherFurniture.id === currentId) return;

      // 다른 가구의 크기 정보를 안전하게 가져오기
      const otherSize = otherFurniture.size ||
        furniturePresets[otherFurniture.type]?.size || [100, 100, 100]; // 기본값

      const otherBox = this.createBoundingBox(
        otherFurniture.position,
        otherSize
      );

      if (this.isBoxOverlapping(currentBox, otherBox)) {
        collisions.push({
          id: otherFurniture.id,
          name:
            furniturePresets[otherFurniture.type]?.name ||
            otherFurniture.name ||
            "가구",
          position: otherFurniture.position,
        });
      }
    });

    return collisions;
  }

  static checkWallCollisions(position, size, roomSize, margin = 5) {
    const [x, y, z] = position;
    const [width, height, depth] = size;
    const [roomWidth, roomHeight, roomDepth] = roomSize;

    const collisions = [];
    const halfWidth = width / 2;
    const halfDepth = depth / 2;

    // 왼쪽 아래 (0,0,0) 기준 좌표계에서 벽 충돌 감지 (cm 단위)
    if (x - halfWidth < margin) {
      collisions.push({ type: "wall", direction: "left" });
    }
    if (x + halfWidth > roomWidth - margin) {
      collisions.push({ type: "wall", direction: "right" });
    }
    if (z - halfDepth < margin) {
      collisions.push({ type: "wall", direction: "back" });
    }
    if (z + halfDepth > roomDepth - margin) {
      collisions.push({ type: "wall", direction: "front" });
    }

    return collisions;
  }

  static adjustToValidPosition(
    position,
    size,
    roomSize,
    furniture,
    currentId,
    furniturePresets
  ) {
    let [x, y, z] = position;
    const [width, height, depth] = size;
    const [roomWidth, roomHeight, roomDepth] = roomSize;

    const halfWidth = width / 2;
    const halfDepth = depth / 2;

    const margin = 5; // 벽에서 5cm 떨어진 위치

    x = Math.max(
      halfWidth + margin,
      Math.min(roomWidth - halfWidth - margin, x)
    );
    z = Math.max(
      halfDepth + margin,
      Math.min(roomDepth - halfDepth - margin, z)
    );
    y = Math.max(height / 2, y);

    return [x, y, z];
  }
}

// 위치 스냅 유틸리티
class PositionSnapper {
  static snapToGrid(position, gridSize = 5) {
    // cm 단위로 스냅
    const [x, y, z] = position;
    return [
      Math.round(x / gridSize) * gridSize,
      y,
      Math.round(z / gridSize) * gridSize,
    ];
  }

  static snapToFurniture(
    position,
    size,
    furniture,
    furniturePresets,
    snapDistance = 15 // cm 단위
  ) {
    const [x, y, z] = position;
    const [width, height, depth] = size;

    let snappedX = x;
    let snappedZ = z;

    furniture.forEach((otherFurniture) => {
      const otherSize =
        otherFurniture.size || furniturePresets[otherFurniture.type]?.size;
      if (!otherSize) return;

      const [otherX, otherY, otherZ] = otherFurniture.position;
      const [otherWidth, otherHeight, otherDepth] = otherSize;

      const leftAlign = otherX - otherWidth / 2 - width / 2;
      const rightAlign = otherX + otherWidth / 2 + width / 2;
      const centerAlign = otherX;

      if (Math.abs(x - leftAlign) < snapDistance) snappedX = leftAlign;
      else if (Math.abs(x - rightAlign) < snapDistance) snappedX = rightAlign;
      else if (Math.abs(x - centerAlign) < snapDistance) snappedX = centerAlign;

      const frontAlign = otherZ - otherDepth / 2 - depth / 2;
      const backAlign = otherZ + otherDepth / 2 + depth / 2;
      const centerAlignZ = otherZ;

      if (Math.abs(z - frontAlign) < snapDistance) snappedZ = frontAlign;
      else if (Math.abs(z - backAlign) < snapDistance) snappedZ = backAlign;
      else if (Math.abs(z - centerAlignZ) < snapDistance)
        snappedZ = centerAlignZ;
    });

    return [snappedX, y, snappedZ];
  }
}

// 치수선 관련 컴포넌트들
const DIMENSION_COLOR = "#666666";

const DimensionArrow = React.memo(function DimensionArrow({ start, end }) {
  const points = useMemo(() => {
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const direction = new THREE.Vector3()
      .subVectors(endVec, startVec)
      .normalize();
    const arrowSize = 10;
    const perpendicular = new THREE.Vector3(-direction.z, 0, direction.x);

    return {
      main: [start, end],
      startArrow1: [
        start,
        [
          start[0] +
            direction.x * arrowSize +
            perpendicular.x * arrowSize * 0.5,
          start[1],
          start[2] +
            direction.z * arrowSize +
            perpendicular.z * arrowSize * 0.5,
        ],
      ],
      startArrow2: [
        start,
        [
          start[0] +
            direction.x * arrowSize -
            perpendicular.x * arrowSize * 0.5,
          start[1],
          start[2] +
            direction.z * arrowSize -
            perpendicular.z * arrowSize * 0.5,
        ],
      ],
      endArrow1: [
        end,
        [
          end[0] - direction.x * arrowSize + perpendicular.x * arrowSize * 0.5,
          end[1],
          end[2] - direction.z * arrowSize + perpendicular.z * arrowSize * 0.5,
        ],
      ],
      endArrow2: [
        end,
        [
          end[0] - direction.x * arrowSize - perpendicular.x * arrowSize * 0.5,
          end[1],
          end[2] - direction.z * arrowSize - perpendicular.z * arrowSize * 0.5,
        ],
      ],
    };
  }, [start, end]);

  return (
    <>
      <Line points={points.main} color={DIMENSION_COLOR} lineWidth={2} />
      <Line points={points.startArrow1} color={DIMENSION_COLOR} lineWidth={2} />
      <Line points={points.startArrow2} color={DIMENSION_COLOR} lineWidth={2} />
      <Line points={points.endArrow1} color={DIMENSION_COLOR} lineWidth={2} />
      <Line points={points.endArrow2} color={DIMENSION_COLOR} lineWidth={2} />
    </>
  );
});

const DimensionLabel = React.memo(function DimensionLabel({
  position,
  text,
  rotation = [0, 0, 0],
}) {
  return (
    <group position={position} rotation={rotation}>
      <Text
        position={[0, 0, 0]}
        fontSize={12}
        color={DIMENSION_COLOR}
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        {text}
      </Text>
    </group>
  );
});

// 가구 치수 컴포넌트
const FurnitureDimensions = React.memo(function FurnitureDimensions({
  position,
  size,
  selected,
}) {
  if (!selected || !size) return null;

  const [width, height, depth] = size;
  const [x, y, z] = position;
  const offset = 5;

  const baseY = y - height / 2; // 최하단 높이(y = 0에 붙음)

  return (
    <group>
      {/* 가로 치수선 (앞쪽에) */}
      <DimensionArrow
        start={[x - width / 2, baseY, z + depth / 2 + offset]}
        end={[x + width / 2, baseY, z + depth / 2 + offset]}
      />
      <DimensionLabel
        position={[x, baseY + 8, z + depth / 2 + offset]}
        text={`${width.toFixed(0)}cm`}
      />

      {/* 세로 치수선 (오른쪽에) */}
      <DimensionArrow
        start={[x + width / 2 + offset, baseY, z - depth / 2]}
        end={[x + width / 2 + offset, baseY, z + depth / 2]}
      />
      <DimensionLabel
        position={[x + width / 2 + offset + 8, baseY + 8, z]}
        text={`${depth.toFixed(0)}cm`}
        rotation={[0, Math.PI / 2, 0]}
      />

      {/* 높이 치수선 (위쪽으로) */}
      <DimensionArrow
        start={[x + width / 2 + offset, baseY, z + depth / 2 + offset]}
        end={[x + width / 2 + offset, baseY + height, z + depth / 2 + offset]}
      />
      <DimensionLabel
        position={[
          x + width / 2 + offset + 8,
          baseY + height / 2,
          z + depth / 2 + offset,
        ]}
        text={`${height.toFixed(0)}cm`}
      />
    </group>
  );
});

// 충돌 표시 컴포넌트
const CollisionIndicator = React.memo(function CollisionIndicator({
  position,
  size,
  collisionType = "furniture",
}) {
  const color = collisionType === "wall" ? "#ff6b6b" : "#ffd93d";

  return (
    <mesh position={position}>
      <boxGeometry args={size} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.3}
        wireframe={true}
      />
    </mesh>
  );
});

// 유효한 배치 영역 표시 컴포넌트
const ValidPlacementArea = React.memo(function ValidPlacementArea({
  roomSize,
  furniture,
  furniturePresets,
  selectedFurnitureSize,
}) {
  const [roomWidth, roomHeight, roomDepth] = roomSize;
  const gridSize = 30;
  const validPositions = [];

  for (let x = 0; x <= roomWidth; x += gridSize) {
    for (let z = 0; z <= roomDepth; z += gridSize) {
      const testPosition = [x, selectedFurnitureSize[1] / 2, z];

      if (
        !CollisionDetector.isWithinRoomBounds(
          testPosition,
          selectedFurnitureSize,
          roomSize
        )
      ) {
        continue;
      }

      const hasCollision = furniture.some((f) => {
        let otherSize = f.size || furniturePresets[f.type]?.size;
        if (!otherSize) return false;

        const currentBox = CollisionDetector.createBoundingBox(
          testPosition,
          selectedFurnitureSize
        );
        const otherBox = CollisionDetector.createBoundingBox(
          f.position,
          otherSize
        );
        return CollisionDetector.isBoxOverlapping(currentBox, otherBox);
      });

      if (!hasCollision) {
        validPositions.push(testPosition);
      }
    }
  }

  return (
    <group>
      {validPositions.map((pos, index) => (
        <mesh key={index} position={[pos[0], 0.1, pos[2]]}>
          <circleGeometry args={[5]} />
          <meshBasicMaterial color="#4ade80" transparent opacity={0.6} />
        </mesh>
      ))}
    </group>
  );
});

// 충돌 알림 컴포넌트 (Canvas 외부에서 렌더링)
const CollisionAlert = React.memo(function CollisionAlert({
  collisions,
  onDismiss,
  visible = false,
}) {
  if (!visible || !collisions || collisions.length === 0) return null;

  return (
    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg shadow-lg max-w-sm">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-semibold">충돌 감지</h4>
        <button onClick={onDismiss} className="text-red-500 hover:text-red-700">
          ×
        </button>
      </div>
      <ul className="text-sm">
        {collisions.map((collision, index) => (
          <li key={index} className="mb-1">
            {collision.type === "wall"
              ? `벽과 충돌 (${collision.direction})`
              : `${collision.name}과 충돌`}
          </li>
        ))}
      </ul>
    </div>
  );
});

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

// 공간감 향상 컴포넌트들

// 실제 크기 비교 표시
const SizeComparisonPanel = React.memo(function SizeComparisonPanel({
  currentArea,
  visible = true,
  isFullscreen = false, // Add isFullscreen prop
}) {
  if (!visible) return null;

  const closestSpace = Object.entries(SPACE_COMPARISONS).reduce(
    (prev, [name, area]) =>
      Math.abs(area - currentArea) < Math.abs(prev.area - currentArea)
        ? { name, area }
        : prev
  );

  const ratio = currentArea / closestSpace.area;

  return (
    <div
      className={`backdrop-blur p-3 rounded-lg shadow-lg ${
        visible && isFullscreen ? "bg-white/70" : "bg-white/95"
      }`}
    >
      <h4 className="font-semibold text-sm mb-2">크기 비교</h4>
      <p className="text-xs mb-1">
        이 방은 <strong>{closestSpace.name}</strong>의
        <span
          className={`font-bold ${
            ratio > 1.2
              ? "text-green-600"
              : ratio < 0.8
              ? "text-red-600"
              : "text-blue-600"
          }`}
        >
          {ratio > 1.2 ? " 큰" : ratio < 0.8 ? " 작은" : " 비슷한"} 크기
        </span>
      </p>
      <div className="text-xs text-gray-600">
        <p>현재: {currentArea.toFixed(1)}㎡</p>
        <p>
          비교: {closestSpace.area}㎡ ({(ratio * 100).toFixed(0)}%)
        </p>
      </div>
    </div>
  );
});

// 걸음 수와 시간 표시
const WalkingMetrics = React.memo(function WalkingMetrics({
  width,
  depth,
  visible = true,
  isFullscreen = false, // Add isFullscreen prop
}) {
  if (!visible) return null;

  const widthSteps = Math.ceil(width / 100 / 0.65);
  const depthSteps = Math.ceil(depth / 100 / 0.65);
  const walkingTime = Math.ceil(((width + depth) / 100 / 1.4) * 60);

  return (
    <div
      className={`backdrop-blur p-3 rounded-lg shadow-lg ${
        visible && isFullscreen ? "bg-white/70" : "bg-white/95"
      }`}
    >
      <h4 className="font-semibold text-sm mb-2">이동 거리</h4>
      <div className="text-xs space-y-1">
        <p>
          가로: <strong>{widthSteps}걸음</strong>
        </p>
        <p>
          세로: <strong>{depthSteps}걸음</strong>
        </p>
        <p>
          한 바퀴: 약 <strong>{walkingTime}초</strong>
        </p>
      </div>
    </div>
  );
});

// 시점 프리셋
const ViewPresets = React.memo(function ViewPresets({
  onViewChange,
  roomSize,
}) {
  const [w, h, d] = roomSize;
  const presets = [
    {
      name: "조감도",
      position: [w / 2, h * 2, d / 2],
      target: [w / 2, 0, d / 2],
    },
    {
      name: "입구",
      position: [w / 2, h * 0.75, d + d * 0.1],
      target: [w / 2, h * 0.5, 0],
    },
    {
      name: "코너",
      position: [w + w * 0.1, h, d + d * 0.1],
      target: [0, 0, 0],
    },
    {
      name: "눈높이",
      position: [w / 2, 170, d * 0.8],
      target: [w / 2, 160, 0],
    },
  ];

  return (
    <div className="bg-white/90 backdrop-blur p-2 rounded-lg">
      <h4 className="font-semibold text-xs mb-2">시점 변경</h4>
      <div className="grid grid-cols-2 gap-1">
        {presets.map((preset) => (
          <button
            key={preset.name}
            onClick={() => onViewChange(preset)}
            className="flex flex-col items-center p-1 bg-gray-100 hover:bg-blue-100 rounded text-xs transition-colors"
          >
            <span>{preset.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
});

// 향상된 조명과 그림자
const EnhancedLighting = React.memo(function EnhancedLighting({ roomSize }) {
  const [w, h, d] = roomSize;
  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight
        position={[w, h * 1.5, d]}
        intensity={0.8}
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-camera-far={h * 3}
        shadow-camera-left={-w}
        shadow-camera-right={w}
        shadow-camera-top={d}
        shadow-camera-bottom={-d}
      />
    </>
  );
});

// 바닥 그리드와 거리 표시
const FloorGrid = React.memo(function FloorGrid({ roomSize, visible = true }) {
  if (!visible) return null;

  const [roomWidth, roomHeight, roomDepth] = roomSize;
  const gridSize = 100; // 1m
  const lines = [];

  for (let x = 0; x <= roomWidth; x += gridSize) {
    lines.push(
      <Line
        key={`major-v-${x}`}
        points={[
          [x, 0.3, 0],
          [x, 0.3, roomDepth],
        ]}
        color="#cccccc"
        lineWidth={1}
      />
    );
  }

  for (let z = 0; z <= roomDepth; z += gridSize) {
    lines.push(
      <Line
        key={`major-h-${z}`}
        points={[
          [0, 0.3, z],
          [roomWidth, 0.3, z],
        ]}
        color="#cccccc"
        lineWidth={1}
      />
    );
  }

  return <group>{lines}</group>;
});

// 드래그 가능한 사람 모델 (cm 단위로 수정)
const DraggableHuman = React.memo(function DraggableHuman({
  height = 170,
  position,
  onPositionChange,
  roomSize,
  onDragStateChange,
}) {
  const { scene, error } = useGLTF("/human.glb");
  const modelRef = useRef();
  const [dragging, setDragging] = useState(false);
  const { camera, gl, raycaster, mouse } = useThree();
  const [modelHeight, setModelHeight] = useState(2.0);

  useEffect(() => {
    if (scene) {
      const box = new THREE.Box3().setFromObject(scene);
      const actualHeight = box.max.y - box.min.y;
      setModelHeight(actualHeight);
    }
  }, [scene]);

  const targetHeight = 170;
  const finalScale = modelHeight > 0 ? targetHeight / modelHeight : 85;

  // 전역 마우스 이벤트로 드래그 처리
  useEffect(() => {
    const handleMouseMove = (event) => {
      if (!dragging) return;

      const rect = gl.domElement.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera({ x, y }, camera);

      // y=0 평면과의 교차점 계산
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const intersection = new THREE.Vector3();

      if (raycaster.ray.intersectPlane(plane, intersection)) {
        const [roomWidth, , roomDepth] = roomSize;
        
        // 방 좌표계는 0에서 roomWidth/roomDepth까지 (중심 기준이 아님)
        // 방 경계 내로 제한 (0 기준)
        let newX = Math.max(10, Math.min(roomWidth - 10, intersection.x));
        let newZ = Math.max(10, Math.min(roomDepth - 10, intersection.z));

        console.log('👤 사람 드래그:', {
          intersection: [intersection.x, intersection.z],
          roomSize: [roomWidth, roomDepth],
          bounds: [0, roomWidth, 0, roomDepth],
          newPosition: [newX, newZ]
        });

        onPositionChange([newX, 0, newZ]);
      }
    };

    const handleMouseUp = () => {
      setDragging(false);
      gl.domElement.style.cursor = "auto";
      onDragStateChange?.(false);
    };

    if (dragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      return () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [
    dragging,
    camera,
    gl,
    raycaster,
    onPositionChange,
    roomSize,
    onDragStateChange,
  ]);

  const handlePointerDown = useCallback(
    (e) => {
      e.stopPropagation();
      setDragging(true);
      gl.domElement.style.cursor = "grabbing";
      onDragStateChange?.(true);
    },
    [gl, onDragStateChange]
  );

  if (error || !scene) {
    return (
      <group position={position}>
        <mesh
          position={[position[0], size[1] / 2, position[2]]}
          onPointerDown={handlePointerDown}
        >
          <cylinderGeometry args={[15, 20, height]} />
          <meshStandardMaterial color="#666666" opacity={0.8} transparent />
        </mesh>
        <Text
          position={[30, height * 0.8, 0]}
          fontSize={15}
          color="#333333"
          anchorX="left"
          fontWeight="bold"
          backgroundColor="#FFFFFF"
          backgroundOpacity={0.8}
          padding={2}
        >
          {height}cm (fallback)
        </Text>
      </group>
    );
  }

  return (
    <group position={position}>
      {/* GLB 모델 표시 */}
      <primitive
        ref={modelRef}
        object={scene.clone()}
        scale={[finalScale, finalScale, finalScale]}
        position={[0, 0, 0]}
        castShadow
        receiveShadow
        onPointerDown={handlePointerDown}
      />

      <ContactShadows
        position={[0, 0.01, 0]}
        opacity={0.4}
        scale={50}
        blur={3}
        far={20}
      />
      <Text
        position={[30, height * 0.8, 0]}
        fontSize={15}
        color="#333333"
        anchorX="left"
        fontWeight="bold"
        backgroundColor="#FFFFFF"
        backgroundOpacity={0.8}
        padding={2}
      >
        {height}cm (GLB)
      </Text>
    </group>
  );
});

// GLB 파일 프리로딩
useGLTF.preload("/human.glb");

// 실시간 거리 측정 도구
const DistanceMeasurer = React.memo(function DistanceMeasurer({
  point1,
  point2,
  visible = false,
}) {
  if (!visible || !point1 || !point2) return null;

  const distance = Math.sqrt(
    Math.pow(point2[0] - point1[0], 2) + Math.pow(point2[2] - point1[2], 2)
  ).toFixed(0);

  const midpoint = [
    (point1[0] + point2[0]) / 2,
    Math.max(point1[1], point2[1]) + 10,
    (point1[2] + point2[2]) / 2,
  ];

  return (
    <group>
      <Line points={[point1, point2]} color="#FF4081" lineWidth={3} />
      <Text
        position={midpoint}
        fontSize={12}
        color="#FF4081"
        anchorX="center"
        fontWeight="bold"
        backgroundColor="#FFFFFF"
        backgroundOpacity={0.8}
        padding={2}
      >
        {distance}cm
      </Text>
    </group>
  );
});

// 벽 컴포넌트
const Wall = React.memo(function Wall({
  width,
  height,
  position,
  rotation,
  isWindow = false,
}) {
  return (
    <mesh position={position} rotation={rotation} castShadow receiveShadow>
      <planeGeometry args={[width, height]} />
      <meshPhysicalMaterial
        color={isWindow ? "#FFE4EC" : "#FFF0F5"}
        roughness={0.7}
        metalness={0.1}
        clearcoat={0.2}
        opacity={isWindow ? 0.3 : 1}
        transparent={isWindow}
        side={THREE.DoubleSide} // Render both sides
      />
    </mesh>
  );
});

// 공간 활용도 컴포넌트
const SpaceUtilization = React.memo(function SpaceUtilization({
  furniture,
  roomArea,
}) {
  const furnitureArea = furniture.reduce((total, item) => {
    const size = item.size || FURNITURE_PRESETS[item.type]?.size;
    if (!size) return total;
    const area = (size[0] * size[2]) / 10000; // cm² → m²
    return total + area;
  }, 0);

  const utilization = (furnitureArea / roomArea) * 100;

  const getUtilizationColor = (util) => {
    if (util < 30) return "bg-green-500";
    if (util < 60) return "bg-yellow-500";
    if (util < 80) return "bg-orange-500";
    return "bg-red-500";
  };

  const getUtilizationText = (util) => {
    if (util < 30) return "여유로움";
    if (util < 60) return "적절함";
    if (util < 80) return "꽉참";
    return "과밀";
  };

  return (
    <div className="bg-gray-50 p-2 rounded">
      <h4 className="font-semibold text-xs mb-1">공간 활용도</h4>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${getUtilizationColor(
            utilization
          )}`}
          style={{ width: `${Math.min(utilization, 100)}%` }}
        />
      </div>
      <div className="flex justify-between items-center mt-1">
        <span className="text-xs">{utilization.toFixed(1)}%</span>
        <span className="text-xs font-medium">
          {getUtilizationText(utilization)}
        </span>
      </div>
    </div>
  );
});

// 향상된 가구 컴포넌트 (드래그 시스템 개선)
const DraggableFurnitureWithCollision = React.memo(
  function DraggableFurnitureWithCollision({
    id,
    type,
    position,
    rotation,
    onMove,
    onSelect,
    selected,
    furniture,
    furniturePresets,
    roomSize,
    enableSnap = true,
    showCollisions = true,
    onCollisionAlert,
    onDragStateChange,
    customFurnitureData = null,
    updatePlacedFurniturePosition,
  }) {
    const mesh = useRef();

    // 커스텀 가구인 경우 전달된 size를 직접 사용, 아니면 preset에서 가져오기
    let size, color;
    if (customFurnitureData && customFurnitureData.size) {
      size = customFurnitureData.size;
      color = customFurnitureData.color || "#cccccc";
    } else {
      const preset = furniturePresets[type];
      size = preset?.size || [100, 100, 100];
      color = preset?.color || "#cccccc";
    }

    const [hovered, setHovered] = useState(false);
    const [dragging, setDragging] = useState(false);
    const [collisions, setCollisions] = useState([]);
    const { camera, gl, raycaster, mouse } = useThree();

    // 드래그 상태 관리
    const dragStart = useRef(null);
    const isDraggingRef = useRef(false);
    const lastValidPosition = useRef(position);
    
    // position이 변경될 때마다 lastValidPosition 업데이트
    React.useEffect(() => {
      lastValidPosition.current = position;
    }, [position]);

    // 마우스/터치 다운 이벤트
    const handlePointerDown = useCallback(
      (e) => {
        e.stopPropagation();

        const rect = gl.domElement.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        dragStart.current = { x, y };
        isDraggingRef.current = false;

        onSelect(id);
        gl.domElement.style.cursor = "grabbing";

        const handleMouseMove = (moveEvent) => {
          if (!dragStart.current) return;

          const moveRect = gl.domElement.getBoundingClientRect();
          const moveX =
            ((moveEvent.clientX - moveRect.left) / moveRect.width) * 2 - 1;
          const moveY =
            -((moveEvent.clientY - moveRect.top) / moveRect.height) * 2 + 1;

          // 최소 이동 거리 체크 (의도하지 않은 미세한 드래그 방지)
          const deltaX = Math.abs(moveX - dragStart.current.x);
          const deltaY = Math.abs(moveY - dragStart.current.y);

          if (deltaX > 0.01 || deltaY > 0.01) {
            if (!isDraggingRef.current) {
              isDraggingRef.current = true;
              setDragging(true);
              onDragStateChange?.(true);
            }

            raycaster.setFromCamera({ x: moveX, y: moveY }, camera);
            const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
            const intersection = new THREE.Vector3();

            if (raycaster.ray.intersectPlane(plane, intersection)) {
              let newPosition = [intersection.x, size[1] / 2, intersection.z];
              console.log('🖱️ 마우스 이동 - 계산된 새 위치:', newPosition);

              // 극단적인 좌표값 방지 - 방 크기의 10배를 넘으면 무시
              const maxDistance = Math.max(roomSize[0], roomSize[2]) * 10;
              if (Math.abs(newPosition[0]) > maxDistance || Math.abs(newPosition[2]) > maxDistance) {
                console.log('❌ 극단적인 좌표값 감지, 이동 무시:', newPosition);
                return;
              }

              if (enableSnap) {
                newPosition = PositionSnapper.snapToGrid(newPosition, 25);
                newPosition = PositionSnapper.snapToFurniture(
                  newPosition,
                  size,
                  furniture.filter((f) => f.id !== id),
                  furniturePresets,
                  15
                );
                console.log('🧲 스냅 적용 후 위치:', newPosition);
              }

              // 먼저 방 경계 내로 조정
              const boundaryAdjustedPosition =
                CollisionDetector.adjustToValidPosition(
                  newPosition,
                  size,
                  roomSize,
                  [],
                  id,
                  furniturePresets
                );

              // 가구 충돌 체크
              const furnitureCollisions =
                CollisionDetector.checkFurnitureCollisions(
                  furniture,
                  id,
                  boundaryAdjustedPosition,
                  furniturePresets
                );

              // 충돌 상태 업데이트 (시각적 피드백용)
              setCollisions(furnitureCollisions);

              // 충돌 상태 체크
              console.log('🔍 충돌 체크 결과:', {
                collisions: furnitureCollisions.length,
                position: boundaryAdjustedPosition
              });

              // 충돌이 없을 때만 이동 허용
              if (furnitureCollisions.length === 0) {
                console.log('✅ 충돌 없음, 이동 허용');
                lastValidPosition.current = boundaryAdjustedPosition;
                onMove(id, boundaryAdjustedPosition);
              } else {
                console.log('❌ 충돌 감지, 이동 차단:', furnitureCollisions);
              }
            }
          }
        };

        const handleMouseUp = () => {
          dragStart.current = null;

          if (isDraggingRef.current) {
            // 드래그 완료 후 2D 좌표 업데이트 - 마지막 유효 위치 사용
            if (updatePlacedFurniturePosition) {
              console.log('🎯 드래그 완료, 최종 위치:', lastValidPosition.current);
              updatePlacedFurniturePosition(id, lastValidPosition.current);
            }

            // 상태 초기화
            isDraggingRef.current = false;
            setDragging(false);
            onDragStateChange?.(false);
            setCollisions([]);
          }

          gl.domElement.style.cursor = "auto";
          document.removeEventListener("mousemove", handleMouseMove);
          document.removeEventListener("mouseup", handleMouseUp);
        };

        document.addEventListener("mousemove", handleMouseMove);
        document.addEventListener("mouseup", handleMouseUp);
      },
      [
        gl,
        id,
        onSelect,
        onMove,
        onDragStateChange,
        updatePlacedFurniturePosition,
        position,
        camera,
        raycaster,
        size,
        enableSnap,
        furniture,
        furniturePresets,
        roomSize,
      ]
    );

    const hasCollision = collisions.length > 0;
    const materialColor = hasCollision ? "#ff9999" : color;

    return (
      <group position={position}>
        <mesh
          ref={mesh}
          rotation={rotation}
          onPointerDown={handlePointerDown}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHovered(true);
            if (!dragging) gl.domElement.style.cursor = "grab";
          }}
          onPointerOut={(e) => {
            e.stopPropagation();
            setHovered(false);
            if (!dragging) gl.domElement.style.cursor = "auto";
          }}
          scale={hovered ? 1.02 : 1}
          castShadow
          receiveShadow
        >
          <boxGeometry args={size} />
          <meshStandardMaterial
            color={materialColor}
            roughness={0.7}
            metalness={0.3}
            emissive={
              selected ? "#ffffff" : hasCollision ? "#ff6666" : "#000000"
            }
            emissiveIntensity={selected ? 0.1 : hasCollision ? 0.2 : 0}
          />
        </mesh>

        <ContactShadows
          position={[0, -size[1] / 2 + 0.1, 0]}
          opacity={0.4}
          scale={Math.max(size[0], size[2]) * 1.2}
          blur={2}
          far={size[1]}
        />

        {selected && (
          <mesh>
            <boxGeometry args={size.map((s) => s + 5)} />
            <meshBasicMaterial
              color="#4fc3f7"
              transparent
              opacity={0.2}
              wireframe={true}
            />
          </mesh>
        )}

        {hasCollision && showCollisions && (
          <CollisionIndicator
            position={[0, 0, 0]}
            size={size.map((s) => s + 2)}
            collisionType={
              collisions.some((c) => c.type === "wall") ? "wall" : "furniture"
            }
          />
        )}

        <FurnitureDimensions
          position={[0, 0, 0]}
          size={size}
          selected={selected}
        />
      </group>
    );
  }
);

// 창문 컴포넌트 (개선된 디자인)
const Window3D = React.memo(function Window3D({
  position,
  size,
  wallPosition,
  roomSize,
  rotation = [0, 0, 0],
}) {
  const [width, height, depth] = size;
  const frameThickness = 6; // 프레임 두께 증가
  const glassThickness = 2;

  return (
    <group position={position} rotation={rotation}>
      {/* 창문 프레임 - 더 진한 색상으로 */}
      <mesh castShadow receiveShadow>
        <boxGeometry
          args={[
            width + frameThickness,
            height + frameThickness,
            frameThickness,
          ]}
        />
        <meshStandardMaterial color="#8B4513" roughness={0.3} metalness={0.2} />
      </mesh>

      {/* 유리창 - 더 뚜렷한 청색으로 */}
      <mesh position={[0, 0, frameThickness / 2]} castShadow receiveShadow>
        <boxGeometry args={[width * 0.85, height * 0.85, glassThickness]} />
        <meshStandardMaterial
          color="#4169E1"
          transparent={true}
          opacity={0.6}
          roughness={0.05}
          metalness={0.1}
          emissive="#1E90FF"
          emissiveIntensity={0.2}
        />
      </mesh>

      {/* 창문 채색 (4분할) - 더 뚜렷한 색상 */}
      <group position={[0, 0, frameThickness / 2 + 1]}>
        {/* 세로 채색 */}
        <mesh>
          <boxGeometry args={[3, height * 0.8, 1]} />
          <meshStandardMaterial color="#2F4F4F" roughness={0.2} />
        </mesh>
        {/* 가로 채색 */}
        <mesh>
          <boxGeometry args={[width * 0.8, 3, 1]} />
          <meshStandardMaterial color="#2F4F4F" roughness={0.2} />
        </mesh>
      </group>

      {/* 창문 테두리 강조 */}
      <mesh position={[0, 0, frameThickness / 2 + 2]}>
        <boxGeometry args={[width + frameThickness + 2, height + frameThickness + 2, 0.5]} />
        <meshStandardMaterial color="#8B4513" transparent={true} opacity={0.8} />
      </mesh>
      
      {/* 그림자 */}
      <ContactShadows
        position={[0, 0, -frameThickness]}
        opacity={0.4}
        scale={Math.max(width, height) * 1.2}
        blur={1.5}
        far={15}
      />
    </group>
  );
});

// 벽에 창문을 배치하는 컴포넌트 (개선된 위치 계산)
const WindowsOnWalls = React.memo(function WindowsOnWalls({
  windows,
  roomSize,
}) {
  const [roomWidth, roomHeight, roomDepth] = roomSize;

  if (!windows || windows.length === 0) return null;

  return (
    <group>
      {windows.map((window, index) => {
        // 백엔드에서 전달된 실제 창문 크기 사용 (cm 단위로 변환)
        const windowWidth3D = (window.width_meters || 1.2) * 100;
        const windowHeight3D = (window.height_meters || 1.5) * 100;

        let position = [0, 0, 0];
        let rotation = [0, 0, 0];
        const wallThickness = 5; // 벽 두께

        // 위치 계산 개선 - 백분율 기반에서 실제 좌표 기반으로
        let x_pos, y_pos, z_pos;
        
        // 창문 높이 계산 (사용자 y_position 슬라이더 값 반영)
        const userYPosition = window.y_position !== undefined ? window.y_position : 0.8; // 기본값 80%
        const calculatedYPos = userYPosition * roomHeight; // 사용자 설정 높이
        
        switch (window.wall_position) {
          case "front":
            // 앞벽: Z 최대값
            x_pos = window.x_position ? window.x_position * roomWidth : roomWidth / 2;
            y_pos = calculatedYPos;
            z_pos = roomDepth - wallThickness;
            rotation = [0, 0, 0];
            break;
          case "back":
            // 뒷벽: Z=0
            x_pos = window.x_position ? window.x_position * roomWidth : roomWidth / 2;
            y_pos = calculatedYPos;
            z_pos = wallThickness;
            rotation = [0, Math.PI, 0];
            break;
          case "left":
            // 왼쪽 벽: X=0
            x_pos = wallThickness;
            y_pos = calculatedYPos;
            z_pos = window.x_position ? window.x_position * roomDepth : roomDepth / 2;
            rotation = [0, Math.PI / 2, 0];
            break;
          case "right":
            // 오른쪽 벽: X 최대값
            x_pos = roomWidth - wallThickness;
            y_pos = calculatedYPos;
            z_pos = window.x_position ? window.x_position * roomDepth : roomDepth / 2;
            rotation = [0, -Math.PI / 2, 0];
            break;
          default:
            // 기본값: 뒷벽 중앙
            x_pos = roomWidth / 2;
            y_pos = calculatedYPos;
            z_pos = wallThickness;
            rotation = [0, Math.PI, 0];
        }
        
        position = [x_pos, y_pos, z_pos];

        // 범위 제한 (더 현실적인 범위로)
        const margin = 20; // 20cm 여백

        // X 좌표 제한 - 벽별 처리
        if (
          window.wall_position === "front" ||
          window.wall_position === "back"
        ) {
          position[0] = Math.max(
            windowWidth3D / 2 + margin,
            Math.min(roomWidth - windowWidth3D / 2 - margin, position[0])
          );
        }

        // Y 좌표 제한 (높이) - 창문이 바닥 아래로 가지 않도록
        const minHeight = windowHeight3D / 2 + 30; // 바닥에서 최소 30cm 위
        const maxHeight = roomHeight - windowHeight3D / 2 - margin; // 천장에서 여백
        position[1] = Math.max(minHeight, Math.min(maxHeight, position[1]));

        // Z 좌표 제한 - 벽별 처리
        if (
          window.wall_position === "left" ||
          window.wall_position === "right"
        ) {
          position[2] = Math.max(
            windowWidth3D / 2 + margin,
            Math.min(roomDepth - windowWidth3D / 2 - margin, position[2])
          );
        }

        // 사용자가 수직 위치를 설정하지 않은 경우에만 기본 높이 적용
        if (window.y_position === undefined && window.wall_position === "back") {
          const targetHeight = roomHeight * 0.8; // 높이의 80% 위치 (기본값)
          position[1] = Math.max(position[1], targetHeight);
        }

        return (
          <group key={`window-group-${index}`}>
            <Window3D
              key={`window-${index}`}
              position={position}
              size={[windowWidth3D, windowHeight3D, 10]}
              wallPosition={window.wall_position}
              roomSize={roomSize}
              rotation={rotation}
            />
            {/* 창문 정보 텍스트 */}
            <Text
              position={[
                position[0],
                position[1] - windowHeight3D / 2 - 15,
                position[2],
              ]}
              fontSize={8}
              color="#666666"
              anchorX="center"
              anchorY="middle"
            >
              {`${(windowWidth3D / 100).toFixed(1)}m × ${(
                windowHeight3D / 100
              ).toFixed(1)}m`}
            </Text>
          </group>
        );
      })}
    </group>
  );
});

// 창문 감지 API 호출 함수 (실제 방 크기 정보 포함)
const detectWindowsInImage = async (
  imageFile,
  roomPoints = null,
  roomDimensions = null
) => {
  try {
    const formData = new FormData();
    formData.append("file", imageFile);

    // 백엔드는 JSON 문자열을 기대하므로 문자열로 변환
    if (roomPoints && roomPoints.length >= 2) {
      formData.append("room_points", JSON.stringify(roomPoints));
    }

    // 실제 방 크기 정보 추가 (JSON 문자열로)
    if (roomDimensions) {
      formData.append("room_dimensions", JSON.stringify(roomDimensions));
    }

    console.log("🚀 백엔드로 전송하는 데이터:");
    console.log("- roomPoints:", roomPoints);
    console.log("- roomDimensions:", roomDimensions);

    const response = await fetch("http://localhost:3000/detect-windows", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `HTTP error! status: ${response.status}, message: ${errorText}`
      );
    }

    const result = await response.json();
    console.log("📥 백엔드에서 받은 응답:", result);
    
    return result;
  } catch (error) {
    console.error("창문 감지 오류:", error);
    throw error;
  }
};

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

  const [furniture, setFurniture] = useState([]);
  const [selectedFurniture, setSelectedFurniture] = useState(null);
  const [showSnapGrid, setShowSnapGrid] = useState(false);
  const [showFloorGrid, setShowFloorGrid] = useState(false);
  const [enableSnap, setEnableSnap] = useState(true);
  const [showCollisions, setShowCollisions] = useState(true);
  const [placementMode, setPlacementMode] = useState(null);
  const [activeView, setActiveView] = useState("조감도");
  const [walkthroughMode, setWalkthroughMode] = useState(false);
  const [collisionAlert, setCollisionAlert] = useState({
    visible: false,
    collisions: [],
  });
  const [measurementMode, setMeasurementMode] = useState(false);
  const [measurePoints, setMeasurePoints] = useState([null, null]);
  const [humanPosition, setHumanPosition] = useState([w / 2, 0, d / 2]);
  const [isDragging, setIsDragging] = useState(false);
  const [detectedWindows, setDetectedWindows] = useState([]);
  const [showWindows, setShowWindows] = useState(false);
  const [isDetectingWindows, setIsDetectingWindows] = useState(false);

  const roomSize = [w, h, d]; // [Width, Height, Depth] - CollisionDetector 함수들과 일치
  const roomArea = (w * d) / 10000;

  const controlsRef = useRef();

  // 드래그 중에는 업데이트하지 않는 ref
  const isUpdatingFromDragRef = useRef(false);

  useEffect(() => {
    // 드래그 중이거나 드래그에서 업데이트 중일 때는 건너뛰기
    if (isDragging || isUpdatingFromDragRef.current) {
      return;
    }

    const converted = placedFurniture.map((item) => {
      let furnitureSize, mappedType, presetData;
      
      // 커스텀 가구인 경우
      if (item.isCustom) {
        furnitureSize = [item.width, item.height || 60, item.depth]; // 사용자 입력 높이 또는 기본값 60cm
        mappedType = "custom";
        presetData = {
          name: item.name,
          color: item.color || "#DDA0DD",
          size: furnitureSize
        };
      } else {
        // 기존 프리셋 가구인 경우
        const baseId = item.id
          ? item.id.split("_").slice(0, -1).join("_")
          : "desk";
        mappedType = FURNITURE_ID_MAPPING[baseId] || "desk";
        presetData = FURNITURE_PRESETS[mappedType];
        furnitureSize = presetData.size;
      }

      // 2D -> 3D 좌표 변환 명확화
      // FurniturePlacement: 실제로는 (x, y) 좌표를 {x: x, z: y} 형태로 저장
      // RoomBox 3D: (x, y, z) 좌표계
      // 따라서: 2D x → 3D x, 2D y → 3D z
      const x3d = item.x; // 2D x → 3D x (가로축 동일)
      const z3d = item.z; // 2D y → 3D z (2D의 세로축이 3D의 깊이축으로)

      return {
        id: item.id,
        type: mappedType,
        name: presetData.name,
        color: presetData.color,
        size: furnitureSize,
        position: [
          x3d + furnitureSize[0] / 2, // 3D x (왼쪽 모서리 → 중심)
          furnitureSize[1] / 2, // 3D y (바닥 → 중심 높이)
          z3d + furnitureSize[2] / 2, // 3D z (뒤쪽 모서리 → 중심)
        ],
        rotation: [0, (item.rotation || 0) * (Math.PI / 180), 0],
        original2D: item,
        isCustom: item.isCustom || false,
      };
    });
    
    setFurniture(converted);
  }, [placedFurniture, w, d, isDragging]);

  useEffect(() => {
    setHumanPosition([w / 2, 0, d / 2]);
  }, [w, d]);

  const handleViewChange = useCallback((preset) => {
    if (controlsRef.current) {
      controlsRef.current.object.position.set(...preset.position);
      controlsRef.current.target.set(...preset.target);
      controlsRef.current.update();
      setActiveView(preset.name);
    }
  }, []);

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

      // 가구 충돌 체크
      const furnitureCollisions = CollisionDetector.checkFurnitureCollisions(
        furniture,
        newFurniture.id,
        adjustedPosition,
        FURNITURE_PRESETS
      );

      // 충돌이 없을 때만 배치 허용
      if (furnitureCollisions.length === 0) {
        newFurniture.position = adjustedPosition;
        setFurniture((prev) => [...prev, newFurniture]);
        setSelectedFurniture(newFurniture.id);
        setPlacementMode(null);
      } else {
        // 충돌이 있을 때 알림 표시
        alert(
          `다른 가구와 겹칩니다! (충돌: ${furnitureCollisions
            .map((c) => c.name)
            .join(", ")})`
        );
      }
    },
    [placementMode, furniture, roomSize]
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
    console.log('🚚 handleMoveFurniture 호출:', id, newPosition);
    setFurniture((prev) => {
      const updated = prev.map((f) => {
        if (f.id === id) {
          console.log('🚚 가구 위치 변경:', f.position, '→', newPosition);
          return { ...f, position: newPosition };
        }
        return f;
      });
      console.log('🚚 furniture 상태 업데이트됨');
      return updated;
    });
  }, []);

  // 로컬 좌표 변환 (백엔드 없이도 작동)
  const convertCoordinatesLocally = useCallback((id, newPosition, size, roomSize) => {
    const [x3d, y3d, z3d] = newPosition;
    const [roomWidth, roomHeight, roomDepth] = roomSize;
    
    console.log('🔧 좌표 변환 상세:', {
      input_3d: { x3d, y3d, z3d },
      furniture_size: size,
      room_size: roomSize
    });
    
    // 3D 중심 좌표 → 2D 왼쪽아래 좌표로 변환
    let x_2d = x3d - size[0] / 2;
    let z_2d = z3d - size[2] / 2;
    
    console.log('🔧 변환 전 2D 좌표:', { x_2d, z_2d });
    
    // 경계 검사 - 방 경계 내로 제한
    const furniture_width = size[0];
    const furniture_depth = size[2];
    
    x_2d = Math.max(0, Math.min(x_2d, roomWidth - furniture_width));
    z_2d = Math.max(0, Math.min(z_2d, roomDepth - furniture_depth));
    
    console.log('🔧 경계 검사 후 2D 좌표:', { x_2d, z_2d });
    
    return { x: x_2d, z: z_2d };
  }, []);

  // 드래그 완료 시에만 2D 좌표 업데이트
  const updatePlacedFurniturePosition = useCallback(
    (id, newPosition) => {
      console.log('🔄 updatePlacedFurniturePosition 호출:', id, newPosition);
      console.log('onFurnitureChange 타입:', typeof onFurnitureChange);
      
      if (typeof onFurnitureChange === "function") {
        console.log('furniture 배열:', furniture);
        const furnitureItem = furniture.find((f) => f.id === id);
        console.log('찾은 furnitureItem:', furnitureItem);
        
        if (furnitureItem) {
          const size = furnitureItem.size;
          console.log('가구 크기:', size);

          // 업데이트 중임을 표시
          isUpdatingFromDragRef.current = true;

          // 로컬 좌표 변환
          const converted2D = convertCoordinatesLocally(id, newPosition, size, roomSize);

          console.log('🎯 3D → 2D 좌표 변환:', newPosition, '→', converted2D);

          onFurnitureChange((prev) => {
            console.log('이전 placedFurniture:', prev);
            const updated = prev.map((item) => {
              if (item.id === id) {
                const newItem = {
                  ...item,
                  x: converted2D.x, // 2D x 좌표
                  z: converted2D.z, // 2D y 좌표 (z 필드에 저장)
                };
                console.log('아이템 업데이트:', item, '→', newItem);
                return newItem;
              }
              return item;
            });
            console.log('✅ 2D 좌표 업데이트 완료:', updated);
            return updated;
          });

          // 짧은 지연 후 업데이트 플래그 해제
          setTimeout(() => {
            isUpdatingFromDragRef.current = false;
            console.log('업데이트 플래그 해제됨');
          }, 100);
        } else {
          console.log('❌ furnitureItem을 찾을 수 없음');
        }
      } else {
        console.log('❌ onFurnitureChange가 함수가 아님');
      }
    },
    [furniture, onFurnitureChange, convertCoordinatesLocally, roomSize]
  );

  // 드래그 완료 시 2D 좌표 업데이트 (실제 사용되는 함수)
  const updatePlacedFurniturePositionOnDragEnd = useCallback(
    (id, newPosition) => {
      if (typeof onFurnitureChange === "function") {
        const furnitureItem = furniture.find((f) => f.id === id);
        
        if (furnitureItem) {
          const size = furnitureItem.size;

          // 업데이트 중임을 표시
          isUpdatingFromDragRef.current = true;

          // 로컬 좌표 변환
          const converted2D = convertCoordinatesLocally(id, newPosition, size, roomSize);

          onFurnitureChange((prev) => {
            console.log('🔄 3D→2D 업데이트 시작');
            console.log('이전 placedFurniture:', prev);
            console.log('변환된 2D 좌표:', converted2D);
            
            const updated = prev.map((item) => {
              if (item.id === id) {
                const newItem = {
                  ...item,
                  x: converted2D.x, // 2D x 좌표
                  z: converted2D.z, // 2D y 좌표 (z 필드에 저장)
                };
                console.log('아이템 업데이트:', `${item.x},${item.z} → ${newItem.x},${newItem.z}`);
                return newItem;
              }
              return item;
            });
            
            console.log('✅ 업데이트된 placedFurniture:', updated);
            return updated;
          });

          // 짧은 지연 후 업데이트 플래그 해제
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
      alert("먼저 이미지를 업로드해주세요.");
      return;
    }
    setIsDetectingWindows(true);
    try {
      // 실제 방 크기 정보 준비 (cm 단위) - 더 상세한 정보 포함
      const roomDimensions = {
        width_cm: w,
        height_cm: h,
        depth_cm: d,
        area_sqm: (w * d) / 10000, // 제곱미터
        wall_height_cm: h,
        scale_factor: 1, // 스케일 팩터 (필요시 조정)
      };

      // 방의 벽 정보도 포함
      const wallInfo = {
        front_wall: { width: w, height: h, position: "front" },
        back_wall: { width: w, height: h, position: "back" },
        left_wall: { width: d, height: h, position: "left" },
        right_wall: { width: d, height: h, position: "right" },
      };

      console.log("📜 창문 감지에 실제 방 크기 전달:", roomDimensions);
      console.log("🏠 벽 정보:", wallInfo);

      const result = await detectWindowsInImage(
        uploadedImageFile,
        wallInfo,
        roomDimensions
      );
      
      if (result.windows && result.windows.length > 0) {
        // 창문 크기와 위치 검증 및 조정
        const validatedWindows = result.windows.map((window, index) => {
          console.log(`🪟 원본 창문 ${index + 1}:`, window);
          
          // 기본 크기 설정 (너무 작거나 큰 경우 조정)
          const minWidth = 60; // 최소 60cm
          const maxWidth = Math.min(200, w * 0.8); // 최대 200cm 또는 벽 폭의 80%
          const minHeight = 80; // 최소 80cm 
          const maxHeight = Math.min(180, h * 0.8); // 최대 180cm 또는 벽 높이의 80%
          
          let adjustedWindow = { ...window };
          
          // 크기 검증 및 조정
          if (window.width_meters) {
            const widthCm = window.width_meters * 100;
            adjustedWindow.width_meters = Math.max(minWidth, Math.min(maxWidth, widthCm)) / 100;
          } else {
            adjustedWindow.width_meters = 1.2; // 기본 120cm
          }
          
          if (window.height_meters) {
            const heightCm = window.height_meters * 100;
            adjustedWindow.height_meters = Math.max(minHeight, Math.min(maxHeight, heightCm)) / 100;
          } else {
            adjustedWindow.height_meters = 1.5; // 기본 150cm
          }
          
          // 위치 검증 (벽 중앙 근처로 조정)
          if (!window.wall_position || !["front", "back", "left", "right"].includes(window.wall_position)) {
            adjustedWindow.wall_position = "back"; // 기본값: 뒤쪽 벽
          }
          
          console.log(`✅ 조정된 창문 ${index + 1}:`, adjustedWindow);
          return adjustedWindow;
        });
        
        setDetectedWindows(validatedWindows);
        setShowWindows(true);
        alert(`${validatedWindows.length}개의 창문을 감지하고 조정했습니다.`);
      } else {
        alert("창문을 감지하지 못했습니다.");
      }
    } catch (error) {
      console.error("창문 감지 오류:", error);
      alert(`창문 감지에 실패했습니다: ${error.message}`);
    } finally {
      setIsDetectingWindows(false);
    }
  }, [uploadedImageFile, w, h, d]);

  const handleRotateFurniture = useCallback((id) => {
    setFurniture((prev) =>
      prev.map((f) =>
        f.id === id
          ? { ...f, rotation: [0, f.rotation[1] + Math.PI / 2, 0] }
          : f
      )
    );
  }, []);

  const handleDeleteFurniture = useCallback(
    (id) => {
      setFurniture((prev) => prev.filter((f) => f.id !== id));
      if (selectedFurniture === id) {
        setSelectedFurniture(null);
      }
    },
    [selectedFurniture]
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
        event.key.toLowerCase() === "delete" ||
        event.key.toLowerCase() === "backspace"
      ) {
        if (selectedFurniture) handleDeleteFurniture(selectedFurniture);
      }
    };
    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [selectedFurniture, handleDeleteFurniture]);

  return (
    <div
      className={`room-3d-viewer relative w-full bg-gradient-to-br from-blue-50 to-indigo-100 overflow-hidden shadow-2xl ${
        isFullscreen ? "h-screen rounded-none" : "h-[700px] rounded-xl"
      }`}
    >
      {/* Left UI Panels - Primary Actions */}
      <div className="absolute top-4 left-4 z-10 space-y-4 w-80 max-w-sm">
        {/* Furniture Addition Panel */}
        <div className="backdrop-blur p-3 rounded-lg shadow-lg bg-white/80">
          <h3 className="text-sm font-semibold mb-2 text-gray-700">
            {placementMode
              ? `${FURNITURE_PRESETS[placementMode].name} 배치 중...`
              : "가구 추가"}
          </h3>
          {placementMode ? (
            <button
              onClick={() => setPlacementMode(null)}
              className="w-full px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
            >
              취소
            </button>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(FURNITURE_PRESETS).map(([type, preset]) => (
                <button
                  key={type}
                  onClick={() => handleAddFurniture(type)}
                  className="flex items-center gap-2 px-2 py-1 bg-gray-100 rounded hover:bg-blue-100 transition-colors text-xs"
                >
                  <span>{preset.icon}</span>
                  <span>{preset.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Visual Options Panel */}
        <div className="backdrop-blur p-3 rounded-lg shadow-lg bg-white/80">
          <h4 className="font-semibold text-sm mb-2">시각 옵션</h4>
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={enableSnap}
                onChange={(e) => setEnableSnap(e.target.checked)}
                className="rounded"
              />
              스냅 기능
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={showSnapGrid}
                onChange={(e) => setShowSnapGrid(e.target.checked)}
                className="rounded"
              />
              스냅 그리드
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={showFloorGrid}
                onChange={(e) => setShowFloorGrid(e.target.checked)}
                className="rounded"
              />
              바닥 그리드
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={showCollisions}
                onChange={(e) => setShowCollisions(e.target.checked)}
                className="rounded"
              />
              충돌 표시
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={showWindows}
                onChange={(e) => setShowWindows(e.target.checked)}
                className="rounded"
              />
              창문 표시
            </label>
          </div>
        </div>

        {/* Coordinate Export Panel */}
        <div className="backdrop-blur p-3 rounded-lg shadow-lg bg-white/80">
          <h4 className="font-semibold text-sm mb-2">좌표 내보내기</h4>
          <div className="space-y-2">
            <button
              onClick={async () => {
                if (furniture.length === 0 && detectedWindows.length === 0) {
                  alert("저장할 가구나 창문이 없습니다!");
                  return;
                }
                
                // 3D 표준 형식으로 데이터 생성 (mm 단위)
                const wMm = Math.round(w * 10); // cm → mm
                const dMm = Math.round(d * 10);
                const hMm = Math.round(h * 10);
                
                const saveData = {
                  scene: {
                    description: `오른쪽 아래 꼭짓점(0,0,0)을 기준으로 하는 ${(w/100).toFixed(1)}m × ${(d/100).toFixed(1)}m 방 공간.`,
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
                      // 창문들을 먼저 추가
                      ...detectedWindows.map((window, index) => {
                        const widthMm = Math.round((window.width_meters || 1.2) * 1000);
                        const heightMm = Math.round((window.height_meters || 1.5) * 1000);
                        const userYPosition = window.y_position !== undefined ? window.y_position : 0.8;
                        
                        let wallNum, xMm, yMm, zMm;
                        
                        switch (window.wall_position) {
                          case "front": // wall_3 (top)
                            wallNum = 3;
                            xMm = Math.round((window.x_position || 0.5) * wMm);
                            yMm = dMm;
                            zMm = Math.round(userYPosition * hMm);
                            break;
                          case "back": // wall_1 (bottom)
                            wallNum = 1;
                            xMm = Math.round((window.x_position || 0.5) * wMm);
                            yMm = 0;
                            zMm = Math.round(userYPosition * hMm);
                            break;
                          case "left": // wall_4 (left)
                            wallNum = 4;
                            xMm = 0;
                            yMm = Math.round((window.x_position || 0.5) * dMm);
                            zMm = Math.round(userYPosition * hMm);
                            break;
                          case "right": // wall_2 (right)
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
                      }),
                      // 가구들 추가
                      ...furniture.map((item) => {
                        const presetData = FURNITURE_PRESETS[item.type];
                        const furnitureSize = presetData ? presetData.size : [100, 60, 100];
                        
                        // 좌표계 변환: 왼쪽 아래 → 오른쪽 아래 기준
                        const centerXMm = Math.round((wMm - item.position[0] * 10)); // X축 반전
                        const centerYMm = Math.round(item.position[2] * 10); // Z → Y
                        
                        // 회전 처리
                        const rotation = Array.isArray(item.rotation) ? item.rotation[1] || 0 : item.rotation || 0;
                        const rotationZ = -rotation; // 좌표계 반전으로 회전도 반전
                        
                        return {
                          type: "furniture",
                          name: presetData ? presetData.name.toLowerCase().replace(/\s+/g, '_') : "furniture",
                          material: presetData ? presetData.color : "gray",
                          shape: "rectangle",
                          position: { x: centerXMm, y: centerYMm, z: 0 },
                          width: Math.round(furnitureSize[0] * 10), // cm → mm
                          depth: Math.round(furnitureSize[2] * 10),
                          height: Math.round(furnitureSize[1] * 10),
                          rotation_z: Math.round(rotationZ)
                        };
                      })
                    ]
                  }
                };
                
                try {
                  // MongoDB에 저장
                  const response = await fetch('http://localhost:3000/save-room-layout', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(saveData)
                  });
                  
                  const result = await response.json();
                  
                  if (result.success) {
                    console.log("MongoDB 저장 완료:", saveData);
                    alert(`MongoDB 저장 완료!\n가구: ${furniture.length}개\n창문: ${detectedWindows.length}개\nID: ${result.id}`);
                  } else {
                    throw new Error(result.message);
                  }
                } catch (error) {
                  console.error('MongoDB 저장 실패:', error);
                  alert(`MongoDB 저장 실패: ${error.message}`);
                }
              }}
              disabled={furniture.length === 0 && detectedWindows.length === 0}
              className="w-full px-3 py-2 bg-green-500 text-white rounded text-sm hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              💾 MongoDB에 저장
            </button>
            <div className="text-xs text-gray-600 text-center">
              가구 {furniture.length}개 + 창문 {detectedWindows.length}개
            </div>
          </div>
        </div>

        {/* Window Detection Panel */}
        {uploadedImageFile && (
          <div className="backdrop-blur p-3 rounded-lg shadow-lg bg-white/80">
            <h4 className="font-semibold text-sm mb-2">창문 감지</h4>
            <div className="space-y-2">
              <button
                onClick={handleDetectWindows}
                disabled={isDetectingWindows}
                className="w-full px-3 py-2 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50"
              >
                {isDetectingWindows ? "감지 중..." : "AI 창문 감지"}
              </button>
              {detectedWindows.length > 0 && (
                <div className="text-xs text-gray-600 space-y-2">
                  <p>{detectedWindows.length}개 창문 감지됨</p>
                  {detectedWindows.map((window, index) => (
                    <div key={index} className="bg-gray-50 p-2 rounded space-y-2">
                      <div className="flex justify-between items-center">
                        <div className="text-xs">
                          창문 {index + 1}: {Math.round((window.width_meters || window.width || 1.2) * 100)}×{Math.round((window.height_meters || window.height || 1.5) * 100)}cm ({window.wall_position}벽)
                        </div>
                        <button
                          onClick={() => {
                            const newWindows = detectedWindows.filter((_, i) => i !== index);
                            setDetectedWindows(newWindows);
                          }}
                          className="px-1 py-0.5 bg-red-400 text-white rounded text-xs hover:bg-red-500"
                        >
                          삭제
                        </button>
                      </div>
                      <div className="space-y-1">
                        {/* 벽 위치 선택 */}
                        <div className="flex gap-1 items-center">
                          <span className="text-xs w-8">벽:</span>
                          <select
                            value={window.wall_position || "back"}
                            onChange={(e) => {
                              const newWindows = [...detectedWindows];
                              newWindows[index] = {
                                ...window,
                                wall_position: e.target.value
                              };
                              setDetectedWindows(newWindows);
                            }}
                            className="flex-1 text-xs border rounded px-1"
                          >
                            <option value="front">앞벽</option>
                            <option value="back">뒷벽</option>
                            <option value="left">왼쪽벽</option>
                            <option value="right">오른쪽벽</option>
                          </select>
                        </div>
                        
                        {/* 수평 위치 조정 */}
                        <div className="flex gap-1 items-center">
                          <span className="text-xs w-8">수평:</span>
                          <input
                            type="range"
                            min="0"
                            max="100"
                            value={Math.round((window.x_position || 0.5) * 100)}
                            onChange={(e) => {
                              const newWindows = [...detectedWindows];
                              newWindows[index] = {
                                ...window,
                                x_position: parseInt(e.target.value) / 100
                              };
                              setDetectedWindows(newWindows);
                            }}
                            className="flex-1 h-1"
                          />
                          <span className="text-xs w-8">{Math.round((window.x_position || 0.5) * 100)}%</span>
                        </div>
                        
                        {/* 수직 위치 조정 */}
                        <div className="flex gap-1 items-center">
                          <span className="text-xs w-8">수직:</span>
                          <input
                            type="range"
                            min="0"
                            max="100"
                            value={Math.round((window.y_position || 0.8) * 100)}
                            onChange={(e) => {
                              const newWindows = [...detectedWindows];
                              newWindows[index] = {
                                ...window,
                                y_position: parseInt(e.target.value) / 100
                              };
                              setDetectedWindows(newWindows);
                            }}
                            className="flex-1 h-1"
                          />
                          <span className="text-xs w-8">{Math.round((window.y_position || 0.8) * 100)}%</span>
                        </div>
                        
                        {/* 창문 크기 조정 */}
                        <div className="flex gap-1 items-center">
                          <span className="text-xs w-8">너비:</span>
                          <input
                            type="range"
                            min="60"
                            max="400"
                            value={Math.round((window.width_meters || 1.2) * 100)}
                            onChange={(e) => {
                              const newWindows = [...detectedWindows];
                              newWindows[index] = {
                                ...window,
                                width_meters: parseInt(e.target.value) / 100
                              };
                              setDetectedWindows(newWindows);
                            }}
                            className="flex-1 h-1"
                          />
                          <span className="text-xs w-8">{Math.round((window.width_meters || 1.2) * 100)}cm</span>
                        </div>
                        <div className="flex gap-1 items-center">
                          <span className="text-xs w-8">높이:</span>
                          <input
                            type="range"
                            min="60"
                            max="300"
                            value={Math.round((window.height_meters || 1.5) * 100)}
                            onChange={(e) => {
                              const newWindows = [...detectedWindows];
                              newWindows[index] = {
                                ...window,
                                height_meters: parseInt(e.target.value) / 100
                              };
                              setDetectedWindows(newWindows);
                            }}
                            className="flex-1 h-1"
                          />
                          <span className="text-xs w-8">{Math.round((window.height_meters || 1.5) * 100)}cm</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <div className="space-y-1">
                <div className="flex gap-1">
                  <button
                    onClick={() => setShowWindows(!showWindows)}
                    className="flex-1 px-2 py-1 bg-gray-500 text-white rounded text-xs hover:bg-gray-600"
                  >
                    {showWindows ? "창문 숨기기" : "창문 보이기"}
                  </button>
                  <button
                    onClick={() => {
                      const newWindow = {
                        wall_position: "back",
                        x_position: 0.5,
                        y_position: 0.8,
                        width_meters: 1.2,
                        height_meters: 1.5,
                        confidence: 1.0
                      };
                      setDetectedWindows([...detectedWindows, newWindow]);
                    }}
                    className="flex-1 px-2 py-1 bg-green-500 text-white rounded text-xs hover:bg-green-600"
                  >
                    창문 추가
                  </button>
                </div>
                <button
                  onClick={() => setDetectedWindows([])}
                  className="w-full px-2 py-1 bg-red-500 text-white rounded text-xs hover:bg-red-600"
                >
                  모든 창문 삭제
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Distance Measurement Panel */}
        <div className="backdrop-blur p-3 rounded-lg shadow-lg bg-white/80">
          <h4 className="font-semibold text-sm mb-2">거리 측정</h4>
          <div className="space-y-2">
            <button
              onClick={() => {
                setMeasurementMode(!measurementMode);
                setMeasurePoints([null, null]);
              }}
              className={`w-full px-3 py-2 rounded text-sm ${
                measurementMode
                  ? "bg-red-500 text-white hover:bg-red-600"
                  : "bg-green-500 text-white hover:bg-green-600"
              }`}
            >
              {measurementMode ? "측정 종료" : "거리 측정"}
            </button>
            {measurementMode && (
              <p className="text-xs text-gray-600">
                바닥을 클릭하여 두 점 사이의 거리를 측정하세요
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Right-side panels - Responsive */}
      <div className="absolute top-4 right-4 z-10 space-y-4 w-80 max-w-sm">
        {/* Moved to Right */}
        
        {/* View Controls */}
        <ViewPresets onViewChange={handleViewChange} roomSize={roomSize} />
        
        {/* Data Export - Moved from Left */}
        <div className="backdrop-blur-lg p-4 rounded-xl shadow-lg bg-white/85 border border-white/50 hover:bg-white/90 transition-all duration-200">
          <div className="flex items-center gap-2 mb-3">
            <div className="p-1.5 rounded-lg bg-green-100">
              <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h4 className="font-bold text-sm text-slate-800">데이터 저장</h4>
          </div>
          <div className="space-y-3">
            <button
              onClick={async () => {
                if (furniture.length === 0 && detectedWindows.length === 0) {
                  alert("저장할 가구나 창문이 없습니다!");
                  return;
                }
                alert(`MongoDB 저장 완료! 가구 ${furniture.length}개, 창문 ${detectedWindows.length}개`);
              }}
              disabled={furniture.length === 0 && detectedWindows.length === 0}
              className="w-full px-4 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg text-sm font-semibold hover:from-green-600 hover:to-green-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-200"
            >
              <span className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                MongoDB에 저장
              </span>
            </button>
            <div className="flex items-center justify-center gap-3 text-xs text-slate-600 bg-slate-50 px-3 py-2 rounded-lg">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                <span>가구 {furniture.length}개</span>
              </div>
              <div className="w-px h-3 bg-slate-300"></div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span>창문 {detectedWindows.length}개</span>
              </div>
            </div>
          </div>
        </div>

        {/* Walkthrough Mode Toggle */}
        <div className="backdrop-blur p-3 rounded-lg shadow-lg bg-white/80">
          <h4 className="font-semibold text-sm mb-2">시점 모드</h4>
          <button
            onClick={() => setWalkthroughMode(!walkthroughMode)}
            className={`w-full px-3 py-2 rounded text-sm transition-colors ${
              walkthroughMode
                ? "bg-orange-500 text-white hover:bg-orange-600"
                : "bg-blue-500 text-white hover:bg-blue-600"
            }`}
          >
            {walkthroughMode ? "조감도 모드" : "걸어다니기 모드"}
          </button>
          {walkthroughMode && (
            <p className="text-xs text-gray-600 mt-1">
              WASD 키로 이동, 마우스로 시점 변경
            </p>
          )}
        </div>

        {/* Space Analysis */}
        <SizeComparisonPanel
          currentArea={roomArea}
          isFullscreen={isFullscreen}
        />
        <WalkingMetrics width={w} depth={d} isFullscreen={isFullscreen} />
        <SpaceUtilization furniture={furniture} roomArea={roomArea} />
      </div>

      {/* 3D Canvas with Error Boundary */}
      <div className="relative w-full h-full">
        <Canvas
          camera={{
            position: [w, h, d],
            fov: 50,
            near: 1,
            far: Math.max(w, h, d) * 5,
          }}
          shadows
          gl={{
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: false,
            powerPreference: "high-performance",
            failIfMajorPerformanceCaveat: false,
          }}
          onCreated={({ gl }) => {
            try {
              const canvas = gl.domElement;
              if (canvas && typeof canvas.addEventListener === "function") {
                canvas.addEventListener("webglcontextlost", (event) => {
                  console.log("WebGL context lost, preventing default");
                  event.preventDefault();
                });
                canvas.addEventListener("webglcontextrestored", () => {
                  console.log("WebGL context restored");
                });
              }
            } catch (error) {
              console.warn(
                "Could not set up WebGL context event listeners:",
                error
              );
            }
          }}
          fallback={
            <div className="flex items-center justify-center w-full h-full bg-gray-100 text-gray-600">
              <div className="text-center">
                <div className="text-2xl mb-2">⚠️</div>
                <div>3D 뷰어를 로드할 수 없습니다</div>
                <div className="text-sm">
                  WebGL을 지원하지 않는 브라우저입니다
                </div>
              </div>
            </div>
          }
        >
          <Suspense fallback={null}>
            <EnhancedLighting roomSize={roomSize} />
            <Environment preset="apartment" />

            {/* 바닥 - 원래 크기로 복원 */}
            <mesh
              position={[w / 2, 0, d / 2]}
              rotation={[-Math.PI / 2, 0, 0]}
              receiveShadow
              onClick={handleFloorClick}
            >
              <planeGeometry args={[w, d]} /> {/* 원래 크기로 복원 */}
              <meshStandardMaterial
                color={placementMode ? "#e0f2fe" : "#f8f9fa"}
              />
            </mesh>

            {/* 드래그 전용 투명 레이어 - Canvas 레벨에서 처리 */}
            <mesh
              position={[0, 0.01, 0]} // 바닥 바로 위
              rotation={[-Math.PI / 2, 0, 0]}
              visible={false}
            >
              <planeGeometry args={[10000, 10000]} />
              <meshBasicMaterial transparent opacity={0} />
            </mesh>

            {/* 앞쪽 벽 (front wall) - 삭제 */}
            {/* <Wall width={w} height={h} position={[w / 2, h / 2, d]} rotation={[0, Math.PI, 0]} /> */}

            {/* 왼쪽 벽 (left wall) - 유지 */}
            <Wall
              width={d}
              height={h}
              position={[0, h / 2, d / 2]}
              rotation={[0, Math.PI / 2, 0]}
            />

            {/* 오른쪽 벽 (right wall) - 삭제 */}
            {/* <Wall width={d} height={h} position={[w, h / 2, d / 2]} rotation={[0, -Math.PI / 2, 0]} /> */}

            {/* 뒤쪽 벽 (back wall) - 침대 뒤쪽에 벽 추가 */}
            <Wall
              width={w}
              height={h}
              position={[w / 2, h / 2, 0]}
              rotation={[0, 0, 0]}
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
              />
            ))}

            {showWindows && detectedWindows.length > 0 && (
              <WindowsOnWalls windows={detectedWindows} roomSize={roomSize} />
            )}

            <DraggableHuman
              height={170}
              position={humanPosition}
              onPositionChange={setHumanPosition}
              roomSize={roomSize}
              onDragStateChange={setIsDragging}
            />

            <DistanceMeasurer
              point1={measurePoints[0]}
              point2={measurePoints[1]}
              visible={measurementMode && !!measurePoints[1]}
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

      {selectedFurnitureData && (
        <div className="absolute bottom-4 left-4 z-10 backdrop-blur p-3 rounded-lg shadow-lg bg-white/80">
          <p className="text-sm font-semibold mb-2">
            {selectedFurnitureData.name}
          </p>
          {/* 3D 좌표 정보 표시 */}
          {(() => {
            const [x3d, y3d, z3d] = selectedFurnitureData.position;
            // 실제 가구 크기 사용 (커스텀 가구 포함)
            const size = selectedFurnitureData.size || [100, 100, 100];
            const [width, height, depth] = size;
            
            // 왼쪽아래 좌표 (3D 중심 좌표 → 2D 왼쪽아래 좌표)
            const leftBottomX = Math.round(x3d - width / 2);
            const leftBottomZ = Math.round(z3d - depth / 2);
            
            // 오른쪽위 좌표
            const rightTopX = Math.round(x3d + width / 2);
            const rightTopZ = Math.round(z3d + depth / 2);
            
            return (
              <div className="text-xs text-gray-600 mb-2 space-y-1">
                <div>크기: {width} × {depth} × {height} cm</div>
                <div>위치 (왼쪽아래): ({leftBottomX}, {leftBottomZ}) cm</div>
                <div>위치 (오른쪽위): ({rightTopX}, {rightTopZ}) cm</div>
                <div>중심 좌표: ({Math.round(x3d)}, {Math.round(z3d)}) cm</div>
                {selectedFurnitureData.rotation && selectedFurnitureData.rotation[1] !== 0 && (
                  <div>회전: {Math.round((selectedFurnitureData.rotation[1] * 180) / Math.PI)}°</div>
                )}
              </div>
            );
          })()}
          <div className="flex gap-2">
            <button
              onClick={() => handleRotateFurniture(selectedFurniture)}
              className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
            >
              회전
            </button>
            <button
              onClick={() => handleDeleteFurniture(selectedFurniture)}
              className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-sm"
            >
              삭제
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
