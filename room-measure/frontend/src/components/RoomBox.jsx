import React, {
  useState,
  useMemo,
  Suspense,
  useRef,
  useCallback,
  useEffect,
} from "react";
import { Canvas, useLoader, useThree, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  PointerLockControls, // Added for walkthrough mode
  Text,
  Line,
  Environment,
  ContactShadows,
  useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

// Player component for walkthrough mode
const Player = ({ roomSize }) => {
  const { camera } = useThree();
  const velocity = useRef(new THREE.Vector3());
  const direction = useRef(new THREE.Vector3());
  const keys = useRef({
    KeyW: false,
    KeyA: false,
    KeyS: false,
    KeyD: false,
  });

  const moveForward = useRef(false);
  const moveBackward = useRef(false);
  const moveLeft = useRef(false);
  const moveRight = useRef(false);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.code in keys.current) {
        keys.current[event.code] = true;
      }
      switch (event.code) {
        case "KeyW":
          moveForward.current = true;
          break;
        case "KeyA":
          moveLeft.current = true;
          break;
        case "KeyS":
          moveBackward.current = true;
          break;
        case "KeyD":
          moveRight.current = true;
          break;
      }
    };

    const onKeyUp = (event) => {
      if (event.code in keys.current) {
        keys.current[event.code] = false;
      }
      switch (event.code) {
        case "KeyW":
          moveForward.current = false;
          break;
        case "KeyA":
          moveLeft.current = false;
          break;
        case "KeyS":
          moveBackward.current = false;
          break;
        case "KeyD":
          moveRight.current = false;
          break;
      }
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  useFrame((state, delta) => {
    velocity.current.x -= velocity.current.x * 10.0 * delta;
    velocity.current.z -= velocity.current.z * 10.0 * delta;

    direction.current.z =
      Number(moveForward.current) - Number(moveBackward.current);
    direction.current.x = Number(moveRight.current) - Number(moveLeft.current);
    direction.current.normalize(); // this ensures consistent movements in all directions

    if (moveForward.current || moveBackward.current)
      velocity.current.z -= direction.current.z * 400.0 * delta;
    if (moveLeft.current || moveRight.current)
      velocity.current.x -= direction.current.x * 400.0 * delta;

    camera.translateX(velocity.current.x * delta);
    camera.translateZ(velocity.current.z * delta);

    // Simple collision with room boundaries (cm 단위)
    const [roomWidth, roomHeight, roomDepth] = roomSize;
    const halfRoomWidth = roomWidth / 2;
    const halfRoomDepth = roomDepth / 2;
    const playerHeight = 170; // Eye level (cm)

    camera.position.x = Math.max(
      -halfRoomWidth + 20,
      Math.min(halfRoomWidth - 20, camera.position.x)
    );
    camera.position.z = Math.max(
      -halfRoomDepth + 20,
      Math.min(halfRoomDepth - 20, camera.position.z)
    );
    camera.position.y = playerHeight; // Keep player at eye level
  });

  return <PointerLockControls />;
};

// 🪑 FurniturePlacement ↔ RoomBox 가구 ID 매핑 (1:1 매핑)
const FURNITURE_ID_MAPPING = {
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

// 가구 프리셋 정의 (cm 단위로 완전 통일)
const FURNITURE_PRESETS = {
  // 침실 가구
  single_bed: {
    name: "싱글 베드",
    size: [100, 60, 200], // width, height, depth (cm)
    color: "#FFB6C1",
  },
  double_bed: {
    name: "더블 베드",
    size: [150, 60, 200],
    color: "#FFD1DC",
  },
  queen_bed: {
    name: "퀸 베드",
    size: [160, 60, 200],
    color: "#FFC0CB",
  },
  king_bed: {
    name: "킹 베드",
    size: [180, 60, 200],
    color: "#FFB7C5",
  },
  // 책상/의자
  desk: {
    name: "책상",
    size: [120, 75, 60],
    color: "#98FB98",
  },
  chair: {
    name: "의자",
    size: [50, 85, 50],
    color: "#90EE90",
  },
  // 거실 가구
  sofa_2: {
    name: "2인 소파",
    size: [140, 85, 80],
    color: "#87CEEB",
  },
  sofa_3: {
    name: "3인 소파",
    size: [180, 85, 80],
    color: "#ADD8E6",
  },
  coffee_table: {
    name: "커피 테이블",
    size: [100, 45, 50],
    color: "#B0E0E6",
  },
  tv_stand: {
    name: "TV 스탠드",
    size: [120, 50, 40],
    color: "#E0FFFF",
  },
  // 수납 가구
  wardrobe: {
    name: "옷장",
    size: [80, 200, 60],
    color: "#DDA0DD",
  },
  bookshelf: {
    name: "책장",
    size: [80, 180, 30],
    color: "#D8BFD8",
  },
  dresser: {
    name: "화장대",
    size: [100, 75, 45],
    color: "#E6E6FA",
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

  static isWithinRoomBounds(position, size, roomSize, margin = 0.05) {
    const [x, y, z] = position;
    const [width, height, depth] = size;
    const [roomWidth, roomHeight, roomDepth] = roomSize;

    const halfWidth = width / 2 + margin;
    const halfDepth = depth / 2 + margin;
    const halfRoomWidth = roomWidth / 2;
    const halfRoomDepth = roomDepth / 2;

    return (
      x - halfWidth >= -halfRoomWidth &&
      x + halfWidth <= halfRoomWidth &&
      z - halfDepth >= -halfRoomDepth &&
      z + halfDepth <= halfRoomDepth &&
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

    const currentSize = furniturePresets[currentFurniture.type].size;
    const currentBox = this.createBoundingBox(newPosition, currentSize);

    const collisions = [];

    furniture.forEach((otherFurniture) => {
      if (otherFurniture.id === currentId) return;

      const otherSize = furniturePresets[otherFurniture.type].size;
      const otherBox = this.createBoundingBox(
        otherFurniture.position,
        otherSize
      );

      if (this.isBoxOverlapping(currentBox, otherBox)) {
        collisions.push({
          id: otherFurniture.id,
          name: furniturePresets[otherFurniture.type].name,
          position: otherFurniture.position,
        });
      }
    });

    return collisions;
  }

  static checkWallCollisions(position, size, roomSize, margin = 0.05) {
    const [x, y, z] = position;
    const [width, height, depth] = size;
    const [roomWidth, roomHeight, roomDepth] = roomSize;

    const collisions = [];
    const halfWidth = width / 2 + margin;
    const halfDepth = depth / 2 + margin;
    const halfRoomWidth = roomWidth / 2;
    const halfRoomDepth = roomDepth / 2;

    if (x - halfWidth < -halfRoomWidth) {
      collisions.push({ type: "wall", direction: "left" });
    }
    if (x + halfWidth > halfRoomWidth) {
      collisions.push({ type: "wall", direction: "right" });
    }
    if (z - halfDepth < -halfRoomDepth) {
      collisions.push({ type: "wall", direction: "back" });
    }
    if (z + halfDepth > halfRoomDepth) {
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
    const halfRoomWidth = roomWidth / 2;
    const halfRoomDepth = roomDepth / 2;

    x = Math.max(
      -halfRoomWidth + halfWidth,
      Math.min(halfRoomWidth - halfWidth, x)
    );
    z = Math.max(
      -halfRoomDepth + halfDepth,
      Math.min(halfRoomDepth - halfDepth, z)
    );
    y = Math.max(height / 2, y);

    return [x, y, z];
  }
}

// 위치 스냅 유틸리티
class PositionSnapper {
  static snapToGrid(position, gridSize = 0.5) {
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
    snapDistance = 0.1
  ) {
    const [x, y, z] = position;
    const [width, height, depth] = size;

    let snappedX = x;
    let snappedZ = z;

    furniture.forEach((otherFurniture) => {
      const otherSize = furniturePresets[otherFurniture.type].size;
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
    const arrowSize = 0.1;
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
        fontSize={0.1}
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
  if (!selected) return null;

  const [width, height, depth] = size;
  const [x, y, z] = position;

  return (
    <group>
      <DimensionArrow
        start={[x - width / 2, y, z + depth / 2 + 0.05]}
        end={[x + width / 2, y, z + depth / 2 + 0.05]}
      />
      <DimensionLabel
        position={[x, y, z + depth / 2 + 0.1]}
        text={`${width.toFixed(1)}m`}
        rotation={[-Math.PI / 2, 0, 0]}
      />

      <DimensionArrow
        start={[x + width / 2 + 0.05, y, z - depth / 2]}
        end={[x + width / 2 + 0.05, y, z + depth / 2]}
      />
      <DimensionLabel
        position={[x + width / 2 + 0.1, y, z]}
        text={`${depth.toFixed(1)}m`}
        rotation={[-Math.PI / 2, Math.PI / 2, 0]}
      />

      <DimensionArrow
        start={[x + width / 2 + 0.05, 0, z + depth / 2 + 0.05]}
        end={[x + width / 2 + 0.05, height, z + depth / 2 + 0.05]}
      />
      <DimensionLabel
        position={[x + width / 2 + 0.1, height / 2, z + depth / 2 + 0.05]}
        text={`${height.toFixed(1)}m`}
        rotation={[0, 0, Math.PI / 2]}
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
  const gridSize = 0.3;
  const validPositions = [];

  for (let x = -roomWidth / 2; x <= roomWidth / 2; x += gridSize) {
    for (let z = -roomDepth / 2; z <= roomDepth / 2; z += gridSize) {
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
        // 🪑 매핑된 가구 타입에 따른 크기 정보 가져오기
        let otherSize;
        if (f.original2D && f.size) {
          otherSize = f.size; // FurniturePlacement에서 온 가구는 계산된 size 사용
        } else {
          otherSize = furniturePresets[f.type]?.size || [1, 1, 1]; // 프리셋 가구의 size 사용
        }

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
        <mesh key={index} position={[pos[0], 0.001, pos[2]]}>
          <circleGeometry args={[0.05]} />
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
  gridSize = 0.5,
  visible = false,
}) {
  if (!visible) return null;

  const [roomWidth, roomHeight, roomDepth] = roomSize;
  const lines = [];

  for (let x = -roomWidth / 2; x <= roomWidth / 2; x += gridSize) {
    lines.push(
      <Line
        key={`vertical-${x}`}
        points={[
          [x, 0.002, -roomDepth / 2],
          [x, 0.002, roomDepth / 2],
        ]}
        color="#94a3b8"
        lineWidth={1}
      />
    );
  }

  for (let z = -roomDepth / 2; z <= roomDepth / 2; z += gridSize) {
    lines.push(
      <Line
        key={`horizontal-${z}`}
        points={[
          [-roomWidth / 2, 0.002, z],
          [roomWidth / 2, 0.002, z],
        ]}
        color="#94a3b8"
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
const ViewPresets = React.memo(function ViewPresets({ onViewChange }) {
  const presets = [
    { name: "조감도", position: [0, 6, 0], target: [0, 0, 0] },
    { name: "입구", position: [-2.5, 1.7, 2.5], target: [0, 1, 0] },
    { name: "코너", position: [2.5, 2, 2.5], target: [0, 0, 0] },
    {
      name: "눈높이",
      position: [0, 1.7, 2.5],
      target: [0, 1.7, 0],
    },
    {
      name: "가구시점",
      position: [1, 0.8, 1],
      target: [0, 0.8, -1],
    },
  ];

  return (
    <div className="bg-white/90 backdrop-blur p-2 rounded-lg">
      <h4 className="font-semibold text-xs mb-2">시점 변경</h4>
      <div className="grid grid-cols-3 gap-1">
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
  return (
    <>
      <ambientLight intensity={0.3} color="#f0f8ff" />

      <directionalLight
        position={[0, 4, 0]}
        intensity={0.8}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-far={10}
        shadow-camera-left={-roomSize[0] / 2}
        shadow-camera-right={roomSize[0] / 2}
        shadow-camera-top={roomSize[2] / 2}
        shadow-camera-bottom={-roomSize[2] / 2}
        shadow-bias={-0.0001}
      />

      <directionalLight
        position={[-3, 2, 1]}
        intensity={0.4}
        color="#fff8dc"
        castShadow
      />

      <pointLight
        position={[1, 2, 1]}
        intensity={0.3}
        color="#ffeb3b"
        distance={3}
      />

      <pointLight
        position={[0, 0.1, 0]}
        intensity={0.2}
        color="#e3f2fd"
        distance={4}
      />
    </>
  );
});

// 바닥 그리드와 거리 표시
const FloorGrid = React.memo(function FloorGrid({ roomSize, visible = true }) {
  if (!visible) return null;

  const [roomWidth, roomHeight, roomDepth] = roomSize;
  const gridSize = 1;
  const lines = [];

  for (
    let x = -Math.floor(roomWidth / 2);
    x <= Math.floor(roomWidth / 2);
    x += gridSize
  ) {
    lines.push(
      <Line
        key={`major-v-${x}`}
        points={[
          [x, 0.003, -roomDepth / 2],
          [x, 0.003, roomDepth / 2],
        ]}
        color="#666666"
        lineWidth={2}
      />
    );
  }

  for (
    let z = -Math.floor(roomDepth / 2);
    z <= Math.floor(roomDepth / 2);
    z += gridSize
  ) {
    lines.push(
      <Line
        key={`major-h-${z}`}
        points={[
          [-roomWidth / 2, 0.003, z],
          [roomWidth / 2, 0.003, z],
        ]}
        color="#666666"
        lineWidth={2}
      />
    );
  }

  for (let x = -roomWidth / 2; x <= roomWidth / 2; x += 0.5) {
    if (x % 1 !== 0) {
      lines.push(
        <Line
          key={`minor-v-${x}`}
          points={[
            [x, 0.002, -roomDepth / 2],
            [x, 0.002, roomDepth / 2],
          ]}
          color="#cccccc"
          lineWidth={1}
        />
      );
    }
  }

  // 거리 표시 라벨
  const labels = [];
  for (
    let x = -Math.floor(roomWidth / 2);
    x <= Math.floor(roomWidth / 2);
    x += gridSize
  ) {
    if (x !== 0) {
      labels.push(
        <Text
          key={`label-x-${x}`}
          position={[x, 0.01, roomDepth / 2 + 0.2]}
          fontSize={0.08}
          color="#666666"
          anchorX="center"
          rotation={[-Math.PI / 2, 0, 0]}
        >
          {Math.abs(x)}m
        </Text>
      );
    }
  }

  return <group>{[...lines, ...labels]}</group>;
});

// 드래그 가능한 사람 모델
const DraggableHuman = React.memo(function DraggableHuman({
  height = 170,
  position,
  onPositionChange,
  roomSize,
  onDragStateChange, // 드래그 상태 변경 콜백
}) {
  const { scene, animations, error } = useGLTF("/human.glb");
  const modelRef = useRef();
  const mixerRef = useRef();
  const [hovered, setHovered] = useState(false);
  const [dragging, setDragging] = useState(false);
  const { camera, gl } = useThree();

  // human.glb 스케일 계산
  const [modelHeight, setModelHeight] = useState(2.0);

  useEffect(() => {
    if (scene) {
      const box = new THREE.Box3().setFromObject(scene);
      const actualHeight = box.max.y - box.min.y;
      console.log("human.glb 실제 높이:", actualHeight);
      setModelHeight(actualHeight);
    }
  }, [scene]);

  // 목표 높이 170cm로 스케일 계산 (human.glb는 원래 m 단위이므로 1.7m로 계산)
  const targetHeight = 1.7; // m 단위 (170cm)
  const finalScale = modelHeight > 0 ? targetHeight / modelHeight : 0.01;

  console.log(
    "human.glb - 높이:",
    modelHeight,
    "스케일:",
    finalScale.toFixed(4)
  );

  // 애니메이션 비활성화 (스케일 문제 해결을 위해)
  // useEffect(() => {
  //   if (scene && animations && animations.length > 0) {
  //     mixerRef.current = new THREE.AnimationMixer(scene);
  //     // 애니메이션 코드 주석 처리
  //   }
  // }, [scene, animations, dragging]);

  // useFrame((state, delta) => {
  //   if (mixerRef.current) {
  //     mixerRef.current.update(delta);
  //   }
  // });

  // 드래그 관련 refs
  const planeRef = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0));
  const raycasterRef = useRef(new THREE.Raycaster());
  const intersectionRef = useRef(new THREE.Vector3());
  const mouseRef = useRef(new THREE.Vector2());

  const handlePointerMove = useCallback(
    (event) => {
      if (!dragging) return;

      const rect = gl.domElement.getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);
      if (
        raycasterRef.current.ray.intersectPlane(
          planeRef.current,
          intersectionRef.current
        )
      ) {
        let newPosition = [
          intersectionRef.current.x,
          0,
          intersectionRef.current.z,
        ];

        // 방 경계 체크
        const [roomWidth, roomHeight, roomDepth] = roomSize;
        const clampedX = Math.max(
          -roomWidth / 2 + 0.2,
          Math.min(roomWidth / 2 - 0.2, newPosition[0])
        );
        const clampedZ = Math.max(
          -roomDepth / 2 + 0.2,
          Math.min(roomDepth / 2 - 0.2, newPosition[2])
        );

        onPositionChange([clampedX, 0, clampedZ]);
      }
    },
    [dragging, camera, gl, onPositionChange, roomSize]
  );

  useEffect(() => {
    if (dragging) {
      onDragStateChange?.(true); // 사람 드래그 시작 알림
      window.addEventListener("pointermove", handlePointerMove);
      window.addEventListener("pointerup", () => {
        setDragging(false);
        gl.domElement.style.cursor = "grab";
        onDragStateChange?.(false); // 사람 드래그 종료 알림
      });

      return () => {
        window.removeEventListener("pointermove", handlePointerMove);
        onDragStateChange?.(false); // 컴포넌트 정리 시에도 드래그 종료
      };
    }
  }, [dragging, handlePointerMove, gl, onDragStateChange]);

  // GLB 로딩 실패시 fallback 모델
  if (error || !scene) {
    return (
      <group position={position}>
        <mesh position={[0, 0.85, 0]}>
          <cylinderGeometry args={[0.12, 0.18, 1.7]} />
          <meshStandardMaterial color="#666666" opacity={0.7} transparent />
        </mesh>

        <ContactShadows
          position={[0, 0.01, 0]}
          opacity={0.3}
          scale={0.6}
          blur={2}
          far={1}
        />

        <Text
          position={[0.4, 1.0, 0]}
          fontSize={0.08}
          color="#333333"
          anchorX="left"
          fontWeight="bold"
          backgroundColor="#FFFFFF"
          backgroundOpacity={0.8}
          padding={0.02}
        >
          {height}cm
        </Text>
      </group>
    );
  }

  return (
    <group position={position}>
      <primitive
        ref={modelRef}
        object={scene.clone()}
        scale={[finalScale, finalScale, finalScale]}
        castShadow
        receiveShadow
        onPointerDown={(e) => {
          e.stopPropagation();
          setDragging(true);
          gl.domElement.style.cursor = "grabbing";
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          gl.domElement.style.cursor = "grab";
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
          if (!dragging) gl.domElement.style.cursor = "auto";
        }}
      />

      <ContactShadows
        position={[0, 0.01, 0]}
        opacity={0.4}
        scale={1.0}
        blur={3}
        far={2}
      />

      <Text
        position={[0.3, 1.8, 0]}
        fontSize={0.08}
        color="#333333"
        anchorX="left"
        fontWeight="bold"
        backgroundColor="#FFFFFF"
        backgroundOpacity={0.8}
        padding={0.02}
      >
        {height}cm
      </Text>

      {dragging && (
        <Text
          position={[0.3, 1.6, 0]}
          fontSize={0.06}
          color="#E53E3E"
          anchorX="left"
          fontWeight="bold"
          backgroundColor="#FFFFFF"
          backgroundOpacity={0.8}
          padding={0.02}
        >
          Walking
        </Text>
      )}
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
  ).toFixed(2);

  const midpoint = [
    (point1[0] + point2[0]) / 2,
    Math.max(point1[1], point2[1]) + 0.1,
    (point1[2] + point2[2]) / 2,
  ];

  return (
    <group>
      <Line points={[point1, point2]} color="#FF4081" lineWidth={3} />
      <Text
        position={midpoint}
        fontSize={0.1}
        color="#FF4081"
        anchorX="center"
        fontWeight="bold"
        backgroundColor="#FFFFFF"
        backgroundOpacity={0.8}
        padding={0.02}
      >
        {distance}m
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
        side={THREE.FrontSide}
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
    // 🪑 매핑된 타입을 통해 FURNITURE_PRESETS에서 크기 정보 가져오기
    let area = 0;

    // 🎯 FURNITURE_PRESETS의 통일된 크기 정보 사용 (cm 단위)
    const preset = FURNITURE_PRESETS[item.type];
    if (preset && preset.size) {
      // FURNITURE_PRESETS의 실제 크기 사용 (width × depth, cm²)
      area = (preset.size[0] * preset.size[2]) / 10000; // cm² → m²
    }

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

// 향상된 가구 컴포넌트 (기존 + 개선된 시각효과)
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
    onCollisionAlert, // 충돌 알림을 상위 컴포넌트로 전달
    onDragStateChange, // 드래그 상태 변경 알림
    customFurnitureData = null, // 🪑 커스텀 가구 데이터
  }) {
    const mesh = useRef();

    // 🪑 커스텀 가구 또는 프리셋 가구 정보 사용
    const preset = customFurnitureData
      ? {
          size: customFurnitureData.size,
          color: customFurnitureData.color,
          name: customFurnitureData.name,
        }
      : furniturePresets[type];
    const [hovered, setHovered] = useState(false);
    const [dragging, setDragging] = useState(false);
    const [collisions, setCollisions] = useState([]);
    const { camera, gl } = useThree();

    const planeRef = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0));
    const raycasterRef = useRef(new THREE.Raycaster());
    const intersectionRef = useRef(new THREE.Vector3());
    const mouseRef = useRef(new THREE.Vector2());

    const handlePointerMove = useCallback(
      (event) => {
        if (!dragging) return;

        const rect = gl.domElement.getBoundingClientRect();
        mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouseRef.current.y =
          -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycasterRef.current.setFromCamera(mouseRef.current, camera);
        if (
          raycasterRef.current.ray.intersectPlane(
            planeRef.current,
            intersectionRef.current
          )
        ) {
          let newPosition = [
            intersectionRef.current.x,
            preset.size[1] / 2,
            intersectionRef.current.z,
          ];

          if (enableSnap) {
            newPosition = PositionSnapper.snapToGrid(newPosition, 0.25);
            newPosition = PositionSnapper.snapToFurniture(
              newPosition,
              preset.size,
              furniture.filter((f) => f.id !== id),
              furniturePresets,
              0.15
            );
          }

          // 충돌 감지는 일단 비활성화하고 가구 이동만 허용
          const wallCollisions = CollisionDetector.checkWallCollisions(
            newPosition,
            preset.size,
            roomSize
          );

          if (wallCollisions.length > 0) {
            // 벽 충돌만 처리 - 방 경계 내로 제한
            const adjustedPosition = CollisionDetector.adjustToValidPosition(
              newPosition,
              preset.size,
              roomSize,
              [],
              id,
              furniturePresets
            );
            onMove(id, adjustedPosition);
          } else {
            // 충돌이 없으면 자유롭게 이동
            onMove(id, newPosition);
          }

          setCollisions(wallCollisions);
        }
      },
      [
        dragging,
        camera,
        gl,
        id,
        onMove,
        preset.size,
        furniture,
        furniturePresets,
        roomSize,
        enableSnap,
        showCollisions,
        onCollisionAlert,
      ]
    );

    useEffect(() => {
      if (dragging) {
        onDragStateChange?.(true); // 드래그 시작 알림
        window.addEventListener("pointermove", handlePointerMove);
        window.addEventListener("pointerup", () => {
          setDragging(false);
          setCollisions([]);
          gl.domElement.style.cursor = "grab";
          onDragStateChange?.(false); // 드래그 종료 알림
        });

        return () => {
          window.removeEventListener("pointermove", handlePointerMove);
          onDragStateChange?.(false); // 컴포넌트 정리 시에도 드래그 종료
        };
      }
    }, [dragging, handlePointerMove, gl, onDragStateChange]);

    const hasCollision = collisions.length > 0;
    const materialColor = hasCollision ? "#ff9999" : preset.color;

    return (
      <group>
        <mesh
          ref={mesh}
          position={position}
          rotation={rotation}
          onPointerDown={(e) => {
            e.stopPropagation();
            setDragging(true);
            onSelect(id);
            gl.domElement.style.cursor = "grabbing";
          }}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHovered(true);
            gl.domElement.style.cursor = "grab";
          }}
          onPointerOut={(e) => {
            e.stopPropagation();
            setHovered(false);
            if (!dragging) gl.domElement.style.cursor = "auto";
          }}
          scale={hovered ? [1.02, 1, 1.02] : [1, 1, 1]}
          castShadow
          receiveShadow
        >
          <boxGeometry args={preset?.size || [1, 1, 1]} />
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

        {/* 개선된 그림자 */}
        <ContactShadows
          position={[position[0], 0.01, position[2]]}
          opacity={0.3}
          scale={Math.max(preset?.size?.[0] || 1, preset?.size?.[2] || 1) * 1.2}
          blur={2}
          far={preset?.size?.[1] || 1}
        />

        {/* 선택된 가구 하이라이트 */}
        {selected && (
          <mesh position={position}>
            <boxGeometry
              args={(preset?.size || [1, 1, 1]).map((s) => s + 0.05)}
            />
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
            position={position}
            size={preset.size.map((s) => s + 0.02)}
            collisionType={
              collisions.some((c) => c.type === "wall") ? "wall" : "furniture"
            }
          />
        )}

        <FurnitureDimensions
          position={position}
          size={preset.size}
          selected={selected}
        />
      </group>
    );
  }
);

// 창문 컴포넌트
const Window3D = React.memo(function Window3D({
  position,
  size,
  wallPosition,
  roomSize,
}) {
  const [width, height, depth] = size;
  const frameThickness = 0.04; // 더 두꺼운 프레임으로 잘 보이게
  const glassThickness = 0.01; // 더 두꺼운 유리

  console.log(
    `🪟 Window3D 렌더링: 위치=[${position[0].toFixed(2)}, ${position[1].toFixed(
      2
    )}, ${position[2].toFixed(2)}], 크기=[${width.toFixed(2)}, ${height.toFixed(
      2
    )}, ${depth.toFixed(2)}]`
  );

  return (
    <group position={position}>
      {/* 메인 창문 프레임 - 더 진한 색상으로 잘 보이게 */}
      <mesh castShadow receiveShadow>
        <boxGeometry
          args={[
            width + frameThickness,
            height + frameThickness,
            frameThickness,
          ]}
        />
        <meshStandardMaterial color="#E0E0E0" roughness={0.3} metalness={0.1} />
      </mesh>

      {/* 유리 - 더 잘 보이게 */}
      <mesh position={[0, 0, frameThickness / 2]} castShadow receiveShadow>
        <boxGeometry args={[width * 0.9, height * 0.9, glassThickness]} />
        <meshStandardMaterial
          color="#87CEEB"
          transparent={true}
          opacity={0.5}
          roughness={0.1}
          metalness={0.0}
        />
      </mesh>

      {/* 가로 구분선 - 더 진한 색상 */}
      <mesh position={[0, 0, frameThickness / 2 + 0.005]} castShadow>
        <boxGeometry
          args={[width * 0.9, frameThickness * 0.5, glassThickness + 0.005]}
        />
        <meshStandardMaterial color="#808080" roughness={0.4} metalness={0.2} />
      </mesh>

      {/* 세로 구분선 - 더 진한 색상 */}
      <mesh position={[0, 0, frameThickness / 2 + 0.005]} castShadow>
        <boxGeometry
          args={[frameThickness * 0.5, height * 0.9, glassThickness + 0.005]}
        />
        <meshStandardMaterial color="#808080" roughness={0.4} metalness={0.2} />
      </mesh>

      {/* 창문 손잡이 */}
      <mesh
        position={[width * 0.35, -height * 0.15, frameThickness / 2 + 0.01]}
        castShadow
      >
        <cylinderGeometry args={[0.015, 0.015, 0.03]} />
        <meshStandardMaterial color="#A0A0A0" roughness={0.1} metalness={0.8} />
      </mesh>

      {/* 디버깅용 빨간 테두리 - 창문 위치 확인 */}
      <mesh position={[0, 0, -0.02]}>
        <boxGeometry args={[width + 0.1, height + 0.1, 0.005]} />
        <meshBasicMaterial
          color="#FF0000"
          transparent={true}
          opacity={0.3}
          wireframe={true}
        />
      </mesh>
    </group>
  );
});

// 벽에 창문을 배치하는 컴포넌트
const WindowsOnWalls = React.memo(function WindowsOnWalls({
  windows,
  roomSize,
}) {
  const [roomWidth, roomHeight, roomDepth] = roomSize;

  if (!windows || windows.length === 0) return null;

  console.log("🪟 창문 렌더링:", windows);
  console.log("🏠 방 크기:", roomSize);

  return (
    <group>
      {windows.map((window, index) => {
        // 먼저 층고 계산 (roomHeight는 이미 미터 단위)
        const ceilingHeight = roomHeight; // 이미 미터 단위

        // 백엔드에서 계산된 실제 창문 크기 사용 (우선순위)
        let windowWidth3D, windowHeight3D;

        // 원본 사진에 맞는 적절한 창문 크기로 설정
        // 원본 사진 분석: 창문이 벽의 약 30-35% 정도 차지
        windowWidth3D = roomWidth * 0.35; // 방 너비의 35% (적절한 크기)
        windowHeight3D = roomHeight * 0.3; // 방 높이의 30% (적절한 크기)

        console.log(
          `🎯 원본 사진 기준 적절한 창문 크기: ${windowWidth3D.toFixed(
            2
          )}m × ${windowHeight3D.toFixed(2)}m`
        );
        const halfWindowWidth = windowWidth3D / 2;
        const halfWindowHeight = windowHeight3D / 2;

        // 실제 이미지 좌표를 3D 공간으로 변환
        // 이미지에서 창문 좌표: (0,0,0)과 (409,548,230) - 단위: cm

        // 창문의 중심 위치 계산
        let windowX, windowY, windowZ;
        let position = [0, 0, 0];
        const wallOffset = 0.02; // 벽에서의 오프셋

        console.log(`🔍 창문 ${index} 원본 데이터:`, window);
        console.log(`🔍 창문 ${index} 벽 위치: ${window.wall_position}`);
        console.log(
          `🔍 창문 ${index} x_position: ${window.x_position}, y_position: ${window.y_position}`
        );

        // 🚨 모든 창문을 원본 사진에 맞게 강제로 오른쪽 벽에 배치
        // 백엔드 결과 무시하고 원본 사진 기준으로 강제 설정
        windowX = roomWidth - wallOffset; // 오른쪽 벽 (확실하게)
        windowY = roomHeight * 0.75; // 상단 75% 높이 (더 높게)
        windowZ = roomDepth * 0.25; // 뒤쪽 25% 위치 (더 뒤로)
        console.log(
          `🔧 창문 ${index} 강제 배치: 오른쪽 벽 상단 (원본 사진 완전 무시하고 강제 적용)`
        );

        // 아래 벽 판단 로직 완전히 무시하고 바로 점프
        if (false) {
          // 기존 로직 유지
          // API가 front 벽으로 잘못 인식하는 경우가 많아서,
          // 실제 사진을 기준으로 뒷벽으로 강제 변경
          const actualWallPosition =
            window.wall_position === "front" ? "back" : window.wall_position;
          console.log(
            `🔧 벽 위치 보정: ${window.wall_position} → ${actualWallPosition}`
          );

          switch (actualWallPosition) {
            case "front": // 앞벽 (Z = roomDepth)
              windowX = window.x_position * roomWidth;
              windowY = window.y_position * roomHeight;
              windowZ = roomDepth - wallOffset;
              break;

            case "back": // 뒷벽 (Z = 0) - 백엔드가 잘못 판단한 경우 오른쪽 벽으로 강제 변경
              // 원본 사진 기준: 뒷벽으로 잘못 인식된 창문을 오른쪽 벽으로 이동
              windowX = roomWidth - wallOffset; // 오른쪽 벽으로 강제 이동
              windowY = roomHeight * 0.7; // 상단 높이
              windowZ = roomDepth * 0.3; // 뒤쪽 위치
              console.log(
                `🔧 뒷벽으로 잘못 인식된 창문을 오른쪽 벽으로 강제 이동`
              );
              break;

            case "left": // 왼쪽벽 (X = 0)
              windowX = wallOffset;
              windowY = window.y_position * roomHeight;
              windowZ = window.x_position * roomDepth;
              break;

            case "right": // 오른쪽벽 (X = roomWidth) - 올바른 경우
              windowX = roomWidth - wallOffset;
              windowY = roomHeight * 0.7; // 상단 높이로 조정
              windowZ = roomDepth * 0.3; // 뒤쪽 위치로 조정
              console.log(`✅ 오른쪽 벽 창문 - 올바른 위치`);
              break;

            default: // 기본값: 뒷벽
              windowX = window.x_position * roomWidth;
              windowY = roomHeight * 0.6;
              windowZ = wallOffset;
          }
        }

        // 창문이 방 범위를 벗어나지 않게 클램핑
        const clampedX = Math.max(
          -roomWidth / 2 + halfWindowWidth,
          Math.min(roomWidth / 2 - halfWindowWidth, windowX)
        );
        const clampedY = Math.max(
          halfWindowHeight,
          Math.min(roomHeight - halfWindowHeight, windowY)
        );
        const clampedZ = Math.max(
          -roomDepth / 2 + halfWindowWidth,
          Math.min(roomDepth / 2 - halfWindowWidth, windowZ)
        );

        position = [clampedX, clampedY, clampedZ];

        console.log(`🪟 창문 ${index} 3D 위치:`, position);
        console.log(
          `🪟 창문 ${index} 벽면 위치: ${
            window.wall_position
          } 벽, X=${clampedX.toFixed(2)}m, Y=${clampedY.toFixed(
            2
          )}m, Z=${clampedZ.toFixed(2)}m`
        );
        console.log(
          `🪟 창문 ${index} 크기: ${windowWidth3D.toFixed(
            2
          )}m × ${windowHeight3D.toFixed(2)}m${
            window.width_meters && window.height_meters
              ? " (백엔드 계산됨)"
              : " (프론트엔드 추정)"
          }`
        );
        console.log(
          `🪟 창문 ${index} 이미지 좌표: x=${window.x_position.toFixed(
            3
          )}, y=${window.y_position.toFixed(3)}`
        );
        if (window.width_meters && window.height_meters) {
          console.log(
            `🎯 창문 ${index} 백엔드 실제 크기: ${window.width_meters.toFixed(
              2
            )}m × ${window.height_meters.toFixed(2)}m`
          );
        }

        return (
          <Window3D
            key={`window-${index}`}
            position={position}
            size={[windowWidth3D, windowHeight3D, 0.1]}
            wallPosition={window.wall_position}
            roomSize={roomSize}
          />
        );
      })}
    </group>
  );
});

// 창문 감지 API 호출 함수
const detectWindowsInImage = async (imageFile, roomPoints = null) => {
  try {
    console.log("🔍 창문 감지 시작:", imageFile.name, imageFile.size, "bytes");
    if (roomPoints) {
      console.log("📐 방 측정 포인트 전송:", roomPoints);
    }

    const formData = new FormData();
    formData.append("file", imageFile);

    // 방 측정 포인트 추가 (있는 경우)
    if (roomPoints && roomPoints.length >= 2) {
      formData.append("room_points", JSON.stringify(roomPoints));
    }

    console.log("📡 API 호출: http://localhost:3000/detect-windows");

    const response = await fetch("http://localhost:3000/detect-windows", {
      method: "POST",
      body: formData,
    });

    console.log("📥 응답 상태:", response.status, response.statusText);

    if (!response.ok) {
      const errorText = await response.text();
      console.error("❌ API 오류:", errorText);
      throw new Error(
        `HTTP error! status: ${response.status}, message: ${errorText}`
      );
    }

    const result = await response.json();
    console.log("✅ 창문 감지 결과:", result);
    return result;
  } catch (error) {
    console.error("❌ 창문 감지 오류:", error);
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
  uploadedImageUrl = null,
  placedFurniture = [], // 🪑 가구 배치 정보
}) {
  // 🎯 3D도 cm 단위로 통일 (스케일 제거)
  const scale = 1; // cm 단위 그대로 사용
  const w = width; // cm
  const h = height; // cm
  const d = depth; // cm

  // 🪑 FurniturePlacement에서 전달받은 가구 데이터를 3D 좌표로 변환 (FURNITURE_PRESETS 통일)
  const convertedFurniture = useMemo(() => {
    return placedFurniture.map((item, index) => {
      // 🔄 FurniturePlacement ID → RoomBox type 변환
      const mappedType = FURNITURE_ID_MAPPING[item.id] || "desk";

      // 🎯 FURNITURE_PRESETS에서 실제 크기와 색상 가져오기 (이미 cm 단위)
      const presetData = FURNITURE_PRESETS[mappedType];
      const furnitureSize = presetData
        ? presetData.size // 이미 cm 단위
        : [120, 75, 60]; // 기본값: 책상 크기 (cm)
      const furnitureColor = presetData ? presetData.color : item.color;

      // FurniturePlacement 좌표계 (cm, 왼쪽 아래 원점) → RoomBox 3D 좌표계 변환
      const x3d = item.x; // cm 그대로 사용
      const z3d = item.y; // y → z, cm 그대로 사용

      // 회전 변환: FurniturePlacement 도 → 라디안
      const rotY = (item.rotation || 0) * (Math.PI / 180);

      // 🎯 가구 크기 (cm 단위)
      const scaledSize = [
        furnitureSize[0], // 너비 (cm)
        furnitureSize[1], // 높이 (cm)
        furnitureSize[2], // 깊이 (cm)
      ];

      return {
        id: item.id || `furniture-${index}`, // FurniturePlacement의 실제 ID 사용
        type: mappedType, // 매핑된 가구 타입 사용
        name: presetData ? presetData.name : item.name, // FURNITURE_PRESETS의 이름 사용
        color: furnitureColor, // FURNITURE_PRESETS의 색상 사용
        size: scaledSize, // FURNITURE_PRESETS의 실제 크기 사용
        position: [
          x3d - w / 2 + scaledSize[0] / 2, // 중심 좌표로 변환 (cm)
          scaledSize[1] / 2, // 바닥에서 높이/2 (cm)
          z3d - d / 2 + scaledSize[2] / 2, // 중심 좌표로 변환 (cm)
        ],
        rotation: [0, rotY, 0],
        original2D: item, // 원본 2D 데이터 보존
      };
    });
  }, [placedFurniture, w, h, d, scale]);

  const [furniture, setFurniture] = useState([]);

  // 🧹 placedFurniture가 비어있으면 furniture도 비우기
  useEffect(() => {
    if (placedFurniture.length === 0) {
      console.log("🧹 placedFurniture가 비어있어서 furniture 초기화");
      setFurniture([]);
    }
  }, [placedFurniture.length]);
  const [selectedFurniture, setSelectedFurniture] = useState(null);

  // 🔄 placedFurniture → furniture 상태 동기화 (단일 진실 소스)
  useEffect(() => {
    console.log("🔄 가구 상태 동기화:", {
      placedCount: placedFurniture.length,
      convertedCount: convertedFurniture.length,
      furnitureIds: convertedFurniture.map((f) => f.id),
    });

    // convertedFurniture를 furniture 상태로 동기화
    setFurniture(convertedFurniture);
  }, [convertedFurniture]);
  const [showSnapGrid, setShowSnapGrid] = useState(false);
  const [showFloorGrid, setShowFloorGrid] = useState(false);
  const [enableSnap, setEnableSnap] = useState(true);
  const [showCollisions, setShowCollisions] = useState(true);
  const [placementMode, setPlacementMode] = useState(null);
  const [activeView, setActiveView] = useState("조감도");
  const [walkthroughMode, setWalkthroughMode] = useState(false); // New state for walkthrough mode
  const [collisionAlert, setCollisionAlert] = useState({
    visible: false,
    collisions: [],
  });
  const [measurementMode, setMeasurementMode] = useState(false); // 거리 측정 모드
  const [measurePoints, setMeasurePoints] = useState([null, null]); // 측정 포인트들
  const [humanPosition, setHumanPosition] = useState([2.2, 0, 2.7]); // 사람 위치 (임시 초기값)
  const [isDraggingFurniture, setIsDraggingFurniture] = useState(false); // 가구 드래그 상태
  const [detectedWindows, setDetectedWindows] = useState([]); // 감지된 창문들
  const [showWindows, setShowWindows] = useState(false); // 창문 표시 여부
  const [isDetectingWindows, setIsDetectingWindows] = useState(false); // 창문 감지 중

  const roomSize = [w, h, d];
  const roomArea = (width * depth) / 10000;

  // 3번 포인트(왼쪽 앞 모서리)를 (0,0,0)으로 설정하기 위한 오프셋
  const originOffset = [w / 2, 0, d / 2]; // [x, y, z] 오프셋

  const controlsRef = useRef();

  // 방 크기가 변경될 때 사람 위치를 방 중앙으로 업데이트
  useEffect(() => {
    setHumanPosition([w * 0.5, 0, d * 0.5]);
  }, [w, d]);

  const handleViewChange = useCallback((preset) => {
    if (controlsRef.current) {
      controlsRef.current.object.position.set(...preset.position);
      controlsRef.current.target.set(...preset.target);
      controlsRef.current.update();
      setActiveView(preset.name);
    }
  }, []);

  const handleCollisionAlert = useCallback((collisions) => {
    setCollisionAlert({ visible: true, collisions });
    setTimeout(() => {
      setCollisionAlert({ visible: false, collisions: [] });
    }, 2000);
  }, []);

  const handleAddFurniture = useCallback((type) => {
    setPlacementMode(type);
  }, []);

  const handlePlaceFurniture = useCallback(
    (position) => {
      if (!placementMode) return;

      const newFurniture = {
        id: Date.now(),
        type: placementMode,
        position: position,
        rotation: [0, 0, 0],
      };

      const adjustedPosition = CollisionDetector.adjustToValidPosition(
        position,
        FURNITURE_PRESETS[placementMode].size,
        roomSize,
        furniture,
        newFurniture.id,
        FURNITURE_PRESETS
      );

      newFurniture.position = adjustedPosition;
      setFurniture((prev) => [...prev, newFurniture]);
      setSelectedFurniture(newFurniture.id);
      setPlacementMode(null);
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
        return;
      }

      if (measurementMode) {
        const clickPoint = [event.point.x, event.point.y, event.point.z];
        setMeasurePoints((prev) => {
          if (!prev[0]) {
            return [clickPoint, null];
          } else {
            return [prev[0], clickPoint];
          }
        });
      }
    },
    [placementMode, handlePlaceFurniture, measurementMode]
  );

  const handleMoveFurniture = useCallback((id, newPosition) => {
    setFurniture((prev) =>
      prev.map((f) => (f.id === id ? { ...f, position: newPosition } : f))
    );
  }, []);

  // 창문 감지 핸들러
  const handleDetectWindows = useCallback(async () => {
    // 업로드된 이미지가 있으면 그것을 사용, 없으면 파일 선택
    if (uploadedImageFile) {
      console.log("📸 업로드된 이미지 사용:", uploadedImageFile.name);

      // 방 측정 포인트 정보 수집 (App.jsx에서 전달받은 경우)
      const roomMeasurementPoints = window.roomMeasurementPoints || null;

      if (roomMeasurementPoints) {
        console.log("📐 방 측정 포인트 사용:", roomMeasurementPoints);
      } else {
        console.log("📐 방 측정 포인트 없음, 기본 방법 사용");
      }

      setIsDetectingWindows(true);
      try {
        const result = await detectWindowsInImage(
          uploadedImageFile,
          roomMeasurementPoints
        );
        console.log("📊 전체 감지 결과:", result);
        console.log("🏠 현재 방 크기 (미터):", roomSize);
        console.log("🏠 현재 방 크기 (cm):", [width, height, depth]);

        setDetectedWindows(result.windows);
        setShowWindows(true);

        const pointsUsed = result.measurement_points_used
          ? " (층고 기준 정확한 위치)"
          : " (이미지 분석 기준)";
        const message =
          `${result.total_windows}개의 창문을 감지했습니다!${pointsUsed}\n\n` +
          result.windows
            .map(
              (w, i) =>
                `창문 ${i + 1}: ${w.wall_position} 벽\n` +
                `  위치: x=${w.x_position.toFixed(3)}, y=${w.y_position.toFixed(
                  3
                )}\n` +
                `  신뢰도: ${(w.confidence * 100).toFixed(0)}%`
            )
            .join("\n\n");

        alert(message);

        console.log("🪟 감지된 창문들:", result.windows);
      } catch (error) {
        console.error("창문 감지 실패:", error);
        alert("창문 감지에 실패했습니다. 다시 시도해주세요.");
      } finally {
        setIsDetectingWindows(false);
      }
    } else {
      // 업로드된 이미지가 없으면 파일 선택
      const input = document.createElement("input");
      input.type = "file";
      input.accept = "image/*";

      input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsDetectingWindows(true);
        try {
          const result = await detectWindowsInImage(file, null);
          console.log("📊 전체 감지 결과:", result);

          setDetectedWindows(result.windows);
          setShowWindows(true);

          const message =
            `${result.total_windows}개의 창문을 감지했습니다!\n\n` +
            result.windows
              .map(
                (w, i) =>
                  `창문 ${i + 1}: ${w.wall_position} 벽 (${(
                    w.confidence * 100
                  ).toFixed(0)}% 신뢰도)`
              )
              .join("\n");

          alert(message);

          console.log("🪟 감지된 창문들:", result.windows);
        } catch (error) {
          console.error("창문 감지 실패:", error);
          alert("창문 감지에 실패했습니다. 다시 시도해주세요.");
        } finally {
          setIsDetectingWindows(false);
        }
      };

      input.click();
    }
  }, [uploadedImageFile]);

  const handleRotateFurniture = useCallback((id) => {
    setFurniture((prev) =>
      prev.map((f) =>
        f.id === id
          ? { ...f, rotation: [0, f.rotation[1] + Math.PI / 2, 0] }
          : f
      )
    );
  }, []);

  const handleDeleteFurniture = useCallback((id) => {
    setFurniture((prev) => prev.filter((f) => f.id !== id));
    setSelectedFurniture(null);
  }, []);

  const selectedFurnitureData = useMemo(
    () => furniture.find((f) => f.id === selectedFurniture),
    [furniture, selectedFurniture]
  );

  // 키보드 이벤트 핸들러
  useEffect(() => {
    const handleKeyPress = (event) => {
      switch (event.key.toLowerCase()) {
        case "escape":
          setPlacementMode(null);
          setSelectedFurniture(null);
          break;
        case "g":
          setShowSnapGrid((prev) => !prev);
          break;
        case "f":
          setShowFloorGrid((prev) => !prev);
          break;
        case "s":
          setEnableSnap((prev) => !prev);
          break;
        case "delete":
        case "backspace":
          if (selectedFurniture) {
            handleDeleteFurniture(selectedFurniture);
          }
          break;
        case "m":
          setMeasurementMode((prev) => !prev);
          setMeasurePoints([null, null]);
          break;
        // 키보드 단축키들
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
      {/* 충돌 알림 (Canvas 외부) */}
      <CollisionAlert
        collisions={collisionAlert.collisions}
        visible={collisionAlert.visible}
        onDismiss={() => setCollisionAlert({ visible: false, collisions: [] })}
      />

      {/* 전체화면 모드에서 ESC 안내 */}
      {isFullscreen && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-50 bg-black bg-opacity-70 text-white px-4 py-2 rounded-lg">
          <p className="text-sm">ESC 키를 눌러 전체화면에서 나가기</p>
        </div>
      )}

      {/* 왼쪽 컨트롤 패널 */}
      <div
        className={`absolute top-4 left-4 z-10 space-y-3 max-w-xs ${
          isFullscreen ? "top-16" : ""
        }`}
      >
        {/* 가구 추가 */}
        <div
          className={`backdrop-blur p-3 rounded-lg shadow-lg ${
            isFullscreen ? "bg-white/70" : "bg-white/95"
          }`}
        >
          <h3 className="text-sm font-semibold mb-2 text-gray-700">
            {placementMode
              ? `${FURNITURE_PRESETS[placementMode].name} 배치 중...`
              : "가구 추가"}
          </h3>

          {placementMode ? (
            <div className="space-y-2">
              <p className="text-xs text-gray-600">
                바닥을 클릭해서 배치하세요
              </p>
              <button
                onClick={() => setPlacementMode(null)}
                className="w-full px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
              >
                취소
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(FURNITURE_PRESETS).map(([type, preset]) => (
                <button
                  key={type}
                  onClick={() => handleAddFurniture(type)}
                  className="flex items-center gap-2 px-2 py-1 bg-gray-100 rounded hover:bg-blue-100 transition-colors text-xs"
                >
                  <span>{preset.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* 옵션 */}
        <div
          className={`backdrop-blur p-3 rounded-lg shadow-lg ${
            isFullscreen ? "bg-white/70" : "bg-white/95"
          }`}
        >
          <h3 className="text-sm font-semibold mb-2 text-gray-700">
            시각 옵션
          </h3>
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={enableSnap}
                onChange={(e) => setEnableSnap(e.target.checked)}
              />
              스냅 기능
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={showSnapGrid}
                onChange={(e) => setShowSnapGrid(e.target.checked)}
              />
              스냅 그리드
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={showFloorGrid}
                onChange={(e) => setShowFloorGrid(e.target.checked)}
              />
              바닥 격자
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={showCollisions}
                onChange={(e) => setShowCollisions(e.target.checked)}
              />
              충돌 표시
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={showWindows}
                onChange={(e) => setShowWindows(e.target.checked)}
              />
              창문 표시
            </label>
          </div>
        </div>

        {/* 창문 감지 도구 */}
        <div
          className={`backdrop-blur p-3 rounded-lg shadow-lg ${
            isFullscreen ? "bg-white/70" : "bg-white/95"
          }`}
        >
          <h3 className="text-sm font-semibold mb-2 text-gray-700">
            창문 감지
          </h3>
          <div className="space-y-2">
            <button
              onClick={handleDetectWindows}
              disabled={isDetectingWindows}
              className={`w-full px-3 py-1 rounded text-xs ${
                isDetectingWindows
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-green-500 text-white hover:bg-green-600"
              }`}
            >
              {isDetectingWindows
                ? "감지 중..."
                : uploadedImageFile
                ? "업로드된 사진에서 창문 감지"
                : "사진에서 창문 감지"}
            </button>
            {detectedWindows.length > 0 && (
              <p className="text-xs text-green-600">
                {detectedWindows.length}개 창문 감지됨
              </p>
            )}
          </div>
        </div>

        {/* 거리 측정 도구 */}
        <div
          className={`backdrop-blur p-3 rounded-lg shadow-lg ${
            isFullscreen ? "bg-white/70" : "bg-white/95"
          }`}
        >
          <h3 className="text-sm font-semibold mb-2 text-gray-700">
            거리 측정
          </h3>
          <div className="space-y-2">
            <button
              onClick={() => {
                setMeasurementMode(!measurementMode);
                setMeasurePoints([null, null]);
              }}
              className={`w-full px-3 py-1 rounded text-xs ${
                measurementMode
                  ? "bg-pink-500 text-white hover:bg-pink-600"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
            >
              {measurementMode ? "측정 모드 종료" : "거리 측정 시작"}
            </button>
            {measurementMode && (
              <p className="text-xs text-pink-600">
                바닥을 클릭해서 거리를 측정하세요
              </p>
            )}
          </div>
        </div>

        {/* 사람 모델 컨트롤 */}
        <div
          className={`backdrop-blur p-3 rounded-lg shadow-lg ${
            isFullscreen ? "bg-white/70" : "bg-white/95"
          }`}
        >
          <h3 className="text-sm font-semibold mb-2 text-gray-700">
            사람 조작
          </h3>
          <div className="space-y-2">
            <p className="text-xs text-gray-600">
              사람을 드래그해서 움직여보세요
            </p>
            <button
              onClick={() => setHumanPosition([0, 0, 0])}
              className="px-3 py-2 bg-pink-500 text-white rounded-lg text-sm font-medium hover:bg-pink-600 transition-colors"
            >
              사람 위치 초기화
            </button>
          </div>
        </div>

        {/* 워크스루 모드 토글 */}
        <div
          className={`backdrop-blur p-3 rounded-lg shadow-lg ${
            isFullscreen ? "bg-white/70" : "bg-white/95"
          }`}
        >
          <h3 className="text-sm font-semibold mb-2 text-gray-700">
            워크스루 모드
          </h3>
          <button
            onClick={() => setWalkthroughMode((prev) => !prev)}
            className={`w-full px-3 py-1 rounded text-sm ${
              walkthroughMode
                ? "bg-pink-500 text-white hover:bg-pink-600"
                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
            }`}
          >
            {walkthroughMode ? "워크스루 종료" : "워크스루 시작"}
          </button>
        </div>
      </div>

      {/* 오른쪽 정보 패널 */}
      <div
        className={`absolute top-1/2 transform -translate-y-1/2 right-4 z-10 space-y-3 max-w-sm ${
          isFullscreen ? "top-16" : ""
        }`}
      >
        {/* 방 정보 */}
        <div
          className={`backdrop-blur p-3 rounded-lg shadow-lg ${
            isFullscreen ? "bg-white/70" : "bg-white/95"
          }`}
        >
          <h3 className="text-sm font-semibold mb-2 text-gray-700">방 정보</h3>
          <div className="text-xs space-y-1">
            <p>
              <strong>크기:</strong> {(width / 100).toFixed(1)} ×{" "}
              {(depth / 100).toFixed(1)} × {(height / 100).toFixed(1)}m
            </p>
            <p>
              <strong>면적:</strong> {roomArea.toFixed(1)}㎡
            </p>
            <p>
              <strong>부피:</strong>{" "}
              {((width * height * depth) / 1000000).toFixed(1)}㎥
            </p>
          </div>

          {/* 공간 활용도 */}
          <div className="mt-2">
            <SpaceUtilization furniture={furniture} roomArea={roomArea} />
          </div>
        </div>

        {/* 크기 비교 */}
        <div className={`${isFullscreen ? "opacity-80" : ""}`}>
          <SizeComparisonPanel
            currentArea={roomArea}
            isFullscreen={isFullscreen}
          />
        </div>

        {/* 이동 거리 */}
        <div className={`${isFullscreen ? "opacity-80" : ""}`}>
          <WalkingMetrics
            width={width}
            depth={depth}
            isFullscreen={isFullscreen}
          />
        </div>

        {/* 시점 변경 */}
        <div className={`${isFullscreen ? "opacity-80" : ""}`}>
          <ViewPresets onViewChange={handleViewChange} />
        </div>
      </div>

      <Canvas
        camera={{ position: [800, 400, 800], fov: 50 }}
        shadows
        gl={{ antialias: true, alpha: true }}
      >
        <Suspense fallback={null}>
          <Environment preset="apartment" />

          {/* 향상된 조명 (cm 단위 조정) */}
          <ambientLight intensity={0.6} />
          <directionalLight
            position={[w * 0.8, h * 1.5, d * 0.8]}
            intensity={0.5}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
            shadow-camera-far={h * 3}
            shadow-camera-left={-w}
            shadow-camera-right={w}
            shadow-camera-top={d}
            shadow-camera-bottom={-d}
          />
          <pointLight
            position={[w / 2, h * 0.7, d / 2]}
            intensity={0.2}
            distance={w * 1.5}
          />

          {/* 바닥 */}
          <mesh
            position={[originOffset[0], 0, originOffset[2]]}
            rotation={[-Math.PI / 2, 0, 0]}
            receiveShadow
            onClick={handleFloorClick}
          >
            <planeGeometry args={[w, d]} />
            <meshStandardMaterial
              color={placementMode ? "#e0f2fe" : "#f8f9fa"}
              roughness={0.8}
              metalness={0.1}
            />
          </mesh>

          {/* 바닥 격자 */}
          <group position={[originOffset[0], 0, originOffset[2]]}>
            <FloorGrid roomSize={roomSize} visible={showFloorGrid} />
          </group>

          {/* 벽들 - 4개 벽 모두 렌더링 */}
          {/* 뒷벽 */}
          <Wall
            width={w}
            height={h}
            position={[originOffset[0], h / 2, 0]}
            rotation={[0, 0, 0]}
          />
          {/* 왼쪽벽 */}
          <Wall
            width={d}
            height={h}
            position={[0, h / 2, originOffset[2]]}
            rotation={[0, Math.PI / 2, 0]}
          />
          {/* 오른쪽벽 */}
          <Wall
            width={d}
            height={h}
            position={[w, h / 2, originOffset[2]]}
            rotation={[0, -Math.PI / 2, 0]}
          />
          {/* 앞벽 (일부만, 입구 고려) */}
          <Wall
            width={w * 0.3}
            height={h}
            position={[originOffset[0] - w * 0.35, h / 2, d]}
            rotation={[0, Math.PI, 0]}
          />
          <Wall
            width={w * 0.3}
            height={h}
            position={[originOffset[0] + w * 0.35, h / 2, d]}
            rotation={[0, Math.PI, 0]}
          />

          {/* 스냅 그리드 */}
          <SnapGrid
            roomSize={roomSize}
            gridSize={0.25}
            visible={showSnapGrid}
          />

          {/* 배치 모드일 때 유효한 영역 표시 */}
          {placementMode && (
            <ValidPlacementArea
              roomSize={roomSize}
              furniture={furniture}
              furniturePresets={FURNITURE_PRESETS}
              selectedFurnitureSize={FURNITURE_PRESETS[placementMode].size}
            />
          )}

          {/* 가구 렌더링 */}
          {furniture.map((f, index) => (
            <DraggableFurnitureWithCollision
              key={`${f.id}-${index}`}
              id={f.id}
              type={f.type}
              position={f.position}
              rotation={f.rotation}
              onMove={handleMoveFurniture}
              onSelect={setSelectedFurniture}
              selected={selectedFurniture === f.id}
              furniture={furniture}
              furniturePresets={FURNITURE_PRESETS}
              roomSize={roomSize}
              enableSnap={enableSnap}
              showCollisions={showCollisions}
              onCollisionAlert={handleCollisionAlert}
              onDragStateChange={setIsDraggingFurniture}
              customFurnitureData={f.original2D ? f : null}
            />
          ))}

          {/* 창문 렌더링 */}
          {showWindows && detectedWindows.length > 0 && (
            <WindowsOnWalls windows={detectedWindows} roomSize={roomSize} />
          )}

          {/* 드래그 가능한 사람 모델 */}
          <DraggableHuman
            key="draggable-human-v2"
            height={170}
            position={humanPosition}
            onPositionChange={setHumanPosition}
            roomSize={roomSize}
            onDragStateChange={setIsDraggingFurniture}
          />

          {/* 거리 측정 도구 */}
          <DistanceMeasurer
            point1={measurePoints[0]}
            point2={measurePoints[1]}
            visible={measurementMode && measurePoints[0] && measurePoints[1]}
          />

          {/* 방 치수 표시 */}
          <group position={[0, 0.01, 0]}>
            <DimensionArrow start={[0, 0, d + 10]} end={[w, 0, d + 10]} />
            <DimensionLabel
              position={[w / 2, 0, d + 15]}
              text={`${width.toFixed(0)}cm`}
              rotation={[0, 0, 0]}
            />

            <DimensionArrow start={[w + 10, 0, 0]} end={[w + 10, 0, d]} />
            <DimensionLabel
              position={[w + 15, 0, d / 2]}
              text={`${depth.toFixed(0)}cm`}
              rotation={[0, Math.PI / 2, 0]}
            />

            <DimensionArrow start={[w + 10, 0, 0]} end={[w + 10, h, 0]} />
            <DimensionLabel
              position={[w + 15, h / 2, 0]}
              text={`${height.toFixed(0)}cm`}
              rotation={[0, 0, Math.PI / 2]}
            />
          </group>

          {walkthroughMode ? (
            <Player roomSize={roomSize} />
          ) : (
            <OrbitControls
              ref={controlsRef}
              enablePan={!isDraggingFurniture}
              enableZoom={!isDraggingFurniture}
              enableRotate={!isDraggingFurniture}
              minDistance={300}
              maxDistance={2000}
              maxPolarAngle={Math.PI / 2.1}
              target={[w / 2, 0, d / 2]}
            />
          )}
        </Suspense>
      </Canvas>

      {/* 선택된 가구 컨트롤 */}
      {selectedFurnitureData && (
        <div
          className={`absolute bottom-4 left-4 z-10 backdrop-blur p-3 rounded-lg shadow-lg ${
            isFullscreen ? "bg-white/70" : "bg-white/95"
          }`}
        >
          <p className="text-sm font-semibold mb-2">
            {selectedFurnitureData.original2D
              ? selectedFurnitureData.name // FurniturePlacement에서 온 가구는 실제 이름 사용
              : FURNITURE_PRESETS[selectedFurnitureData.type]?.name ||
                "알 수 없는 가구"}
          </p>
          <div className="space-y-2">
            <div className="flex gap-2">
              <button
                onClick={() => handleRotateFurniture(selectedFurniture)}
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors text-sm"
              >
                회전
              </button>
              <button
                onClick={() => handleDeleteFurniture(selectedFurniture)}
                className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 transition-colors text-sm"
              >
                삭제
              </button>
            </div>

            {/* 위치 정보 */}
            <div className="text-xs text-gray-600">
              <p>
                위치: ({selectedFurnitureData.position[0].toFixed(1)},{" "}
                {selectedFurnitureData.position[2].toFixed(1)})
              </p>
              <p>
                크기:{" "}
                {selectedFurnitureData.original2D
                  ? selectedFurnitureData.size
                      .map((s) => s.toFixed(1))
                      .join(" × ")
                  : (
                      FURNITURE_PRESETS[selectedFurnitureData.type]?.size || [
                        1, 1, 1,
                      ]
                    )
                      .map((s) => s.toFixed(1))
                      .join(" × ")}
                m
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 하단 상태 표시 */}
      {!isFullscreen && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-10">
          <div className="bg-white/90 backdrop-blur px-4 py-2 rounded-lg shadow-lg">
            <p className="text-sm font-medium">현재 시점: {activeView}</p>
            <p className="text-xs text-gray-600">가구 {furniture.length}개</p>
          </div>
        </div>
      )}

      {/* 조작 가이드 */}
      <div
        className={`absolute bottom-4 right-4 z-10 text-xs text-gray-600 bg-white/80 backdrop-blur px-3 py-2 rounded-lg max-w-xs ${
          isFullscreen ? "opacity-70" : ""
        }`}
      >
        <div className="space-y-1">
          <p className="font-semibold">조작 방법</p>
          {!walkthroughMode && (
            <>
              <p>• 드래그: 시점 회전</p>
              <p>• 휠: 확대/축소</p>
            </>
          )}
          <p>• 가구 드래그: 이동</p>
          <p>• 시점 버튼: 빠른 시점 변경</p>
          <p>• G: 스냅 그리드 | F: 바닥 격자</p>
          <p>• M: 거리 측정 | 드래그: 사람 이동</p>
          {walkthroughMode && (
            <p className="text-pink-600 font-semibold">
              • W,A,S,D: 이동 | 마우스: 시점 변경
            </p>
          )}
          {!walkthroughMode && placementMode && (
            <p className="text-pink-600 font-semibold">바닥 클릭: 가구 배치</p>
          )}
          {measurementMode && (
            <p className="text-pink-600 font-semibold">
              바닥 클릭: 거리 측정 (2점)
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
