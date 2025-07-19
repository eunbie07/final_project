import React, { useState, useMemo, Suspense, useRef, useCallback } from "react";
import { Canvas, useLoader, useThree, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Line, Environment } from "@react-three/drei";
import * as THREE from "three";

// 가구 프리셋 정의
const FURNITURE_PRESETS = {
  bed_single: {
    name: "싱글 침대",
    size: [1, 0.5, 2],
    color: "#FFB7D1", // 연한 핑크
    icon: "🛏️",
  },
  bed_double: {
    name: "더블 침대",
    size: [1.5, 0.5, 2],
    color: "#FFC4D6", // 살구색 핑크
    icon: "🛏️",
  },
  desk: {
    name: "책상",
    size: [1.2, 0.75, 0.6],
    color: "#FFE4E1", // 미스티 로즈
    icon: "🪑",
  },
  wardrobe: {
    name: "옷장",
    size: [1, 2, 0.6],
    color: "#FFDAB9", // 피치 퍼프
    icon: "🗄️",
  },
  sofa: {
    name: "소파",
    size: [1.8, 0.8, 0.85],
    color: "#FFE4E8", // 라이트 핑크
    icon: "🛋️",
  },
};

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
      {/* 가로 치수 */}
      <DimensionArrow
        start={[x - width / 2, y, z + depth / 2 + 0.05]}
        end={[x + width / 2, y, z + depth / 2 + 0.05]}
      />
      <DimensionLabel
        position={[x, y, z + depth / 2 + 0.1]}
        text={`${width.toFixed(1)}m`}
        rotation={[-Math.PI / 2, 0, 0]}
      />

      {/* 세로 치수 */}
      <DimensionArrow
        start={[x + width / 2 + 0.05, y, z - depth / 2]}
        end={[x + width / 2 + 0.05, y, z + depth / 2]}
      />
      <DimensionLabel
        position={[x + width / 2 + 0.1, y, z]}
        text={`${depth.toFixed(1)}m`}
        rotation={[-Math.PI / 2, Math.PI / 2, 0]}
      />

      {/* 높이 치수 */}
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

// 드래그 가능한 가구 컴포넌트 (개선된 버전)
const DraggableFurniture = React.memo(function DraggableFurniture({
  id,
  type,
  position,
  rotation,
  onMove,
  onSelect,
  selected,
}) {
  const mesh = useRef();
  const preset = FURNITURE_PRESETS[type];
  const [hovered, setHovered] = useState(false);
  const [dragging, setDragging] = useState(false);
  const { camera, gl } = useThree();

  // 성능 최적화를 위해 레퍼런스로 관리
  const planeRef = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0));
  const raycasterRef = useRef(new THREE.Raycaster());
  const intersectionRef = useRef(new THREE.Vector3());
  const mouseRef = useRef(new THREE.Vector2());

  // 마우스 이동 핸들러
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
        onMove(id, [
          intersectionRef.current.x,
          preset.size[1] / 2,
          intersectionRef.current.z,
        ]);
      }
    },
    [dragging, camera, gl, id, onMove, preset.size]
  );

  // 이벤트 리스너 관리
  React.useEffect(() => {
    if (dragging) {
      window.addEventListener("pointermove", handlePointerMove);
      window.addEventListener("pointerup", () => {
        setDragging(false);
        gl.domElement.style.cursor = "grab";
      });

      return () => {
        window.removeEventListener("pointermove", handlePointerMove);
      };
    }
  }, [dragging, handlePointerMove, gl]);

  return (
    <>
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
        <boxGeometry args={preset.size} />
        <meshStandardMaterial
          color={preset.color}
          roughness={0.7}
          metalness={0.3}
          emissive={selected ? "#ffffff" : "#000000"}
          emissiveIntensity={selected ? 0.1 : 0}
        />
      </mesh>
      <FurnitureDimensions
        position={position}
        size={preset.size}
        selected={selected}
      />
    </>
  );
});

// 사람 실루엣 컴포넌트 (에러 처리 추가)
const HumanSilhouette = React.memo(function HumanSilhouette({ height = 170 }) {
  const [texture, setTexture] = useState(null);
  const scale = height / 100;

  // 텍스처 로딩 시도
  React.useEffect(() => {
    new THREE.TextureLoader().load(
      "/human.png",
      (tex) => setTexture(tex),
      undefined,
      () => {
        console.warn("Human texture not found, using fallback");
      }
    );
  }, []);

  // 텍스처가 없을 경우 간단한 실린더로 대체
  if (!texture) {
    return (
      <group position={[1.2, scale / 2, 0.8]}>
        <mesh>
          <cylinderGeometry args={[0.15, 0.15, scale]} />
          <meshBasicMaterial color="#666666" opacity={0.7} transparent />
        </mesh>
        <Text
          position={[0, scale / 2 + 0.1, 0]}
          fontSize={0.1}
          color="#333333"
          anchorX="center"
        >
          170cm
        </Text>
      </group>
    );
  }

  return (
    <mesh position={[1.2, scale / 2, 0.8]}>
      <planeGeometry args={[0.6, scale]} />
      <meshBasicMaterial map={texture} transparent />
    </mesh>
  );
});

// 치수선 색상 설정
const DIMENSION_COLOR = "#666666"; // 그레이 색상

// 치수 화살표 컴포넌트
const DimensionArrow = React.memo(function DimensionArrow({ start, end }) {
  const points = useMemo(() => {
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const direction = new THREE.Vector3()
      .subVectors(endVec, startVec)
      .normalize();
    const arrowSize = 0.1;

    // 수직 벡터 계산
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

// 치수 레이블 컴포넌트
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

// 바닥 컴포넌트
const Floor = React.memo(function Floor({ width, depth }) {
  const [hovered, setHovered] = useState(false);

  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      receiveShadow
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <planeGeometry args={[width, depth]} />
      <meshStandardMaterial
        color={hovered ? "#FFF0F5" : "#FFF5F8"} // 라이트 핑크
        roughness={0.5}
        metalness={0.1}
      />
    </mesh>
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
        color={isWindow ? "#FFE4EC" : "#FFF0F5"} // 창문: 연한 핑크, 벽: 라이트 핑크
        roughness={0.7}
        metalness={0.1}
        clearcoat={0.2}
        opacity={isWindow ? 0.3 : 1}
        transparent={isWindow}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
});

// 로딩 화면
function LoadingScreen() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-80">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-gray-600">3D 환경 로딩 중...</p>
      </div>
    </div>
  );
}

// 메인 컴포넌트
export default function RoomBox({ width = 400, height = 230, depth = 400 }) {
  // 기본 가구 설정
  const initialFurniture = useMemo(
    () => [
      {
        id: "default-bed",
        type: "bed_single",
        position: [-1, FURNITURE_PRESETS["bed_single"].size[1] / 2, -1],
        rotation: [0, 0, 0],
      },
    ],
    []
  );

  const [furniture, setFurniture] = useState(initialFurniture);
  const [selectedFurniture, setSelectedFurniture] = useState(null);

  const scale = 0.01;
  const w = width * scale;
  const h = height * scale;
  const d = depth * scale;

  // 가구 추가 핸들러
  const handleAddFurniture = useCallback((type) => {
    const newFurniture = {
      id: Date.now(),
      type,
      position: [0, FURNITURE_PRESETS[type].size[1] / 2, 0],
      rotation: [0, 0, 0],
    };
    setFurniture((prev) => [...prev, newFurniture]);
    setSelectedFurniture(newFurniture.id);
  }, []);

  // 가구 이동 핸들러
  const handleMoveFurniture = useCallback((id, newPosition) => {
    setFurniture((prev) =>
      prev.map((f) => (f.id === id ? { ...f, position: newPosition } : f))
    );
  }, []);

  // 가구 회전 핸들러
  const handleRotateFurniture = useCallback((id) => {
    setFurniture((prev) =>
      prev.map((f) =>
        f.id === id
          ? { ...f, rotation: [0, f.rotation[1] + Math.PI / 2, 0] }
          : f
      )
    );
  }, []);

  // 가구 삭제 핸들러
  const handleDeleteFurniture = useCallback((id) => {
    setFurniture((prev) => prev.filter((f) => f.id !== id));
    setSelectedFurniture(null);
  }, []);

  // 선택된 가구 정보
  const selectedFurnitureData = useMemo(
    () => furniture.find((f) => f.id === selectedFurniture),
    [furniture, selectedFurniture]
  );

  // 조명 설정
  const lights = useMemo(
    () => (
      <>
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[5, 8, 5]}
          intensity={0.7}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-far={50}
          shadow-camera-left={-10}
          shadow-camera-right={10}
          shadow-camera-top={10}
          shadow-camera-bottom={-10}
        />
        <pointLight position={[0, 6, 0]} intensity={0.4} color="#FFF5E0" />
      </>
    ),
    []
  );

  return (
    <div className="relative w-full h-[600px] bg-gradient-to-br from-rose-50 to-pink-100 rounded-xl overflow-hidden shadow-2xl">
      {/* 가구 선택 UI */}
      <div className="absolute top-4 left-4 z-10 bg-white/95 backdrop-blur p-3 rounded-lg shadow-lg">
        <h3 className="text-sm font-semibold mb-2 text-gray-700">가구 추가</h3>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(FURNITURE_PRESETS).map(([type, preset]) => (
            <button
              key={type}
              onClick={() => handleAddFurniture(type)}
              className="flex items-center gap-2 px-3 py-2 bg-gray-100 rounded-lg hover:bg-blue-100 transition-colors"
            >
              <span className="text-xl">{preset.icon}</span>
              <span className="text-sm">{preset.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* 방 정보 표시 */}
      <div className="absolute top-4 right-4 z-10 bg-white/95 backdrop-blur p-3 rounded-lg shadow-lg">
        <h3 className="text-sm font-semibold mb-1 text-gray-700">방 크기</h3>
        <p className="text-xs text-gray-600">
          {(width / 100).toFixed(1)}m × {(depth / 100).toFixed(1)}m ×{" "}
          {(height / 100).toFixed(1)}m
        </p>
        <p className="text-xs text-gray-500 mt-1">
          면적: {((width * depth) / 10000).toFixed(1)}㎡
        </p>
      </div>

      <Canvas
        camera={{ position: [3, 3, 3], fov: 60 }}
        shadows
        gl={{ antialias: true, alpha: true }}
      >
        <Suspense fallback={null}>
          {/* 환경광 */}
          <Environment preset="apartment" />

          {/* 조명 */}
          {lights}

          {/* 바닥 */}
          <Floor width={w} depth={d} />

          {/* 벽들 */}
          <Wall
            width={w}
            height={h}
            position={[0, h / 2, -d / 2]}
            rotation={[0, 0, 0]}
          />
          <Wall
            width={d}
            height={h}
            position={[-w / 2, h / 2, 0]}
            rotation={[0, Math.PI / 2, 0]}
          />

          {/* 가구 렌더링 */}
          {furniture.map((f) => (
            <DraggableFurniture
              key={f.id}
              id={f.id}
              type={f.type}
              position={f.position}
              rotation={f.rotation}
              onMove={handleMoveFurniture}
              onSelect={setSelectedFurniture}
              selected={selectedFurniture === f.id}
            />
          ))}

          {/* 사람 실루엣 */}
          <HumanSilhouette height={170} />

          {/* 치수 표시 */}
          <group position={[0, 0.01, 0]}>
            {/* 가로 치수 */}
            <DimensionArrow
              start={[-w / 2, 0, d / 2 + 0.1]}
              end={[w / 2, 0, d / 2 + 0.1]}
            />
            <DimensionLabel
              position={[0, 0, d / 2 + 0.15]}
              text={`${(width / 100).toFixed(1)}m`}
              rotation={[0, 0, 0]}
            />

            {/* 세로 치수 */}
            <DimensionArrow
              start={[w / 2 + 0.1, 0, -d / 2]}
              end={[w / 2 + 0.1, 0, d / 2]}
            />
            <DimensionLabel
              position={[w / 2 + 0.15, 0, 0]}
              text={`${(depth / 100).toFixed(1)}m`}
              rotation={[0, Math.PI / 2, 0]} // 회전 수정
            />

            {/* 높이 치수 */}
            <DimensionArrow
              start={[w / 2 + 0.1, 0, -d / 2]}
              end={[w / 2 + 0.1, h, -d / 2]}
            />
            <DimensionLabel
              position={[w / 2 + 0.15, h / 2, -d / 2]}
              text={`${(height / 100).toFixed(1)}m`}
              rotation={[0, 0, Math.PI / 2]}
            />
          </group>

          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={2}
            maxDistance={10}
          />
        </Suspense>
      </Canvas>

      {/* 선택된 가구 컨트롤 */}
      {selectedFurnitureData && (
        <div className="absolute bottom-4 left-4 z-10 bg-white/95 backdrop-blur p-3 rounded-lg shadow-lg">
          <p className="text-sm font-semibold mb-2">
            {FURNITURE_PRESETS[selectedFurnitureData.type].name}
          </p>
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
        </div>
      )}

      {/* 조작 설명 */}
      <div className="absolute bottom-4 right-4 z-10 text-xs text-gray-600 bg-white/80 backdrop-blur px-2 py-1 rounded">
        마우스 드래그: 회전 • 스크롤: 확대/축소 • 가구 클릭 후 드래그: 이동
      </div>
    </div>
  );
}
