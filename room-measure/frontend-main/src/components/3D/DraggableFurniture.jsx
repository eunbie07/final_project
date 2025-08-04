import React, { useState, useRef, useCallback, useEffect } from "react";
import { useThree } from "@react-three/fiber";
import { ContactShadows, Text } from "@react-three/drei";
import * as THREE from "three";
import CollisionDetector from "../../utils/CollisionDetector";
import PositionSnapper from "../../utils/PositionSnapper";
import { FurnitureDimensions } from "./DimensionComponents";
import { CollisionIndicator } from "./CollisionComponents";
import { FurnitureModel } from "./FurnitureModel";

// 향상된 가구 컴포넌트 (드래그 시스템 개선)
export const DraggableFurnitureWithCollision = React.memo(
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
    use3DModels = false, // 3D 모델 사용 여부
  }) {
    const mesh = useRef();

    // 커스텀 가구인 경우 전달된 size를 직접 사용, 아니면 preset에서 가져오기
    let size, color, preset;
    if (customFurnitureData && customFurnitureData.size) {
      size = customFurnitureData.size;
      color = customFurnitureData.color || "#cccccc";
      preset = null;
    } else {
      preset = furniturePresets[type];
      size = preset?.size || [100, 100, 100];
      color = preset?.color || "#cccccc";
    }

    // 3D 모델 사용 가능 여부 체크
    const canUse3DModel = use3DModels && preset && preset.model3D && !customFurnitureData;

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

              // 극단적인 좌표값 방지 - 방 크기의 10배를 넘으면 무시
              const maxDistance = Math.max(roomSize[0], roomSize[2]) * 10;
              if (Math.abs(newPosition[0]) > maxDistance || Math.abs(newPosition[2]) > maxDistance) {
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
              }

              // 먼저 방 경계 내로 조정 (회전 정보 포함)
              const boundaryAdjustedPosition =
                CollisionDetector.adjustToValidPosition(
                  newPosition,
                  size,
                  roomSize,
                  [],
                  id,
                  furniturePresets,
                  rotation
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

              // 충돌이 없을 때만 이동 허용
              if (furnitureCollisions.length === 0) {
                lastValidPosition.current = boundaryAdjustedPosition;
                onMove(id, boundaryAdjustedPosition);
              }
            }
          }
        };

        const handleMouseUp = () => {
          dragStart.current = null;

          if (isDraggingRef.current) {
            // 드래그 완료 후 2D 좌표 업데이트 - 마지막 유효 위치와 회전 정보 사용
            if (updatePlacedFurniturePosition) {
              updatePlacedFurniturePosition(id, lastValidPosition.current, rotation);
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
        <group
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
        >
          {canUse3DModel ? (
            // 3D 모델 렌더링
            <FurnitureModel
              modelConfig={preset.model3D}
              size={size}
              color={color}
              selected={selected}
              hasCollision={hasCollision}
              isChildComponent={true}
            />
          ) : (
            // 기본 박스 렌더링
            <mesh castShadow receiveShadow>
              <boxGeometry args={size} />
              <meshStandardMaterial
                color={materialColor}
                roughness={0.7}
                metalness={0.3}
                emissive={
                  selected ? "white" : hasCollision ? "#EF4444" : "#000000"
                }
                emissiveIntensity={selected ? 0.1 : hasCollision ? 0.2 : 0}
              />
            </mesh>
          )}
        </group>

        {/* 가구 이름 텍스트 */}
        <Text
          position={[0, size[1] / 2 + 10, 0]} // 가구 위에 표시
          rotation={[0, -rotation[1], 0]} // 가구 회전과 반대로 회전하여 항상 정면을 보도록
          fontSize={15}
          color="white"
          anchorX="center"
          anchorY="middle"
          depthTest={false} // 다른 오브젝트에 가려지지 않도록
        >
          {customFurnitureData?.name || furniturePresets[type]?.name}
        </Text>

        {/* 가구 치수 텍스트 */}
        <Text
          position={[0, size[1] / 2 - 5, 0]} // 가구 이름 아래에 표시
          rotation={[0, -rotation[1], 0]} // 가구 회전과 반대로 회전하여 항상 정면을 보도록
          fontSize={10}
          color="white"
          anchorX="center"
          anchorY="middle"
          depthTest={false} // 다른 오브젝트에 가려지지 않도록
        >
          {`${size[0]}x${size[2]}x${size[1]}cm`}
        </Text>

        <ContactShadows
          position={[0, -size[1] / 2 + 0.1, 0]}
          opacity={0.4}
          scale={Math.max(size[0], size[2]) * 1.2}
          blur={2}
          far={size[1]}
        />

        {selected && (
          <mesh rotation={rotation}>
            <boxGeometry args={size.map((s) => s + 5)} />
            <meshBasicMaterial
              color="#334155"
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
          rotation={rotation}
          selected={selected}
        />
      </group>
    );
  }
);