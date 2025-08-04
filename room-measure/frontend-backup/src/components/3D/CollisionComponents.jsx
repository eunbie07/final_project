import React from "react";
import CollisionDetector from "../../utils/CollisionDetector";

// 충돌 표시 컴포넌트
export const CollisionIndicator = React.memo(function CollisionIndicator({
  position,
  size,
  collisionType = "furniture",
}) {
  const color = collisionType === "wall" ? "#EF4444" : "#FFD700";

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
export const ValidPlacementArea = React.memo(function ValidPlacementArea({
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
          <meshBasicMaterial color="#10B981" transparent opacity={0.6} />
        </mesh>
      ))}
    </group>
  );
});

// 충돌 알림 컴포넌트 (Canvas 외부에서 렌더링)
export const CollisionAlert = React.memo(function CollisionAlert({
  collisions,
  onDismiss,
  visible = false,
}) {
  if (!visible || !collisions || collisions.length === 0) return null;

  return (
    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 bg-danger border border-danger text-white px-4 py-3 rounded-lg shadow-lg max-w-sm">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-semibold">충돌 감지</h4>
        <button onClick={onDismiss} className="text-white/80 hover:text-white">
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