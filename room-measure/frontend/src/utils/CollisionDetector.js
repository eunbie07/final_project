// 충돌 검사 유틸리티 클래스
class CollisionDetector {
  static createBoundingBox(position, size, rotation = [0, 0, 0]) {
    const [x, y, z] = position;
    let [width, height, depth] = size;
    
    // Y축 회전(rotation[1])을 고려하여 가로/세로 치수 조정
    const yRotation = rotation[1] || 0;
    const rotationDegrees = Math.abs(yRotation * 180 / Math.PI) % 180;
    
    // 90도 회전 시 width와 depth를 바꿈
    if (rotationDegrees > 45 && rotationDegrees < 135) {
      [width, depth] = [depth, width];
    }

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

    // 현재 가구의 회전 정보 가져오기
    const currentRotation = currentFurniture.rotation || [0, 0, 0];
    const currentBox = this.createBoundingBox(newPosition, currentSize, currentRotation);

    const collisions = [];

    furniture.forEach((otherFurniture) => {
      if (otherFurniture.id === currentId) return;

      // 다른 가구의 크기 정보를 안전하게 가져오기
      const otherSize = otherFurniture.size ||
        furniturePresets[otherFurniture.type]?.size || [100, 100, 100]; // 기본값

      // 다른 가구의 회전 정보 가져오기
      const otherRotation = otherFurniture.rotation || [0, 0, 0];
      const otherBox = this.createBoundingBox(
        otherFurniture.position,
        otherSize,
        otherRotation
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
    furniturePresets,
    rotation = [0, 0, 0]
  ) {
    let [x, y, z] = position;
    let [width, height, depth] = size;
    const [roomWidth, roomHeight, roomDepth] = roomSize;

    // 회전을 고려한 크기 조정
    const yRotation = rotation[1] || 0;
    const rotationDegrees = Math.abs(yRotation * 180 / Math.PI) % 180;
    
    // 90도 회전 시 width와 depth를 바꿈
    if (rotationDegrees > 45 && rotationDegrees < 135) {
      [width, depth] = [depth, width];
    }

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

export default CollisionDetector;