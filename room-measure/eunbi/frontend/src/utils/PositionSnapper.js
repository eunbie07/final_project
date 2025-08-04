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

export default PositionSnapper;