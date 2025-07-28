import React from "react";

const SpaceUtilization = React.memo(function SpaceUtilization({
  furniture,
  roomArea,
  furniturePresets,
}) {
  const furnitureArea = furniture.reduce((total, item) => {
    const size = item.size || furniturePresets[item.type]?.size;
    if (!size) return total;
    const area = (size[0] * size[2]) / 10000; // cm² → m²
    return total + area;
  }, 0);

  const utilization = (furnitureArea / roomArea) * 100;

  const getUtilizationColor = (util) => {
    if (util < 30) return "bg-accent";
    if (util < 60) return "bg-primary";
    if (util < 80) return "bg-secondary";
    return "bg-danger";
  };

  const getUtilizationText = (util) => {
    if (util < 30) return "여유로움";
    if (util < 60) return "적절함";
    if (util < 80) return "꽉참";
    return "과밀";
  };

  return (
    <div className="bg-background p-2 rounded">
      <h4 className="font-semibold text-xs mb-1 text-text-primary">공간 활용도</h4>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${getUtilizationColor(
            utilization
          )}`}
          style={{ width: `${Math.min(utilization, 100)}%` }}
        />
      </div>
      <div className="flex justify-between items-center mt-1">
        <span className="text-xs text-text-secondary">{utilization.toFixed(1)}%</span>
        <span className="text-xs font-medium text-text-secondary">
          {getUtilizationText(utilization)}
        </span>
      </div>
    </div>
  );
});

export default SpaceUtilization;