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

export default SpaceUtilization;