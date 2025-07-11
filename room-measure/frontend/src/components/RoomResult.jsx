// frontend/src/components/RoomResult.jsx
import React from "react";
import RoomCanvas from "./RoomCanvas";

const RoomResult = ({ x, y, depthImageUrl }) => {
  return (
    <div className="mt-10">
      <RoomCanvas x={x} y={y} />
      
      {depthImageUrl && (
        <div className="mt-8">
          <h2 className="font-bold mb-2">Depth Map</h2>
          <img
            src={depthImageUrl}
            alt="Depth Map"
            className="w-full max-w-md border border-gray-400"
          />
        </div>
      )}
    </div>
  );
};

export default RoomResult;
