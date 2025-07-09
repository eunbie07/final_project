import React, { useState } from "react";

const CeilingFloorSelector = ({ imageUrl, onPointsSelected }) => {
  const [points, setPoints] = useState({ ceilingY: null, floorY: null });

  const handleClick = (e) => {
    const rect = e.target.getBoundingClientRect();
    const y = e.clientY - rect.top;

    if (points.ceilingY === null) {
      setPoints((prev) => ({ ...prev, ceilingY: Math.round(y) }));
    } else if (points.floorY === null) {
      const newPoints = {
        ceilingY: points.ceilingY,
        floorY: Math.round(y),
      };
      setPoints(newPoints);
      onPointsSelected(newPoints); // 두 좌표가 모두 선택된 경우에만 전달
    }
  };

  const resetPoints = () => {
    setPoints({ ceilingY: null, floorY: null });
  };

  return (
    <div>
      <h3>천장과 바닥을 순서대로 클릭해주세요</h3>
      <div style={{ position: "relative", display: "inline-block" }}>
        <img
          src={imageUrl}
          onClick={handleClick}
          style={{ width: "100%", maxWidth: "500px", border: "1px solid #ccc" }}
          alt="Room Preview"
        />
        {points.ceilingY !== null && (
          <div
            style={{
              position: "absolute",
              top: points.ceilingY,
              left: 0,
              width: "100%",
              height: "2px",
              background: "red",
            }}
          />
        )}
        {points.floorY !== null && (
          <div
            style={{
              position: "absolute",
              top: points.floorY,
              left: 0,
              width: "100%",
              height: "2px",
              background: "blue",
            }}
          />
        )}
      </div>

      <div style={{ marginTop: "10px" }}>
        {points.ceilingY !== null && <p>천장 Y좌표: {points.ceilingY}</p>}
        {points.floorY !== null && <p>바닥 Y좌표: {points.floorY}</p>}
        <button onClick={resetPoints}>다시 선택</button>
      </div>
    </div>
  );
};

export default CeilingFloorSelector;
