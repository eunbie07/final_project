import React, { useState } from "react";

const RoomPointSelector = ({ imageUrl, onPointsSelected }) => {
  const [points, setPoints] = useState({
    ceilingY: null,
    floorY: null,
    leftX: null,
    rightX: null,
  });

  const handleClick = (e) => {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (points.ceilingY === null) {
      setPoints({ ...points, ceilingY: y });
    } else if (points.floorY === null) {
      setPoints({ ...points, floorY: y });
    } else if (points.leftX === null) {
      setPoints({ ...points, leftX: x });
    } else if (points.rightX === null) {
      const newPoints = { ...points, rightX: x };
      setPoints(newPoints);
      onPointsSelected(newPoints); // 부모 컴포넌트에 전달
    }
  };

  const resetPoints = () => {
    setPoints({ ceilingY: null, floorY: null, leftX: null, rightX: null });
  };

  return (
    <div>
      <h3>천장, 바닥, 왼쪽 벽, 오른쪽 벽 순서로 클릭해주세요</h3>
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
        {points.leftX !== null && (
          <div
            style={{
              position: "absolute",
              left: points.leftX,
              top: 0,
              height: "100%",
              width: "2px",
              background: "green",
            }}
          />
        )}
        {points.rightX !== null && (
          <div
            style={{
              position: "absolute",
              left: points.rightX,
              top: 0,
              height: "100%",
              width: "2px",
              background: "orange",
            }}
          />
        )}
      </div>

      <div style={{ marginTop: "10px" }}>
        <p>천장 Y좌표: {points.ceilingY !== null ? Math.round(points.ceilingY) : "-"}</p>
        <p>바닥 Y좌표: {points.floorY !== null ? Math.round(points.floorY) : "-"}</p>
        <p>왼쪽 X좌표: {points.leftX !== null ? Math.round(points.leftX) : "-"}</p>
        <p>오른쪽 X좌표: {points.rightX !== null ? Math.round(points.rightX) : "-"}</p>
        <button onClick={resetPoints}>다시 선택</button>
      </div>
    </div>
  );
};

export default RoomPointSelector;
