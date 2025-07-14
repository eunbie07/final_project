import React, { useState, useRef } from "react";
import axios from "axios";

const ImageClickArea = ({ imageUrl, onComplete, depthWidth, depthHeight }) => {
  const [points, setPoints] = useState([]);
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  const handleClick = async (e) => {
    if (points.length >= 4) return;
    if (!imageRef.current || depthWidth === 0 || depthHeight === 0) {
      console.warn("깊이 정보가 아직 준비되지 않았습니다.");
      return;
    }

    const rect = imageRef.current.getBoundingClientRect();
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;

    const displayWidth = imageRef.current.width;
    const displayHeight = imageRef.current.height;

    const x = Math.round((displayX / displayWidth) * depthWidth);
    const y = Math.round((displayY / displayHeight) * depthHeight);

    if (isNaN(x) || isNaN(y)) {
      alert("좌표 변환 오류");
      return;
    }

    try {
      const res = await axios.get("http://localhost:3000/get-depth-at-point", {
        params: { x, y },
      });

      const depth = res.data.depth;
      if (isNaN(depth) || depth <= 0) {
        alert("깊이 정보를 불러올 수 없습니다. 다른 위치를 클릭해 주세요.");
        return;
      }

      const newPoints = [...points, { x, y, z: depth }];
      setPoints(newPoints);

      if (newPoints.length === 4) {
        onComplete(newPoints);
      }
    } catch (error) {
      console.error("깊이 정보 요청 실패:", error);
      alert("서버 오류로 깊이 정보를 불러오지 못했습니다.");
    }
  };


  return (
    <div className="mb-4">
      <p className="mb-2">
        ①하단 → ②상단 → ③왼쪽 바닥 → ④오른쪽 바닥 순서로 클릭하세요.
      </p>

      <div
        ref={containerRef}
        style={{ position: "relative", display: "inline-block" }}
        onClick={handleClick}
      >
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Uploaded"
          style={{
            width: "100%",
            maxWidth: "800px",
            display: "block",
            cursor: "crosshair",
          }}
        />

        {points.map((pt, index) => {
          const img = imageRef.current;
          if (!img) return null;

          const displayX = (pt.x / depthWidth) * img.width;
          const displayY = (pt.y / depthHeight) * img.height;

          return (
            <div
              key={index}
              style={{
                position: "absolute",
                left: `${displayX}px`,
                top: `${displayY}px`,
                width: "10px",
                height: "10px",
                backgroundColor: "red",
                borderRadius: "50%",
                transform: "translate(-50%, -50%)",
              }}
            >
              <span
                style={{
                  position: "absolute",
                  top: "-20px",
                  left: "-10px",
                  color: "white",
                  backgroundColor: "black",
                  fontSize: "10px",
                  padding: "1px 3px",
                  borderRadius: "4px",
                }}
              >
                {index + 1}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ImageClickArea;
