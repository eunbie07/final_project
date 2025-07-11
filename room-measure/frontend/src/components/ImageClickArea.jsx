import React, { useState, useRef, useEffect } from "react";

const ImageClickArea = ({ imageUrl, onComplete }) => {
  const [points, setPoints] = useState([]);
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  const handleClick = (e) => {
    if (points.length >= 4) return;

    const rect = imageRef.current.getBoundingClientRect();

    const scaleX = imageRef.current.naturalWidth / imageRef.current.width;
    const scaleY = imageRef.current.naturalHeight / imageRef.current.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const newPoints = [...points, { x, y }];
    setPoints(newPoints);

    if (newPoints.length === 4) {
      onComplete(newPoints);
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
          // 마커 위치는 클릭 좌표 (원본 해상도 기준)를 이미지 렌더링 사이즈 기준으로 환산해서 표시
          const img = imageRef.current;
          if (!img) return null;
          const scaleX = img.width / img.naturalWidth;
          const scaleY = img.height / img.naturalHeight;
          const x = pt.x * scaleX;
          const y = pt.y * scaleY;

          return (
            <div
              key={index}
              style={{
                position: "absolute",
                left: `${x}px`,
                top: `${y}px`,
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
