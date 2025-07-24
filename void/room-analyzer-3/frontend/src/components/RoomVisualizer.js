import React, { useRef, useEffect, useState } from 'react';

function RoomVisualizer({ imageUrl, onPointsSelected }) {
  const canvasRef = useRef(null);
  const [points, setPoints] = useState([]);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageRatio, setImageRatio] = useState(1);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = imageUrl;

    img.onload = () => {
      const maxWidth = 800;
      let drawWidth = img.width;
      let drawHeight = img.height;

      if (img.width > maxWidth) {
        drawWidth = maxWidth;
        drawHeight = (img.height / img.width) * maxWidth;
      }

      setImageRatio(img.width / drawWidth); 
      
      canvas.width = drawWidth;
      canvas.height = drawHeight;
      ctx.drawImage(img, 0, 0, drawWidth, drawHeight);
      setImageLoaded(true);

      points.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x / imageRatio, p.y / imageRatio, 5, 0, 2 * Math.PI); 
        ctx.fillStyle = 'red';
        ctx.fill();
        ctx.closePath();
      });
    };

    if (imageUrl) {
        img.src = imageUrl;
    }

  }, [imageUrl, points, imageRatio]);

  const handleClick = (event) => {
    if (!imageLoaded) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const actualX = Math.round(x * imageRatio);
    const actualY = Math.round(y * imageRatio);

    const newPoints = [...points, { x: actualX, y: actualY }];
    setPoints(newPoints);
    onPointsSelected(newPoints);
  };

  const clearPoints = () => {
    setPoints([]);
    onPointsSelected([]);
  };

  return (
    <div>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{ border: '1px solid black', cursor: 'crosshair' }}
      ></canvas>
      <p>사진 위에서 점을 클릭하여 측정에 필요한 지점들을 지정하세요. (예: 바닥 코너 4개)</p>
      <button onClick={clearPoints}>점 초기화</button>
      <p>선택된 점 개수: {points.length}</p>
    </div>
  );
}

export default RoomVisualizer;