// frontend/src/RoomVisualizer.js

import React, { useRef, useEffect, useState, useCallback } from 'react';

function RoomVisualizer({ imageUrl, onPointsSelected }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [points, setPoints] = useState([]);
  const [canvasScale, setCanvasScale] = useState(1);
  const [isImageReady, setIsImageReady] = useState(false); // 이미지 준비 상태

  // 1. 점들을 캔버스에 그리는 함수 (useCallback으로 최적화)
  const drawPointsOnCanvas = useCallback((currentPoints, scale, imgElement) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height); // 캔버스 초기화
    if (imgElement) {
      ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height); // 이미지 그리기
    } else if (imageRef.current) {
        ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
    }

    currentPoints.forEach(point => {
      ctx.beginPath();
      ctx.arc(point.x * scale, point.y * scale, 5, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
      ctx.closePath();
    });
  }, []);


  // 2. 이미지가 로드될 때만 한 번 실행되어 캔버스 크기 및 이미지 그리기
  useEffect(() => {
    const img = imageRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) {
        console.log("useEffect: img or canvas ref is null on initial render. img:", !!img, "canvas:", !!canvas);
        return;
    }

    const ctx = canvas.getContext('2d');

    img.onload = () => {
      console.log("Image loaded successfully:", imageUrl);
      const maxWidth = 800;
      let drawWidth = img.naturalWidth;
      let drawHeight = img.naturalHeight;

      let scale = 1;
      if (img.naturalWidth > maxWidth) {
        scale = maxWidth / img.naturalWidth;
        drawWidth = maxWidth;
        drawHeight = img.naturalHeight * scale;
      }
      
      setCanvasScale(scale);

      canvas.width = drawWidth;
      canvas.height = drawHeight;
      ctx.drawImage(img, 0, 0, drawWidth, drawHeight);
      setIsImageReady(true); // 이미지가 캔버스에 성공적으로 그려지면 true로 설정
      console.log("Image ready, isImageReady set to true. Canvas dimensions:", canvas.width, "x", canvas.height);

      // 이미지가 로드되면 이전에 찍었던 점들을 다시 그림 (이미지 변경 시 points는 비어있을 것임)
      drawPointsOnCanvas(points, scale, img);
    };

    // imageUrl이 변경될 때마다 img src를 업데이트 (새 이미지 로드 트리거)
    // 이전에 생성된 URL은 해제
    img.src = imageUrl;

    return () => {
      if (img) {
          img.onload = null; // 이전 onload 핸들러 제거
      }
      setIsImageReady(false); // 새로운 이미지 로드 전에 상태 초기화
      // setPoints([]); // 🚨🚨🚨 이 줄을 제거하여 무한 루프를 방지합니다. 🚨🚨🚨
      console.log("useEffect cleanup: isImageReady set to false."); // 로그 메시지 수정
    };
  }, [imageUrl, points, drawPointsOnCanvas]);


  // 3. points 상태가 변경될 때마다 부모 컴포넌트로 전달 및 캔버스 업데이트
  useEffect(() => {
    onPointsSelected(points);
    drawPointsOnCanvas(points, canvasScale, imageRef.current);
  }, [points, onPointsSelected, canvasScale, drawPointsOnCanvas]);


  const handleClick = (event) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    
    if (!canvas || !img || !isImageReady) {
        console.log("handleClick: Canvas or Image not ready or isImageReady is false. Cannot click.");
        console.log(`Debug: canvas=${!!canvas}, img=${!!img}, isImageReady=${isImageReady}`);
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const clientX = event.clientX;
    const clientY = event.clientY;

    const canvasX = clientX - rect.left;
    const canvasY = clientY - rect.top;

    const newPoint = { x: Math.round(canvasX / canvasScale), y: Math.round(canvasY / canvasScale) };
    setPoints(prevPoints => [...prevPoints, newPoint]);
    console.log("Point added:", newPoint, "Total points:", points.length + 1);
  };

  const handleClearPoints = () => {
    setPoints([]);
    onPointsSelected([]); // 부모에게도 초기화 알림
    console.log("Points cleared.");
  };

  return (
    <div className="room-visualizer-container">
      {imageUrl && (
        <div style={{ position: 'relative', display: 'inline-block' }}>
          <img
            ref={imageRef}
            src={imageUrl}
            alt="Room to analyze"
            style={{ display: 'none' }}
          />
          <canvas
            ref={canvasRef}
            style={{
              cursor: 'crosshair',
              border: '1px solid #ddd'
            }}
            onClick={handleClick}
          />
          <button onClick={handleClearPoints} style={{ marginTop: '10px' }}>
            점 초기화 (Visualizer 내부)
          </button>
          <p>Visualizer 내부 선택된 점 개수: {points.length}</p>
        </div>
      )}
      <p>사진 위에서 점을 클릭하여 측정에 필요한 지점들을 지정하세요. (예: 바닥 코너 4개)</p>
    </div>
  );
}

export default RoomVisualizer;