// frontend/src/FloorPlanViewer.js

import React, { useRef, useEffect } from 'react';

function FloorPlanViewer({ vertices, width, height }) {
  const canvasRef = useRef(null);

  const metersToPixels = 50; // 예: 1미터당 50픽셀

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !vertices || vertices.length === 0) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    canvas.width = width * metersToPixels;
    canvas.height = height * metersToPixels;

    ctx.fillStyle = '#f0f0f0'; // 배경색
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = 'blue'; // 선 색상
    ctx.lineWidth = 2; // 선 두께

    if (vertices.length > 0) {
      ctx.beginPath();
      ctx.moveTo(vertices[0].x * metersToPixels, vertices[0].y * metersToPixels);

      for (let i = 1; i < vertices.length; i++) {
        ctx.lineTo(vertices[i].x * metersToPixels, vertices[i].y * metersToPixels);
      }
      ctx.closePath();
      ctx.stroke();

      ctx.fillStyle = 'red';
      vertices.forEach(v => {
        ctx.beginPath();
        ctx.arc(v.x * metersToPixels, v.y * metersToPixels, 4, 0, 2 * Math.PI);
        ctx.fill();
      });

      ctx.fillStyle = 'black';
      ctx.font = '14px Arial';
      
      // 가로 길이 (첫 번째 변)
      if (vertices.length >= 2) {
        const p1 = vertices[0];
        const p2 = vertices[1];
        const midX = ((p1.x + p2.x) / 2) * metersToPixels;
        const midY = ((p1.y + p2.y) / 2) * metersToPixels - 10;
        const distance = Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2)).toFixed(2);
        ctx.fillText(`${distance}m`, midX, midY);
      }

      // 세로 길이 (두 번째 변)
      if (vertices.length >= 3) {
        const p2 = vertices[1];
        const p3 = vertices[2];
        const midX = ((p2.x + p3.x) / 2) * metersToPixels + 10;
        const midY = ((p2.y + p3.y) / 2) * metersToPixels;
        const distance = Math.sqrt(Math.pow(p3.x - p2.x, 2) + Math.pow(p3.y - p2.y, 2)).toFixed(2);
        ctx.fillText(`${distance}m`, midX, midY);
      }
    }
  }, [vertices, width, height, metersToPixels]); // metersToPixels 의존성 추가

  if (!vertices || vertices.length === 0) {
    return <p>2D 평면도를 생성할 수 없습니다. 4개 이상의 점을 선택해주세요.</p>;
  }

  return (
    <div className="floor-plan-viewer-container">
      <h3>추정된 2D 평면도</h3>
      <canvas ref={canvasRef} style={{ border: '1px solid #ccc' }}></canvas>
      <p>평면도 스케일: 1m = {metersToPixels}px</p>
    </div>
  );
}

export default FloorPlanViewer;