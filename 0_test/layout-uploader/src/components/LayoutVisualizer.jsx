// src/components/LayoutVisualizer.jsx
import React, { useState } from 'react';

const LayoutVisualizer = ({ imageUrl, layout }) => {
  const [imgSize, setImgSize] = useState({ width: 0, height: 0 });

  if (!imageUrl || !layout || layout.length === 0) return null;

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      {/* 원본 이미지 */}
      <img
        src={imageUrl}
        alt="업로드 이미지"
        style={{ width: '100%', maxWidth: '600px', display: 'block' }}
        onLoad={(e) => {
          const { width, height } = e.target.getBoundingClientRect();
          setImgSize({ width, height });
        }}
      />

      {/* 감지된 객체 박스 */}
      {imgSize.width > 0 &&
        layout.map((item, index) => (
          <div
            key={index}
            style={{
              position: 'absolute',
              left: `${item.x * imgSize.width}px`,
              top: `${item.y * imgSize.height}px`,
              width: `${item.w * imgSize.width}px`,
              height: `${item.h * imgSize.height}px`,
              border: '2px solid red',
              color: 'red',
              fontSize: '12px',
              fontWeight: 'bold',
              pointerEvents: 'none',
            }}
          >
            <div style={{ background: 'white', padding: '0 2px' }}>
              {item.label}
            </div>
          </div>
        ))}
    </div>
  );
};

export default LayoutVisualizer;
