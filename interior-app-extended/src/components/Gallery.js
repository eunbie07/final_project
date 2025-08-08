import React from 'react';

function Gallery({ images }) {
  return (
    <div style={{ marginTop: '20px' }}>
      <h3>결과 갤러리</h3>
      <div style={{ display: 'flex', flexWrap: 'wrap' }}>
        {images.map((img, idx) => (
          <img key={idx} src={img} alt="generated" style={{ width: '150px', margin: '5px' }} />
        ))}
      </div>
    </div>
  );
}

export default Gallery;
