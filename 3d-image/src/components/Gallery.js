import React, { useState } from "react";
import "./Gallery.css";

const Gallery = ({ images }) => {
  const [selectedImage, setSelectedImage] = useState(null);

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleString("ko-KR");
  };

  const downloadImage = (imageUrl, filename) => {
    const link = document.createElement("a");
    link.href = imageUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (images.length === 0) {
    return null;
  }

  return (
    <div className="gallery">
      <h2>🖼️ 생성된 이미지 갤러리</h2>

      <div className="gallery-grid">
        {images.map((image, index) => (
          <div key={index} className="gallery-item">
            <div className="image-comparison">
              <div className="image-section">
                <h4>원본</h4>
                <img
                  src={image.original}
                  alt="Original"
                  onClick={() =>
                    setSelectedImage({
                      url: image.original,
                      title: "원본 이미지",
                    })
                  }
                />
              </div>

              <div className="image-section">
                <h4>스타일 변경 ({image.style})</h4>
                <img
                  src={image.styled}
                  alt="Styled"
                  onClick={() =>
                    setSelectedImage({
                      url: image.styled,
                      title: `${image.style} 스타일`,
                    })
                  }
                />
              </div>

              <div className="image-section">
                <h4>실사화</h4>
                <img
                  src={image.photorealistic}
                  alt="Photorealistic"
                  onClick={() =>
                    setSelectedImage({
                      url: image.photorealistic,
                      title: "실사화 이미지",
                    })
                  }
                />
              </div>
            </div>

            <div className="image-info">
              <p>
                <strong>스타일:</strong> {image.style}
              </p>
              <p>
                <strong>생성 시간:</strong> {formatDate(image.timestamp)}
              </p>
              <p>
                <strong>프롬프트:</strong> {image.prompt}
              </p>

              <div className="download-buttons">
                <button
                  onClick={() =>
                    downloadImage(
                      image.styled,
                      `styled_${image.style}_${Date.now()}.png`
                    )
                  }
                >
                  스타일 변경 이미지 다운로드
                </button>
                <button
                  onClick={() =>
                    downloadImage(
                      image.photorealistic,
                      `photorealistic_${Date.now()}.png`
                    )
                  }
                >
                  실사화 이미지 다운로드
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 모달 */}
      {selectedImage && (
        <div className="modal" onClick={() => setSelectedImage(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <span className="close" onClick={() => setSelectedImage(null)}>
              &times;
            </span>
            <h3>{selectedImage.title}</h3>
            <img src={selectedImage.url} alt="Selected" />
            <button
              onClick={() =>
                downloadImage(selectedImage.url, `vertex_ai_${Date.now()}.png`)
              }
            >
              다운로드
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Gallery;

