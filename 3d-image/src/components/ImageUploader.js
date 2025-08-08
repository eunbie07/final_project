import React, { useRef } from "react";
import "./ImageUploader.css";

const ImageUploader = ({ onImageUpload, selectedImage }) => {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onImageUpload({
          file: file,
          url: e.target.result,
          name: file.name,
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onImageUpload({
          file: file,
          url: e.target.result,
          name: file.name,
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <div className="image-uploader">
      <h2>📸 이미지 업로드</h2>
      <div
        className="upload-area"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current.click()}
      >
        {selectedImage ? (
          <div className="selected-image">
            <img src={selectedImage.url} alt="Selected" />
            <p className="image-name">{selectedImage.name}</p>
          </div>
        ) : (
          <div className="upload-placeholder">
            <div className="upload-icon">📁</div>
            <p>클릭하거나 이미지를 드래그하여 업로드하세요</p>
            <p className="upload-hint">JPG, PNG, GIF 파일 지원</p>
          </div>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
      {selectedImage && (
        <button
          className="change-image-btn"
          onClick={() => fileInputRef.current.click()}
        >
          다른 이미지 선택
        </button>
      )}
    </div>
  );
};

export default ImageUploader;

