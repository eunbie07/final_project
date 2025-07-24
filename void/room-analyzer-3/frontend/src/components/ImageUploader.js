import React from 'react';

function ImageUploader({ onImageUpload }) {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onImageUpload(file);
    }
  };

  return (
    <div className="file-input-container">
      <input type="file" accept="image/*" onChange={handleFileChange} />
    </div>
  );
}

export default ImageUploader;