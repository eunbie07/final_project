// frontend/src/components/ImageUploader.jsx
import React from "react";

const ImageUploader = ({ onUpload }) => {
  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
    }
  };

  return (
    <div className="mb-4">
      <label className="block mb-2 font-semibold">방 사진 업로드</label>
      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="border p-2 rounded"
      />
    </div>
  );
};

export default ImageUploader;
