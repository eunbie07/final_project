// frontend/src/components/ImageUploader.jsx
import React, { useRef } from "react";

const ImageUploader = ({ onUpload }) => {
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div className="w-full flex flex-col items-center justify-center">
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
      />
      <button
        type="button"
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
        className="bg-white border border-gray-400 text-gray-800 font-semibold px-6 py-2 rounded-lg shadow-sm hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-400 transition mb-2"
      >
        Upload
      </button>
      <span className="text-xs text-gray-500 text-center mt-1">
        (jpg, png 등 이미지 파일만 업로드 가능)
      </span>
    </div>
  );
};

export default ImageUploader;
