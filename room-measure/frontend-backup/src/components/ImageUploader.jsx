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
        className="bg-primary text-white font-semibold px-6 py-2 rounded-lg shadow-sm hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary transition mb-2"
      >
        Upload
      </button>
      <span className="text-xs text-text-secondary text-center mt-1">
        (JPG, PNG and other image files only)
      </span>
    </div>
  );
};

export default ImageUploader;
