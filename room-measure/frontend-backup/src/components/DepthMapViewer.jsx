// frontend/src/components/DepthMapViewer.jsx
import React, { useState, useRef } from "react";
import { generateDepthMap, getDepthMapImage } from "../utils/api";

const DepthMapViewer = () => {
  const [depthUrl, setDepthUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleDepthUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      await generateDepthMap();
      const imageBlob = await getDepthMapImage();
      const url = URL.createObjectURL(imageBlob);
      setDepthUrl(url);
    } catch (err) {
      console.error("깊이 추정 실패:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-6 p-6 bg-surface rounded-xl shadow-lg border border-border">
      <h2 className="text-2xl font-bold mb-4 text-text-primary">Depth Map 보기</h2>
      <input type="file" accept="image/*" onChange={handleDepthUpload} ref={fileInputRef} className="hidden" />
      <button
        type="button"
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
        className="px-4 py-2 bg-primary hover:bg-secondary text-white rounded-lg font-medium transition-colors mb-4"
      >
        깊이 맵 이미지 업로드
      </button>
      {loading && <p className="mt-2 text-text-secondary">분석 중입니다...</p>}
      {depthUrl && (
        <div className="mt-4 p-4 border border-border rounded-lg bg-background">
          <img src={depthUrl} alt="depth map" className="max-w-full h-auto rounded-lg" />
        </div>
      )}
    </div>
  );
};

export default DepthMapViewer;
