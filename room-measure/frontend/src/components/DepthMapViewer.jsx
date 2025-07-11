// frontend/src/components/DepthMapViewer.jsx
import React, { useState } from "react";
import axios from "axios";

const DepthMapViewer = () => {
  const [depthUrl, setDepthUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleDepthUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:3000/depth-map", formData, {
        responseType: "blob", // 이미지 응답
      });

      const url = URL.createObjectURL(res.data);
      setDepthUrl(url);
    } catch (err) {
      console.error("깊이 추정 실패:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-6">
      <h2 className="text-lg font-bold mb-2">Depth Map 보기</h2>
      <input type="file" accept="image/*" onChange={handleDepthUpload} />
      {loading && <p className="mt-2">분석 중입니다...</p>}
      {depthUrl && (
        <div className="mt-4">
          <img src={depthUrl} alt="depth map" style={{ maxWidth: "100%" }} />
        </div>
      )}
    </div>
  );
};

export default DepthMapViewer;
