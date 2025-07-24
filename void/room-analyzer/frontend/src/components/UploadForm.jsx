import React, { useState } from "react";
import CeilingFloorSelector from "./CeilingFloorSelector";

const UploadForm = ({ onResult }) => {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [ceiling, setCeiling] = useState("230");
  const [points, setPoints] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setImageUrl(URL.createObjectURL(selectedFile)); // 미리보기용 URL 생성
      setPoints(null); // 새 이미지 업로드 시 좌표 초기화
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !points || points.ceilingY == null || points.floorY == null) {
      alert("파일과 천장/바닥 위치를 모두 선택해주세요.");
      return;
    }

    const formData = new FormData();
    formData.append("photo", file);
    formData.append("ceiling", ceiling);
    formData.append("ceiling_y", points.ceilingY);  // 백엔드와 키 맞춤
    formData.append("floor_y", points.floorY);      // 백엔드와 키 맞춤

    setLoading(true);
    try {
      const res = await fetch("http://13.55.21.100:3000/analyze-room", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}`);
      }

      const data = await res.json();
      onResult(data);
    } catch (error) {
      console.error("Upload error:", error);
      alert("분석 요청 중 오류가 발생했습니다.");
    }
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginBottom: "20px" }}>
      <h3>방 사진 업로드</h3>
      <input type="file" accept="image/*" onChange={handleFileChange} />

      <br />
      <label>
        층고(cm):&nbsp;
        <input
          type="number"
          value={ceiling}
          onChange={(e) => setCeiling(e.target.value)}
          min="200"
          max="300"
        />
      </label>

      <br />
      {imageUrl && (
        <CeilingFloorSelector
          imageUrl={imageUrl}
          onPointsSelected={setPoints}
        />
      )}

      <br />
      <button type="submit" disabled={loading || !points}>
        {loading ? "분석 중..." : "분석 시작"}
      </button>
    </form>
  );
};

export default UploadForm;
