// App.jsx
import React, { useState } from "react";
import axios from "axios";
import ImageUploader from "./components/ImageUploader";
import ImageClickArea from "./components/ImageClickArea";
import RoomResult from "./components/RoomResult";

function App() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [depthImageUrl, setDepthImageUrl] = useState(null);

  const handleImageUpload = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      await axios.post("http://localhost:3000/undistort", formData);
      setImage(file);
      setImageUrl(URL.createObjectURL(file));
      setResult(null);
      setDepthImageUrl(null);  // 초기화
    } catch (error) {
      console.error("업로드 실패:", error);
    }
  };

  const handlePointsSubmit = async (points) => {
    try {
      const res1 = await axios.post("http://localhost:3000/estimate-room-size", {
        points: points,
      });
      setResult(res1.data);

      const formData = new FormData();
      formData.append("file", image);
      const res2 = await axios.post("http://localhost:3000/depth-map", formData, {
        responseType: "blob",
      });

      const blobUrl = URL.createObjectURL(res2.data);
      setDepthImageUrl(blobUrl);
    } catch (error) {
      console.error("거리 계산 또는 depth map 실패:", error);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-4">방 크기 추정기</h1>
      <ImageUploader onUpload={handleImageUpload} />
      {imageUrl && (
        <ImageClickArea imageUrl={imageUrl} onComplete={handlePointsSubmit} />
      )}
      {result && (
        <RoomResult
          x={result.x_cm}
          y={result.y_cm}
          depthImageUrl={depthImageUrl}
        />
      )}
    </div>
  );
}

export default App;
