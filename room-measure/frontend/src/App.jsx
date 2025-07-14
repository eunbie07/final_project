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
  const [depthSize, setDepthSize] = useState({ width: 0, height: 0 });

  const handleImageUpload = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log("🔄 1. undistort 요청 시작");
      await axios.post("http://localhost:3000/undistort", formData);
      console.log("✅ 1. undistort 완료");

      setImage(file);
      setImageUrl(URL.createObjectURL(file));
      setResult(null);
      setDepthImageUrl(null);

      console.log("🔄 2. depth-map 요청 시작");
      // 1단계: depth-map → meta 정보 포함된 JSON 응답 받기
      const res = await axios.post("http://localhost:3000/depth-map", formData);
      console.log("✅ 2. depth-map 응답:", res.data);
      
      const { depth_image_url, depth_width, depth_height } = res.data;
      setDepthSize({ width: depth_width, height: depth_height });

      console.log("🔄 3. depth-map-image 요청 시작");
      // 2단계: 실제 시각화용 이미지 요청
      try {
        const imageRes = await axios.get("http://localhost:3000/depth-map-image", {
          responseType: "blob",
        });
        console.log("✅ 3. depth-map-image 응답 받음:", imageRes.status);
        
        const blobUrl = URL.createObjectURL(imageRes.data);
        setDepthImageUrl(blobUrl);
        console.log("✅ 모든 처리 완료");
      } catch (imageError) {
        console.error("❌ depth-map-image 요청 실패:", imageError);
        console.log("🔍 서버 응답:", imageError.response?.status, imageError.response?.data);
        
        // 이미지 로딩 실패해도 좌표 클릭은 가능하도록 처리
        alert("깊이 이미지 로딩에 실패했지만, 좌표 클릭은 가능합니다.");
      }
    } catch (error) {
      console.error("❌ 업로드 실패:", error);
      console.log("🔍 에러 세부사항:", {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        url: error.config?.url
      });
      alert(`업로드 실패: ${error.message}`);
    }
  };

  const handlePointsSubmit = async (points) => {
    try {
      console.log("서버로 보낼 좌표:", points);

      const payload = {
        points: points.map((pt) => ({
          x: parseFloat(pt.x),
          y: parseFloat(pt.y),
          z: parseFloat(pt.z),
        })),
      };

      const res1 = await axios.post(
        "http://localhost:3000/estimate-room-size",
        payload,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      setResult(res1.data);
    } catch (error) {
      console.error("거리 계산 또는 depth map 실패:", error);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-4">방 크기 추정기</h1>
      
      {/* 서버 상태 확인 버튼 추가 */}
      <button 
        onClick={async () => {
          try {
            const res = await axios.get("http://localhost:3000/health");
            console.log("서버 상태:", res.data);
            alert("서버 연결 정상");
          } catch (error) {
            console.error("서버 연결 실패:", error);
            alert("서버에 연결할 수 없습니다");
          }
        }}
        className="mb-4 px-4 py-2 bg-blue-500 text-white rounded"
      >
        서버 상태 확인
      </button>
      
      <ImageUploader onUpload={handleImageUpload} />
      {imageUrl && (
        <ImageClickArea
          imageUrl={imageUrl}
          onComplete={handlePointsSubmit}
          depthWidth={depthSize.width}
          depthHeight={depthSize.height}
        />
      )}
      {result && (
        <RoomResult
          result={result}
          depthImageUrl={depthImageUrl}
        />
      )}
    </div>
  );
}

export default App;