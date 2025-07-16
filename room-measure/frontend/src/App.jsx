import React, { useState } from "react";
import axios from "axios";
import ImageUploader from "./components/ImageUploader";
import ImageClickArea from "./components/ImageClickArea";
import RoomResult from "./components/RoomResult";
import RoomBox from "./components/RoomBox";

const HOUSING_TYPES = [
  {
    label: "주택, 아파트, 빌라",
    value: "house",
    defaultCeiling: 230,
    description: "단독주택, 빌라",
  },
  {
    label: "오피스텔",
    value: "officetel",
    defaultCeiling: 235,
    description: "오피스텔",
  },
  {
    label: "사무실",
    value: "office",
    defaultCeiling: 240,
    description: "상업용 건물",
  },
];

const LoadingSpinner = ({ message }) => (
  <div className="flex items-center justify-center p-8">
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
      <p className="text-gray-600 font-medium">{message}</p>
    </div>
  </div>
);

const UploadStatus = ({ status, error }) => {
  if (error) {
    return (
      <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-center gap-2">
          <span className="text-red-600">업로드 실패</span>
        </div>
        <p className="text-red-700 mt-1 text-sm">{error}</p>
      </div>
    );
  }

  if (status) {
    return (
      <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
        <div className="flex items-center gap-2">
          <span className="text-green-600">업로드 성공</span>
        </div>
        <p className="text-green-700 mt-1 text-sm">{status}</p>
      </div>
    );
  }

  return null;
};

function App() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [depthImageUrl, setDepthImageUrl] = useState(null);
  const [depthSize, setDepthSize] = useState({ width: 0, height: 0 });
  const [housingType, setHousingType] = useState(HOUSING_TYPES[0].value);
  const [ceilingHeight, setCeilingHeight] = useState(
    HOUSING_TYPES[0].defaultCeiling
  );
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleImageUpload = async (file) => {
    if (!file) return;

    setIsProcessing(true);
    setUploadError(null);
    setUploadStatus(null);
    setResult(null);
    setDepthImageUrl(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploadStatus("이미지 전처리 중...");
      await axios.post("http://localhost:3000/undistort", formData);

      setImage(file);
      setImageUrl(URL.createObjectURL(file));

      setUploadStatus("AI 깊이 분석 중... (30초 소요)");
      const res = await axios.post("http://localhost:3000/depth-map", formData);
      const { depth_width, depth_height } = res.data;
      setDepthSize({ width: depth_width, height: depth_height });

      setUploadStatus("깊이 이미지 생성 중...");
      try {
        const imageRes = await axios.get(
          "http://localhost:3000/depth-map-image",
          { responseType: "blob" }
        );
        const blobUrl = URL.createObjectURL(imageRes.data);
        setDepthImageUrl(blobUrl);
        setUploadStatus("업로드 완료! 이제 사진에서 4개 점을 클릭해주세요.");
      } catch {
        setUploadStatus("깊이 이미지 로딩에 실패했지만, 측정은 가능합니다.");
      }
    } catch (error) {
      let errorMessage = "업로드에 실패했습니다.";
      if (error.response?.status === 500)
        errorMessage = "서버 오류입니다. 다른 이미지를 시도해보세요.";
      else if (error.code === "NETWORK_ERROR")
        errorMessage = "네트워크 연결을 확인해주세요.";
      setUploadError(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePointsSubmit = async (points) => {
    try {
      const payload = {
        points: points.map((pt) => ({
          x: parseFloat(pt.x),
          y: parseFloat(pt.y),
          z: parseFloat(pt.z),
        })),
        target_height: parseFloat(ceilingHeight) / 100,
      };
      const res = await axios.post(
        "http://localhost:3000/estimate-room-size",
        payload,
        {
          headers: { "Content-Type": "application/json" },
        }
      );
      setResult(res.data);
    } catch (error) {
      const message = error.response?.data?.error || "측정에 실패했습니다.";
      alert(`측정 실패: ${message}`);
    }
  };

  const handleHousingTypeChange = (e) => {
    const type = e.target.value;
    setHousingType(type);
    const found = HOUSING_TYPES.find((h) => h.value === type);
    if (found) setCeilingHeight(found.defaultCeiling);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-10">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            AI 방 크기 측정기
          </h1>
          <p className="text-gray-600 text-lg">
            방 사진을 업로드하고 4개 점을 클릭하면 정확한 크기를 측정해드립니다
          </p>
        </div>

        <div className="bg-white border border-gray-200 rounded-2xl shadow-xl p-8 mb-8">
          <div className="mb-8">
            <div className="w-full border-2 border-dashed border-gray-300 rounded-xl p-8 flex flex-col items-center justify-center bg-gray-50 hover:bg-gray-100">
              <span className="font-semibold text-xl mb-2 text-gray-700">
                방 사진을 여기에 드래그하세요
              </span>
              <span className="text-gray-500 text-base mb-4 text-center">
                정면 또는 비스듬한 각도의 사진을 권장합니다
              </span>
              <ImageUploader onUpload={handleImageUpload} />
            </div>
            {isProcessing && <LoadingSpinner message={uploadStatus} />}
            <UploadStatus status={uploadStatus} error={uploadError} />
          </div>

          <div className="border-t pt-6">
            <h3 className="font-semibold text-lg mb-4 text-gray-700">
              측정 설정
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block font-medium text-gray-700 mb-2">
                  건물 유형
                </label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-4 py-3"
                  value={housingType}
                  onChange={handleHousingTypeChange}
                >
                  {HOUSING_TYPES.map((h) => (
                    <option key={h.value} value={h.value}>
                      {h.label} ({h.description})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block font-medium text-gray-700 mb-2">
                  천장 높이
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="number"
                    value={ceilingHeight}
                    min={180}
                    max={300}
                    step={5}
                    onChange={(e) => setCeilingHeight(e.target.value)}
                    className="flex-1 border border-gray-300 rounded-lg px-4 py-3"
                  />
                  <span className="text-gray-500 text-base font-medium">
                    cm
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {imageUrl && !isProcessing && (
          <ImageClickArea
            imageUrl={imageUrl}
            onComplete={handlePointsSubmit}
            depthWidth={depthSize.width}
            depthHeight={depthSize.height}
          />
        )}

        {result && (
          <>
            <RoomResult result={result} depthImageUrl={depthImageUrl} />
            <RoomBox
              width={result.width_cm}
              depth={result.depth_cm}
              height={result.height_cm}
            />
          </>
        )}

        <div className="mt-12 text-center">
          <div className="bg-white rounded-lg p-6 shadow-md">
            <h3 className="font-semibold text-gray-700 mb-3">측정 팁</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
              <div className="flex items-start gap-2">
                📐 직사각형 방이 가장 정확하게 측정됩니다
              </div>
              <div className="flex items-start gap-2">
                📷 조명이 밝고 선명한 사진을 사용하세요
              </div>
              <div className="flex items-start gap-2">
                🎯 벽면의 모서리가 잘 보이게 촬영하세요
              </div>
              <div className="flex items-start gap-2">
                ⚡ 가구가 가리지 않은 벽면을 선택하세요
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
