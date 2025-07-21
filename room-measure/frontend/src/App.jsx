import React, { useState } from "react";
import axios from "axios";
import ImageUploader from "./components/ImageUploader";
import ImageClickArea from "./components/ImageClickArea";
import RoomResult from "./components/RoomResult";
import RoomBox from "./components/RoomBox";
import FurniturePlacement from "./components/FurniturePlacement";

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
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-pink-400 mx-auto mb-4"></div>
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
      <div className="mt-4 p-4 bg-pink-50 border border-pink-200 rounded-lg">
        <div className="flex items-center gap-2">
          <span className="text-pink-600">업로드 성공</span>
        </div>
        <p className="text-pink-700 mt-1 text-sm">{status}</p>
      </div>
    );
  }

  return null;
};

const HelpTips = ({ isOpen, onClose }) => (
  <div
    className={`fixed inset-0 z-50 ${
      isOpen ? "flex" : "hidden"
    } items-center justify-center bg-black bg-opacity-50`}
  >
    <div className="bg-white rounded-xl p-6 shadow-xl max-w-lg mx-4 relative">
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
      >
        ✕
      </button>
      <h3 className="font-semibold text-gray-700 mb-4 text-lg">
        측정 가이드
      </h3>
      <div className="space-y-3">
        <div className="bg-pink-50/50 p-3 rounded-lg">
          <div className="font-medium text-gray-700 mb-1">
            최적의 방 형태
          </div>
          <p className="text-sm text-gray-600">
            직사각형 방이 가장 정확하게 측정됩니다
          </p>
        </div>
        <div className="bg-pink-50/50 p-3 rounded-lg">
          <div className="font-medium text-gray-700 mb-1">사진 촬영 팁</div>
          <p className="text-sm text-gray-600">
            조명이 밝고 선명한 사진을 사용하세요
          </p>
        </div>
        <div className="bg-pink-50/50 p-3 rounded-lg">
          <div className="font-medium text-gray-700 mb-1">모서리 포인트</div>
          <p className="text-sm text-gray-600">
            벽면의 모서리가 잘 보이게 촬영하세요
          </p>
        </div>
        <div className="bg-pink-50/50 p-3 rounded-lg">
          <div className="font-medium text-gray-700 mb-1">가구 배치</div>
          <p className="text-sm text-gray-600">
            가구가 가리지 않은 벽면을 선택하세요
          </p>
        </div>
      </div>
    </div>
  </div>
);

function App() {
  const [isHelpOpen, setIsHelpOpen] = useState(false);
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
  const [activeTab, setActiveTab] = useState("analysis");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [manualResult, setManualResult] = useState(null);
  const [autoResult, setAutoResult] = useState(null);
  const [selectedMethod, setSelectedMethod] = useState("manual"); // "manual" or "auto"

  const handleTabClick = (tabName) => {
    setActiveTab(tabName);
    if (tabName === "3d") {
      // 3D 뷰어 탭으로 전환될 때 전체화면 요청
      setTimeout(() => {
        const element = document.querySelector('.room-3d-viewer');
        if (element) {
          if (element.requestFullscreen) {
            element.requestFullscreen();
          } else if (element.webkitRequestFullscreen) {
            element.webkitRequestFullscreen();
          } else if (element.mozRequestFullScreen) {
            element.mozRequestFullScreen();
          } else if (element.msRequestFullscreen) {
            element.msRequestFullscreen();
          }
        }
      }, 100); // DOM 업데이트를 기다리기 위해 약간의 지연
    }
  };

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

  const handlePointsSubmit = async (points, method = "manual") => {
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
      
      // 결과를 방법에 따라 저장
      const resultData = { ...res.data, detectionMethod: method };
      
      if (method === "auto") {
        setAutoResult(resultData);
        setSelectedMethod("auto");
      } else {
        setManualResult(resultData);
        setSelectedMethod("manual");
      }
      
      // 기본 result는 선택된 방법으로 설정
      setResult(resultData);
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

  // 전체화면 상태 감지
  React.useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-50 to-pink-100 py-4 sm:py-6 md:py-8">
      {/* 전체 폭 사용하도록 max-w 제거하고 px 조정 */}
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* 헤더 */}
        <div className="text-center mb-8 md:mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-3">
            <strong>Room Measurement Tool</strong>
          </h1>
          <div className="flex items-center justify-center gap-2">
            <p className="text-gray-600 text-base sm:text-lg max-w-2xl">
              Measure accurate room dimensions from a single photo using artificial intelligence
            </p>
            <button
              onClick={() => setIsHelpOpen(true)}
              className="ml-2 p-2 text-pink-600 hover:text-pink-700 transition-colors"
              title="측정 가이드"
            >
              ❔
            </button>
          </div>
        </div>

        {/* 이미지 업로드 섹션 - max-w-6xl로 확장 */}
        {!result && (
          <div className="max-w-6xl mx-auto">
            <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-4 sm:p-6 lg:p-8 mb-6">
              <div className="mb-6">
                <div className="w-full border-2 border-dashed border-pink-200 rounded-xl p-6 sm:p-8 flex flex-col items-center justify-center bg-pink-50/50 hover:bg-pink-50 transition-colors">
                  <span className="font-medium text-lg sm:text-xl mb-2 text-gray-700">
                    <strong>Upload Photo</strong>
                  </span>
                  <span className="text-gray-500 text-sm sm:text-base mb-4 text-center">
                    Please select a photo where the room corners are clearly visible
                  </span>
                  <ImageUploader onUpload={handleImageUpload} />
                </div>
                {isProcessing && <LoadingSpinner message={uploadStatus} />}
                <UploadStatus status={uploadStatus} error={uploadError} />
              </div>

              <div className="border-t border-pink-100 pt-6">
                <h3 className="font-medium text-lg mb-4 text-gray-800">
                  <strong>측정 설정</strong>
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                  <div>
                    <label className="block font-medium text-gray-700 mb-2">
                      <strong>건물 유형</strong>
                    </label>
                    <select
                      className="w-full border border-pink-200 rounded-lg px-4 py-3 focus:border-pink-300 focus:ring focus:ring-pink-200 focus:ring-opacity-50"
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
                      <strong>천장 높이</strong>
                    </label>
                    <div className="flex items-center gap-3">
                      <input
                        type="number"
                        value={ceilingHeight}
                        min={180}
                        max={300}
                        step={5}
                        onChange={(e) => setCeilingHeight(e.target.value)}
                        className="flex-1 border border-pink-200 rounded-lg px-4 py-3 focus:border-pink-300 focus:ring focus:ring-pink-200 focus:ring-opacity-50"
                      />
                      <span className="text-gray-500 text-base font-medium">
                        cm
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 이미지 클릭 영역 - 전체 폭 사용 */}
        {imageUrl && !isProcessing && !result && (
          <div className="max-w-7xl mx-auto">
            <ImageClickArea
              imageUrl={imageUrl}
              onComplete={handlePointsSubmit}
              depthWidth={depthSize.width}
              depthHeight={depthSize.height}
            />
          </div>
        )}

        {/* 결과 섹션 - 전체 폭 사용 */}
        {result && (
          <div className="space-y-6">
            <div className="bg-white/90 backdrop-blur border border-pink-100 rounded-2xl shadow-xl overflow-visible">
              <div className="flex flex-wrap border-b border-pink-100">
                <button
                  onClick={() => setActiveTab("analysis")}
                  className={`flex-1 min-w-[120px] px-4 sm:px-6 py-3 sm:py-4 text-center font-medium transition-colors text-sm sm:text-base ${
                    activeTab === "analysis"
                      ? "bg-pink-50 text-pink-600 border-b-2 border-pink-500"
                      : "text-gray-600 hover:text-gray-800 hover:bg-pink-50/50"
                  }`}
                >
                  <strong>Analysis</strong>
                </button>
                <button
                  onClick={() => setActiveTab("2d")}
                  className={`flex-1 min-w-[120px] px-4 sm:px-6 py-3 sm:py-4 text-center font-medium transition-colors text-sm sm:text-base ${
                    activeTab === "2d"
                      ? "bg-pink-50 text-pink-600 border-b-2 border-pink-500"
                      : "text-gray-600 hover:text-gray-800 hover:bg-pink-50/50"
                  }`}
                >
                  <strong>2D Floor Plan</strong>
                </button>
                <button
                  onClick={() => setActiveTab("furniture")}
                  className={`flex-1 min-w-[120px] px-4 sm:px-6 py-3 sm:py-4 text-center font-medium transition-colors text-sm sm:text-base ${
                    activeTab === "furniture"
                      ? "bg-pink-50 text-pink-600 border-b-2 border-pink-500"
                      : "text-gray-600 hover:text-gray-800 hover:bg-pink-50/50"
                  }`}
                >
                  <strong>Furniture Layout</strong>
                </button>
                <button
                  onClick={() => handleTabClick("3d")}
                  className={`flex-1 min-w-[120px] px-4 sm:px-6 py-3 sm:py-4 text-center font-medium transition-colors text-sm sm:text-base ${
                    activeTab === "3d"
                      ? "bg-pink-50 text-pink-600 border-b-2 border-pink-500"
                      : "text-gray-600 hover:text-gray-800 hover:bg-pink-50/50"
                  }`}
                >
                  <strong>3D Viewer</strong>
                </button>
              </div>

              <div className="p-4 sm:p-6 lg:p-8">
                {activeTab === "analysis" && (
                  <div className="max-w-6xl mx-auto space-y-6">
                    {/* 측정 방법 비교 (자동/수동 모두 있을 때만) */}
                    {manualResult && autoResult && (
                      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 shadow-sm border border-blue-200">
                        <h3 className="text-xl font-bold mb-4 text-gray-800">
                          <strong>Measurement Comparison</strong>
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          {/* 수동 방식 결과 */}
                          <div className="bg-white rounded-lg p-4 border-2 border-gray-200">
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="font-semibold text-gray-700">📍 Manual Detection</h4>
                              <button
                                onClick={() => {
                                  setSelectedMethod("manual");
                                  setResult(manualResult);
                                }}
                                className={`px-3 py-1 rounded text-sm font-medium ${
                                  selectedMethod === "manual"
                                    ? "bg-pink-600 text-white"
                                    : "bg-gray-200 text-gray-600 hover:bg-gray-300"
                                }`}
                              >
                                {selectedMethod === "manual" ? "✓ Selected" : "Select"}
                              </button>
                            </div>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span>Width:</span>
                                <span className="font-medium">{(manualResult.width_cm / 100).toFixed(1)}m</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Depth:</span>
                                <span className="font-medium">{(manualResult.depth_cm / 100).toFixed(1)}m</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Area:</span>
                                <span className="font-medium">{((manualResult.width_cm * manualResult.depth_cm) / 10000).toFixed(1)}㎡</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Confidence:</span>
                                <span className="font-medium">{(manualResult.confidence * 100).toFixed(0)}%</span>
                              </div>
                            </div>
                          </div>

                          {/* 자동 방식 결과 */}
                          <div className="bg-white rounded-lg p-4 border-2 border-blue-200">
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="font-semibold text-gray-700">🤖 Auto Detection (RoomNet)</h4>
                              <button
                                onClick={() => {
                                  setSelectedMethod("auto");
                                  setResult(autoResult);
                                }}
                                className={`px-3 py-1 rounded text-sm font-medium ${
                                  selectedMethod === "auto"
                                    ? "bg-blue-600 text-white"
                                    : "bg-gray-200 text-gray-600 hover:bg-gray-300"
                                }`}
                              >
                                {selectedMethod === "auto" ? "✓ Selected" : "Select"}
                              </button>
                            </div>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span>Width:</span>
                                <span className="font-medium">{(autoResult.width_cm / 100).toFixed(1)}m</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Depth:</span>
                                <span className="font-medium">{(autoResult.depth_cm / 100).toFixed(1)}m</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Area:</span>
                                <span className="font-medium">{((autoResult.width_cm * autoResult.depth_cm) / 10000).toFixed(1)}㎡</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Confidence:</span>
                                <span className="font-medium">{(autoResult.confidence * 100).toFixed(0)}%</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* 차이점 분석 */}
                        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                          <h5 className="font-medium text-gray-700 mb-2">Difference Analysis</h5>
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                              <span className="text-gray-600">Width Difference:</span>
                              <span className="ml-2 font-medium">
                                {Math.abs((autoResult.width_cm - manualResult.width_cm) / 100).toFixed(2)}m
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-600">Depth Difference:</span>
                              <span className="ml-2 font-medium">
                                {Math.abs((autoResult.depth_cm - manualResult.depth_cm) / 100).toFixed(2)}m
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-600">Area Difference:</span>
                              <span className="ml-2 font-medium">
                                {Math.abs(((autoResult.width_cm * autoResult.depth_cm) - (manualResult.width_cm * manualResult.depth_cm)) / 10000).toFixed(2)}㎡
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* 분석 개요 */}
                    <div className="bg-gradient-to-r from-pink-50 to-rose-50 rounded-lg p-6 shadow-sm">
                      <h3 className="text-xl font-bold mb-3 text-gray-800">
                        <strong>AI Room Measurement Analysis</strong>
                        {result?.detectionMethod && (
                          <span className="ml-3 text-sm font-normal">
                            ({result.detectionMethod === "auto" ? "🤖 RoomNet Auto-Detection" : "📍 Manual 4-Point"})
                          </span>
                        )}
                      </h3>
                      <p className="text-gray-600 mb-4">
                        Advanced computer vision and deep learning technology for accurate room size measurement from a single photo.
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-pink-500 rounded-full"></span>
                          <span>Accuracy: ±5~10cm</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-rose-500 rounded-full"></span>
                          <span>Processing time: ~30 seconds</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-pink-500 rounded-full"></span>
                          <span>Supported: Rectangular rooms</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-rose-500 rounded-full"></span>
                          <span>Required: Ceiling height</span>
                        </div>
                      </div>
                    </div>

                    {/* 원본 이미지와 깊이 분석 이미지 비교 */}
                    <div className="bg-white rounded-lg p-4 shadow-sm">
                      <h3 className="text-lg font-semibold mb-3 text-gray-700">                      <strong>이미지 분석 과정</strong></h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* 원본 이미지 */}
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2">                          <strong>원본 이미지</strong></h4>
                          {imageUrl && (
                            <div className="relative">
                              <img 
                                src={imageUrl} 
                                alt="원본 이미지" 
                                className="w-full rounded-lg shadow-md"
                              />
                              <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
                                INPUT
                              </div>
                            </div>
                          )}
                          <p className="text-sm text-gray-600 mt-2">
                            사용자가 업로드한 방 사진
                          </p>
                        </div>

                        {/* AI 깊이 분석 이미지 */}
                        {depthImageUrl && (
                          <div>
                            <h4 className="font-medium text-gray-700 mb-2">                            <strong>AI 깊이 분석</strong></h4>
                            <div className="relative">
                              <img 
                                src={depthImageUrl} 
                                alt="깊이 분석 이미지" 
                                className="w-full rounded-lg shadow-md"
                              />
                              <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
                                DEPTH MAP
                              </div>
                            </div>
                            <p className="text-sm text-gray-600 mt-2">
                              밝은 부분은 가까운 거리, 어두운 부분은 먼 거리를 나타냄
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* 분석 과정 설명 */}
                    <div className="bg-white rounded-lg p-4 shadow-sm">
                      <h3 className="text-lg font-semibold mb-3 text-gray-700">4단계 분석 과정</h3>
                      <div className="space-y-4">
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 bg-rose-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</div>
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-700">이미지 전처리</h4>
                            <p className="text-sm text-gray-600 mb-1">카메라 왜곡 보정 및 이미지 최적화</p>
                            <div className="text-xs text-gray-500 bg-rose-50 p-2 rounded">
                              • 렌즈 왜곡 제거 • 노이즈 감소 • 해상도 최적화
                            </div>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 bg-pink-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</div>
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-700">AI 깊이 분석</h4>
                            <p className="text-sm text-gray-600 mb-1">MiDaS 딥러닝 모델로 픽셀별 깊이 정보 추출</p>
                            <div className="text-xs text-gray-500 bg-pink-50 p-2 rounded">
                              • Transformer 기반 네트워크 • 단안 깊이 추정 • 상대적 깊이 계산
                            </div>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 bg-rose-600 text-white rounded-full flex items-center justify-center text-sm font-bold">3</div>
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-700">모서리 포인트 매핑</h4>
                            <p className="text-sm text-gray-600 mb-1">사용자가 선택한 4개 모서리와 깊이 정보 연결</p>
                            <div className="text-xs text-gray-500 bg-rose-50 p-2 rounded">
                              • 픽셀 좌표 변환 • 깊이 값 추출 • 원근 보정
                            </div>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 bg-pink-600 text-white rounded-full flex items-center justify-center text-sm font-bold">4</div>
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-700">실제 크기 계산</h4>
                            <p className="text-sm text-gray-600 mb-1">천장 높이 기준으로 상대적 깊이를 절대적 크기로 변환</p>
                            <div className="text-xs text-gray-500 bg-pink-50 p-2 rounded">
                              • 스케일 팩터 팩터 계산 • 비례 관계 적용 • 오차 보정
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* 분석 결과 요약 */}
                    <div className="bg-white rounded-lg p-4 shadow-sm">
                      <h3 className="text-lg font-semibold mb-3 text-gray-700">Measurement Results</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="text-center p-3 bg-rose-50 rounded-lg">
                          <div className="text-2xl font-bold text-rose-600">{(result.width_cm / 100).toFixed(1)}m</div>
                          <div className="text-sm text-gray-600">Width</div>
                        </div>
                        <div className="text-center p-3 bg-pink-50 rounded-lg">
                          <div className="text-2xl font-bold text-pink-600">{(result.depth_cm / 100).toFixed(1)}m</div>
                          <div className="text-sm text-gray-600">Depth</div>
                        </div>
                        <div className="text-center p-3 bg-rose-100 rounded-lg">
                          <div className="text-2xl font-bold text-rose-700">{(result.height_cm / 100).toFixed(1)}m</div>
                          <div className="text-sm text-gray-600">Height</div>
                        </div>
                      </div>
                      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="text-center p-3 bg-pink-100 rounded-lg">
                          <div className="text-xl font-bold text-pink-700">
                            {((result.width_cm * result.depth_cm) / 10000).toFixed(1)}㎡
                          </div>
                          <div className="text-sm text-gray-600">Total Area</div>
                        </div>
                        <div className="text-center p-3 bg-rose-200 rounded-lg">
                          <div className="text-xl font-bold text-rose-800">
                            {((result.width_cm * result.depth_cm * result.height_cm) / 1000000).toFixed(1)}㎥
                          </div>
                          <div className="text-sm text-gray-600">Total Volume</div>
                        </div>
                      </div>
                    </div>

                    {/* 기술적 세부사항 */}
                    <div className="bg-white rounded-lg p-4 shadow-sm">
                      <h3 className="text-lg font-semibold mb-3 text-gray-700">Technical Information</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2">Technologies Used</h4>
                          <ul className="text-sm text-gray-600 space-y-1">
                            <li>• MiDaS v3.1 (Mixed Data Sampling)</li>
                            <li>• Vision Transformer Architecture</li>
                            <li>• Monocular Depth Estimation</li>
                            <li>• Camera Distortion Correction</li>
                            <li>• 3D Geometric Transformation</li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-700 mb-2">Measurement Details</h4>
                          <div className="text-sm text-gray-600 space-y-1">
                            <div className="flex justify-between">
                              <span>Depth Map Resolution:</span>
                              <span>{depthSize.width} × {depthSize.height}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Ceiling Height:</span>
                              <span>{(ceilingHeight / 100).toFixed(1)}m</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Building Type:</span>
                              <span>{HOUSING_TYPES.find(h => h.value === housingType)?.label}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Status:</span>
                              <span className="text-pink-600 font-medium">✓ Complete</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* 정확도 및 한계 정보 */}
                    <div className="bg-gradient-to-r from-pink-50 to-rose-50 rounded-lg p-4 shadow-sm border border-pink-200">
                      <h3 className="text-lg font-semibold mb-3 text-pink-800">Accuracy & Limitations</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-medium text-pink-700 mb-2">Accuracy</h4>
                          <ul className="text-sm text-pink-700 space-y-1">
                            <li>• Typical error range: ±5~10cm</li>
                            <li>• Best performance in rectangular rooms</li>
                            <li>• Better lighting improves accuracy</li>
                            <li>• Clear corner lines required</li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium text-pink-700 mb-2">Limitations</h4>
                          <ul className="text-sm text-pink-700 space-y-1">
                            <li>• Circular or polygonal rooms may be inaccurate</li>
                            <li>• Too much furniture hinders measurement</li>
                            <li>• Limited in extremely dark environments</li>
                            <li>• Depends on ceiling height accuracy</li>
                          </ul>
                        </div>
                      </div>
                      <div className="mt-3 p-3 bg-white bg-opacity-50 rounded border border-pink-300">
                        <p className="text-xs text-pink-800">
                          <strong>Tip:</strong> For more accurate measurements, take photos from angles where room corners are clearly visible, 
                          and ensure furniture doesn't block the walls.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === "2d" && (
                  <div className="max-w-6xl mx-auto">
                    <RoomResult result={result} depthImageUrl={depthImageUrl} />
                  </div>
                )}
                
                {activeTab === "3d" && (
                  <div className="w-full">
                    {/* 3D 뷰어 - 높이 자동 조절 */}
                    <div className="relative">
                      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-50">
                        <button
                          onClick={() => {
                            const element = document.querySelector('.room-3d-viewer');
                            if (element.requestFullscreen) {
                              element.requestFullscreen();
                            } else if (element.webkitRequestFullscreen) {
                              element.webkitRequestFullscreen();
                            } else if (element.mozRequestFullScreen) {
                              element.mozRequestFullScreen();
                            } else if (element.msRequestFullscreen) {
                              element.msRequestFullscreen();
                            }
                          }}
                          className="px-4 py-2 bg-black bg-opacity-50 text-white rounded-lg hover:bg-opacity-70 transition-all duration-200 flex items-center gap-2"
                          title="전체화면으로 보기"
                        >
                          <span>⛶</span>
                          <span className="text-sm">전체화면</span>
                        </button>
                      </div>
                      <div className={`room-3d-viewer bg-black rounded-lg ${
                        isFullscreen 
                          ? 'fixed inset-0 w-screen h-screen z-[9999]' 
                          : 'min-h-[500px] md:min-h-[600px] lg:min-h-[700px]'
                      }`}>
                        <RoomBox
                          width={result.width_cm}
                          depth={result.depth_cm}
                          height={result.height_cm}
                          isFullscreen={isFullscreen}
                        />
                      </div>
                    </div>
                  </div>
                )}
                
                {activeTab === "furniture" && (
                  <div className="w-full">
                    {/* 가구 배치 - 전체 폭 활용 */}
                    <FurniturePlacement
                      roomWidth={result.width_cm}
                      roomHeight={result.depth_cm}
                    />
                  </div>
                )}
              </div>
            </div>

            {/* 새로 측정하기 버튼 */}
            <div className="text-center mt-8">
              <button
                onClick={() => {
                  setResult(null);
                  setManualResult(null);
                  setAutoResult(null);
                  setSelectedMethod("manual");
                  setImage(null);
                  setImageUrl(null);
                  setDepthImageUrl(null);
                  setUploadStatus(null);
                  setUploadError(null);
                }}
                className="px-6 py-3 bg-pink-500 hover:bg-pink-600 text-white font-medium rounded-lg transition-colors shadow-lg"
              >
                새로 측정하기
              </button>
            </div>
          </div>
        )}

        <HelpTips isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
      </div>
    </div>
  );
}

export default App;