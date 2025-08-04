import React, { useState, useEffect } from "react";
import axios from "axios";
import { undistortImage, generateDepthMap, getDepthMapImage, getDepthMeta, estimateRoomSize } from "../utils/api";
import ImageUploader from "../components/ImageUploader";
import ImageClickArea from "../components/ImageClickArea";
import RoomResult from "../components/RoomResult";
import RoomBox from "../components/RoomBox";
import FurniturePlacement from "../components/FurniturePlacement";

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
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
      <p className="text-text-secondary font-medium">{message}</p>
    </div>
  </div>
);

const UploadStatus = ({ status, error }) => {
  if (error) {
    const errorMessage = typeof error === 'string' ? error : 
                        error?.message || 
                        (error?.detail && typeof error.detail === 'string' ? error.detail : '알 수 없는 오류가 발생했습니다.');
    
    return (
      <div className="mt-4 p-4 bg-danger border border-danger rounded-lg">
        <div className="flex items-center gap-2">
          <span className="text-white">업로드 실패</span>
        </div>
        <p className="text-white mt-1 text-sm">{errorMessage}</p>
      </div>
    );
  }

  if (status) {
    const statusMessage = typeof status === 'string' ? status : '처리 중...';
    
    return (
      <div className="mt-4 p-4 bg-primary border border-primary rounded-lg">
        <div className="flex items-center gap-2">
          <span className="text-white">업로드 성공</span>
        </div>
        <p className="text-white mt-1 text-sm">{statusMessage}</p>
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
    <div className="bg-surface rounded-xl p-6 max-w-md mx-4 relative">
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-text-secondary hover:text-text-primary"
      >
        ✕
      </button>
      <h3 className="font-semibold text-text-primary mb-4 text-lg">측정 가이드</h3>
      <div className="space-y-3">
        <div className="bg-background p-3 rounded-lg">
          <div className="font-medium text-text-primary mb-1">최적의 방 형태</div>
          <p className="text-sm text-text-secondary">
            직사각형 방이 가장 정확하게 측정됩니다
          </p>
        </div>
        <div className="bg-background p-3 rounded-lg">
          <div className="font-medium text-text-primary mb-1">사진 촬영 팁</div>
          <p className="text-sm text-text-secondary">
            조명이 밝고 선명한 사진을 사용하세요
          </p>
        </div>
        <div className="bg-background p-3 rounded-lg">
          <div className="font-medium text-text-primary mb-1">모서리 포인트</div>
          <p className="text-sm text-text-secondary">
            벽면의 모서리가 잘 보이게 촬영하세요
          </p>
        </div>
        <div className="bg-background p-3 rounded-lg">
          <div className="font-medium text-text-primary mb-1">가구 배치</div>
          <p className="text-sm text-text-secondary">
            가구가 가리지 않은 벽면을 선택하세요
          </p>
        </div>
      </div>
    </div>
  </div>
);

function RoomPlannerPage() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);

  const [placedFurniture, setPlacedFurniture] = useState([]);

  useEffect(() => {
    if (process.env.NODE_ENV === "development") {
      console.log("🪑 Placed furniture updated:", placedFurniture);
    }
  }, [placedFurniture]);

  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [depthImageUrl, setDepthImageUrl] = useState(null);
  const [depthSize, setDepthSize] = useState({ width: 0, height: 0 });

  const [housingType, setHousingType] = useState("house");
  const [ceilingHeight, setCeilingHeight] = useState(230);

  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const [activeTab, setActiveTab] = useState("analysis");
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [manualResult, setManualResult] = useState(null);
  const [autoResult, setAutoResult] = useState(null);
  const [selectedMethod, setSelectedMethod] = useState("manual");
  const [roomMeasurementPoints, setRoomMeasurementPoints] = useState(null);

  const updateRoomMeasurementPoints = (points) => {
    setRoomMeasurementPoints(points);
    window.roomMeasurementPoints = points;
    console.log("📐 방 측정 포인트 저장됨:", points);
  };

  const handleTabClick = (tabName) => {
    setActiveTab(tabName);
    if (tabName === "3d") {
      setTimeout(() => {
        const element = document.querySelector(".room-3d-viewer");
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
      }, 100);
    }
  };

  const handleImageUpload = async (file) => {
    setUploadError(null);
    setUploadStatus(null);
    setIsProcessing(true);

    // 파일 유효성 검사
    if (!file) {
      setUploadError("파일이 선택되지 않았습니다.");
      setIsProcessing(false);
      return;
    }

    // 이미지 파일 타입 검사
    if (!file.type.startsWith('image/')) {
      setUploadError("이미지 파일만 업로드 가능합니다.");
      setIsProcessing(false);
      return;
    }

    // 파일 크기 검사 (10MB 제한)
    if (file.size > 10 * 1024 * 1024) {
      setUploadError("파일 크기는 10MB 이하여야 합니다.");
      setIsProcessing(false);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploadStatus("이미지 전처리 중...");
      console.log("Uploading file:", file.name, "Size:", file.size, "Type:", file.type);
      console.log("FormData entries:");
      for (let [key, value] of formData.entries()) {
        console.log(key, value);
      }
      
      const responseBlob = await undistortImage(file);

      setImage(file);
      setImageUrl(URL.createObjectURL(file));

      setUploadStatus("AI 깊이 분석 중... (30초 소요)");
      await generateDepthMap();
      
      // 깊이 맵 크기 정보 가져오기
      try {
        const metaData = await getDepthMeta();
        if (metaData.success) {
          setDepthSize({ width: metaData.width, height: metaData.height });
        }
      } catch (error) {
        console.error("깊이 맵 메타 정보 가져오기 실패:", error);
      }

      try {
        const imageBlob = await getDepthMapImage();
        const imageObjectURL = URL.createObjectURL(imageBlob);
        setDepthImageUrl(imageObjectURL);
      } catch (error) {
        console.warn("깊이 맵 이미지 로드 실패:", error);
      }

      setUploadStatus("이미지 업로드 완료! 방 모서리를 클릭하세요.");
    } catch (error) {
      console.error("Upload failed:", error);
      let message = "알 수 없는 오류가 발생했습니다.";
      
      if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          message = error.response.data.detail;
        } else if (Array.isArray(error.response.data.detail)) {
          message = error.response.data.detail.map(item => 
            typeof item === 'string' ? item : item.msg || '오류'
          ).join(', ');
        }
      } else if (error.message) {
        message = error.message;
      } else if (error.code === 'ERR_NETWORK') {
        message = "서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.";
      }
      
      setUploadError(message);
      alert(`측정 실패: ${message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePointsSubmit = async (points, method = "manual") => {
    try {
      updateRoomMeasurementPoints(points);

      const payload = {
        housing_type: housingType,
        points: points.map((pt) => ({
          x: parseFloat(pt.x),
          y: parseFloat(pt.y),
          z: parseFloat(pt.z),
        })),
        target_height: parseFloat(ceilingHeight) / 100,
      };
      
      console.log("API 요청 payload:", payload);
      const res = await estimateRoomSize(payload.points, payload.target_height);
      console.log("API 응답 전문:", res);

      const resultData = { ...res, detectionMethod: method };

      if (method === "manual") {
        setManualResult(resultData);
      } else {
        setAutoResult(resultData);
      }

      setResult(resultData);
    } catch (error) {
      console.error("Room size estimation failed:", error);
      let message = "방 크기 계산에 실패했습니다.";
      
      if (error.message.includes('HTTP error')) {
        message = `서버 오류: ${error.message}`;
      } else if (error.message) {
        message = error.message;
      } else if (error.code === 'ERR_NETWORK') {
        message = "서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.";
      }
      
      alert(`측정 실패: ${message}`);
    }
  };

  React.useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement !== null);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
    document.addEventListener("mozfullscreenchange", handleFullscreenChange);
    document.addEventListener("MSFullscreenChange", handleFullscreenChange);

    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange);
      document.removeEventListener("mozfullscreenchange", handleFullscreenChange);
      document.removeEventListener("MSFullscreenChange", handleFullscreenChange);
    };
  }, []);

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="container mx-auto px-4">
        <div className="mb-8 pt-24 md:pt-28 max-w-4xl">
          <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-6 leading-tight">
            Room <span className="text-primary">Planner</span>
          </h1>
          <p className="text-lg text-text-secondary mb-8">
            Measure accurate room dimensions from a single photo using artificial intelligence
          </p>
        </div>

      {!result && (
        <div className="max-w-6xl mx-auto">
          <div className="bg-surface border border-border rounded-lg shadow-sm p-4 sm:p-6 lg:p-8 mb-6">
            <div className="mb-6">
              <div className="w-full border-2 border-dashed border-border rounded-xl p-6 sm:p-8 flex flex-col items-center justify-center bg-background hover:bg-gray-100 transition-colors">
                <span className="font-medium text-lg sm:text-xl mb-2 text-text-primary">
                  Upload Photo
                </span>
                <span className="text-text-secondary text-sm sm:text-base mb-4 text-center">
                  Please select a photo where the room corners are clearly
                  visible
                </span>
                <ImageUploader onUpload={handleImageUpload} />
              </div>
              {isProcessing && <LoadingSpinner message={uploadStatus} />}
              <UploadStatus status={uploadStatus} error={uploadError} />
            </div>

            <div className="border-t border-border pt-6">
              <h3 className="font-medium text-lg mb-4 text-text-primary">
                측정 설정
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                <div>
                  <label className="block font-medium text-text-primary mb-2">
                    건물 유형
                  </label>
                  <select
                    className="w-full border border-border rounded-lg px-4 py-3 focus:border-blue-600 focus:ring focus:ring-blue-600 focus:ring-opacity-50"
                    value={housingType}
                    onChange={(e) => setHousingType(e.target.value)}
                  >
                    {HOUSING_TYPES.map((h) => (
                      <option key={h.value} value={h.value}>
                        {h.label} ({h.description})
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block font-medium text-text-primary mb-2">
                    천장 높이
                  </label>
                  <div className="flex items-center gap-3">
                    <input
                      type="number"
                      min="200"
                      max="400"
                      step="5"
                      value={ceilingHeight}
                      onChange={(e) => setCeilingHeight(e.target.value)}
                      className="flex-1 border border-border rounded-lg px-4 py-3 focus:border-blue-600 focus:ring focus:ring-blue-600 focus:ring-opacity-50"
                    />
                    <span className="text-text-secondary text-base font-medium">
                      cm
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

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

      {result && (
        <div className="space-y-6">
          <div className="bg-surface rounded-lg shadow-sm border border-border overflow-visible">
            <div className="flex flex-wrap border-b border-border">
              <button
                onClick={() => setActiveTab("analysis")}
                className={`flex-1 min-w-[120px] px-4 sm:px-6 py-3 sm:py-4 text-center font-medium transition-colors text-sm sm:text-base ${
                  activeTab === "analysis"
                    ? "bg-surface text-primary border-b-2 border-blue-600"
                    : "text-text-secondary hover:text-text-primary hover:bg-background"
                }`}
              >
                Analysis
              </button>
              <button
                onClick={() => setActiveTab("2d")}
                className={`flex-1 min-w-[120px] px-4 sm:px-6 py-3 sm:py-4 text-center font-medium transition-colors text-sm sm:text-base ${
                  activeTab === "2d"
                    ? "bg-surface text-primary border-b-2 border-blue-600"
                    : "text-text-secondary hover:text-text-primary hover:bg-background"
                }`}
              >
                2D Floor Plan
              </button>
              <button
                onClick={() => setActiveTab("furniture")}
                className={`flex-1 min-w-[120px] px-4 sm:px-6 py-3 sm:py-4 text-center font-medium transition-colors text-sm sm:text-base ${
                  activeTab === "furniture"
                    ? "bg-surface text-primary border-b-2 border-blue-600"
                    : "text-text-secondary hover:text-text-primary hover:bg-background"
                }`}
              >
                Furniture Layout
              </button>
              <button
                onClick={() => handleTabClick("3d")}
                className={`flex-1 min-w-[120px] px-4 sm:px-6 py-3 sm:py-4 text-center font-medium transition-colors text-sm sm:text-base ${
                  activeTab === "3d"
                    ? "bg-surface text-primary border-b-2 border-blue-600"
                    : "text-text-secondary hover:text-text-primary hover:bg-background"
                }`}
              >
                3D Viewer
              </button>
            </div>

            <div className="p-4 sm:p-6 lg:p-8">
              {activeTab === "analysis" && (
                <div className="max-w-6xl mx-auto space-y-6">
                  <div className="bg-surface rounded-lg p-6 shadow-sm border border-border">
                    <h3 className="text-2xl font-bold mb-4 text-text-primary">
                      AI Room Measurement Analysis
                      {result?.detectionMethod && (
                        <span className="ml-3 text-sm font-normal text-text-secondary">
                          ({result.detectionMethod === "auto" ? "🤖 Auto Detection" : "📍 Manual 4-Point"})
                        </span>
                      )}
                    </h3>
                    <p className="text-text-secondary mb-4">
                      Advanced computer vision and deep learning technology
                      for accurate room size measurement from a single photo.
                    </p>

                    {/* 측정 결과 */}
                    <div className="grid grid-cols-3 gap-4 mb-6">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">
                          {((result.dimensions?.width_cm || result.width_cm) / 100).toFixed(1)}m
                        </div>
                        <div className="text-sm text-text-secondary">Width</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">
                          {((result.dimensions?.depth_cm || result.depth_cm) / 100).toFixed(1)}m
                        </div>
                        <div className="text-sm text-text-secondary">Depth</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">
                          {((result.dimensions?.height_cm || result.height_cm) / 100).toFixed(1)}m
                        </div>
                        <div className="text-sm text-text-secondary">Height</div>
                      </div>
                    </div>

                    {/* 계산된 값들 */}
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div className="text-center p-3 bg-background rounded border border-border">
                        <div className="text-xl font-bold text-text-primary">
                          {(result.calculated_values?.area_sqm || (((result.dimensions?.width_cm || result.width_cm) * (result.dimensions?.depth_cm || result.depth_cm)) / 10000)).toFixed(1)}㎡
                        </div>
                        <div className="text-sm text-text-secondary">Floor Area</div>
                      </div>
                      <div className="text-center p-3 bg-background rounded border border-border">
                        <div className="text-xl font-bold text-text-primary">
                          {(result.calculated_values?.volume_cum || (((result.dimensions?.width_cm || result.width_cm) * (result.dimensions?.depth_cm || result.depth_cm) * (result.dimensions?.height_cm || result.height_cm)) / 1000000)).toFixed(1)}㎥
                        </div>
                        <div className="text-sm text-text-secondary">Volume</div>
                      </div>
                    </div>
                  </div>

                  {/* 측정 정확도 및 신뢰도 */}
                  <div className="bg-surface rounded-lg p-6 shadow-sm border border-border">
                    <h4 className="text-lg font-semibold text-text-primary mb-3">Measurement Accuracy</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-primary rounded-full"></span>
                        <span className="text-text-secondary">Accuracy: ±5~10cm</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-gray-600 rounded-full"></span>
                        <span className="text-text-secondary">Confidence: {(result.confidence * 100).toFixed(0)}%</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-primary rounded-full"></span>
                        <span className="text-text-secondary">Processing time: ~30 seconds</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-gray-600 rounded-full"></span>
                        <span className="text-text-secondary">Method: {result.detectionMethod === "auto" ? "Automatic" : "Manual"}</span>
                      </div>
                    </div>
                  </div>

                  {/* 원본 이미지와 깊이 분석 이미지 비교 */}
                  {depthImageUrl && (
                    <div className="bg-surface rounded-lg p-6 shadow-sm border border-border">
                      <h4 className="text-lg font-semibold text-text-primary mb-4">Image Analysis Process</h4>
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="space-y-3">
                          <h5 className="font-medium text-text-primary">Original Image</h5>
                          <div className="relative">
                            <div className="aspect-video rounded-lg overflow-hidden border border-border">
                              <img
                                src={imageUrl}
                                alt="Original Image"
                                className="w-full h-full object-cover"
                              />
                              <div className="absolute top-2 left-2 bg-primary/80 text-white px-2 py-1 rounded text-xs">
                                Original
                              </div>
                            </div>
                          </div>
                          <p className="text-sm text-text-secondary">
                            User uploaded room image
                          </p>
                        </div>

                        <div className="space-y-3">
                          <h5 className="font-medium text-text-primary">AI Depth Analysis</h5>
                          <div className="relative">
                            <div className="aspect-video rounded-lg overflow-hidden border border-border">
                              <img
                                src={depthImageUrl}
                                alt="Depth Map"
                                className="w-full h-full object-cover"
                              />
                              <div className="absolute top-2 left-2 bg-primary/80 text-white px-2 py-1 rounded text-xs">
                                Depth Map
                              </div>
                            </div>
                          </div>
                          <p className="text-sm text-text-secondary">
                            AI analyzed depth information and 3D structure
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* 측정 설정 정보 */}
                  <div className="bg-surface rounded-lg p-6 shadow-sm border border-border">
                    <h4 className="text-lg font-semibold text-text-primary mb-4">Measurement Parameters</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm text-text-secondary">
                      <div className="flex justify-between">
                        <span>Ceiling Height:</span>
                        <span className="font-medium">{(ceilingHeight / 100).toFixed(1)}m</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Building Type:</span>
                        <span className="font-medium">
                          {HOUSING_TYPES.find((h) => h.value === housingType)?.label}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Depth Map Size:</span>
                        <span className="font-medium">{depthSize.width}×{depthSize.height}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Detection Method:</span>
                        <span className="font-medium">
                          {result.detectionMethod === "auto" ? "Automatic Detection" : "Manual 4-Point"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* 기술 정보 */}
                  <div className="bg-surface rounded-lg p-6 shadow-sm border border-border">
                    <h4 className="text-lg font-semibold text-text-primary mb-4">Technology Details</h4>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <div className="space-y-3">
                        <h5 className="font-medium text-text-primary">AI Models Used</h5>
                        <ul className="space-y-1 text-sm text-text-secondary">
                          <li>• MiDaS v3.1 (Depth Estimation)</li>
                          <li>• Vision Transformer Architecture</li>
                          <li>• Monocular Depth Estimation</li>
                          <li>• Camera Distortion Correction</li>
                          <li>• 3D Geometric Transformation</li>
                        </ul>
                      </div>
                      <div className="space-y-3">
                        <h5 className="font-medium text-text-primary">Limitations</h5>
                        <ul className="space-y-1 text-sm text-text-secondary">
                          <li>• Optimized for rectangular rooms</li>
                          <li>• Better lighting improves accuracy</li>
                          <li>• Clear corner visibility required</li>
                          <li>• Depends on ceiling height accuracy</li>
                        </ul>
                      </div>
                    </div>
                    
                    <div className="mt-4 p-4 bg-blue-50 rounded-lg border-l-4 border-blue-600">
                      <p className="text-sm text-text-primary">
                        <strong>Tip:</strong> For more accurate measurements, ensure good lighting 
                        and clear visibility of room corners and edges.
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
                  <div className="relative">
                    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-50">
                      <button
                        onClick={() => {
                          const element = document.querySelector(".room-3d-viewer");
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
                        className="px-4 py-2 bg-primary/50 text-white rounded-lg hover:bg-primary/70 transition-all duration-200 flex items-center gap-2"
                        title="전체화면으로 보기"
                      >
                        <span>⛶</span>
                        <span className="text-sm">전체화면</span>
                      </button>
                    </div>
                    <div
                      className={`room-3d-viewer bg-black rounded-lg ${
                        isFullscreen
                          ? "fixed inset-0 w-screen h-screen z-[9999]"
                          : "min-h-[500px] md:min-h-[600px] lg:min-h-[700px]"
                      }`}
                    >
                      <RoomBox
                        width={result.dimensions?.width_cm || result.width_cm}
                        depth={result.dimensions?.depth_cm || result.depth_cm}
                        height={result.dimensions?.height_cm || result.height_cm}
                        isFullscreen={isFullscreen}
                        uploadedImageFile={image}
                        uploadedImageUrl={imageUrl}
                        placedFurniture={placedFurniture}
                        onFurnitureChange={setPlacedFurniture}
                      />
                    </div>
                  </div>
                </div>
              )}

              {activeTab === "furniture" && (
                <div className="w-full">
                  <FurniturePlacement
                    roomWidth={result.dimensions?.width_cm || result.width_cm}
                    roomDepth={result.dimensions?.depth_cm || result.depth_cm}
                    roomHeight={result.dimensions?.height_cm || result.height_cm}
                    placedFurniture={placedFurniture}
                    onFurnitureChange={setPlacedFurniture}
                    detectedWindows={[]}
                  />
                </div>
              )}
            </div>
          </div>

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
                setPlacedFurniture([]);
              }}
              className="px-6 py-3 bg-primary hover:bg-secondary text-white font-medium rounded-lg transition-colors shadow-lg"
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

export default RoomPlannerPage;