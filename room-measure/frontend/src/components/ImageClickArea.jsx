// frontend/src/components/ImageClickArea.jsx
import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

const CLICK_INSTRUCTIONS = [
  {
    step: 1,
    text: "바닥과 벽이 만나는 모서리 클릭",
    icon: "📍",
    color: "bg-red-100 border-red-300 text-red-800",
    detail: "층고 측정의 기준점이 되는 바닥 모서리",
  },
  {
    step: 2,
    text: "천장과 벽이 만나는 모서리 클릭 (같은 벽)",
    icon: "📍",
    color: "bg-blue-100 border-blue-300 text-blue-800",
    detail: "첫 번째 점과 수직선상에 있는 천장 모서리",
  },
  {
    step: 3,
    text: "왼쪽 벽의 바닥 모서리 클릭",
    icon: "📍",
    color: "bg-green-100 border-green-300 text-green-800",
    detail: "방의 세로 길이를 측정하기 위한 점",
  },
  {
    step: 4,
    text: "오른쪽 벽의 바닥 모서리 클릭",
    icon: "📍",
    color: "bg-purple-100 border-purple-300 text-purple-800",
    detail: "방의 가로 길이를 측정하기 위한 점",
  },
];

const ClickGuide = ({ currentStep, warnings }) => (
  <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-xl mb-6 border border-blue-200">
    <h3 className="font-bold text-lg mb-4 text-gray-800 flex items-center gap-2">
      📋 클릭 순서 가이드
      <span className="text-sm font-normal text-gray-600">
        ({currentStep}/4 완료)
      </span>
    </h3>

    <div className="space-y-3">
      {CLICK_INSTRUCTIONS.map((instruction, idx) => (
        <div
          key={idx}
          className={`flex items-start gap-3 p-3 rounded-lg border transition-all duration-200 ${
            idx === currentStep
              ? `${instruction.color} shadow-md transform scale-105`
              : idx < currentStep
              ? "bg-gray-100 border-gray-300 text-gray-600"
              : "bg-white border-gray-200 text-gray-500"
          }`}
        >
          <span className="text-xl flex-shrink-0 mt-0.5">
            {idx < currentStep ? "✅" : instruction.icon}
          </span>
          <div className="flex-1">
            <div
              className={`font-semibold ${
                idx === currentStep ? "text-lg" : ""
              }`}
            >
              {instruction.step}. {instruction.text}
            </div>
            <div className="text-sm mt-1 opacity-75">{instruction.detail}</div>
          </div>
        </div>
      ))}
    </div>

    {warnings.length > 0 && (
      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
        <div className="font-semibold text-yellow-800 mb-2">⚠️ 주의사항</div>
        <ul className="text-sm text-yellow-700 space-y-1">
          {warnings.map((warning, idx) => (
            <li key={idx}>• {warning}</li>
          ))}
        </ul>
      </div>
    )}
  </div>
);

const PointMarker = ({ point, index, isActive }) => {
  const colors = ["red", "blue", "green", "purple"];
  const color = colors[index] || "gray";

  return (
    <div
      className={`absolute transform -translate-x-1/2 -translate-y-1/2 ${
        isActive ? "animate-pulse" : ""
      }`}
      style={{ left: point.x, top: point.y }}
    >
      <div
        className={`w-4 h-4 bg-${color}-500 border-2 border-white rounded-full shadow-lg`}
      />
      <div
        className={`absolute -top-8 left-1/2 transform -translate-x-1/2 bg-${color}-600 text-white text-xs font-bold px-2 py-1 rounded shadow-lg`}
      >
        {index + 1}
      </div>
    </div>
  );
};

const validateClickedPoints = (points) => {
  const warnings = [];

  if (points.length >= 2) {
    const verticalDistance = Math.sqrt(
      Math.pow(points[1].x - points[0].x, 2) +
        Math.pow(points[1].y - points[0].y, 2)
    );

    if (verticalDistance < 50) {
      warnings.push(
        "수직 거리가 너무 짧습니다. 더 멀리 떨어진 점을 선택해주세요."
      );
    }

    // 수직선 체크 (점1과 점2가 거의 수직선상에 있는지)
    const horizontalDiff = Math.abs(points[1].x - points[0].x);
    if (horizontalDiff > verticalDistance * 0.3) {
      warnings.push("천장 점이 바닥 점과 수직선상에 있지 않습니다.");
    }
  }

  if (points.length >= 3) {
    const depthDistance = Math.sqrt(
      Math.pow(points[2].x - points[0].x, 2) +
        Math.pow(points[2].y - points[0].y, 2)
    );

    if (depthDistance < 30) {
      warnings.push("세로 방향 거리가 너무 짧습니다.");
    }
  }

  if (points.length >= 4) {
    const widthDistance = Math.sqrt(
      Math.pow(points[3].x - points[0].x, 2) +
        Math.pow(points[3].y - points[0].y, 2)
    );

    if (widthDistance < 30) {
      warnings.push("가로 방향 거리가 너무 짧습니다.");
    }

    // 기하학적 일관성 체크
    const allDistances = [];
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        const dist = Math.sqrt(
          Math.pow(points[j].x - points[i].x, 2) +
            Math.pow(points[j].y - points[i].y, 2)
        );
        allDistances.push(dist);
      }
    }

    const minDistance = Math.min(...allDistances);
    if (minDistance < 20) {
      warnings.push("일부 점들이 너무 가깝습니다. 더 넓게 분포시켜 주세요.");
    }
  }

  return warnings;
};

const ImageClickArea = ({ imageUrl, onComplete, depthWidth, depthHeight }) => {
  const [points, setPoints] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [warnings, setWarnings] = useState([]);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [depthMeta, setDepthMeta] = useState({ width: 0, height: 0 });
  const imageRef = useRef(null);

  const currentStep = points.length;

  useEffect(() => {
    const newWarnings = validateClickedPoints(points);
    setWarnings(newWarnings);
  }, [points]);

  // 깊이 맵 메타 정보 가져오기
  useEffect(() => {
    const fetchDepthMeta = async () => {
      try {
        const response = await axios.get("http://localhost:3000/depth-meta");
        console.log("📊 Depth meta 정보:", response.data);
        setDepthMeta(response.data);
      } catch (error) {
        console.error("❌ Depth meta 조회 실패:", error);
        // 기본값 사용
        setDepthMeta({ width: depthWidth || 256, height: depthHeight || 192 });
      }
    };

    fetchDepthMeta();
  }, [depthWidth, depthHeight]);

  // 이미지 로드 시 크기 정보 저장
  const handleImageLoad = () => {
    if (imageRef.current) {
      const { naturalWidth, naturalHeight, clientWidth, clientHeight } =
        imageRef.current;
      console.log("🖼️ 이미지 크기 정보:");
      console.log("   원본 크기:", naturalWidth, "x", naturalHeight);
      console.log("   표시 크기:", clientWidth, "x", clientHeight);
      console.log("   Depth 크기:", depthMeta.width, "x", depthMeta.height);

      setImageSize({
        naturalWidth,
        naturalHeight,
        clientWidth,
        clientHeight,
      });
    }
  };

  // 좌표 변환 함수
  const convertToDepthCoordinates = (displayX, displayY) => {
    if (
      !imageSize.clientWidth ||
      !imageSize.clientHeight ||
      !depthMeta.width ||
      !depthMeta.height
    ) {
      console.warn("⚠️ 좌표 변환에 필요한 정보가 부족합니다");
      return { x: Math.round(displayX), y: Math.round(displayY) };
    }

    // 표시된 이미지 좌표 → 깊이 맵 좌표 변환
    const scaleX = depthMeta.width / imageSize.clientWidth;
    const scaleY = depthMeta.height / imageSize.clientHeight;

    const depthX = Math.round(displayX * scaleX);
    const depthY = Math.round(displayY * scaleY);

    // 경계값 체크
    const clampedX = Math.max(0, Math.min(depthX, depthMeta.width - 1));
    const clampedY = Math.max(0, Math.min(depthY, depthMeta.height - 1));

    console.log("🔄 좌표 변환:");
    console.log(
      `   표시 좌표: (${displayX.toFixed(1)}, ${displayY.toFixed(1)})`
    );
    console.log(`   스케일: (${scaleX.toFixed(3)}, ${scaleY.toFixed(3)})`);
    console.log(`   깊이 좌표: (${clampedX}, ${clampedY})`);

    return { x: clampedX, y: clampedY };
  };

  const handleImageClick = async (e) => {
    if (points.length >= 4) return;

    const rect = e.target.getBoundingClientRect();
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;

    console.log(
      `🖱️ 클릭 ${points.length + 1}: 표시 좌표 (${displayX.toFixed(
        1
      )}, ${displayY.toFixed(1)})`
    );

    // 좌표 변환
    const depthCoords = convertToDepthCoordinates(displayX, displayY);

    try {
      console.log(`📡 깊이 값 요청: (${depthCoords.x}, ${depthCoords.y})`);

      const depthResponse = await axios.get(
        `http://localhost:3000/get-depth-at-point?x=${depthCoords.x}&y=${depthCoords.y}`
      );

      console.log("✅ 깊이 값 응답:", depthResponse.data);

      // 표시용으로는 원래 클릭 좌표 사용, z값만 깊이 맵에서 가져옴
      const newPoint = {
        x: displayX, // 표시용 좌표
        y: displayY, // 표시용 좌표
        z: depthResponse.data.depth, // 깊이 값
        // 실제 계산용 깊이 맵 좌표도 저장
        depthX: depthCoords.x,
        depthY: depthCoords.y,
      };

      const newPoints = [...points, newPoint];
      setPoints(newPoints);

      console.log("📝 업데이트된 points:", newPoints);
    } catch (error) {
      console.error("❌ 깊이 값 조회 실패:", error);

      // 에러 메시지 개선
      let errorMessage = "깊이 값을 가져올 수 없습니다.";
      if (error.response?.status === 400) {
        errorMessage =
          "클릭한 위치의 깊이 정보를 읽을 수 없습니다. 다른 지점을 클릭해주세요.";
      } else if (error.response?.status === 404) {
        errorMessage =
          "깊이 맵이 생성되지 않았습니다. 이미지를 다시 업로드해주세요.";
      }

      alert(errorMessage);
    }
  };

  const handleSubmit = async () => {
    if (points.length !== 4) {
      alert("4개의 점을 모두 클릭해주세요.");
      return;
    }

    if (warnings.length > 0) {
      const proceed = confirm(
        `경고사항이 있습니다:\n${warnings.join("\n")}\n\n계속 진행하시겠습니까?`
      );
      if (!proceed) return;
    }

    setIsLoading(true);
    try {
      // 깊이 맵 좌표계를 사용하여 전송
      const convertedPoints = points.map((point) => ({
        x: point.depthX || point.x, // 깊이 맵 좌표 우선 사용
        y: point.depthY || point.y, // 깊이 맵 좌표 우선 사용
        z: point.z,
      }));

      console.log("📤 서버로 전송할 좌표 (깊이 맵 기준):", convertedPoints);
      await onComplete(convertedPoints);
    } catch (error) {
      console.error("❌ 측정 실패:", error);
      alert("측정에 실패했습니다. 다시 시도해주세요.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setPoints([]);
    setWarnings([]);
  };

  return (
    <div className="mt-8">
      <ClickGuide currentStep={currentStep} warnings={warnings} />

      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-800">
            📷 방 사진에서 포인트 선택
          </h3>
          <div className="flex gap-2">
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg text-sm font-medium transition-colors"
              disabled={points.length === 0}
            >
              🔄 다시 시작
            </button>
            <button
              onClick={handleSubmit}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                points.length === 4 && warnings.length === 0
                  ? "bg-green-600 hover:bg-green-700 text-white shadow-lg"
                  : points.length === 4
                  ? "bg-yellow-600 hover:bg-yellow-700 text-white"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }`}
              disabled={points.length !== 4 || isLoading}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  측정 중...
                </span>
              ) : warnings.length > 0 ? (
                "⚠️ 경고 무시하고 측정"
              ) : (
                "📏 방 크기 측정하기"
              )}
            </button>
          </div>
        </div>

        {/* 좌표 변환 정보 표시 (디버깅용) */}
        {depthMeta.width > 0 && imageSize.clientWidth > 0 && (
          <div className="mb-4 p-3 bg-gray-50 rounded-lg text-sm">
            <div className="font-medium text-gray-700 mb-1">
              📊 좌표 변환 정보
            </div>
            <div className="text-gray-600 space-y-1">
              <div>
                표시 크기: {imageSize.clientWidth} × {imageSize.clientHeight}
              </div>
              <div>
                깊이 맵 크기: {depthMeta.width} × {depthMeta.height}
              </div>
              <div>
                변환 비율:{" "}
                {(depthMeta.width / imageSize.clientWidth).toFixed(3)} ×{" "}
                {(depthMeta.height / imageSize.clientHeight).toFixed(3)}
              </div>
            </div>
          </div>
        )}

        <div className="relative inline-block border-2 border-gray-300 rounded-lg overflow-hidden">
          <img
            ref={imageRef}
            src={imageUrl}
            alt="측정할 방 이미지"
            className="max-w-full h-auto cursor-crosshair"
            onClick={handleImageClick}
            onLoad={handleImageLoad}
            style={{ maxHeight: "600px" }}
          />

          {/* 클릭된 점들 표시 */}
          {points.map((point, index) => (
            <PointMarker
              key={index}
              point={point}
              index={index}
              isActive={index === currentStep - 1}
            />
          ))}

          {/* 진행률 표시 */}
          <div className="absolute top-4 left-4 bg-black bg-opacity-70 text-white px-3 py-2 rounded-lg">
            <div className="text-sm font-medium">진행률: {points.length}/4</div>
            <div className="w-24 h-2 bg-gray-600 rounded-full mt-1">
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-300"
                style={{ width: `${(points.length / 4) * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-semibold text-gray-700 mb-2">💡 측정 팁</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>모서리가 명확하게 보이는 지점을 선택하세요</li>
            <li>첫 번째와 두 번째 점은 수직선상에 있어야 합니다</li>
            <li>가구나 물건이 가리지 않는 벽면을 선택하세요</li>
            <li>정면에서 찍은 사진일수록 정확도가 높습니다</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ImageClickArea;
