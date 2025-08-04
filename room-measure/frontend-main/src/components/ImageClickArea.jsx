// frontend/src/components/ImageClickArea.jsx
import React, { useState, useRef, useEffect } from "react";
import { getDepthMeta, getDepthAtPoint, autoDetectRoom } from "../utils/api";

const CLICK_INSTRUCTIONS = [
  {
    step: 1,
    text: "Floor-wall corner",
    icon: "1",
    detail: "Base point for height measurement",
  },
  {
    step: 2,
    text: "Ceiling-wall corner (same wall)",
    icon: "2",
    detail: "Ceiling corner vertically above first point",
  },
  {
    step: 3,
    text: "Left wall floor corner",
    icon: "3",
    detail: "Point for measuring room depth",
  },
  {
    step: 4,
    text: "Right wall floor corner",
    icon: "4",
    detail: "Point for measuring room width",
  },
];

const ClickGuide = ({ currentStep, warnings }) => (
  <div className="bg-surface p-4 rounded-xl border border-border">
    <h3 className="text-2xl font-bold mb-4 text-text-primary flex items-center gap-2">
      <strong>Click Guide</strong>
      <span className="text-sm font-normal text-text-secondary">
        ({currentStep}/4 completed)
      </span>
    </h3>

    <div className="space-y-3">
      {CLICK_INSTRUCTIONS.map((instruction, idx) => (
        <div
          key={idx}
          className={`flex items-start gap-3 p-3 rounded-lg border transition-all duration-200 ${
            idx === currentStep
              ? `bg-primary text-white shadow-md transform scale-105`
              : idx < currentStep
              ? "bg-background border-border text-text-secondary"
              : "bg-surface border-border text-text-secondary"
          }`}
        >
          <span className="text-xl flex-shrink-0 mt-0.5">
            {idx < currentStep ? "✓" : instruction.icon}
          </span>
          <div className="flex-1">
            <div
              className={`font-semibold ${
                idx === currentStep ? "text-lg" : ""
              }`}
            >
              {instruction.text}
            </div>
            <div className="text-sm mt-1 opacity-75">{instruction.detail}</div>
          </div>
        </div>
      ))}
    </div>

    {warnings.length > 0 && (
      <div className="mt-4 p-3 bg-danger border border-danger rounded-lg">
        <div className="font-semibold text-white mb-2">주의사항</div>
        <ul className="text-sm text-white space-y-1">
          {warnings.map((warning, idx) => (
            <li key={idx}>• {warning}</li>
          ))}
        </ul>
      </div>
    )}
  </div>
);

const PointMarker = ({ point, index, isActive }) => {
  const colors = ["primary", "secondary", "accent", "danger"];
  const color = colors[index] || "text-text-secondary";
  const isAutoDetected = point.autoDetected;

  return (
    <div
      className={`absolute transform -translate-x-1/2 -translate-y-1/2 ${
        isActive ? "animate-pulse" : ""
      } ${isAutoDetected ? "animate-bounce" : ""}`}
      style={{ left: point.x, top: point.y }}
    >
      <div
        className={`w-4 h-4 ${
          isAutoDetected
            ? `bg-primary border-2 border-primary/50`
            : `bg-${color} border-2 border-surface`
        } rounded-full shadow-lg`}
      />
      <div
        className={`absolute -top-8 left-1/2 transform -translate-x-1/2 ${
          isAutoDetected ? "bg-primary" : `bg-${color}`
        } text-white text-xs font-bold px-2 py-1 rounded shadow-lg`}
      >
        {point.pointNumber || index + 1}
      </div>
    </div>
  );
};

const calculateDynamicThreshold = (canvas) => {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  
  // 이미지 품질 분석
  let brightness = 0;
  const values = [];
  let edgePixels = 0;
  
  // 밝기와 엣지 픽셀 계산
  for (let i = 0; i < data.length; i += 4) {
    const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    values.push(gray);
    brightness += gray;
    
    // 인접 픽셀과의 차이가 큰 경우 엣지로 판단
    if (i > 4 && Math.abs(gray - values[values.length - 2]) > 30) {
      edgePixels++;
    }
  }
  
  brightness /= values.length;
  const edgeRatio = edgePixels / values.length;
  
  // 표준편차 (대비) 계산
  const variance = values.reduce((sum, val) => sum + Math.pow(val - brightness, 2), 0) / values.length;
  const contrast = Math.sqrt(variance);
  
  // 동적 임계값 계산 (더 보수적으로)
  let threshold = 0.6; // 기본값을 낮춤
  
  // 밝기에 따른 조정
  if (brightness < 70) {
    threshold = 0.4; // 어두운 이미지
  } else if (brightness > 190) {
    threshold = 0.7; // 매우 밝은 이미지
  } else if (brightness > 150) {
    threshold = 0.65; // 밝은 이미지
  }
  
  // 대비에 따른 조정
  if (contrast < 25) {
    threshold -= 0.15; // 매우 낮은 대비
  } else if (contrast < 40) {
    threshold -= 0.1; // 낮은 대비
  } else if (contrast > 90) {
    threshold += 0.1; // 높은 대비
  }
  
  // 엣지 비율에 따른 조정
  if (edgeRatio < 0.1) {
    threshold -= 0.1; // 엣지가 적으면 임계값 낮춤
  } else if (edgeRatio > 0.3) {
    threshold += 0.05; // 엣지가 많으면 약간 높임
  }
  
  // 경계값 제한 (더 넓은 범위)
  return Math.max(0.2, Math.min(0.8, threshold));
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
  const [isAutoDetecting, setIsAutoDetecting] = useState(false);
  const [detectionMethod, setDetectionMethod] = useState("manual"); // "manual" or "auto"
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
        const data = await getDepthMeta();
        console.log("Depth meta 정보:", data);
        setDepthMeta(data);
      } catch (error) {
        console.error("Depth meta 조회 실패:", error);
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
      console.log("이미지 크기 정보:");
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
      console.warn("좌표 변환에 필요한 정보가 부족합니다");
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

    console.log("좌표 변환:");
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
      `클릭 ${points.length + 1}: 표시 좌표 (${displayX.toFixed(
        1
      )}, ${displayY.toFixed(1)})`
    );

    // 좌표 변환
    const depthCoords = convertToDepthCoordinates(displayX, displayY);

    try {
      console.log(`깊이 값 요청: (${depthCoords.x}, ${depthCoords.y})`);

      const depthData = await getDepthAtPoint(depthCoords.x, depthCoords.y);

      console.log("깊이 값 응답:", depthData);

      // 표시용으로는 원래 클릭 좌표 사용, z값만 깊이 맵에서 가져옴
      const newPoint = {
        x: displayX, // 표시용 좌표
        y: displayY, // 표시용 좌표
        z: depthData.depth, // 깊이 값
        // 실제 계산용 깊이 맵 좌표도 저장
        depthX: depthCoords.x,
        depthY: depthCoords.y,
        pointNumber: points.length + 1, // 수동 클릭 포인트에도 순서 번호 추가 (1, 2, 3, 4)
      };

      const newPoints = [...points, newPoint];
      setPoints(newPoints);

      console.log("업데이트된 points:", newPoints);
    } catch (error) {
      console.error("깊이 값 조회 실패:", error);

      // 에러 메시지 개선
      let errorMessage = "깊이 값을 가져올 수 없습니다.";
      if (error.message.includes('400')) {
        errorMessage =
          "클릭한 위치의 깊이 정보를 읽을 수 없습니다. 다른 지점을 클릭해주세요.";
      } else if (error.message.includes('404')) {
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

      console.log("서버로 전송할 좌표 (깊이 맵 기준):", convertedPoints);
      await onComplete(convertedPoints, detectionMethod);
    } catch (error) {
      console.error("측정 실패:", error);
      alert("측정에 실패했습니다. 다시 시도해주세요.");
    } finally {
      setIsLoading(false);
    }
  };

  const enhancedImagePreprocessing = async (img, canvas, ctx) => {
    // 이미지를 캔버스에 그리기
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    // 백엔드에서 수행할 전처리와 유사하게 간단하게 처리
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // 간단한 대비 개선 및 그레이스케일 변환
    for (let i = 0; i < data.length; i += 4) {
      // RGB를 그레이스케일로 변환
      const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      
      // 간단한 대비 개선 (선형 스트레칭)
      const enhanced = Math.min(255, Math.max(0, gray * 1.1 + 10));
      
      data[i] = enhanced;     // Red
      data[i + 1] = enhanced; // Green  
      data[i + 2] = enhanced; // Blue
      // Alpha는 그대로 유지
    }

    // 처리된 데이터를 캔버스에 적용
    ctx.putImageData(imageData, 0, 0);
  };

  const handleAutoDetect = async () => {
    setIsAutoDetecting(true);
    setDetectionMethod("auto");
    setPoints([]);
    setWarnings([]);

    try {
      // 이미지를 Canvas로 로드하고 고급 전처리 적용
      const response = await fetch(imageUrl);
      const blob = await response.blob();

      // 이미지를 Canvas에 그려서 고급 전처리 수행
      const img = new Image();
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      await new Promise((resolve) => {
        img.onload = resolve;
        img.src = URL.createObjectURL(blob);
      });

      // Canvas 크기를 이미지 크기로 설정
      canvas.width = img.width;
      canvas.height = img.height;

      // 고급 이미지 전처리 적용
      await enhancedImagePreprocessing(img, canvas, ctx);

      // Canvas를 Blob으로 변환
      const grayscaleBlob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.9);
      });

      const file = new File([grayscaleBlob], "room-image-grayscale.jpg", {
        type: "image/jpeg",
      });

      const formData = new FormData();
      formData.append("file", file);
      formData.append("confidence_threshold", "0.7");

      console.log("Enhanced RoomNet 자동 감지 시작 (개선된 전처리 적용)...");

      // 동적 신뢰도 임계값 계산
      const dynamicThreshold = calculateDynamicThreshold(canvas);
      console.log(`동적 신뢰도 임계값: ${dynamicThreshold}`);

      const autoDetectData = await autoDetectRoom(file, dynamicThreshold);

      if (autoDetectData.success) {
        console.log("RoomNet 감지 성공:", autoDetectData);

        // 자동 감지된 포인트들을 표시용으로 변환 (좌표 스케일 조정)
        console.log("원본 백엔드 포인트들:", autoDetectData.detected_points);
        console.log("현재 이미지 표시 크기:", imageSize);
        console.log("깊이 맵 크기:", depthMeta);
        
        const detectedPoints = autoDetectData.detected_points.map(
          (point, index) => {
            // 백엔드에서 반환된 좌표는 이미 원본 이미지 크기 기준
            // 프론트엔드 표시 크기로 변환 필요
            let displayX = point.x;
            let displayY = point.y;
            
            // 좌표 변환: 원본 이미지 크기 -> 표시 크기
            if (imageSize.naturalWidth > 0 && imageSize.naturalHeight > 0) {
              const scaleX = imageSize.clientWidth / imageSize.naturalWidth;
              const scaleY = imageSize.clientHeight / imageSize.naturalHeight;
              
              displayX = point.x * scaleX;
              displayY = point.y * scaleY;
              
              console.log(`✅ 포인트 ${index}: (${point.x}, ${point.y}) -> (${displayX.toFixed(1)}, ${displayY.toFixed(1)}) [스케일: ${scaleX.toFixed(2)}, ${scaleY.toFixed(2)}]`);
            } else {
              console.log(`⚠️ 포인트 ${index}: 스케일링 정보 없음, 원본 좌표 사용 (${point.x}, ${point.y})`);
            }
            
            return {
              x: displayX,
              y: displayY,
              z: point.z,
              depthX: point.x,  // 서버로 전송시 사용할 원본 좌표
              depthY: point.y,  // 서버로 전송시 사용할 원본 좌표
              autoDetected: true,
              pointNumber: index + 1,  // AI 감지 포인트에도 순서 번호 추가 (1, 2, 3, 4)
            };
          }
        );

        setPoints(detectedPoints);

        // 자동 감지 결과를 표시하고 사용자가 확인할 수 있도록 대기
        console.log("자동 감지 완료! 사용자 확인 대기 중...");
      } else {
        console.warn("RoomNet 감지 실패:", autoDetectData);
        alert(
          `자동 감지에 실패했습니다: ${autoDetectData.error}\n\n수동 4포인트 방식을 사용해주세요.`
        );
        setDetectionMethod("manual");
      }
    } catch (error) {
      console.error("자동 감지 오류:", error);
      let errorMessage = "자동 감지 중 오류가 발생했습니다.";

      if (error.message.includes('422')) {
        errorMessage = "이미지에서 방 경계를 자동으로 감지할 수 없습니다.";
      } else if (error.message.includes('500')) {
        errorMessage = "서버 오류가 발생했습니다.";
      }

      alert(`${errorMessage}\n\n수동 4포인트 방식을 사용해주세요.`);
      setDetectionMethod("manual");
    } finally {
      setIsAutoDetecting(false);
    }
  };

  const handleReset = () => {
    setPoints([]);
    setWarnings([]);
    setDetectionMethod("manual");
  };

  return (
    <div className="mt-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 왼쪽: 클릭 가이드 */}
        <div className="lg:col-span-1">
          <ClickGuide currentStep={currentStep} warnings={warnings} />
        </div>

        {/* 오른쪽: 이미지 클릭 영역 */}
        <div className="lg:col-span-2">
          <div className="bg-surface rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-2xl font-bold text-text-primary">
                <strong>Select Points on Room Photo</strong>
              </h3>
              <div className="flex gap-2">
                {/* Auto Detect Button */}
                <button
                  onClick={handleAutoDetect}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    isAutoDetecting
                      ? "bg-primary/50 text-white cursor-not-allowed"
                      : "bg-primary hover:bg-secondary text-white shadow-md hover:shadow-lg transform hover:scale-105"
                  }`}
                  disabled={isAutoDetecting || isLoading}
                  title="AI로 방 경계를 자동 감지합니다"
                >
                  {isAutoDetecting ? (
                    <span className="flex items-center gap-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      <span className="text-xs">AI 분석 중...</span>
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      AI
                      <strong>Auto Detect</strong>
                    </span>
                  )}
                </button>

                <button
                  onClick={handleReset}
                  className="px-4 py-2 bg-secondary hover:bg-primary text-white rounded-lg text-sm font-medium transition-colors"
                  disabled={points.length === 0 && !isAutoDetecting}
                >
                  <strong>Reset</strong>
                </button>
                <button
                  onClick={handleSubmit}
                  className={`px-6 py-2 rounded-lg font-medium transition-all ${
                    points.length === 4 && warnings.length === 0
                      ? "bg-primary hover:bg-secondary text-white shadow-lg"
                      : points.length === 4
                      ? "bg-primary/70 hover:bg-primary text-white"
                      : "bg-background text-text-secondary cursor-not-allowed"
                  }`}
                  disabled={points.length !== 4 || isLoading || isAutoDetecting}
                >
                  {isLoading ? (
                    <span className="flex items-center gap-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      Measuring...
                    </span>
                  ) : warnings.length > 0 ? (
                    <strong>Continue with Warnings</strong>
                  ) : (
                    <strong>Measure Room Size</strong>
                  )}
                </button>
              </div>
            </div>

            {/* Detection Method Indicator */}
            {detectionMethod === "auto" && points.length > 0 && (
              <div className="mb-4 p-4 bg-background border border-border rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-primary">AI</span>
                    <span className="font-medium text-text-primary">
                      RoomNet 자동 감지 완료!
                    </span>
                  </div>
                  <span className="text-text-secondary text-sm font-medium">
                    {points.length}/4 포인트 감지됨
                  </span>
                </div>
                <div className="bg-surface p-3 rounded border border-border">
                  <p className="text-sm text-text-secondary mb-2">
                    <strong>자동 감지된 포인트들을 확인해주세요:</strong>
                  </p>
                  <ul className="text-xs text-text-secondary space-y-1">
                    <li>• 파란색 AI 마커들이 AI가 감지한 방 모서리입니다</li>
                    <li>
                      • 포인트들이 정확한지 확인 후 "Measure Room Size" 버튼을
                      눌러주세요
                    </li>
                    <li>• 부정확하면 "Reset" 후 수동으로 다시 선택하세요</li>
                  </ul>
                </div>
              </div>
            )}

            {/* 좌표 변환 정보 표시 (디버깅용) */}
            {depthMeta.width > 0 && imageSize.clientWidth > 0 && (
              <div className="mb-4 p-3 bg-background rounded-lg text-sm">
                <div className="font-medium text-text-primary mb-1">
                  <strong>좌표 변환 정보</strong>
                </div>
                <div className="text-text-secondary space-y-1">
                  <div>
                    표시 크기: {imageSize.clientWidth} ×{" "}
                    {imageSize.clientHeight}
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

            <div className="relative inline-block border-2 border-border rounded-lg overflow-hidden">
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
              <div className="absolute top-4 left-4 bg-primary/70 text-white px-3 py-2 rounded-lg">
                <div className="text-sm font-medium">
                  Progress: {points.length}/4
                </div>
                <div className="w-24 h-2 bg-secondary rounded-full mt-1">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-300"
                    style={{ width: `${(points.length / 4) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="mt-4 p-4 bg-background rounded-lg">
              <h4 className="text-lg font-semibold text-text-primary mb-2">
                <strong>Measurement Tips</strong>
              </h4>
              <ul className="text-sm text-text-secondary space-y-1">
                <li>
                  AI <strong>Auto Detect:</strong> Try AI-powered automatic room
                  detection first
                </li>
                <li>
                  <strong>Manual:</strong> Select 4 corner points if auto
                  detection fails
                </li>
                <li>First and second points should be vertically aligned</li>
                <li>Choose walls not blocked by furniture or objects</li>
                <li>Front-facing photos provide better accuracy</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageClickArea;
