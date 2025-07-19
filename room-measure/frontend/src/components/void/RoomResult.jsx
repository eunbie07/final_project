// frontend/src/components/RoomResult.jsx
import React from "react";
import RoomCanvas from "./RoomCanvas";
// import Room3DViewer from './Room3DViewer';

const ConfidenceIndicator = ({ confidence, reliability }) => {
  const getColorClasses = (conf) => {
    if (conf > 0.8)
      return {
        bg: "bg-green-50",
        border: "border-green-200",
        text: "text-green-800",
        bar: "bg-green-500",
        icon: "🎯",
      };
    if (conf > 0.6)
      return {
        bg: "bg-yellow-50",
        border: "border-yellow-200",
        text: "text-yellow-800",
        bar: "bg-yellow-500",
        icon: "⚠️",
      };
    return {
      bg: "bg-red-50",
      border: "border-red-200",
      text: "text-red-800",
      bar: "bg-red-500",
      icon: "❌",
    };
  };

  const colors = getColorClasses(confidence);
  const percentage = Math.round(confidence * 100);

  return (
    <div className={`p-4 rounded-lg border ${colors.bg} ${colors.border} mb-4`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className={`font-semibold flex items-center gap-2 ${colors.text}`}>
          {colors.icon} 측정 신뢰도
        </h3>
        <span className={`font-bold text-lg ${colors.text}`}>
          {percentage}%
        </span>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
        <div
          className={`h-3 rounded-full transition-all duration-500 ${colors.bar}`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      <div className={`text-sm ${colors.text}`}>
        <div className="font-medium mb-1">신뢰도: {reliability}</div>
        {confidence < 0.7 && (
          <div className="text-xs opacity-75">
            💡 더 정확한 측정을 위해 다른 각도에서 시도해보세요
          </div>
        )}
        {confidence >= 0.8 && (
          <div className="text-xs opacity-75">
            ✨ 높은 정확도로 측정되었습니다!
          </div>
        )}
      </div>
    </div>
  );
};

const MeasurementDetails = ({ result }) => {
  const hasPixelData = result.pixel_distances;
  const hasPerspectiveData = result.perspective_correction;

  return (
    <div className="bg-gray-50 rounded-lg p-4 mt-4">
      <h4 className="font-semibold mb-3 text-gray-700 flex items-center gap-2">
        🔍 측정 세부사항
        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
          {result.method || "improved_midas_relative"}
        </span>
      </h4>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 픽셀 거리 정보 */}
        {hasPixelData && (
          <div className="bg-white p-3 rounded border">
            <h5 className="font-medium text-gray-600 mb-2">📏 픽셀 거리</h5>
            <div className="space-y-1 text-sm">
              <div>수직: {result.pixel_distances.vertical} px</div>
              <div>가로: {result.pixel_distances.horizontal} px</div>
              <div>세로: {result.pixel_distances.depth} px</div>
            </div>
          </div>
        )}

        {/* 스케일 정보 */}
        <div className="bg-white p-3 rounded border">
          <h5 className="font-medium text-gray-600 mb-2">📐 스케일 정보</h5>
          <div className="space-y-1 text-sm">
            <div>스케일 팩터: {result.scale_factor} cm/px</div>
            <div>기준 높이: {result.height_cm} cm</div>
          </div>
        </div>

        {/* 원근 보정 정보 */}
        {hasPerspectiveData && (
          <div className="bg-white p-3 rounded border">
            <h5 className="font-medium text-gray-600 mb-2">🎯 원근 보정</h5>
            <div className="space-y-1 text-sm">
              <div>
                가로 보정: {result.perspective_correction.horizontal_factor}
              </div>
              <div>세로 보정: {result.perspective_correction.depth_factor}</div>
              <div>깊이 범위: {result.perspective_correction.depth_range}</div>
            </div>
          </div>
        )}

        {/* 측정 품질 */}
        {result.measurement_quality && (
          <div className="bg-white p-3 rounded border">
            <h5 className="font-medium text-gray-600 mb-2">⭐ 측정 품질</h5>
            <div className="space-y-1 text-sm">
              <div>
                신뢰도 점수: {result.measurement_quality.confidence_score}
              </div>
              <div>품질 등급: {result.measurement_quality.reliability}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const RoomResult = ({ result, depthImageUrl }) => {
  if (!result) return null;

  // 새로운 API 응답 형태에 맞춰 데이터 추출
  const width = result.width_cm;
  const depth = result.depth_cm;
  const confidence = result.confidence || 0;
  const reliability = result.measurement_quality?.reliability || "알 수 없음";

  // 디버깅 로그
  console.log("🔍 RoomResult received:", result);
  console.log("🔍 width:", width, "depth:", depth, "confidence:", confidence);

  // NaN 체크 및 기본값 설정
  const validWidth = isNaN(width) ? 0 : width;
  const validDepth = isNaN(depth) ? 0 : depth;

  // 평방미터 계산
  const area_m2 = (validWidth * validDepth) / 10000;

  // 평 계산 (1평 = 3.3058㎡)
  const area_pyeong = area_m2 / 3.3058;

  return (
    <div className="mt-10">
      {/* 신뢰도 표시 */}
      <ConfidenceIndicator confidence={confidence} reliability={reliability} />

      {/* 경고 메시지 */}
      {result.warning && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center gap-2">
            <span className="text-yellow-600">⚠️</span>
            <span className="font-medium text-yellow-800">주의사항</span>
          </div>
          <p className="text-yellow-700 mt-1">{result.warning}</p>
        </div>
      )}

      {/* 기존 RoomCanvas - x, y 파라미터로 전달 */}
      <RoomCanvas x={validWidth} y={validDepth} />

      {/* 새로운 측정 결과 표시 */}
      <div className="mt-6 p-6 border rounded-xl bg-gradient-to-br from-white to-gray-50 shadow-lg">
        <h2 className="text-xl font-bold mb-6 text-gray-800 flex items-center gap-2">
          📏 방 크기 측정 결과
          <span className="text-sm font-normal bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
            개선된 알고리즘
          </span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white p-4 rounded-lg border shadow-sm">
            <div className="text-sm text-gray-600 mb-1">가로 (Width)</div>
            <div className="text-2xl font-bold text-blue-600 mb-1">
              {width?.toFixed(1)} cm
            </div>
            <div className="text-sm text-gray-500">
              {(width / 100)?.toFixed(2)} m
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border shadow-sm">
            <div className="text-sm text-gray-600 mb-1">세로 (Depth)</div>
            <div className="text-2xl font-bold text-green-600 mb-1">
              {depth?.toFixed(1)} cm
            </div>
            <div className="text-sm text-gray-500">
              {(depth / 100)?.toFixed(2)} m
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border shadow-sm mb-6">
          <h3 className="font-semibold mb-3 text-gray-700">📐 면적 계산</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="text-center p-3 bg-purple-50 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">제곱미터</div>
              <div className="text-xl font-bold text-purple-600">
                {area_m2.toFixed(2)} ㎡
              </div>
            </div>
            <div className="text-center p-3 bg-orange-50 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">평수</div>
              <div className="text-xl font-bold text-orange-600">
                {area_pyeong.toFixed(1)} 평
              </div>
            </div>
          </div>
        </div>

        {/* 측정 세부사항 */}
        <MeasurementDetails result={result} />

        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start gap-2">
            <span className="text-blue-600 mt-0.5">💡</span>
            <div>
              <div className="font-medium text-blue-800 mb-1">참고사항</div>
              <p className="text-sm text-blue-700">
                이 측정은 개선된 알고리즘으로 층고 {result.height_cm}cm를
                기준으로 계산되었습니다. 신뢰도가 높을수록 실제 크기에 가까운
                결과입니다.
              </p>
            </div>
          </div>
        </div>
      </div>

      {depthImageUrl && (
        <div className="mt-8 bg-white p-6 rounded-xl shadow-lg border">
          <h2 className="font-bold mb-4 text-gray-800 flex items-center gap-2">
            🎨 Depth Map 시각화
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
              MiDaS 출력
            </span>
          </h2>
          <div className="text-center">
            <img
              src={depthImageUrl}
              alt="Depth Map"
              className="max-w-full h-auto border border-gray-300 rounded-lg shadow-sm mx-auto"
              style={{ maxHeight: "400px" }}
            />
            <p className="text-xs text-gray-500 mt-3">
              * 따뜻한 색상(빨강/노랑)은 가까운 거리, 차가운 색상(파랑/보라)은
              먼 거리를 나타냅니다
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default RoomResult;
