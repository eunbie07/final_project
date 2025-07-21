// frontend/src/components/RoomResult.jsx
import React from "react";
import RoomCanvas from "./RoomCanvas";

const ConfidenceIndicator = ({ confidence, reliability }) => {
  const getColorClasses = (conf) => {
    if (conf > 0.8)
      return {
        bg: "bg-rose-50",
        border: "border-rose-200",
        text: "text-rose-800",
        bar: "bg-rose-500",
      };
    if (conf > 0.6)
      return {
        bg: "bg-pink-50",
        border: "border-pink-200",
        text: "text-pink-800",
        bar: "bg-pink-500",
      };
    return {
      bg: "bg-red-50",
      border: "border-red-200",
      text: "text-red-800",
      bar: "bg-red-500",
    };
  };

  const colors = getColorClasses(confidence);
  const percentage = Math.round(confidence * 100);

  return (
    <div className={`p-4 rounded-lg border ${colors.bg} ${colors.border} mb-4`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className={`font-semibold flex items-center gap-2 ${colors.text}`}>
          <strong>측정 신뢰도</strong>
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
        <div className="font-medium mb-1"><strong>신뢰도:</strong> {reliability}</div>
        {confidence < 0.7 && (
          <div className="text-xs opacity-75">
            💡 더 정확한 측정을 위해 다른 각도에서 시도해보세요
          </div>
        )}
        {confidence >= 0.8 && (
          <div className="text-xs opacity-75">
            <strong>높은 정확도로 측정되었습니다!</strong>
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
    <div className="bg-rose-50 rounded-lg p-4 mt-4">
      <h4 className="font-semibold mb-3 text-gray-700 flex items-center gap-2">
        <strong>측정 세부사항</strong>
        <span className="text-xs bg-pink-100 text-pink-800 px-2 py-1 rounded-full">
          {result.method || "improved_midas_relative"}
        </span>
      </h4>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 픽셀 거리 정보 */}
        {hasPixelData && (
          <div className="bg-white p-3 rounded border">
            <h5 className="font-medium text-gray-600 mb-2">픽셀 거리</h5>
            <div className="space-y-1 text-sm">
              <div>수직: {result.pixel_distances.vertical} px</div>
              <div>가로: {result.pixel_distances.horizontal} px</div>
              <div>세로: {result.pixel_distances.depth} px</div>
            </div>
          </div>
        )}

        {/* 스케일 정보 */}
        <div className="bg-white p-3 rounded border">
          <h5 className="font-medium text-gray-600 mb-2">스케일 정보</h5>
          <div className="space-y-1 text-sm">
            <div>스케일 팩터: {result.scale_factor} cm/px</div>
            <div>기준 높이: {Math.round(result.height_cm)} cm</div>
          </div>
        </div>

        {/* 원근 보정 정보 */}
        {hasPerspectiveData && (
          <div className="bg-white p-3 rounded border">
            <h5 className="font-medium text-gray-600 mb-2">원근 보정</h5>
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
            <h5 className="font-medium text-gray-600 mb-2">측정 품질</h5>
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
  console.log("RoomResult received:", result);
  console.log("width:", width, "depth:", depth, "confidence:", confidence);

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
        <div className="mb-6 p-4 bg-rose-50 border border-rose-200 rounded-lg">
          <div className="flex items-center gap-2">
            <span className="text-rose-600"></span>
            <span className="font-medium text-rose-800"><strong>주의사항</strong></span>
          </div>
          <p className="text-rose-700 mt-1">{result.warning}</p>
        </div>
      )}

      {/* 2D 평면도 내용 */}
      <div>
          {/* 기존 RoomCanvas */}
          <RoomCanvas x={validWidth} y={validDepth} />

          {/* 측정 결과 표시 */}
          <div className="mt-6 p-6 border rounded-xl bg-gradient-to-br from-white to-rose-50 shadow-lg">
            <h2 className="text-xl font-bold mb-6 text-gray-800 flex items-center gap-2">
              <strong>방 크기 측정 결과</strong>
              <span className="text-sm font-normal bg-pink-100 text-pink-800 px-2 py-1 rounded-full">
                개선된 알고리즘
              </span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className="bg-white p-4 rounded-lg border shadow-sm">
                <div className="text-sm text-gray-600 mb-1"><strong>가로 (Width)</strong></div>
                <div className="text-2xl font-bold text-rose-600 mb-1">
                  {width?.toFixed(1)} cm
                </div>
                <div className="text-sm text-gray-500">
                  {(width / 100)?.toFixed(2)} m
                </div>
              </div>

              <div className="bg-white p-4 rounded-lg border shadow-sm">
                <div className="text-sm text-gray-600 mb-1"><strong>세로 (Depth)</strong></div>
                <div className="text-2xl font-bold text-pink-600 mb-1">
                  {depth?.toFixed(1)} cm
                </div>
                <div className="text-sm text-gray-500">
                  {(depth / 100)?.toFixed(2)} m
                </div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg border shadow-sm mb-6">
              <h3 className="font-semibold mb-3 text-gray-700"><strong>면적 계산</strong></h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="text-center p-3 bg-rose-50 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1"><strong>제곱미터</strong></div>
                  <div className="text-xl font-bold text-rose-600">
                    {area_m2.toFixed(2)} ㎡
                  </div>
                </div>
                <div className="text-center p-3 bg-pink-50 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1"><strong>평수</strong></div>
                  <div className="text-xl font-bold text-pink-600">
                    {area_pyeong.toFixed(1)} 평
                  </div>
                </div>
              </div>
            </div>

            {/* 측정 세부사항 */}
            <MeasurementDetails result={result} />

            <div className="mt-6 p-4 bg-pink-50 border border-pink-200 rounded-lg">
              <div className="flex items-start gap-2">
                <span className="text-pink-600 mt-0.5"></span>
                <div>
                  <div className="font-medium text-pink-800 mb-1"><strong>참고사항</strong></div>
                  <p className="text-sm text-pink-700">
                    이 측정은 개선된 알고리즘으로 층고{" "}
                    {Math.round(result.height_cm)}cm를 기준으로 계산되었습니다.
                    신뢰도가 높을수록 실제 크기에 가까운 결과입니다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
    </div>
  );
};

export default RoomResult;
