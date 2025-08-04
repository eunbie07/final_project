// frontend/src/components/RoomResult.jsx
import React from "react";
import RoomCanvas from "./RoomCanvas";

const ConfidenceIndicator = ({ confidence, reliability }) => {
  const getColorClasses = (conf) => {
    if (conf > 0.8)
      return {
        bg: "bg-green-500",
        border: "border-green-500",
        text: "text-white",
        bar: "bg-green-500",
      };
    if (conf > 0.6)
      return {
        bg: "bg-primary",
        border: "border-blue-600",
        text: "text-white",
        bar: "bg-primary",
      };
    return {
      bg: "bg-danger",
      border: "border-danger",
      text: "text-white",
      bar: "bg-danger",
    };
  };

  const colors = getColorClasses(confidence);
  const percentage = Math.round(confidence * 100);

  return (
    <div className={`p-4 rounded-lg border ${colors.bg} ${colors.border} mb-4`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className={`font-semibold flex items-center gap-2 ${colors.text}`}>
          측정 신뢰도
        </h3>
        <span className={`font-bold text-lg ${colors.text}`}>
          {percentage}%
        </span>
      </div>

      <div className="w-full bg-border rounded-full h-3 mb-2">
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
            높은 정확도로 측정되었습니다!
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
    <div className="bg-surface rounded-lg p-4 mt-4 border border-border">
      <h4 className="text-lg font-semibold mb-3 text-text-primary flex items-center gap-2">
        측정 세부사항
        <span className="text-xs bg-primary text-white px-2 py-1 rounded-full">
          {result.method || "improved_midas_relative"}
        </span>
      </h4>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 픽셀 거리 정보 */}
        {hasPixelData && (
          <div className="bg-background p-3 rounded border border-border">
            <h5 className="font-medium text-text-primary mb-2">픽셀 거리</h5>
            <div className="space-y-1 text-sm text-text-secondary">
              <div>수직: {result.pixel_distances.height_pixels} px</div>
              <div>가로: {result.pixel_distances.width_pixels} px</div>
              <div>세로: {result.pixel_distances.depth_pixels} px</div>
            </div>
          </div>
        )}

        {/* 스케일 정보 */}
        <div className="bg-background p-3 rounded border border-border">
          <h5 className="font-medium text-text-primary mb-2">스케일 정보</h5>
          <div className="space-y-1 text-sm text-text-secondary">
            <div>스케일 팩터: {(result.calculated_values?.pixels_per_meter / 100)?.toFixed(3)} cm/px</div>
            <div>기준 높이: {Math.round(result.dimensions?.height_cm || result.height_cm)} cm</div>
          </div>
        </div>

        {/* 원근 보정 정보 */}
        {hasPerspectiveData && (
          <div className="bg-background p-3 rounded border border-border">
            <h5 className="font-medium text-text-primary mb-2">원근 보정</h5>
            <div className="space-y-1 text-sm text-text-secondary">
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
          <div className="bg-background p-3 rounded border border-border">
            <h5 className="font-medium text-text-primary mb-2">측정 품질</h5>
            <div className="space-y-1 text-sm text-text-secondary">
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
  const width = result.dimensions?.width_cm || result.width_cm;
  const depth = result.dimensions?.depth_cm || result.depth_cm;
  const height = result.dimensions?.height_cm || result.height_cm;
  const confidence = result.confidence || 0;
  const area = result.calculated_values?.area_sqm || result.area_sqm;
  const volume = result.calculated_values?.volume_cum || result.volume_cum;
  const reliability = result.measurement_quality?.reliability || 
    (confidence >= 0.8 ? "높음" : confidence >= 0.6 ? "보통" : "낮음");

  // 디버깅 로그
  console.log("RoomResult received:", result);
  console.log("width:", width, "depth:", depth, "height:", height, "confidence:", confidence);

  // NaN 체크 및 기본값 설정
  const validWidth = isNaN(width) ? 0 : width;
  const validDepth = isNaN(depth) ? 0 : depth;
  const validHeight = isNaN(height) ? 0 : height;

  // 백엔드에서 계산된 값이 있으면 사용, 없으면 프론트엔드에서 계산
  const area_m2 = area || (validWidth * validDepth) / 10000;
  const volume_m3 = volume || (validWidth * validDepth * validHeight) / 1000000;

  // 평 계산 (1평 = 3.3058㎡)
  const area_pyeong = area_m2 / 3.3058;

  return (
    <div className="mt-10">
      {/* 신뢰도 표시 */}
      <ConfidenceIndicator confidence={confidence} reliability={reliability} />

      {/* 경고 메시지 */}
      {result.warning && (
        <div className="mb-6 p-4 bg-danger border border-danger rounded-lg">
          <div className="flex items-center gap-2">
            <span className="text-white"></span>
            <span className="font-medium text-white">주의사항</span>
          </div>
          <p className="text-white mt-1">{result.warning}</p>
        </div>
      )}

      {/* 2D 평면도 내용 */}
      <div>
          {/* 기존 RoomCanvas */}
          <RoomCanvas x={validWidth} y={validDepth} />

          {/* 측정 결과 표시 */}
          <div className="mt-6 p-6 border border-border rounded-xl bg-surface shadow-lg">
            <h2 className="text-xl font-bold mb-6 text-text-primary flex items-center gap-2">
              방 크기 측정 결과
              <span className="text-sm font-normal bg-primary text-white px-2 py-1 rounded-full">
                개선된 알고리즘
              </span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-background p-4 rounded-lg border border-border shadow-sm">
                <div className="text-sm text-text-secondary mb-1">가로 (Width)</div>
                <div className="text-2xl font-bold text-primary mb-1">
                  {validWidth?.toFixed(1)} cm
                </div>
                <div className="text-sm text-text-secondary">
                  {(validWidth / 100)?.toFixed(2)} m
                </div>
              </div>

              <div className="bg-background p-4 rounded-lg border border-border shadow-sm">
                <div className="text-sm text-text-secondary mb-1">세로 (Depth)</div>
                <div className="text-2xl font-bold text-primary mb-1">
                  {validDepth?.toFixed(1)} cm
                </div>
                <div className="text-sm text-text-secondary">
                  {(validDepth / 100)?.toFixed(2)} m
                </div>
              </div>

              <div className="bg-background p-4 rounded-lg border border-border shadow-sm">
                <div className="text-sm text-text-secondary mb-1">높이 (Height)</div>
                <div className="text-2xl font-bold text-primary mb-1">
                  {validHeight?.toFixed(1)} cm
                </div>
                <div className="text-sm text-text-secondary">
                  {(validHeight / 100)?.toFixed(2)} m
                </div>
              </div>
            </div>

            <div className="bg-surface p-4 rounded-lg border border-border shadow-sm mb-6">
              <h3 className="text-lg font-semibold mb-3 text-text-primary">면적 및 부피</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-3 bg-background rounded-lg border border-border">
                  <div className="text-sm text-text-secondary mb-1">제곱미터</div>
                  <div className="text-xl font-bold text-primary">
                    {area_m2.toFixed(2)} ㎡
                  </div>
                </div>
                <div className="text-center p-3 bg-background rounded-lg border border-border">
                  <div className="text-sm text-text-secondary mb-1">평수</div>
                  <div className="text-xl font-bold text-primary">
                    {area_pyeong.toFixed(1)} 평
                  </div>
                </div>
                <div className="text-center p-3 bg-background rounded-lg border border-border">
                  <div className="text-sm text-text-secondary mb-1">부피</div>
                  <div className="text-xl font-bold text-primary">
                    {volume_m3.toFixed(2)} ㎥
                  </div>
                </div>
              </div>
            </div>

            {/* 측정 세부사항 */}
            <MeasurementDetails result={result} />

            <div className="mt-6 p-4 bg-window-fill border border-window-stroke rounded-lg">
              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5"></span>
                <div>
                  <div className="font-medium text-accent-dark mb-1">참고사항</div>
                  <p className="text-sm text-secondary">
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
