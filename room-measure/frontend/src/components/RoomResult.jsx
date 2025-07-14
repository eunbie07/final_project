// frontend/src/components/RoomResult.jsx
import React from "react";
import RoomCanvas from "./RoomCanvas";

const RoomResult = ({ result, depthImageUrl }) => {
  if (!result) return null;

  // 새로운 API 응답 형태에 맞춰 데이터 추출
  const width = result.width_cm;
  const depth = result.depth_cm;
  
  // 디버깅 로그
  console.log("🔍 RoomResult received:", result);
  console.log("🔍 width:", width, "depth:", depth);
  
  // NaN 체크 및 기본값 설정
  const validWidth = isNaN(width) ? 0 : width;
  const validDepth = isNaN(depth) ? 0 : depth;
  
  // 평방미터 계산
  const area_m2 = (validWidth * validDepth) / 10000;
  
  // 평 계산 (1평 = 3.3058㎡)
  const area_pyeong = area_m2 / 3.3058;

  return (
    <div className="mt-10">
      {/* 기존 RoomCanvas - x, y 파라미터로 전달 */}
      <RoomCanvas x={validWidth} y={validDepth} />
      
      {/* 새로운 측정 결과 표시 */}
      <div className="mt-6 p-4 border rounded-lg bg-gray-50">
        <h2 className="text-lg font-bold mb-4">📏 방 크기 측정 결과</h2>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-white p-3 rounded border">
            <div className="text-sm text-gray-600">가로 (Width)</div>
            <div className="text-xl font-bold text-blue-600">
              {width?.toFixed(1)} cm
            </div>
            <div className="text-sm text-gray-500">
              {(width / 100)?.toFixed(2)} m
            </div>
          </div>
          
          <div className="bg-white p-3 rounded border">
            <div className="text-sm text-gray-600">세로 (Depth)</div>
            <div className="text-xl font-bold text-green-600">
              {depth?.toFixed(1)} cm
            </div>
            <div className="text-sm text-gray-500">
              {(depth / 100)?.toFixed(2)} m
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded border mb-4">
          <h3 className="font-semibold mb-2">📐 면적 계산</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-gray-600">제곱미터</div>
              <div className="text-lg font-bold text-purple-600">
                {area_m2.toFixed(2)} ㎡
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-600">평수</div>
              <div className="text-lg font-bold text-orange-600">
                {area_pyeong.toFixed(1)} 평
              </div>
            </div>
          </div>
        </div>

        {/* 계산 세부사항 */}
        {result.cm_per_pixel && (
          <div className="bg-gray-100 p-3 rounded mb-4">
            <h4 className="font-semibold mb-2">🔍 계산 세부사항</h4>
            <div className="text-sm space-y-1">
              <div>픽셀당 실제 거리: {result.cm_per_pixel} cm/pixel</div>
              {result.vertical_pixels && (
                <div>수직 픽셀 거리: {result.vertical_pixels} px (층고 230cm 기준)</div>
              )}
              {result.horizontal_pixels && (
                <div>가로 픽셀 거리: {result.horizontal_pixels} px</div>
              )}
              {result.depth_pixels && (
                <div>세로 픽셀 거리: {result.depth_pixels} px</div>
              )}
            </div>
          </div>
        )}
        
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
          <p className="text-sm text-yellow-800">
            ⚠️ <strong>참고사항:</strong> 이 측정은 층고 230cm를 기준으로 한 추정값입니다. 
            실제 측정과 차이가 있을 수 있습니다.
          </p>
        </div>
      </div>
             
      {depthImageUrl && (
        <div className="mt-8">
          <h2 className="font-bold mb-2">🎨 Depth Map 시각화</h2>
          <img
            src={depthImageUrl}
            alt="Depth Map"
            className="w-full max-w-md border border-gray-400"
          />
          <p className="text-xs text-gray-500 mt-2">
            * 색상이 따뜻할수록 가까운 거리, 차가울수록 먼 거리를 의미합니다.
          </p>
        </div>
      )}
    </div>
  );
};

export default RoomResult;