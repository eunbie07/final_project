import React from "react";

// 향상된 조명과 그림자
const EnhancedLighting = React.memo(function EnhancedLighting({ roomSize }) {
  const [w, h, d] = roomSize;
  return (
    <>
      {/* 부드러운 주변 조명 - 재질 효과를 위해 줄임 */}
      <ambientLight intensity={0.15} color="#f5f0e8" />
      
      {/* 따뜻한 메인 조명 (햇빛) - 재질 차이를 보기 위해 강화 */}
      <directionalLight
        position={[w * 0.8, h * 2, d * 0.6]}
        intensity={1.5}
        color="#fff4e6"
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-far={h * 4}
        shadow-camera-left={-w * 1.2}
        shadow-camera-right={w * 1.2}
        shadow-camera-top={d * 1.2}
        shadow-camera-bottom={-d * 1.2}
        shadow-bias={-0.0001}
      />
      
      {/* 보조 조명 (반대편에서 부드럽게) */}
      <directionalLight
        position={[-w * 0.3, h * 1.2, -d * 0.3]}
        intensity={0.4}
        color="#e6f3ff"
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-camera-far={h * 2}
        shadow-camera-left={-w * 0.8}
        shadow-camera-right={w * 0.8}
        shadow-camera-top={d * 0.8}
        shadow-camera-bottom={-d * 0.8}
      />
      
      {/* 포인트 조명 (실내 조명 느낌) */}
      <pointLight
        position={[w * 0.3, h * 0.8, d * 0.3]}
        intensity={0.6}
        color="#fff8dc"
        distance={w * 2}
        decay={2}
      />
      
      <pointLight
        position={[w * 0.7, h * 0.8, d * 0.7]}
        intensity={0.5}
        color="#ffeaa7"
        distance={w * 1.5}
        decay={2}
      />
    </>
  );
});

export default EnhancedLighting;