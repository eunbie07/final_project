import React from "react";

// 향상된 조명과 그림자
const EnhancedLighting = React.memo(function EnhancedLighting({ roomSize }) {
  const [w, h, d] = roomSize;
  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight
        position={[w, h * 1.5, d]}
        intensity={0.8}
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-camera-far={h * 3}
        shadow-camera-left={-w}
        shadow-camera-right={w}
        shadow-camera-top={d}
        shadow-camera-bottom={-d}
      />
    </>
  );
});

export default EnhancedLighting;