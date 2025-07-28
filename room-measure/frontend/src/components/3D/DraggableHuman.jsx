import React, { useState, useRef, useCallback, useEffect } from "react";
import { useThree } from "@react-three/fiber";
import { ContactShadows, Text, useGLTF } from "@react-three/drei";
import * as THREE from "three";

// 드래그 가능한 사람 모델 (cm 단위로 수정)
export const DraggableHuman = React.memo(function DraggableHuman({
  height = 170,
  position,
  onPositionChange,
  roomSize,
  onDragStateChange,
}) {
  const { scene, error } = useGLTF("/human.glb");
  const modelRef = useRef();
  const [dragging, setDragging] = useState(false);
  const { camera, gl, raycaster, mouse } = useThree();
  const [modelHeight, setModelHeight] = useState(2.0);

  useEffect(() => {
    if (scene) {
      const box = new THREE.Box3().setFromObject(scene);
      const actualHeight = box.max.y - box.min.y;
      setModelHeight(actualHeight);
    }
  }, [scene]);

  const targetHeight = 170;
  const finalScale = modelHeight > 0 ? targetHeight / modelHeight : 85;

  // 전역 마우스 이벤트로 드래그 처리
  useEffect(() => {
    const handleMouseMove = (event) => {
      if (!dragging) return;

      const rect = gl.domElement.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera({ x, y }, camera);

      // y=0 평면과의 교차점 계산
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const intersection = new THREE.Vector3();

      if (raycaster.ray.intersectPlane(plane, intersection)) {
        const [roomWidth, , roomDepth] = roomSize;
        
        // 방 좌표계는 0에서 roomWidth/roomDepth까지 (중심 기준이 아님)
        // 방 경계 내로 제한 (0 기준)
        let newX = Math.max(10, Math.min(roomWidth - 10, intersection.x));
        let newZ = Math.max(10, Math.min(roomDepth - 10, intersection.z));

        onPositionChange([newX, 0, newZ]);
      }
    };

    const handleMouseUp = () => {
      setDragging(false);
      gl.domElement.style.cursor = "auto";
      onDragStateChange?.(false);
    };

    if (dragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      return () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [
    dragging,
    camera,
    gl,
    raycaster,
    onPositionChange,
    roomSize,
    onDragStateChange,
  ]);

  const handlePointerDown = useCallback(
    (e) => {
      e.stopPropagation();
      setDragging(true);
      gl.domElement.style.cursor = "grabbing";
      onDragStateChange?.(true);
    },
    [gl, onDragStateChange]
  );

  if (error || !scene) {
    return (
      <group position={position}>
        <mesh
          position={[0, height / 2, 0]}
          onPointerDown={handlePointerDown}
        >
          <cylinderGeometry args={[15, 20, height]} />
          <meshStandardMaterial color="#64748B" opacity={0.6} transparent />
        </mesh>
        <Text
          position={[30, height * 0.8, 0]}
          fontSize={15}
          color="#1E293B"
          anchorX="left"
          fontWeight="bold"
          backgroundColor="#FFFFFF"
          backgroundOpacity={0.8}
          padding={2}
        >
          {height}cm (fallback)
        </Text>
      </group>
    );
  }

  return (
    <group position={position}>
      {/* GLB 모델 표시 */}
      <primitive
        ref={modelRef}
        object={scene.clone()}
        scale={[finalScale, finalScale, finalScale]}
        position={[0, 0, 0]}
        castShadow
        receiveShadow
        onPointerDown={handlePointerDown}
      />

      <ContactShadows
        position={[0, 0.01, 0]}
        opacity={0.4}
        scale={50}
        blur={3}
        far={20}
      />
      <Text
        position={[30, height * 0.8, 0]}
        fontSize={15}
        color="#1E293B"
        anchorX="left"
        fontWeight="bold"
        backgroundColor="#FFFFFF"
        backgroundOpacity={0.8}
        padding={2}
      >
        {height}cm (GLB)
      </Text>
    </group>
  );
});

// GLB 파일 프리로딩
useGLTF.preload("/human.glb");