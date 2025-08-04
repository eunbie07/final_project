import React, { useRef, useEffect, useState } from "react";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";

// 재질 설정 헬퍼 함수
const setupMaterial = (material, selected, hasCollision) => {
  try {
    if (selected) {
      if (material.emissive) material.emissive.setHex(0x333333);
      if (material.emissiveIntensity !== undefined)
        material.emissiveIntensity = 0.1;
    }

    if (hasCollision) {
      if (material.color) material.color.setHex(0xff9999);
      if (material.emissive) material.emissive.setHex(0xff0000);
      if (material.emissiveIntensity !== undefined)
        material.emissiveIntensity = 0.2;
    }

    return material;
  } catch (error) {
    console.warn("재질 속성 설정 실패:", error);
    return material;
  }
};

// 3D 모델 로더 컴포넌트
export const FurnitureModel = React.memo(function FurnitureModel({
  modelConfig,
  size,
  color,
  selected,
  hasCollision,
  isChildComponent = false, // 부모 컴포넌트 내에서 사용되는지 여부
}) {
  // 실시간 offset 조정을 위한 상태
  const [dynamicOffset, setDynamicOffset] = useState(() => {
    const initialOffset = modelConfig.offset || [0, 0, 0];
    // Y축 방향을 올바르게 설정 (양수 = 위로, 음수 = 아래로)
    return [initialOffset[0], initialOffset[1], initialOffset[2]];
  });
  const group = useRef();
  const [loadedModel, setLoadedModel] = useState(null);
  const [modelMeshes, setModelMeshes] = useState([]);

  // GLTF 모델 로드
  const { scene, error } = useGLTF(modelConfig.path, true, true, (loader) => {
    loader.manager.onError = (url) => {
      console.warn("Failed to load 3D model:", url);
    };
  });

  // 개발자 모드에서 실시간 조정을 위한 전역 함수 등록
  useEffect(() => {
    if (typeof window !== "undefined") {
      // 현재 모델의 offset을 조정하는 함수
      window.adjustModelOffset = (x = 0, y = 0, z = 0) => {
        setDynamicOffset([x, y, z]);
        console.log(`Offset 조정됨: [${x}, ${y}, ${z}]`);
      };

      // 현재 offset 값을 확인하는 함수
      window.getCurrentOffset = () => {
        console.log("현재 offset:", dynamicOffset);
        return dynamicOffset;
      };

      // offset을 초기화하는 함수
      window.resetOffset = () => {
        setDynamicOffset(modelConfig.offset || [0, 0, 0]);
        console.log("Offset 초기화됨");
      };

      console.log("🎮 실시간 모델 조정 도구 준비 완료!");
      console.log("사용법:");
      console.log("- adjustModelOffset(x, y, z): offset 조정");
      console.log("- getCurrentOffset(): 현재 offset 확인");
      console.log("- resetOffset(): offset 초기화");
    }
  }, [modelConfig.offset]);

  // 모델 로드 및 메시 찾기
  useEffect(() => {
    if (!scene || error) {
      setLoadedModel(null);
      setModelMeshes([]);
      return;
    }

    try {
      // 간단 모드: meshNames가 ["*"] 이면 전체 scene 사용
      if (modelConfig.meshNames && modelConfig.meshNames.includes("*")) {
        console.log("전체 GLB 파일 로드 모드");
        setLoadedModel(scene);
        setModelMeshes([scene]); // 전체 scene을 하나의 "메시"처럼 취급
        return;
      }

      const foundMeshes = [];

      // 모든 메시 이름들을 로그로 출력 (디버깅용)
      const allMeshNames = [];
      scene.traverse((child) => {
        if (child.isMesh) {
          allMeshNames.push(child.name);
        }
      });
      console.log("=== GLB 파일 내 모든 메시 이름들 ===");
      console.log(allMeshNames);
      console.log("찾고 있는 키워드들:", modelConfig.meshNames);

      // 설정된 메시 이름들을 찾기
      scene.traverse((child) => {
        if (child.isMesh) {
          const childNameLower = child.name.toLowerCase();

          // 메시 이름이 설정된 키워드와 일치하는지 확인
          const matches = modelConfig.meshNames.some((keyword) =>
            childNameLower.includes(keyword.toLowerCase())
          );

          console.log(`메시 "${child.name}" - 매칭: ${matches}`);

          if (matches) {
            // 메시를 복제하여 사용
            const clonedMesh = child.clone();

            // 메시 위치와 회전 초기화 (조립 문제 해결)
            clonedMesh.position.set(0, 0, 0);
            clonedMesh.rotation.set(0, 0, 0);
            clonedMesh.scale.set(1, 1, 1);

            // 안전한 재질 설정
            try {
              if (clonedMesh.material) {
                // 재질이 배열인 경우 처리
                if (Array.isArray(clonedMesh.material)) {
                  clonedMesh.material = clonedMesh.material.map((mat) => {
                    const newMat = mat.clone();
                    return setupMaterial(newMat, selected, hasCollision);
                  });
                } else {
                  const newMaterial = clonedMesh.material.clone();
                  clonedMesh.material = setupMaterial(
                    newMaterial,
                    selected,
                    hasCollision
                  );
                }
              } else {
                // 재질이 없는 경우 기본 재질 생성
                clonedMesh.material = new THREE.MeshStandardMaterial({
                  color: selected
                    ? "#ffffff"
                    : hasCollision
                    ? "#ff9999"
                    : "#cccccc",
                  roughness: 0.7,
                  metalness: 0.1,
                });
              }
            } catch (materialError) {
              console.warn("재질 설정 오류, 기본 재질 사용:", materialError);
              clonedMesh.material = new THREE.MeshStandardMaterial({
                color: selected
                  ? "#ffffff"
                  : hasCollision
                  ? "#ff9999"
                  : "#cccccc",
                roughness: 0.7,
                metalness: 0.1,
              });
            }

            foundMeshes.push(clonedMesh);
          }
        }
      });

      if (foundMeshes.length > 0) {
        setModelMeshes(foundMeshes);
        setLoadedModel(scene);
        console.log(`Found ${foundMeshes.length} meshes for furniture model`);
      } else {
        console.warn("No matching meshes found for:", modelConfig.meshNames);
        setModelMeshes([]);
        setLoadedModel(null);
      }
    } catch (err) {
      console.error("Error processing 3D model:", err);
      setLoadedModel(null);
      setModelMeshes([]);
    }
  }, [scene, error, modelConfig.meshNames, selected, hasCollision]);

  // 모델 크기 조정
  useEffect(() => {
    if (!group.current || modelMeshes.length === 0) return;

    try {
      // 바운딩 박스 계산
      const box = new THREE.Box3();
      modelMeshes.forEach((mesh) => {
        const meshBox = new THREE.Box3().setFromObject(mesh);
        box.union(meshBox);
      });

      const modelSize = box.getSize(new THREE.Vector3());

      // 타겟 크기와 모델 크기 비율 계산
      const scaleX = size[0] / modelSize.x;
      const scaleY = size[1] / modelSize.y;
      const scaleZ = size[2] / modelSize.z;

      // 균등 스케일링 (가장 작은 비율 사용)
      const uniformScale = Math.min(scaleX, scaleY, scaleZ);

      group.current.scale.set(
        uniformScale * modelConfig.scale[0],
        uniformScale * modelConfig.scale[1],
        uniformScale * modelConfig.scale[2]
      );

      // 회전 적용
      group.current.rotation.set(...modelConfig.rotation);

      // 모델을 바닥에 맞춤 - 위치 조정 수정
      const scaledBox = box.clone();
      scaledBox.min.multiplyScalar(uniformScale);
      scaledBox.max.multiplyScalar(uniformScale);

      // 모델을 가구의 중심에 맞춤 (Y축은 바닥 기준)
      const centerY = (scaledBox.max.y + scaledBox.min.y) / 2;

      // 부모 컴포넌트 내에서 사용되는 경우 위치 조정을 하지 않음
      if (!isChildComponent) {
        group.current.position.y = -centerY + dynamicOffset[1];

        // X, Z축도 중심에 맞춤
        const centerX = (scaledBox.max.x + scaledBox.min.x) / 2;
        const centerZ = (scaledBox.max.z + scaledBox.min.z) / 2;
        group.current.position.x = -centerX;
        group.current.position.z = -centerZ;
      } else {
        // 부모 컴포넌트 내에서 사용되는 경우 - 더 직관적인 위치 조정
        // 모델을 바닥에 놓고 offset만 적용
        group.current.position.y = size[1] / 2 + (dynamicOffset[1] || 0);

        // X, Z축 오프셋 적용
        group.current.position.x = dynamicOffset[0] || 0;
        group.current.position.z = dynamicOffset[2] || 0;

        console.log("직관적 위치 조정:", {
          modelHeight: size[1],
          baseY: size[1] / 2,
          offsetY: dynamicOffset[1],
          finalY: group.current.position.y,
          finalX: group.current.position.x,
          finalZ: group.current.position.z,
          dynamicOffset: dynamicOffset,
        });
      }

      // 디버깅 정보 출력
      console.log("3D 모델 위치 조정 정보:", {
        modelConfig: modelConfig.path,
        targetSize: size,
        modelSize,
        uniformScale,
        scaledBox: {
          min: scaledBox.min,
          max: scaledBox.max,
          center: {
            x: (scaledBox.max.x + scaledBox.min.x) / 2,
            y: centerY,
            z: (scaledBox.max.z + scaledBox.min.z) / 2,
          },
        },
        finalPosition: {
          x: group.current.position.x,
          y: group.current.position.y,
          z: group.current.position.z,
        },
        dynamicOffset: dynamicOffset,
        centerY: centerY,
        yCalculation: `-${centerY} + ${dynamicOffset[1]} = ${
          -centerY + dynamicOffset[1]
        }`,
        isChildComponent,
      });
    } catch (err) {
      console.error("Error scaling 3D model:", err);
    }
  }, [modelMeshes, size, modelConfig.scale, modelConfig.rotation, isChildComponent, dynamicOffset]);

  // 모델이 로드되지 않은 경우 null 반환 (폴백으로 박스 렌더링)
  if (!loadedModel || modelMeshes.length === 0) {
    return null;
  }

  return (
    <group ref={group}>
      {modelMeshes.map((mesh, index) => (
        <primitive
          key={`mesh-${index}`}
          object={mesh}
          castShadow
          receiveShadow
        />
      ))}
    </group>
  );
});

// GLTF 리소스 정리
useGLTF.preload = () => {};
