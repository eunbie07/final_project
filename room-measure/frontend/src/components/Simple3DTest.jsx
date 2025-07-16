import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";

const Simple3DTest = ({
  roomWidth = 400,
  roomHeight = 300,
  placedFurniture = [],
  onClose,
}) => {
  const mountRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [loadingStep, setLoadingStep] = useState("초기화 중...");

  useEffect(() => {
    let renderer, scene, camera, animationId;

    const init = async () => {
      try {
        setLoadingStep("마운트 확인 중...");
        console.log("🚀 3D 초기화 시작");

        // 마운트 참조 확인 (더 안전하게)
        if (!mountRef.current) {
          console.log("⏳ 마운트 참조 대기 중...");
          // 잠깐 기다린 후 다시 시도
          setTimeout(() => {
            if (mountRef.current) {
              init(); // 재귀 호출
            } else {
              throw new Error(
                "마운트 참조를 찾을 수 없습니다. DOM이 준비되지 않았습니다."
              );
            }
          }, 200);
          return;
        }

        console.log("✅ 마운트 참조 확인됨:", mountRef.current);

        setLoadingStep("Three.js 초기화 중...");

        // Scene
        setLoadingStep("씬 생성 중...");
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        console.log("✅ 씬 생성 완료");

        // Camera
        setLoadingStep("카메라 설정 중...");
        const containerWidth = mountRef.current.clientWidth || 800;
        const containerHeight = mountRef.current.clientHeight || 400;

        camera = new THREE.PerspectiveCamera(
          75,
          containerWidth / containerHeight,
          1,
          1000
        );
        camera.position.set(200, 150, 200);
        camera.lookAt(0, 0, 0);
        console.log("✅ 카메라 설정 완료");

        // Renderer (간단하게)
        setLoadingStep("렌더러 생성 중...");
        renderer = new THREE.WebGLRenderer({ antialias: false }); // 안티앨리어싱 끔
        renderer.setSize(containerWidth, containerHeight);
        // 그림자 비활성화로 성능 향상
        renderer.shadowMap.enabled = false;

        // DOM에 추가하기 전에 한번 더 확인
        if (!mountRef.current) {
          throw new Error("렌더러 추가 중 마운트 참조가 사라졌습니다");
        }

        mountRef.current.appendChild(renderer.domElement);
        console.log("✅ 렌더러 생성 완료");

        // 간단한 조명만
        setLoadingStep("조명 설정 중...");
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);
        console.log("✅ 조명 설정 완료");

        // 방 바닥만 (간단하게)
        setLoadingStep("방 생성 중...");
        const floorGeometry = new THREE.PlaneGeometry(roomWidth, roomHeight);
        const floorMaterial = new THREE.MeshBasicMaterial({ color: 0xeeeeee });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        scene.add(floor);
        console.log("✅ 바닥 생성 완료");

        // 가구들 (매우 간단하게)
        setLoadingStep(`가구 ${placedFurniture.length}개 생성 중...`);
        placedFurniture.forEach((furniture, index) => {
          console.log(`🪑 가구 ${index + 1} 생성: ${furniture.name}`);

          const geometry = new THREE.BoxGeometry(
            furniture.width,
            50,
            furniture.height
          );
          const material = new THREE.MeshBasicMaterial({
            color: furniture.color,
          });
          const mesh = new THREE.Mesh(geometry, material);

          mesh.position.set(
            furniture.x - roomWidth / 2 + furniture.width / 2,
            25,
            furniture.y - roomHeight / 2 + furniture.height / 2
          );

          scene.add(mesh);
        });
        console.log("✅ 모든 가구 생성 완료");

        // 첫 렌더링
        setLoadingStep("첫 렌더링 중...");
        renderer.render(scene, camera);
        console.log("✅ 첫 렌더링 완료");

        // 애니메이션 시작
        const animate = () => {
          animationId = requestAnimationFrame(animate);
          if (renderer && scene && camera) {
            renderer.render(scene, camera);
          }
        };
        animate();

        console.log("🎉 3D 초기화 완전히 완료!");
        setIsLoading(false);
      } catch (err) {
        console.error("❌ 3D 초기화 실패:", err);
        setError(err.message);
        setIsLoading(false);
      }
    };

    // 더 긴 지연 후 초기화 (DOM이 완전히 준비될 때까지)
    const timer = setTimeout(init, 300);

    return () => {
      clearTimeout(timer);
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
      if (renderer && mountRef.current) {
        try {
          mountRef.current.removeChild(renderer.domElement);
          renderer.dispose();
        } catch (e) {
          console.warn("정리 중 에러:", e);
        }
      }
    };
  }, [roomWidth, roomHeight, placedFurniture]);

  if (error) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-red-50 rounded-lg border border-red-200">
        <div className="text-center p-6">
          <div className="text-red-600 text-lg font-semibold mb-2">
            3D 로딩 실패
          </div>
          <div className="text-red-700 text-sm mb-4">{error}</div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg"
          >
            2D로 돌아가기
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-gray-100 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <div className="text-gray-600 mb-2">간단 3D 테스트 중...</div>
          <div className="text-sm text-blue-600 font-medium">{loadingStep}</div>
          <div className="text-xs text-gray-500 mt-2">
            콘솔(F12)에서 진행상황을 확인해보세요
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="p-4 bg-green-50 border-b flex justify-between items-center">
        <div>
          <div className="font-medium text-green-800">
            ✅ 간단 3D 테스트 성공!
          </div>
          <div className="text-sm text-green-600">
            가구 {placedFurniture.length}개 • {roomWidth}×{roomHeight}cm
          </div>
        </div>
        <button
          onClick={onClose}
          className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg"
        >
          ← 2D로 돌아가기
        </button>
      </div>

      <div
        ref={mountRef}
        className="w-full bg-gray-50"
        style={{ height: "400px" }}
      />

      <div className="p-4 bg-blue-50 text-sm">
        <div className="text-blue-800 font-medium mb-1">
          🧪 이것은 테스트 버전입니다
        </div>
        <div className="text-blue-700">
          • 그림자, 텍스처, 고급 조명 제거로 최대한 간단하게 만든 버전
          <br />
          • 이것도 느리면 브라우저나 하드웨어 문제일 수 있음
          <br />• 빠르게 로드되면 원본 버전에 문제가 있는 것
        </div>
      </div>
    </div>
  );
};

export default Simple3DTest;
