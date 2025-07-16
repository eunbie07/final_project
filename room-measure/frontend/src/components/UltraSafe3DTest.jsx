import React, { useRef, useEffect, useState, useCallback } from "react";
import * as THREE from "three";

const UltraSafe3DTest = ({
  roomWidth = 400,
  roomHeight = 300,
  placedFurniture = [],
  onClose,
}) => {
  const mountRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [loadingStep, setLoadingStep] = useState("컴포넌트 초기화 중...");
  const [canStart3D, setCanStart3D] = useState(false);

  // DOM이 완전히 준비되었는지 확인하는 함수
  const checkDOMReady = useCallback(() => {
    console.log("🔍 DOM 준비 상태 확인 중...");

    if (!mountRef.current) {
      console.log("❌ mountRef.current가 null");
      return false;
    }

    if (!mountRef.current.offsetParent && mountRef.current.offsetWidth === 0) {
      console.log("❌ 엘리먼트가 화면에 보이지 않음");
      return false;
    }

    console.log("✅ DOM 준비 완료!", {
      element: mountRef.current,
      width: mountRef.current.clientWidth,
      height: mountRef.current.clientHeight,
      offsetWidth: mountRef.current.offsetWidth,
      offsetHeight: mountRef.current.offsetHeight,
    });

    return true;
  }, []);

  // DOM 준비를 기다리는 useEffect
  useEffect(() => {
    console.log("🚀 컴포넌트 마운트됨");
    setLoadingStep("DOM 준비 대기 중...");

    // 여러 방법으로 DOM이 준비될 때까지 기다리기
    const waitForDOM = () => {
      let attempts = 0;
      const maxAttempts = 50; // 5초 (100ms * 50)

      const checkInterval = setInterval(() => {
        attempts++;
        setLoadingStep(`DOM 확인 중... (${attempts}/${maxAttempts})`);

        if (checkDOMReady()) {
          clearInterval(checkInterval);
          setCanStart3D(true);
          setLoadingStep("DOM 준비 완료! 3D 초기화 시작...");
        } else if (attempts >= maxAttempts) {
          clearInterval(checkInterval);
          setError("DOM이 준비되지 않았습니다. 페이지를 새로고침해보세요.");
          setIsLoading(false);
        }
      }, 100);
    };

    // 약간의 지연 후 DOM 체크 시작
    const timer = setTimeout(waitForDOM, 200);

    return () => {
      clearTimeout(timer);
    };
  }, [checkDOMReady]);

  // 실제 3D 초기화 useEffect
  useEffect(() => {
    if (!canStart3D) return;

    let renderer, scene, camera, animationId;
    let cleanupDone = false;

    const init3D = async () => {
      try {
        console.log("🎮 3D 초기화 시작!");
        setLoadingStep("Three.js 씬 생성 중...");

        // 한번 더 확인
        if (!mountRef.current) {
          throw new Error("3D 시작 시점에 마운트 참조가 없습니다");
        }

        const container = mountRef.current;
        const containerWidth = container.clientWidth || 800;
        const containerHeight = container.clientHeight || 400;

        console.log("📐 컨테이너 정보:", {
          width: containerWidth,
          height: containerHeight,
          element: container,
        });

        // Scene
        setLoadingStep("씬 생성 중...");
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        console.log("✅ 씬 생성");

        // Camera
        setLoadingStep("카메라 생성 중...");
        camera = new THREE.PerspectiveCamera(
          75,
          containerWidth / containerHeight,
          1,
          1000
        );
        camera.position.set(200, 150, 200);
        camera.lookAt(0, 0, 0);
        console.log("✅ 카메라 생성");

        // Renderer
        setLoadingStep("렌더러 생성 중...");
        renderer = new THREE.WebGLRenderer({
          antialias: false,
          alpha: false,
          powerPreference: "high-performance",
        });
        renderer.setSize(containerWidth, containerHeight);
        renderer.shadowMap.enabled = false;
        console.log("✅ 렌더러 생성");

        // DOM에 추가 (가장 중요한 부분)
        setLoadingStep("렌더러를 DOM에 추가 중...");
        if (!mountRef.current) {
          throw new Error("DOM 추가 시점에 마운트 참조가 사라짐");
        }

        // 기존 canvas 제거
        const existingCanvas = mountRef.current.querySelector("canvas");
        if (existingCanvas) {
          mountRef.current.removeChild(existingCanvas);
        }

        mountRef.current.appendChild(renderer.domElement);
        console.log("✅ 렌더러 DOM 추가 완료");

        // 조명
        setLoadingStep("조명 추가 중...");
        const light = new THREE.AmbientLight(0xffffff, 1.0);
        scene.add(light);
        console.log("✅ 조명 추가");

        // 테스트 박스 (빨간색, 회전함)
        setLoadingStep("테스트 박스 생성 중...");
        const testGeometry = new THREE.BoxGeometry(50, 50, 50);
        const testMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const testBox = new THREE.Mesh(testGeometry, testMaterial);
        testBox.position.set(0, 25, 0);
        scene.add(testBox);
        console.log("✅ 테스트 박스 추가");

        // 바닥
        setLoadingStep("바닥 생성 중...");
        const floorGeometry = new THREE.PlaneGeometry(roomWidth, roomHeight);
        const floorMaterial = new THREE.MeshBasicMaterial({ color: 0xcccccc });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        scene.add(floor);
        console.log("✅ 바닥 추가");

        // 실제 가구들 추가
        if (placedFurniture.length > 0) {
          setLoadingStep(`가구 ${placedFurniture.length}개 생성 중...`);
          placedFurniture.forEach((furniture, index) => {
            console.log(`🪑 가구 ${index + 1}: ${furniture.name}`);

            const geometry = new THREE.BoxGeometry(
              furniture.width,
              30,
              furniture.height
            );
            const material = new THREE.MeshBasicMaterial({
              color: furniture.color,
            });
            const mesh = new THREE.Mesh(geometry, material);

            mesh.position.set(
              furniture.x - roomWidth / 2 + furniture.width / 2,
              15,
              furniture.y - roomHeight / 2 + furniture.height / 2
            );

            scene.add(mesh);
          });
          console.log("✅ 모든 가구 추가 완료");
        }

        // 첫 렌더링
        setLoadingStep("첫 렌더링 중...");
        renderer.render(scene, camera);
        console.log("✅ 첫 렌더링 성공");

        // 애니메이션 시작
        setLoadingStep("애니메이션 시작 중...");
        const animate = () => {
          if (cleanupDone) return; // 정리되었으면 중단

          if (renderer && scene && camera && testBox) {
            testBox.rotation.y += 0.01;
            renderer.render(scene, camera);
            animationId = requestAnimationFrame(animate);
          }
        };
        animate();

        console.log("🎉 3D 초기화 완전히 성공!");
        setIsLoading(false);
      } catch (err) {
        console.error("❌ 3D 초기화 실패:", err);
        setError(`3D 초기화 실패: ${err.message}`);
        setIsLoading(false);
      }
    };

    // 3D 초기화 시작
    init3D();

    // 정리 함수
    return () => {
      cleanupDone = true;
      console.log("🧹 3D 정리 시작");

      if (animationId) {
        cancelAnimationFrame(animationId);
      }

      if (renderer && mountRef.current && renderer.domElement) {
        try {
          if (mountRef.current.contains(renderer.domElement)) {
            mountRef.current.removeChild(renderer.domElement);
          }
          renderer.dispose();
        } catch (e) {
          console.warn("정리 중 에러:", e);
        }
      }

      console.log("🧹 3D 정리 완료");
    };
  }, [canStart3D, roomWidth, roomHeight, placedFurniture]);

  if (error) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-red-50 rounded-lg border border-red-200">
        <div className="text-center p-6 max-w-md">
          <div className="text-red-600 text-lg font-semibold mb-2">
            3D 로딩 실패 😞
          </div>
          <div className="text-red-700 text-sm mb-4 whitespace-pre-line">
            {error}
          </div>
          <div className="text-xs text-red-600 mb-4">
            가능한 해결 방법:
            <br />• 페이지 새로고침 (F5)
            <br />• 다른 브라우저 사용
            <br />• 브라우저 콘솔(F12) 확인
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg"
          >
            2D로 돌아가기
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
          <div className="text-gray-700 font-medium mb-2">
            3D 공간 생성 중...
          </div>
          <div className="text-sm text-blue-600 font-medium mb-3">
            {loadingStep}
          </div>
          <div className="text-xs text-gray-500">
            {canStart3D ? "3D 엔진 초기화 중..." : "DOM 준비 대기 중..."}
          </div>
          <div className="text-xs text-gray-400 mt-2">
            문제가 지속되면 F12 → Console에서 오류를 확인해주세요
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="p-4 bg-green-50 border-b flex justify-between items-center">
        <div>
          <div className="font-medium text-green-800 flex items-center gap-2">
            ✅ 3D 테스트 성공!
            <span className="text-xs bg-green-200 px-2 py-1 rounded">
              ULTRA SAFE
            </span>
          </div>
          <div className="text-sm text-green-600">
            가구 {placedFurniture.length}개 • {roomWidth}×{roomHeight}cm • 빨간
            박스가 회전하는지 확인
          </div>
        </div>
        <button
          onClick={onClose}
          className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
        >
          ← 2D로 돌아가기
        </button>
      </div>

      <div
        ref={mountRef}
        className="w-full bg-gradient-to-br from-gray-50 to-gray-100"
        style={{ height: "400px", minHeight: "400px" }}
      />

      <div className="p-4 bg-blue-50 text-sm">
        <div className="text-blue-800 font-medium mb-1">
          🧪 최강 안전 모드 테스트 버전
        </div>
        <div className="text-blue-700">
          • DOM 준비 상태를 50번까지 체크하는 안전 버전
          <br />
          • 빨간 박스가 회전하면 3D 엔진이 정상 작동 중<br />
          • 이것도 실패하면 브라우저/하드웨어 문제일 가능성 높음
          <br />• 성공하면 원본 버전의 복잡도가 문제
        </div>
      </div>
    </div>
  );
};

export default UltraSafe3DTest;
