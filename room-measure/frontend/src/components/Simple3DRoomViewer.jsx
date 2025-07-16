// src/components/Simple3DRoomViewer.jsx
import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";

const Simple3DRoomViewer = ({ roomWidth = 400, roomHeight = 300, onClose }) => {
  const mountRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    console.log("🚀 3D 뷰어 시작!");

    const initViewer = () => {
      try {
        console.log("🔍 컨테이너 찾기 시작...");
        console.log("mountRef.current:", mountRef.current);

        const container = mountRef.current;
        if (!container) {
          console.log("❌ 컨테이너 없음");
          // DOM 상태 디버깅
          console.log("document.body:", document.body ? "존재" : "없음");
          console.log("mountRef:", mountRef);
          return false; // 실패 표시
        }

        console.log("✅ 컨테이너 찾음:", container);
        console.log(
          "컨테이너 크기:",
          container.clientWidth,
          "x",
          container.clientHeight
        );
        console.log(
          "컨테이너 스타일:",
          window.getComputedStyle(container).display
        );

        // 크기 설정 (더 안전하게)
        const rect = container.getBoundingClientRect();
        const width = Math.max(rect.width || container.clientWidth || 600, 400);
        const height = Math.max(
          rect.height || container.clientHeight || 400,
          300
        );

        console.log("📏 사용할 크기:", width, "x", height);

        // Scene 생성
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xe8f4fd);
        console.log("✅ Scene 생성됨");

        // Camera 생성
        const camera = new THREE.PerspectiveCamera(75, width / height, 1, 1000);
        camera.position.set(300, 200, 300);
        camera.lookAt(0, 0, 0);
        console.log("✅ Camera 생성됨");

        // Renderer 생성
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        renderer.setClearColor(0xe8f4fd);
        console.log("✅ Renderer 생성됨");

        // 간단한 조명
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
        directionalLight.position.set(100, 100, 50);
        scene.add(directionalLight);
        console.log("✅ 조명 추가됨");

        // 성공을 알리는 회전하는 박스
        const geometry = new THREE.BoxGeometry(50, 50, 50);
        const material = new THREE.MeshLambertMaterial({ color: 0x00ff00 });
        const cube = new THREE.Mesh(geometry, material);
        cube.position.set(0, 25, 0);
        scene.add(cube);
        console.log("✅ 테스트 박스 추가됨");

        // 바닥 (방 크기로)
        const floorGeometry = new THREE.PlaneGeometry(roomWidth, roomHeight);
        const floorMaterial = new THREE.MeshLambertMaterial({
          color: 0xcccccc,
          transparent: true,
          opacity: 0.8,
        });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        scene.add(floor);
        console.log("✅ 바닥 추가됨");

        // 방 테두리
        const edges = new THREE.EdgesGeometry(floorGeometry);
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0x000000 });
        const wireframe = new THREE.LineSegments(edges, lineMaterial);
        wireframe.rotation.x = -Math.PI / 2;
        scene.add(wireframe);
        console.log("✅ 방 테두리 추가됨");

        // DOM에 추가
        console.log("🎨 Canvas를 DOM에 추가 중...");
        container.appendChild(renderer.domElement);
        console.log("✅ Canvas DOM에 추가됨");

        // 애니메이션
        let animationId;
        const animate = () => {
          cube.rotation.x += 0.01;
          cube.rotation.y += 0.01;

          renderer.render(scene, camera);
          animationId = requestAnimationFrame(animate);
        };
        animate();
        console.log("✅ 애니메이션 시작됨");

        console.log("🎉 3D 뷰어 초기화 완료!");
        setIsLoading(false);

        // 정리 함수
        return () => {
          if (animationId) {
            cancelAnimationFrame(animationId);
          }
          if (
            renderer &&
            container &&
            container.contains(renderer.domElement)
          ) {
            container.removeChild(renderer.domElement);
          }
          if (renderer) {
            renderer.dispose();
          }
          console.log("🧹 3D 뷰어 정리됨");
        };
      } catch (err) {
        console.error("💥 3D 뷰어 에러:", err);
        setError(`3D 초기화 실패: ${err.message}`);
        setIsLoading(false);
        return false;
      }
    };

    // 여러 번 시도하는 방식
    let retryCount = 0;
    const maxRetries = 10;

    const tryInit = () => {
      console.log(`🔄 시도 ${retryCount + 1}/${maxRetries}`);

      const result = initViewer();

      if (result !== false) {
        // 성공
        return result;
      } else {
        // 실패 - 재시도
        retryCount++;
        if (retryCount < maxRetries) {
          console.log(`⏰ ${retryCount * 200}ms 후 재시도...`);
          setTimeout(tryInit, retryCount * 200);
        } else {
          console.log("💀 최대 재시도 횟수 초과");
          setError(
            "DOM 엘리먼트를 찾을 수 없습니다. 컴포넌트 렌더링 문제일 수 있습니다."
          );
          setIsLoading(false);
        }
      }
    };

    // 첫 시도
    const timer = setTimeout(tryInit, 100);

    return () => {
      clearTimeout(timer);
    };
  }, [roomWidth, roomHeight]);

  if (error) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-red-50 rounded-lg border border-red-200">
        <div className="text-center p-6">
          <div className="text-red-600 text-lg font-semibold mb-2">
            3D 로딩 실패 😞
          </div>
          <div className="text-red-700 text-sm mb-4">{error}</div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg"
          >
            ← 2D로 돌아가기
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">⚙️</div>
          <div className="text-gray-700 font-bold">3D 룸 뷰어 로딩 중...</div>
          <div className="text-sm text-blue-600 mt-2">
            개발자 도구(F12) Console에서 로그를 확인하세요
          </div>
          <button
            onClick={onClose}
            className="mt-4 px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg"
          >
            취소
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="p-4 bg-green-50 border-b flex justify-between items-center">
        <div>
          <div className="font-bold text-green-800 flex items-center gap-2">
            🎉 3D 룸 뷰어 성공!
            <span className="text-xs bg-green-200 px-2 py-1 rounded">
              WORKING
            </span>
          </div>
          <div className="text-sm text-green-600">
            {roomWidth}×{roomHeight}cm • 회전하는 초록 박스가 보이나요?
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
        className="w-full bg-gradient-to-br from-gray-50 to-blue-50"
        style={{
          height: "500px",
          width: "100%",
        }}
      />

      <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 text-sm border-t">
        <div className="text-green-800 font-bold mb-2">✅ 테스트 성공!</div>
        <div className="text-gray-700">
          • 회전하는 초록 박스가 보이면 Three.js가 정상 작동 중입니다
          <br />
          • 방 크기에 맞는 바닥과 테두리가 표시됩니다
          <br />• Console(F12)에서 초기화 로그를 확인할 수 있습니다
        </div>
      </div>
    </div>
  );
};

export default Simple3DRoomViewer;
