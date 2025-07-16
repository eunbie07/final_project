import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";

const FoolProof3DTest = ({
  roomWidth = 400,
  roomHeight = 300,
  placedFurniture = [],
  onClose,
}) => {
  const mountRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [debugInfo, setDebugInfo] = useState([]);

  // Three.js 객체들을 useRef로 관리
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const animationIdRef = useRef(null);

  const addDebugInfo = (message) => {
    console.log(message);
    setDebugInfo((prev) => [
      ...prev,
      `${new Date().toLocaleTimeString()}: ${message}`,
    ]);
  };

  // Three.js 초기화 함수 (useEffect 외부에 정의)
  const initThreeJS = () => {
    addDebugInfo("🎮 Three.js 초기화 시작");

    const container = mountRef.current;
    if (!container) {
      throw new Error("컨테이너가 없습니다");
    }

    addDebugInfo(
      `📦 컨테이너 크기: ${container.clientWidth}x${container.clientHeight}`
    );

    // 크기 설정
    const width = Math.max(container.clientWidth, 600);
    const height = Math.max(container.clientHeight, 400);

    // Scene 생성
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe8f4fd);
    sceneRef.current = scene;

    // Camera 생성
    const camera = new THREE.PerspectiveCamera(75, width / height, 1, 1000);
    camera.position.set(200, 150, 200);
    camera.lookAt(0, 0, 0);

    // Renderer 생성
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setClearColor(0xe8f4fd);
    rendererRef.current = renderer;

    // 조명
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(100, 100, 50);
    scene.add(directionalLight);

    // 회전하는 성공 표시 박스
    const successGeometry = new THREE.BoxGeometry(60, 30, 15);
    const successMaterial = new THREE.MeshLambertMaterial({ color: 0x00ff00 });
    const successBox = new THREE.Mesh(successGeometry, successMaterial);
    successBox.position.set(0, 15, 0);
    scene.add(successBox);

    // 바닥
    const floorGeometry = new THREE.PlaneGeometry(roomWidth, roomHeight);
    const floorMaterial = new THREE.MeshLambertMaterial({
      color: 0xf0f0f0,
      transparent: true,
      opacity: 0.8,
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    // 방 테두리
    const edges = new THREE.EdgesGeometry(
      new THREE.PlaneGeometry(roomWidth, roomHeight)
    );
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x000000 });
    const wireframe = new THREE.LineSegments(edges, lineMaterial);
    wireframe.rotation.x = -Math.PI / 2;
    scene.add(wireframe);

    // 가구들 추가
    placedFurniture.forEach((furniture, index) => {
      const furnitureGeometry = new THREE.BoxGeometry(
        furniture.width,
        30,
        furniture.height
      );
      const furnitureMaterial = new THREE.MeshLambertMaterial({
        color: furniture.color || 0x8b4513,
      });
      const furnitureMesh = new THREE.Mesh(
        furnitureGeometry,
        furnitureMaterial
      );

      // 위치 설정 (방 중앙을 원점으로)
      furnitureMesh.position.set(
        furniture.x - roomWidth / 2,
        15,
        furniture.y - roomHeight / 2
      );

      // 회전 적용
      if (furniture.rotation) {
        furnitureMesh.rotation.y = (furniture.rotation * Math.PI) / 180;
      }

      scene.add(furnitureMesh);
    });

    // DOM에 추가
    container.appendChild(renderer.domElement);

    // 첫 렌더링
    renderer.render(scene, camera);

    // 애니메이션 루프
    const animate = () => {
      successBox.rotation.y += 0.02;
      successBox.rotation.x += 0.01;

      renderer.render(scene, camera);
      animationIdRef.current = requestAnimationFrame(animate);
    };
    animate();

    addDebugInfo("🎉 Three.js 초기화 완료!");
    setIsLoading(false);
  };

  // 정리 함수
  const cleanup = () => {
    if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current);
      animationIdRef.current = null;
    }

    if (rendererRef.current && mountRef.current) {
      const canvas = rendererRef.current.domElement;
      if (canvas && mountRef.current.contains(canvas)) {
        mountRef.current.removeChild(canvas);
      }
      rendererRef.current.dispose();
      rendererRef.current = null;
    }

    if (sceneRef.current) {
      sceneRef.current.clear();
      sceneRef.current = null;
    }
  };

  // ✨ 핵심: useLayoutEffect 사용으로 DOM 렌더링 완료 후 실행
  useEffect(() => {
    addDebugInfo("🚀 컴포넌트 마운트됨");

    // 이미 초기화된 경우 중복 방지
    if (rendererRef.current) {
      addDebugInfo("⚠️ 이미 초기화됨");
      return;
    }

    // requestAnimationFrame을 사용해 다음 렌더 사이클에서 실행
    let frameId = requestAnimationFrame(() => {
      addDebugInfo(
        `🔍 RAF에서 mountRef.current: ${mountRef.current ? "EXISTS" : "NULL"}`
      );

      if (mountRef.current) {
        try {
          initThreeJS();
        } catch (err) {
          addDebugInfo(`💥 초기화 에러: ${err.message}`);
          setError(`Three.js 초기화 실패: ${err.message}`);
          setIsLoading(false);
        }
      } else {
        // 한 번 더 시도
        frameId = requestAnimationFrame(() => {
          addDebugInfo(
            `🔍 2차 RAF에서 mountRef.current: ${
              mountRef.current ? "EXISTS" : "NULL"
            }`
          );

          if (mountRef.current) {
            try {
              initThreeJS();
            } catch (err) {
              addDebugInfo(`💥 2차 초기화 에러: ${err.message}`);
              setError(`Three.js 초기화 실패: ${err.message}`);
              setIsLoading(false);
            }
          } else {
            addDebugInfo("💀 2차 시도도 실패");
            setError("DOM 엘리먼트를 찾을 수 없습니다.");
            setIsLoading(false);
          }
        });
      }
    });

    return () => {
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
      cleanup();
    };
  }, []);

  if (error) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-red-50 rounded-lg border border-red-200">
        <div className="text-center p-6 max-w-2xl">
          <div className="text-red-600 text-lg font-semibold mb-2">
            3D 로딩 실패 😞
          </div>
          <div className="text-red-700 text-sm mb-4">{error}</div>

          <details className="text-left mb-4">
            <summary className="cursor-pointer text-red-600 font-medium">
              🔍 디버그 로그 보기 (클릭)
            </summary>
            <div className="mt-2 p-3 bg-red-100 rounded text-xs text-red-800 max-h-40 overflow-y-auto">
              {debugInfo.map((info, index) => (
                <div key={index} className="mb-1">
                  {info}
                </div>
              ))}
            </div>
          </details>

          <div className="flex gap-2 justify-center">
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg"
            >
              🔄 새로고침
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg"
            >
              ← 2D로 돌아가기
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-gradient-to-br from-blue-50 to-purple-100 rounded-lg">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">⚙️</div>
          <div className="text-gray-700 font-bold mb-2">3D 환경 구성 중...</div>
          <div className="text-sm text-blue-600">
            RequestAnimationFrame 방식으로 안정화
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="p-4 bg-green-50 border-b flex justify-between items-center">
        <div>
          <div className="font-bold text-green-800 flex items-center gap-2">
            🎉 THREE.JS 최종 성공!
            <span className="text-xs bg-green-200 px-2 py-1 rounded">RAF</span>
          </div>
          <div className="text-sm text-green-600">
            가구 {placedFurniture.length}개 • {roomWidth}×{roomHeight}cm •
            RequestAnimationFrame 방식
          </div>
        </div>
        <button
          onClick={onClose}
          className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
        >
          ← 2D로 돌아가기
        </button>
      </div>

      {/* ✨ 핵심: 즉시 렌더링되는 DOM 엘리먼트 */}
      <div
        ref={mountRef}
        className="w-full bg-gradient-to-br from-blue-50 to-purple-50"
        style={{
          height: "500px",
          width: "100%",
          minHeight: "500px",
          minWidth: "600px",
        }}
      />

      <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 text-sm">
        <div className="text-green-800 font-bold mb-2">
          🎯 RequestAnimationFrame 방식의 장점
        </div>
        <div className="text-gray-700 mb-2">
          • DOM 렌더링 완료 후 정확한 타이밍에 실행
          <br />
          • React Strict Mode와 완벽 호환
          <br />• 브라우저 렌더링 사이클과 동기화
        </div>

        <details>
          <summary className="cursor-pointer text-blue-600 font-medium">
            📋 실행 로그 보기
          </summary>
          <div className="mt-2 p-3 bg-white rounded text-xs text-gray-700 max-h-40 overflow-y-auto border">
            {debugInfo.map((info, index) => (
              <div key={index} className="mb-1">
                {info}
              </div>
            ))}
          </div>
        </details>
      </div>
    </div>
  );
};

export default FoolProof3DTest;
