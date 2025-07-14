import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

const Room3DViewer = ({ width, height, depth, className = "" }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const animationRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!width || !height || !depth) return;

    // Scene, Camera, Renderer 초기화
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      75,
      800 / 600,
      0.1,
      1000
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(800, 600);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    // 컨테이너에 추가
    if (mountRef.current) {
      mountRef.current.appendChild(renderer.domElement);
    }

    // 조명 설정
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    // 방 크기를 Three.js 단위로 변환 (cm -> 미터 단위)
    const roomWidth = width / 100;
    const roomHeight = height / 100;
    const roomDepth = depth / 100;

    // 바닥 생성
    const floorGeometry = new THREE.PlaneGeometry(roomWidth, roomDepth);
    const floorMaterial = new THREE.MeshLambertMaterial({ 
      color: 0xf4f4f4,
      side: THREE.DoubleSide 
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // 천장 생성
    const ceiling = new THREE.Mesh(floorGeometry, floorMaterial);
    ceiling.rotation.x = Math.PI / 2;
    ceiling.position.y = roomHeight;
    scene.add(ceiling);

    // 벽면 생성
    const wallMaterial = new THREE.MeshLambertMaterial({ color: 0xffffff });

    // 뒷벽
    const backWallGeometry = new THREE.PlaneGeometry(roomWidth, roomHeight);
    const backWall = new THREE.Mesh(backWallGeometry, wallMaterial);
    backWall.position.z = -roomDepth / 2;
    backWall.position.y = roomHeight / 2;
    scene.add(backWall);

    // 앞벽 (투명하게)
    const frontWallMaterial = new THREE.MeshLambertMaterial({ 
      color: 0xffffff, 
      transparent: true, 
      opacity: 0.1 
    });
    const frontWall = new THREE.Mesh(backWallGeometry, frontWallMaterial);
    frontWall.position.z = roomDepth / 2;
    frontWall.position.y = roomHeight / 2;
    frontWall.rotation.y = Math.PI;
    scene.add(frontWall);

    // 왼쪽 벽
    const sideWallGeometry = new THREE.PlaneGeometry(roomDepth, roomHeight);
    const leftWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
    leftWall.position.x = -roomWidth / 2;
    leftWall.position.y = roomHeight / 2;
    leftWall.rotation.y = Math.PI / 2;
    scene.add(leftWall);

    // 오른쪽 벽
    const rightWall = new THREE.Mesh(sideWallGeometry, wallMaterial);
    rightWall.position.x = roomWidth / 2;
    rightWall.position.y = roomHeight / 2;
    rightWall.rotation.y = -Math.PI / 2;
    scene.add(rightWall);

    // 샘플 가구 추가
    addFurniture(scene, roomWidth, roomHeight, roomDepth);

    // 카메라 위치 설정
    camera.position.set(roomWidth * 0.8, roomHeight * 0.8, roomDepth * 0.8);
    camera.lookAt(0, roomHeight / 2, 0);

    // 애니메이션 루프
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      
      // 자동 회전
      const time = Date.now() * 0.0005;
      camera.position.x = Math.cos(time) * (roomWidth * 1.2);
      camera.position.z = Math.sin(time) * (roomDepth * 1.2);
      camera.lookAt(0, roomHeight / 2, 0);
      
      renderer.render(scene, camera);
    };

    animate();
    setIsLoading(false);

    // Cleanup
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [width, height, depth]);

  // 가구 추가 함수
  const addFurniture = (scene, roomWidth, roomHeight, roomDepth) => {
    // 침대
    const bedGeometry = new THREE.BoxGeometry(2, 0.5, 1);
    const bedMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
    const bed = new THREE.Mesh(bedGeometry, bedMaterial);
    bed.position.set(-roomWidth/4, 0.25, -roomDepth/4);
    bed.castShadow = true;
    scene.add(bed);

    // 책상
    const deskGeometry = new THREE.BoxGeometry(1.2, 0.05, 0.6);
    const deskMaterial = new THREE.MeshLambertMaterial({ color: 0x654321 });
    const desk = new THREE.Mesh(deskGeometry, deskMaterial);
    desk.position.set(roomWidth/3, 0.8, roomDepth/3);
    desk.castShadow = true;
    scene.add(desk);

    // 책상 다리
    for (let i = 0; i < 4; i++) {
      const legGeometry = new THREE.BoxGeometry(0.05, 0.8, 0.05);
      const leg = new THREE.Mesh(legGeometry, deskMaterial);
      const x = roomWidth/3 + (i % 2 === 0 ? -0.55 : 0.55);
      const z = roomDepth/3 + (i < 2 ? -0.25 : 0.25);
      leg.position.set(x, 0.4, z);
      leg.castShadow = true;
      scene.add(leg);
    }

    // 의자
    const chairGeometry = new THREE.BoxGeometry(0.4, 0.05, 0.4);
    const chairMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 });
    const chair = new THREE.Mesh(chairGeometry, chairMaterial);
    chair.position.set(roomWidth/3, 0.45, roomDepth/2);
    chair.castShadow = true;
    scene.add(chair);
  };

  return (
    <div className={`relative ${className}`}>
      <div className="bg-gradient-to-br from-slate-100 to-slate-200 rounded-xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            🏠 3D 방 시각화
          </h3>
          <div className="text-sm text-gray-600">
            {width?.toFixed(0)} × {depth?.toFixed(0)} × {height?.toFixed(0)} cm
          </div>
        </div>
        
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 rounded-xl">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
              <p className="text-gray-600">3D 모델 생성 중...</p>
            </div>
          </div>
        )}
        
        <div 
          ref={mountRef} 
          className="w-full h-96 bg-gray-100 rounded-lg overflow-hidden shadow-inner"
        />
        
        <div className="mt-4 text-xs text-gray-500 text-center">
          💡 방이 자동으로 회전하며 3D로 표시됩니다
        </div>
      </div>
    </div>
  );
};

export default Room3DViewer;