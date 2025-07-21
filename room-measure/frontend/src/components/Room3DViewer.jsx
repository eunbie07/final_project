import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";

const Room3DViewer = ({
  roomWidth = 400,
  roomHeight = 300,
  placedFurniture = [],
  onClose,
}) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const animationIdRef = useRef(null);

  const [viewMode, setViewMode] = useState("person"); // 'person', 'overview', 'bird'
  const [personHeight, setPersonHeight] = useState(170); // cm
  const [isLoading, setIsLoading] = useState(true);

  // 유효성 검사
  const validRoomWidth = isNaN(roomWidth) || roomWidth <= 0 ? 400 : roomWidth;
  const validRoomHeight =
    isNaN(roomHeight) || roomHeight <= 0 ? 300 : roomHeight;

  useEffect(() => {
    if (!mountRef.current) return;

    try {
      initializeScene();
      setIsLoading(false);
    } catch (error) {
      console.error("3D 씬 초기화 실패:", error);
      setIsLoading(false);
    }

    return cleanup;
  }, []);

  useEffect(() => {
    if (cameraRef.current) {
      setCameraPosition();
    }
  }, [viewMode, personHeight]);

  const initializeScene = () => {
    // Scene 설정
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    // 카메라 설정
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      1,
      3000
    );
    cameraRef.current = camera;

    // 렌더러 설정
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(
      mountRef.current.clientWidth,
      mountRef.current.clientHeight
    );
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0xf0f0f0);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 조명 설정
    setupLighting(scene);

    // 방 구조 생성
    createRoom(scene);

    // 가구 배치
    createAllFurniture(scene);

    // 사람 추가
    createPerson(scene, personHeight);

    // 카메라 초기 위치
    setCameraPosition();

    // 마우스 컨트롤
    setupMouseControls();

    // 렌더링 시작
    startAnimation();
  };

  const setupLighting = (scene) => {
    // 전체 밝기 (환경광)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // 메인 조명 (창문에서 들어오는 빛)
    const mainLight = new THREE.DirectionalLight(0xffffff, 0.7);
    mainLight.position.set(validRoomWidth / 2, 250, validRoomHeight / 2);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    mainLight.shadow.camera.near = 50;
    mainLight.shadow.camera.far = 500;
    mainLight.shadow.camera.left = -validRoomWidth;
    mainLight.shadow.camera.right = validRoomWidth;
    mainLight.shadow.camera.top = validRoomHeight;
    mainLight.shadow.camera.bottom = -validRoomHeight;
    scene.add(mainLight);

    // 천장 조명
    const ceilLight = new THREE.PointLight(0xfffacd, 0.3, validRoomWidth * 2);
    ceilLight.position.set(0, 220, 0);
    scene.add(ceilLight);
  };

  const createRoom = (scene) => {
    const roomHeight = 230; // 2.3m

    // 바닥 생성 (타일 패턴)
    const floorGeometry = new THREE.PlaneGeometry(
      validRoomWidth,
      validRoomHeight
    );
    const floorMaterial = new THREE.MeshLambertMaterial({ color: 0xf8f8f8 });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // 벽 재질
    const wallMaterial = new THREE.MeshLambertMaterial({
      color: 0xf5f5f0,
      transparent: true,
      opacity: 0.9,
    });

    // 뒷벽
    const backWall = new THREE.Mesh(
      new THREE.PlaneGeometry(validRoomWidth, roomHeight),
      wallMaterial
    );
    backWall.position.set(0, roomHeight / 2, -validRoomHeight / 2);
    scene.add(backWall);

    // 왼쪽 벽
    const leftWall = new THREE.Mesh(
      new THREE.PlaneGeometry(validRoomHeight, roomHeight),
      wallMaterial
    );
    leftWall.position.set(-validRoomWidth / 2, roomHeight / 2, 0);
    leftWall.rotation.y = Math.PI / 2;
    scene.add(leftWall);

    // 오른쪽 벽
    const rightWall = new THREE.Mesh(
      new THREE.PlaneGeometry(validRoomHeight, roomHeight),
      wallMaterial
    );
    rightWall.position.set(validRoomWidth / 2, roomHeight / 2, 0);
    rightWall.rotation.y = -Math.PI / 2;
    scene.add(rightWall);

    // 앞벽 (입구 있는 벽 - 부분적으로)
    const doorWidth = 80; // 문 너비
    const wallSegmentWidth = (validRoomWidth - doorWidth) / 2;

    if (wallSegmentWidth > 0) {
      // 왼쪽 벽 부분
      const frontWallLeft = new THREE.Mesh(
        new THREE.PlaneGeometry(wallSegmentWidth, roomHeight),
        wallMaterial
      );
      frontWallLeft.position.set(
        -doorWidth / 2 - wallSegmentWidth / 2,
        roomHeight / 2,
        validRoomHeight / 2
      );
      frontWallLeft.rotation.y = Math.PI;
      scene.add(frontWallLeft);

      // 오른쪽 벽 부분
      const frontWallRight = new THREE.Mesh(
        new THREE.PlaneGeometry(wallSegmentWidth, roomHeight),
        wallMaterial
      );
      frontWallRight.position.set(
        doorWidth / 2 + wallSegmentWidth / 2,
        roomHeight / 2,
        validRoomHeight / 2
      );
      frontWallRight.rotation.y = Math.PI;
      scene.add(frontWallRight);
    }

    // 천장
    const ceiling = new THREE.Mesh(
      new THREE.PlaneGeometry(validRoomWidth, validRoomHeight),
      new THREE.MeshLambertMaterial({ color: 0xffffff })
    );
    ceiling.position.y = roomHeight;
    ceiling.rotation.x = Math.PI / 2;
    scene.add(ceiling);
  };

  const createAllFurniture = (scene) => {
    placedFurniture.forEach((furniture, index) => {
      try {
        createFurniture3D(scene, furniture, index);
      } catch (error) {
        console.warn(`가구 "${furniture.name}" 생성 실패:`, error);
      }
    });
  };

  const createFurniture3D = (scene, furniture, index) => {
    const { x, y, width, height, name, color, rotation = 0 } = furniture;

    // 가구별 높이 설정
    const furnitureHeights = {
      "싱글 베드": 60,
      "더블 베드": 60,
      "퀸 베드": 60,
      "킹 베드": 60,
      책상: 75,
      의자: 85,
      "2인 소파": 85,
      "3인 소파": 85,
      "커피 테이블": 45,
      "TV 스탠드": 50,
      옷장: 200,
      책장: 180,
      화장대: 75,
    };

    const furnitureHeight = furnitureHeights[name] || 75;

    // 회전 고려한 실제 크기
    const actualWidth = rotation % 180 === 0 ? width : height;
    const actualHeight = rotation % 180 === 0 ? height : width;

    // 가구 메쉬 생성
    const geometry = new THREE.BoxGeometry(
      actualWidth,
      furnitureHeight,
      actualHeight
    );
    const material = new THREE.MeshLambertMaterial({ color: color });
    const mesh = new THREE.Mesh(geometry, material);

    // 위치 설정 (2D 좌표를 3D 중심 좌표로 변환)
    mesh.position.set(
      x - validRoomWidth / 2 + actualWidth / 2,
      furnitureHeight / 2,
      y - validRoomHeight / 2 + actualHeight / 2
    );

    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);

    // 가구 라벨 추가
    createFurnitureLabel(scene, furniture, mesh.position, furnitureHeight);
  };

  const createFurnitureLabel = (scene, furniture, position, height) => {
    // 텍스트 텍스쳐 생성
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    canvas.width = 256;
    canvas.height = 64;

    // 배경
    context.fillStyle = "rgba(255,255,255,0.9)";
    context.fillRect(0, 0, 256, 64);

    // 테두리
    context.strokeStyle = "rgba(0,0,0,0.3)";
    context.lineWidth = 2;
    context.strokeRect(0, 0, 256, 64);

    // 텍스트
    context.fillStyle = "#333333";
    context.font = "bold 18px Arial";
    context.textAlign = "center";
    context.fillText(furniture.name, 128, 40);

    const texture = new THREE.CanvasTexture(canvas);
    const labelMaterial = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      alphaTest: 0.1,
    });
    const labelGeometry = new THREE.PlaneGeometry(40, 10);
    const label = new THREE.Mesh(labelGeometry, labelMaterial);

    label.position.copy(position);
    label.position.y = height + 15;

    // 라벨이 항상 카메라를 향하도록
    label.lookAt(0, label.position.y, 0);

    scene.add(label);
  };

  const createPerson = (scene, height) => {
    const personGroup = new THREE.Group();

    // 몸통 (캡슐 모양)
    const bodyGeometry = new THREE.CapsuleGeometry(8, height * 0.5, 4, 8);
    const personMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 });
    const body = new THREE.Mesh(bodyGeometry, personMaterial);
    body.position.y = height * 0.35;
    personGroup.add(body);

    // 머리
    const headGeometry = new THREE.SphereGeometry(12, 8, 6);
    const head = new THREE.Mesh(headGeometry, personMaterial);
    head.position.y = height * 0.85;
    personGroup.add(head);

    // 위치 설정 (방 중앙 약간 앞쪽)
    personGroup.position.set(0, 0, validRoomHeight * 0.1);
    scene.add(personGroup);

    // 키 표시 라벨
    createHeightLabel(scene, height, personGroup.position);
  };

  const createHeightLabel = (scene, height, position) => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    canvas.width = 128;
    canvas.height = 32;

    context.fillStyle = "rgba(255,255,255,0.9)";
    context.fillRect(0, 0, 128, 32);
    context.strokeStyle = "red";
    context.lineWidth = 1;
    context.strokeRect(0, 0, 128, 32);
    context.fillStyle = "red";
    context.font = "bold 14px Arial";
    context.textAlign = "center";
    context.fillText(`${height}cm`, 64, 20);

    const texture = new THREE.CanvasTexture(canvas);
    const label = new THREE.Mesh(
      new THREE.PlaneGeometry(25, 8),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true })
    );
    label.position.set(position.x, height + 25, position.z);
    scene.add(label);
  };

  const setCameraPosition = () => {
    if (!cameraRef.current) return;

    const camera = cameraRef.current;

    switch (viewMode) {
      case "person":
        // 사람 시점 (방 안에서 생활하는 느낌)
        const personRadius = Math.min(validRoomWidth, validRoomHeight) * 0.25;
        camera.position.set(-personRadius, personHeight, personRadius);
        camera.lookAt(personRadius, personHeight * 0.8, -personRadius);
        break;

      case "overview":
        // 전체 뷰 (방 전체를 보는 시점)
        camera.position.set(
          validRoomWidth * 0.7,
          Math.max(validRoomWidth, validRoomHeight) * 0.5,
          validRoomHeight * 0.7
        );
        camera.lookAt(0, 50, 0);
        break;

      case "bird":
        // 탑뷰 (위에서 내려다보기)
        camera.position.set(0, 400, 0);
        camera.lookAt(0, 0, 0);
        break;
    }
  };

  const setupMouseControls = () => {
    if (!rendererRef.current) return;

    let mouseDown = false;
    let mouseX = 0,
      mouseY = 0;

    const handleMouseDown = (event) => {
      mouseDown = true;
      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleMouseUp = () => {
      mouseDown = false;
    };

    const handleMouseMove = (event) => {
      if (!mouseDown || !cameraRef.current) return;

      const deltaX = event.clientX - mouseX;
      const deltaY = event.clientY - mouseY;

      const camera = cameraRef.current;

      if (viewMode === "person") {
        // 사람 시점에서는 좌우 회전과 약간의 상하 움직임
        const radius = Math.min(validRoomWidth, validRoomHeight) * 0.25;
        const currentAngle = Math.atan2(camera.position.z, camera.position.x);
        const newAngle = currentAngle + deltaX * 0.01;

        camera.position.x = Math.cos(newAngle) * radius;
        camera.position.z = Math.sin(newAngle) * radius;

        // 약간의 상하 움직임
        camera.position.y = Math.max(
          personHeight - 30,
          Math.min(personHeight + 30, camera.position.y - deltaY * 0.5)
        );

        camera.lookAt(0, personHeight * 0.8, 0);
      } else {
        // 다른 시점에서는 자유 회전
        const spherical = new THREE.Spherical();
        spherical.setFromVector3(camera.position);
        spherical.theta -= deltaX * 0.01;
        spherical.phi += deltaY * 0.01;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

        camera.position.setFromSpherical(spherical);
        camera.lookAt(0, 0, 0);
      }

      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleWheel = (event) => {
      if (!cameraRef.current) return;

      const camera = cameraRef.current;
      const distance = camera.position.length();
      const newDistance = distance + event.deltaY * 0.5;
      const minDistance = Math.max(validRoomWidth, validRoomHeight) * 0.1;
      const maxDistance = Math.max(validRoomWidth, validRoomHeight) * 2;

      camera.position
        .normalize()
        .multiplyScalar(
          Math.max(minDistance, Math.min(maxDistance, newDistance))
        );
    };

    const canvas = rendererRef.current.domElement;
    canvas.addEventListener("mousedown", handleMouseDown);
    canvas.addEventListener("mouseup", handleMouseUp);
    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("wheel", handleWheel);
  };

  const startAnimation = () => {
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();
  };

  const cleanup = () => {
    if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current);
    }

    if (rendererRef.current) {
      if (mountRef.current && rendererRef.current.domElement) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
      rendererRef.current.dispose();
    }

    if (sceneRef.current) {
      // 씬의 모든 객체 정리
      while (sceneRef.current.children.length > 0) {
        sceneRef.current.remove(sceneRef.current.children[0]);
      }
    }
  };

  const handleResize = () => {
    if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();
    rendererRef.current.setSize(width, height);
  };

  // 창 크기 변경 처리
  useEffect(() => {
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  if (isLoading) {
    return (
      <div className="w-full h-96 flex items-center justify-center bg-gray-100 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <div className="text-gray-600">3D 공간을 생성하고 있습니다...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg shadow-lg overflow-hidden">
      {/* 컨트롤 패널 */}
      <div className="p-4 bg-gray-50 border-b flex flex-wrap gap-3 items-center">
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode("person")}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === "person"
                ? "bg-blue-500 text-white shadow-md"
                : "bg-white text-gray-700 hover:bg-gray-100 border"
            }`}
          >
            <strong>사람 시점</strong>
          </button>
          <button
            onClick={() => setViewMode("overview")}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === "overview"
                ? "bg-blue-500 text-white shadow-md"
                : "bg-white text-gray-700 hover:bg-gray-100 border"
            }`}
          >
            <strong>전체 뷰</strong>
          </button>
          <button
            onClick={() => setViewMode("bird")}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              viewMode === "bird"
                ? "bg-blue-500 text-white shadow-md"
                : "bg-white text-gray-700 hover:bg-gray-100 border"
            }`}
          >
            <strong>탑뷰</strong>
          </button>
        </div>

        <div className="flex items-center gap-2 bg-white px-3 py-2 rounded-lg border">
          <label className="text-sm font-medium text-gray-700"><strong>키:</strong></label>
          <input
            type="range"
            min="150"
            max="190"
            value={personHeight}
            onChange={(e) => setPersonHeight(Number(e.target.value))}
            className="w-20"
          />
          <span className="text-sm font-medium w-12 text-center">
            {personHeight}cm
          </span>
        </div>

        <button
          onClick={onClose}
          className="ml-auto px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
        >
          <strong>← 2D로 돌아가기</strong>
        </button>
      </div>

      {/* 3D 뷰어 */}
      <div
        ref={mountRef}
        className="w-full cursor-move bg-gray-50"
        style={{ height: "500px" }}
      />

      {/* 정보 패널 */}
      <div className="p-4 bg-blue-50 border-t">
        <div className="flex items-start justify-between">
          <div>
            <div className="font-medium text-blue-800 mb-1">
              <strong>실제 공간감 체험</strong> - {validRoomWidth}×{validRoomHeight}cm (천장
              2.3m)
            </div>
            <div className="text-sm text-blue-700">
              • <strong>사람 시점:</strong> {personHeight}cm 키로 방 안에서
              생활하는 시점 체험
              <br />• <strong>마우스 조작:</strong> 드래그로 시점 변경, 휠로 줌
              조절
              <br />• 실제 가구 크기로 공간이 얼마나 좁은지/넓은지 확인해보세요!
            </div>
          </div>
          <div className="text-right text-sm text-blue-600">
            <div>
              <strong>배치된 가구:</strong> {placedFurniture.length}개
            </div>
            <div>
              <strong>현재 시점:</strong>{" "}
              {viewMode === "person"
                ? "사람"
                : viewMode === "overview"
                ? "전체"
                : "탑뷰"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Room3DViewer;
