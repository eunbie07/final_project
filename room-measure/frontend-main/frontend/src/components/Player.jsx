import React, { useRef, useEffect } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import { PointerLockControls } from "@react-three/drei";
import * as THREE from "three";

// Player component for walkthrough mode
const Player = ({ roomSize }) => {
  const { camera } = useThree();
  const velocity = useRef(new THREE.Vector3());
  const direction = useRef(new THREE.Vector3());
  const keys = useRef({
    KeyW: false,
    KeyA: false,
    KeyS: false,
    KeyD: false,
  });

  const moveForward = useRef(false);
  const moveBackward = useRef(false);
  const moveLeft = useRef(false);
  const moveRight = useRef(false);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.code in keys.current) {
        keys.current[event.code] = true;
      }
      switch (event.code) {
        case "KeyW":
          moveForward.current = true;
          break;
        case "KeyA":
          moveLeft.current = true;
          break;
        case "KeyS":
          moveBackward.current = true;
          break;
        case "KeyD":
          moveRight.current = true;
          break;
      }
    };

    const onKeyUp = (event) => {
      if (event.code in keys.current) {
        keys.current[event.code] = false;
      }
      switch (event.code) {
        case "KeyW":
          moveForward.current = false;
          break;
        case "KeyA":
          moveLeft.current = false;
          break;
        case "KeyS":
          moveBackward.current = false;
          break;
        case "KeyD":
          moveRight.current = false;
          break;
      }
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  useFrame((state, delta) => {
    // walkthrough 모드일 때만 카메라를 직접 조작
    if (
      !state.controls ||
      state.controls.constructor.name !== "OrbitControls"
    ) {
      velocity.current.x -= velocity.current.x * 10.0 * delta;
      velocity.current.z -= velocity.current.z * 10.0 * delta;

      direction.current.z =
        Number(moveForward.current) - Number(moveBackward.current);
      direction.current.x =
        Number(moveRight.current) - Number(moveLeft.current);
      direction.current.normalize();

      if (moveForward.current || moveBackward.current)
        velocity.current.z -= direction.current.z * 400.0 * delta;
      if (moveLeft.current || moveRight.current)
        velocity.current.x -= direction.current.x * 400.0 * delta;

      camera.translateX(velocity.current.x * delta);
      camera.translateZ(velocity.current.z * delta);

      // 방 경계 제한 (cm 단위)
      const [roomWidth, roomHeight, roomDepth] = roomSize;
      const halfRoomWidth = roomWidth / 2;
      const halfRoomDepth = roomDepth / 2;
      const playerHeight = 170; // 눈높이 (cm)

      camera.position.x = Math.max(
        -halfRoomWidth + 20,
        Math.min(halfRoomWidth - 20, camera.position.x)
      );
      camera.position.z = Math.max(
        -halfRoomDepth + 20,
        Math.min(halfRoomDepth - 20, camera.position.z)
      );
      camera.position.y = playerHeight;
    }
  });

  return <PointerLockControls />;
};

export default Player;