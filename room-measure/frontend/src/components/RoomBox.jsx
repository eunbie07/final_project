import React from "react";
import { Canvas, useLoader } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import * as THREE from "three";

function HumanSilhouette({ height = 170 }) {
  const texture = useLoader(THREE.TextureLoader, "/human.png");
  const scale = height / 100;
  return (
    <mesh position={[1.2, scale / 2, 0.8]}>
      <planeGeometry args={[0.6, scale]} />
      <meshBasicMaterial map={texture} transparent />
    </mesh>
  );
}

function FurnitureBox({
  position = [1, 0.25, 2.5],
  size = [1.6, 0.5, 0.9],
  color = "#8b5e3c",
}) {
  return (
    <mesh position={[...position]}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

function Desk({ position = [-1.2, 0.375, -1.2] }) {
  return (
    <>
      <mesh position={[...position]}>
        {" "}
        {/* 책상 */}
        <boxGeometry args={[0.8, 0.75, 0.6]} />
        <meshStandardMaterial color="#c4a484" />
      </mesh>
      <mesh position={[position[0], 0.5, position[2] - 0.4]}>
        {" "}
        {/* 의자 */}
        <boxGeometry args={[0.4, 1, 0.4]} />
        <meshStandardMaterial color="#333" />
      </mesh>
    </>
  );
}

function DimensionLabel({ position, text, rotation = [0, 0, 0] }) {
  return (
    <Text
      position={position}
      fontSize={0.1}
      color="black"
      rotation={rotation}
      anchorX="center"
      anchorY="middle"
    >
      {text}
    </Text>
  );
}

function DimensionArrow({ start, end }) {
  return (
    <Line points={[start, end]} color="black" lineWidth={1} dashed={false} />
  );
}

export default function RoomBox({ width = 400, height = 230, depth = 400 }) {
  const scale = 0.01;

  const w = width * scale;
  const h = height * scale;
  const d = depth * scale;

  return (
    <div className="w-full h-[500px] bg-white rounded-xl overflow-hidden shadow-lg">
      <Canvas camera={{ position: [2.5, 2.5, 2.5], fov: 60 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 10, 5]} intensity={0.5} />

        {/* 바닥 */}
        <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[w, d, 4, 4]} />
          <meshStandardMaterial color="#e5e5e5" wireframe={false} />
        </mesh>

        {/* 벽 */}
        <mesh position={[0, h / 2, -d / 2]}>
          <planeGeometry args={[w, h]} />
          <meshStandardMaterial color="#f5f0e6" side={THREE.DoubleSide} />
        </mesh>
        <mesh position={[-w / 2, h / 2, 0]} rotation={[0, Math.PI / 2, 0]}>
          <planeGeometry args={[d, h]} />
          <meshStandardMaterial color="#d9c9b6" side={THREE.DoubleSide} />
        </mesh>

        {/* 침대 */}
        <FurnitureBox
          position={[1.5, 0.25, 1.5]}
          size={[1.6, 0.5, 0.9]}
          color="#a97b50"
        />

        {/* 책상 + 의자 */}
        <Desk position={[-1.2, 0.375, -1.2]} />

        {/* 사람 */}
        <HumanSilhouette height={170} />

        {/* 거리 치수선 + 라벨 */}
        <DimensionArrow
          start={[-w / 2, 0.01, d / 2 + 0.05]}
          end={[w / 2, 0.01, d / 2 + 0.05]}
        />
        <DimensionLabel
          position={[0, 0.01, d / 2 + 0.15]}
          text={`${width / 100}m`}
          rotation={[-Math.PI / 2, 0, 0]}
        />

        <DimensionArrow
          start={[w / 2 + 0.05, 0.01, d / 2]}
          end={[w / 2 + 0.05, 0.01, -d / 2]}
        />
        <DimensionLabel
          position={[w / 2 + 0.15, 0.01, 0]}
          text={`${depth / 100}m`}
          rotation={[-Math.PI / 2, 0, Math.PI / 2]}
        />

        {/* 높이 치수 (오른쪽 벽 바로 옆 기준) */}
        <DimensionArrow start={[w / 2, 0, 0]} end={[w / 2, h, 0]} />
        <DimensionLabel
          position={[w / 2, h / 2, 0.1]}
          text={`${height / 100}m`}
          rotation={[0, 0, 0]}
        />

        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
      </Canvas>
    </div>
  );
}
