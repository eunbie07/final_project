// frontend/src/App.js

import React, { useState, useRef } from 'react';
import './App.css';
import RoomVisualizer from './RoomVisualizer';
import FloorPlanViewer from './FloorPlanViewer';

function App() {
  const [imageUrl, setImageUrl] = useState(null);
  const [selectedPoints, setSelectedPoints] = useState([]);
  const [measurements, setMeasurements] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setLoading(true);
      setError(null);
      setImageUrl(URL.createObjectURL(file));
      setSelectedPoints([]);
      setMeasurements(null);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('http://localhost:3000/initial-image-analysis', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || '초기 이미지 분석 실패');
        }

        const data = await response.json();
        console.log("Initial analysis response:", data);
        setLoading(false);

      } catch (err) {
        console.error("Error during initial image analysis:", err);
        setError(err.message || "이미지 분석 중 오류가 발생했습니다.");
        setLoading(false);
        setImageUrl(null);
      }
    }
  };

  const handlePointsSelected = (points) => {
    setSelectedPoints(points);
    console.log("Selected points:", points);
  };

  const handleMeasureRoom = async () => {
    if (selectedPoints.length < 4) { // 평면도 생성을 위해 최소 4개 점 요구
      setError("2D 평면도 생성을 위해 최소 4개 이상의 점(방 코너)을 선택해주세요.");
      return;
    }
    if (!fileInputRef.current || !fileInputRef.current.files[0]) {
      setError("측정할 이미지를 먼저 업로드해주세요.");
      return;
    }

    setLoading(true);
    setError(null);

    const file = fileInputRef.current.files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('points', JSON.stringify(selectedPoints));

    try {
      const response = await fetch('http://localhost:3000/measure-room-with-points', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '방 측정 실패');
      }

      const data = await response.json();
      console.log("Measurement response:", data);
      setMeasurements(data);
      setLoading(false);

    } catch (err) {
      console.error("Error during room measurement:", err);
      setError(err.message || "방 측정 중 오류가 발생했습니다.");
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImageUrl(null);
    setSelectedPoints([]);
    setMeasurements(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="App">
      <h1>방 사진 분석으로 2D 평면도 만들기</h1>

      <div className="input-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          ref={fileInputRef}
        />
        {imageUrl && <p>사진 위에서 점을 클릭하여 측정에 필요한 지점을 지정하세요 (예: 바닥 코너 4개)</p>}
        {imageUrl && <button onClick={handleReset}>모두 초기화</button>}
      </div>

      {loading && <p>로딩 중...</p>}
      {error && <p style={{ color: 'red' }}>오류: {error}</p>}

      {imageUrl && (
        <RoomVisualizer imageUrl={imageUrl} onPointsSelected={handlePointsSelected} />
      )}

      {imageUrl && <p>선택된 점 개수: {selectedPoints.length}</p>}

      {imageUrl && selectedPoints.length >= 4 && (
        <button onClick={handleMeasureRoom} disabled={loading}>
          방 측정 및 평면도 생성
        </button>
      )}

      {measurements && (
        <div className="ai-result">
          <h2>AI 분석 결과</h2>
          <p>Image and points processed successfully for measurement.</p>
          <p>이미지 너비 (px): {measurements.image_width_px}</p>
          <p>이미지 높이 (px): {measurements.image_height_px}</p>
          <p>추정된 가로 길이 (m): {measurements.estimated_width_m}</p>
          <p>추정된 세로 길이 (m): {measurements.estimated_height_m}</p>
          <p>Notes: {measurements.notes}</p>

          {measurements.floor_plan_vertices && measurements.floor_plan_vertices.length > 0 &&
           measurements.estimated_width_m !== "N/A" && measurements.estimated_height_m !== "N/A" && (
            <FloorPlanViewer
              vertices={measurements.floor_plan_vertices}
              width={measurements.estimated_width_m}
              height={measurements.estimated_height_m}
            />
          )}
        </div>
      )}

    </div>
  );
}

export default App;