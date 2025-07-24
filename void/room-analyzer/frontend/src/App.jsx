import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import RoomVisualizer from "./components/RoomVisualizer";

const App = () => {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const parseResult = (data) => {
    try {
      const width = Number(data.width_cm ?? 0);
      const height = Number(data.height_cm ?? 0);
      const layout = data.layout ?? null;

      if (isNaN(width) || isNaN(height)) {
        setError("방의 크기를 계산할 수 없습니다.");
        setResult(null);
        return;
      }

      setResult({
        layout,
        roomWidth: width,
        roomHeight: height,
      });
      setError(null);
    } catch (err) {
      console.error("결과 처리 중 오류:", err);
      setError("결과를 처리하는 도중 오류가 발생했습니다.");
      setResult(null);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1>Room Analyzer</h1>
      <UploadForm onResult={parseResult} />

      {error && (
        <div style={{ color: "red", marginTop: "10px" }}>
          <strong>오류:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: "20px" }}>
          <p>방 너비: {result.roomWidth} cm</p>
          <p>방 높이: {result.roomHeight} cm</p>

          {result.layout && result.layout.length > 0 ? (
            <RoomVisualizer {...result} />
          ) : (
            <p>감지된 가구 정보가 없습니다.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default App;
