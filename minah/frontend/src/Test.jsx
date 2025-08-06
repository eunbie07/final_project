import React, { useState } from 'react';
import './style.css'; // <-- 파일 경로 이미지에 따라 'styles.css'에서 'style.css'로 변경되었습니다!

// Test.jsx 파일 내부에서 App 컴포넌트를 정의합니다.
// main.jsx에서 이 App 컴포넌트를 임포트하여 사용해야 합니다.
function App() { 
  const [roomWidth, setRoomWidth] = useState('');
  const [roomHeight, setRoomHeight] = useState('');
  const [furniture, setFurniture] = useState([]);
  const [roomPlanImage, setRoomPlanImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- FastAPI 서버 주소 설정 ---
  // ** 중요: 이 주소는 당신의 FastAPI 서버가 실행되는 실제 IP 주소와 포트여야 합니다! **
  const API_BASE_URL = 'http://localhost:3002'; 

  // --- 가구 추가 핸들러: 숫자 필드에 유효한 기본값 설정 ---
  // 사용자가 가구를 추가할 때부터 유효한 숫자로 시작하도록 합니다.
  const handleAddRectangleFurniture = () => {
    setFurniture([
      ...furniture,
      { 
        type: 'rectangle', 
        x1: 0, y1: 0, x2: 1000, y2: 1000, // 초기값을 유효한 숫자로 설정 (예: 0,0에서 1000,1000 크기의 사각형)
        color: '#4287f5', // 파란색 계열 기본 색상
        label: `직사각형 가구 ${furniture.length + 1}` 
      }
    ]);
  };

  const handleAddCircleFurniture = () => {
    setFurniture([
      ...furniture,
      { 
        type: 'circle', 
        center_x: 500, center_y: 500, radius: 250, // 초기값을 유효한 숫자로 설정 (예: 중심 500,500에 반지름 250)
        color: '#f54242', // 빨간색 계열 기본 색상
        label: `원형 가구 ${furniture.length + 1}` 
      }
    ]);
  };

  const handleFurnitureChange = (index, field, value) => {
    const updatedFurniture = [...furniture];
    // 숫자 필드는 parseFloat으로 변환하고, 아니면 문자열 그대로 저장합니다.
    updatedFurniture[index] = { 
      ...updatedFurniture[index], 
      [field]: ['x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'radius'].includes(field) ? parseFloat(value) : value 
    };
    setFurniture(updatedFurniture);
  };

  const handleRemoveFurniture = (index) => {
    const updatedFurniture = furniture.filter((_, i) => i !== index);
    setFurniture(updatedFurniture);
  };

  const handleSubmit = async (e) => {
    e.preventDefault(); // 폼 기본 제출 동작 방지
    setLoading(true);
    setError(null); // 이전 오류 메시지 초기화
    setRoomPlanImage(null); // 이전 이미지 초기화

    // --- 가구 데이터 정제 (FastAPI로 보내기 전 최종 유효성 검사 및 기본값 처리) ---
    const processedFurniture = furniture.map(item => {
      if (item.type === 'rectangle') {
        return {
          type: item.type,
          // isNaN이면 0으로, 아니면 원래 값 사용
          x1: isNaN(item.x1) ? 0 : item.x1,
          y1: isNaN(item.y1) ? 0 : item.y1,
          x2: isNaN(item.x2) ? 0 : item.x2,
          y2: isNaN(item.y2) ? 0 : item.y2,
          color: item.color || '#4287f5', // 색상이 비어있으면 기본값 사용
          label: item.label || '직사각형 가구' // 라벨이 비어있으면 기본값 사용
        };
      } else { // circle (원형 가구)
        return {
          type: item.type,
          center_x: isNaN(item.center_x) ? 0 : item.center_x,
          center_y: isNaN(item.center_y) ? 0 : item.center_y,
          // 반지름은 0보다 커야 하므로, isNaN이거나 0 이하면 1로 설정합니다.
          radius: (isNaN(item.radius) || item.radius <= 0) ? 1 : item.radius, 
          color: item.color || '#f54242',
          label: item.label || '원형 가구'
        };
      }
    });

    // 디버깅을 위해 전송될 데이터와 JSON 문자열을 콘솔에 출력
    console.log(">>>>> FastAPI로 전송될 요청 본문:", { furniture: processedFurniture }); 
    console.log(">>>>> JSON 문자열:", JSON.stringify({ furniture: processedFurniture })); 

    try {
      // FastAPI 백엔드로 요청 보내기
      const response = await fetch(
        // 방의 가로, 세로 길이는 URL 쿼리 파라미터로 전송
        `${API_BASE_URL}/room-plan/?width=${parseFloat(roomWidth)}&height=${parseFloat(roomHeight)}`,
        {
          method: 'POST', // FastAPI 엔드포인트는 POST 요청을 받음
          headers: {
            'Content-Type': 'application/json', // 요청 본문이 JSON임을 명시
          },
          body: JSON.stringify({ furniture: processedFurniture }), // 가구 데이터를 JSON 문자열로 변환하여 본문에 담기
        }
      );

      if (!response.ok) { // HTTP 응답 상태가 200번대가 아니면 오류 처리
        const errorData = await response.json(); // FastAPI의 상세 에러 메시지를 JSON 형태로 받습니다.
        console.error("FastAPI 상세 에러 응답:", errorData); // 개발자 도구 콘솔에 상세 에러 로깅

        // 422 (Unprocessable Entity) 오류이고 'detail' 필드가 있다면
        if (response.status === 422 && errorData.detail) {
          // 'detail' 배열의 각 오류 메시지를 조합하여 사용자에게 보여줄 메시지를 만듭니다.
          const validationErrors = errorData.detail.map(err => {
            // 오류 위치(loc)를 'body.furniture.0.x1' 같은 문자열로 만듭니다.
            const field = err.loc.length > 1 ? err.loc.slice(1).join('.') : err.loc.join('.');
            return `${field}: ${err.msg}`;
          }).join('\n'); // 각 오류를 새 줄로 구분
          throw new Error(`입력 값 유효성 검사 오류:\n${validationErrors}`);
        } else {
          // 422가 아닌 다른 HTTP 오류이거나 'detail' 필드가 없는 경우
          throw new Error(errorData.detail || `평면도 가져오기 실패: ${response.status} ${response.statusText}`);
        }
      }

      const imageBlob = await response.blob(); // 이미지 데이터를 Blob 형태로 받기
      const imageUrl = URL.createObjectURL(imageBlob); // Blob을 브라우저에서 표시할 수 있는 URL로 변환
      setRoomPlanImage(imageUrl); // 이미지 URL 상태 업데이트
    } catch (err) {
      console.error('평면도 가져오는 중 오류 발생:', err);
      setError(err.message || '예상치 못한 오류가 발생했습니다.'); // 사용자에게 오류 메시지 표시
    } finally {
      setLoading(false); // 로딩 상태 해제
    }
  };

  return (
    <div className="container">
      <h1>🏡 방 평면도 생성기</h1>
      <form onSubmit={handleSubmit} className="form-section">
        {/* 방 크기 입력 부분 */}
        <div className="input-group">
          <label htmlFor="width">방 가로 (mm):</label>
          <input
            id="width"
            type="number"
            value={roomWidth}
            onChange={(e) => setRoomWidth(e.target.value)}
            required // 필수 입력 필드
            min="1" // 최소값 1
            placeholder="예: 5000"
          />
        </div>
        <div className="input-group">
          <label htmlFor="height">방 세로 (mm):</label>
          <input
            id="height"
            type="number"
            value={roomHeight}
            onChange={(e) => setRoomHeight(e.target.value)}
            required
            min="1"
            placeholder="예: 4000"
          />
        </div>

        <h2>가구 추가 🛋️</h2>
        <div className="furniture-controls">
          <button type="button" onClick={handleAddRectangleFurniture}>
            ➕ 직사각형 가구
          </button>
          <button type="button" onClick={handleAddCircleFurniture}>
            ➕ 원형 가구
          </button>
        </div>

        {/* 추가된 가구 목록 렌더링 */}
        {furniture.map((item, index) => (
          <div key={index} className="furniture-item">
            <h3>{item.type === 'rectangle' ? '직사각형 가구' : '원형 가구'} {index + 1}</h3>
            <div className="input-group">
              <label>라벨:</label>
              <input
                type="text"
                value={item.label}
                onChange={(e) => handleFurnitureChange(index, 'label', e.target.value)}
                placeholder="예: 침대, 소파"
              />
            </div>
            {item.type === 'rectangle' ? (
              <>
                <div className="input-group">
                  <label>X1 (mm):</label>
                  <input type="number" value={item.x1} onChange={(e) => handleFurnitureChange(index, 'x1', e.target.value)} required />
                </div>
                <div className="input-group">
                  <label>Y1 (mm):</label>
                  <input type="number" value={item.y1} onChange={(e) => handleFurnitureChange(index, 'y1', e.target.value)} required />
                </div>
                <div className="input-group">
                  <label>X2 (mm):</label>
                  <input type="number" value={item.x2} onChange={(e) => handleFurnitureChange(index, 'x2', e.target.value)} required />
                </div>
                <div className="input-group">
                  <label>Y2 (mm):</label>
                  <input type="number" value={item.y2} onChange={(e) => handleFurnitureChange(index, 'y2', e.target.value)} required />
                </div>
              </>
            ) : (
              <>
                <div className="input-group">
                  <label>중심 X (mm):</label>
                  <input type="number" value={item.center_x} onChange={(e) => handleFurnitureChange(index, 'center_x', e.target.value)} required />
                </div>
                <div className="input-group">
                  <label>중심 Y (mm):</label>
                  <input type="number" value={item.center_y} onChange={(e) => handleFurnitureChange(index, 'center_y', e.target.value)} required />
                </div>
                <div className="input-group">
                  <label>반지름 (mm):</label>
                  <input type="number" value={item.radius} onChange={(e) => handleFurnitureChange(index, 'radius', e.target.value)} required min="1" />
                </div>
              </>
            )}
            <div className="input-group">
              <label>색상:</label>
              <input
                type="color"
                value={item.color}
                onChange={(e) => handleFurnitureChange(index, 'color', e.target.value)}
              />
            </div>
            <button type="button" onClick={() => handleRemoveFurniture(index)} className="remove-button">
              🗑️ 삭제
            </button>
          </div>
        ))}

        <button type="submit" disabled={loading} className="submit-button">
          {loading ? '🎨 평면도 생성 중...' : '✨ 평면도 생성'}
        </button>
      </form>

      {error && <div className="error-message">⚠️ 오류: {error}</div>}

      {roomPlanImage && (
        <div className="image-section">
          <h2>📊 생성된 평면도</h2>
          <img src={roomPlanImage} alt="방 평면도" className="room-image" />
        </div>
      )}
    </div>
  );
}

export default App; // App 컴포넌트를 내보냅니다.