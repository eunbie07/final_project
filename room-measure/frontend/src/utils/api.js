/**
 * API 호출 유틸리티 함수들
 */

// MongoDB 저장 API 호출 함수
export const saveRoomLayoutToMongoDB = async (saveData) => {
  const response = await fetch('http://localhost:3000/save-room-layout', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(saveData)
  });
  
  const result = await response.json();
  
  if (!result.success) {
    throw new Error(result.message);
  }
  
  return result;
};

// 창문 감지 API 호출 함수 (실제 방 크기 정보 포함)
export const detectWindowsInImage = async (
  imageFile,
  roomPoints = null,
  roomDimensions = null
) => {
  try {
    const formData = new FormData();
    formData.append("file", imageFile);

    // 백엔드는 JSON 문자열을 기대하므로 문자열로 변환
    if (roomPoints && roomPoints.length >= 2) {
      formData.append("room_points", JSON.stringify(roomPoints));
    }

    // 실제 방 크기 정보 추가 (JSON 문자열로)
    if (roomDimensions) {
      formData.append("room_dimensions", JSON.stringify(roomDimensions));
    }

    const response = await fetch("http://localhost:3000/detect-windows", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `HTTP error! status: ${response.status}, message: ${errorText}`
      );
    }

    const result = await response.json();
    
    return result;
  } catch (error) {
    throw error;
  }
};