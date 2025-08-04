/**
 * API 호출 유틸리티 함수들
 * 포트 분리: 로컬 이미지 처리(3010), 클라우드 데이터(3000)
 */

// API 엔드포인트 설정 (환경변수 사용)
const LOCAL_API_BASE = import.meta.env.VITE_LOCAL_API_BASE || 'http://localhost:3010';  // 로컬 이미지/AI 처리
const CLOUD_API_BASE = import.meta.env.VITE_CLOUD_API_BASE || 'http://localhost:3000';  // 클라우드 데이터 저장/조회

// 헤더 생성 함수
const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` })
  };
};

// ---------------------------
// 인증 API
// ---------------------------

// 회원가입
export const signup = async (userData) => {
  const response = await fetch(`${CLOUD_API_BASE}/signup`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(userData)
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw { response: { data: error } };
  }
  
  return response.json();
};

// 로그인
export const login = async (userData) => {
  const response = await fetch(`${CLOUD_API_BASE}/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(userData)
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw { response: { data: error } };
  }
  
  return response.json();
};

// 현재 사용자 정보 조회
export const getCurrentUser = async () => {
  const response = await fetch(`${CLOUD_API_BASE}/me`, {
    method: 'GET',
    headers: getAuthHeaders()
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw { response: { data: error } };
  }
  
  return response.json();
};

// ---------------------------
// 클라우드 API (EC2 배포용)
// ---------------------------

// 방 레이아웃 저장 (로그인 사용자)
export const saveRoomLayoutToMongoDB = async (saveData) => {
  const token = localStorage.getItem('token');
  const endpoint = token ? '/save-room-layout' : '/save-room-layout-guest';
  
  const response = await fetch(`${CLOUD_API_BASE}${endpoint}`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify(saveData)
  });
  
  const result = await response.json();
  
  if (!result.success) {
    throw new Error(result.message);
  }
  
  return result;
};

// 사용자별 방 레이아웃 조회
export const getAllRoomLayouts = async () => {
  const token = localStorage.getItem('token');
  const endpoint = token ? '/room-layouts' : '/room-layouts-guest';
  
  const response = await fetch(`${CLOUD_API_BASE}${endpoint}`, {
    method: 'GET',
    headers: getAuthHeaders()
  });
  
  const result = await response.json();
  
  if (!result.success) {
    throw new Error(result.message || 'Failed to fetch layouts');
  }
  
  return result;
};

// 특정 방 레이아웃 조회
export const getRoomLayoutById = async (layoutId) => {
  const response = await fetch(`${CLOUD_API_BASE}/room-layout/${layoutId}`);
  const result = await response.json();
  
  if (!result.success) {
    throw new Error(result.message || 'Failed to fetch layout');
  }
  
  return result;
};

// 가구 좌표 변환
export const convertFurnitureCoordinates = async (furniture2D, roomSize) => {
  const response = await fetch(`${CLOUD_API_BASE}/convert-furniture-coordinates`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      furniture_2d: furniture2D,
      room_size: roomSize
    })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
};

// ---------------------------
// 로컬 API (이미지/AI 처리)
// ---------------------------

// 창문 감지 API 호출 함수
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

    const response = await fetch(`${LOCAL_API_BASE}/detect-windows`, {
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

// 이미지 왜곡 보정
export const undistortImage = async (imageFile) => {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch(`${LOCAL_API_BASE}/undistort`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.blob();
};

// 깊이 맵 생성
export const generateDepthMap = async () => {
  const response = await fetch(`${LOCAL_API_BASE}/depth-map`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

// 깊이 맵 이미지 조회
export const getDepthMapImage = async () => {
  const response = await fetch(`${LOCAL_API_BASE}/depth-map-image`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.blob();
};

// 특정 좌표의 깊이 값 조회
export const getDepthAtPoint = async (x, y) => {
  const response = await fetch(`${LOCAL_API_BASE}/get-depth-at-point?x=${x}&y=${y}`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

// 깊이 맵 메타 정보 조회
export const getDepthMeta = async () => {
  const response = await fetch(`${LOCAL_API_BASE}/depth-meta`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

// AI 자동 방 감지
export const autoDetectRoom = async (imageFile, confidenceThreshold = 0.7) => {
  const formData = new FormData();
  formData.append("file", imageFile);
  formData.append("confidence_threshold", confidenceThreshold.toString());

  const response = await fetch(`${LOCAL_API_BASE}/auto-detect-room`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

// 방 크기 측정
export const estimateRoomSize = async (roomPoints, targetHeight = 2.3) => {
  const response = await fetch(`${LOCAL_API_BASE}/estimate-room-size`, {
    method: "POST",
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      points: roomPoints,
      target_height: targetHeight
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

// 깊이 거리 계산
export const calculateDepthDistance = async (point1, point2) => {
  const response = await fetch(`${LOCAL_API_BASE}/depth-distance`, {
    method: "POST",
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      point1: point1,
      point2: point2
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};