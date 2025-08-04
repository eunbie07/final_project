/**
 * 팀별 API 통합 유틸리티
 * 각 팀원의 서비스별 API 호출 함수들
 */

// 환경변수에서 API URL 가져오기 (fallback 포함)
const EUNBI_API = import.meta.env.VITE_EUNBI_API_URL || 'http://localhost:8000';
const MINAH_API = import.meta.env.VITE_MINAH_API_URL || 'http://localhost:8001';  
const DANBI_API = import.meta.env.VITE_DANBI_API_URL || 'http://localhost:8002';

// 공통 헤더 함수
const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` })
  };
};

// ---------------------------
// 은비 - 방측정/가구배치 서비스
// ---------------------------

export const eunbiApi = {
  // 방 레이아웃 저장
  saveRoomLayout: async (layoutData) => {
    const response = await fetch(`${EUNBI_API}/api/room/save-layout`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(layoutData)
    });
    return response.json();
  },

  // 방 레이아웃 조회
  getRoomLayouts: async () => {
    const response = await fetch(`${EUNBI_API}/api/room/layouts`, {
      headers: getAuthHeaders()
    });
    return response.json();
  },

  // 창문 감지
  detectWindows: async (imageFile, roomPoints = null) => {
    const formData = new FormData();
    formData.append("file", imageFile);
    if (roomPoints) {
      formData.append("room_points", JSON.stringify(roomPoints));
    }

    const response = await fetch(`${EUNBI_API}/api/room/detect-windows`, {
      method: 'POST',
      body: formData
    });
    return response.json();
  },

  // 가구 배치
  placeFurniture: async (furnitureData) => {
    const response = await fetch(`${EUNBI_API}/api/furniture/place`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(furnitureData)
    });
    return response.json();
  }
};

// ---------------------------
// 민아 - AI 인테리어 디자인 서비스
// ---------------------------

export const minahApi = {
  // AI 디자인 생성
  generateDesign: async (designInput) => {
    const response = await fetch(`${MINAH_API}/api/ai-design/generate`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(designInput)
    });
    return response.json();
  },

  // 인테리어 스타일 추천
  recommendStyles: async (roomData) => {
    const response = await fetch(`${MINAH_API}/api/interior/recommend-styles`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(roomData)
    });
    return response.json();
  },

  // 색상 팔레트 생성
  generateColorPalette: async (preferences) => {
    const response = await fetch(`${MINAH_API}/api/ai-design/color-palette`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(preferences)
    });
    return response.json();
  }
};

// ---------------------------
// 단비 - 집찾기/추천 서비스
// ---------------------------

export const danbiApi = {
  // 집 추천
  getHouseRecommendations: async (preferences) => {
    const response = await fetch(`${DANBI_API}/api/recommend/houses`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(preferences)
    });
    return response.json();
  },

  // 집 상세 정보
  getHouseDetails: async (houseId) => {
    const response = await fetch(`${DANBI_API}/api/find-house/details/${houseId}`, {
      headers: getAuthHeaders()
    });
    return response.json();
  },

  // 지역별 집 검색
  searchHousesByLocation: async (location, filters = {}) => {
    const params = new URLSearchParams({ location, ...filters });
    const response = await fetch(`${DANBI_API}/api/find-house/search?${params}`, {
      headers: getAuthHeaders()
    });
    return response.json();
  },

  // 인프라 정보 조회
  getInfrastructureInfo: async (location) => {
    const response = await fetch(`${DANBI_API}/api/recommend/infrastructure`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ location })
    });
    return response.json();
  }
};

// ---------------------------
// 공통 인증 서비스 (클라우드 API 사용)
// ---------------------------

export const authApi = {
  // 회원가입
  signup: async (userData) => {
    const response = await fetch(`${EUNBI_API}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData)
    });
    return response.json();
  },

  // 로그인
  login: async (userData) => {
    const response = await fetch(`${EUNBI_API}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData)
    });
    return response.json();
  },

  // 사용자 정보 조회
  getCurrentUser: async () => {
    const response = await fetch(`${EUNBI_API}/api/auth/me`, {
      headers: getAuthHeaders()
    });
    return response.json();
  }
};

// 통합 API 객체 (외부에서 사용)
export const teamApi = {
  eunbi: eunbiApi,
  minah: minahApi, 
  danbi: danbiApi,
  auth: authApi
};