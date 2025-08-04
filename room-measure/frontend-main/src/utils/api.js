/**
 * API 호출 유틸리티 함수들 (통합 홈페이지용)
 * 인증 관련 API만 포함
 */

// API 엔드포인트 설정
const CLOUD_API_BASE = 'http://localhost:3000';  // 클라우드 데이터 저장/조회

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
// 팀 API 연동 (추후 추가 예정)
// ---------------------------

// AI 인테리어 디자인 API (민아 팀)
// export const generateAIDesign = async (designData) => { ... };

// 집찾기 추천 API (단비 팀)  
// export const searchHouses = async (searchData) => { ... };

// 방측정 서비스 API (은비 팀) - 추후 통합 시 추가
// export const measureRoom = async (roomData) => { ... };