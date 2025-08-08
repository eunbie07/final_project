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

// ---------------------------
// AI 인테리어 API (ai-interior 백엔드 연결)
// ---------------------------

// AI 인테리어 API 기본 설정
const AI_INTERIOR_API_BASE = import.meta.env.VITE_AI_INTERIOR_API_BASE || 'http://localhost:8000';

// MongoDB에 방 데이터 저장 후 AI 인테리어 생성
export const saveRoomDataAndGenerateAI = async (roomData, style = 'scandinavian', generator = 'dalle') => {
  try {
    let layout_id;
    
    // 가구 스타일 변경 모드 감지
    const isFurnitureStyleChange = roomData.mode === "furniture_style_change";
    
    // 이미 mongo_id가 있으면 저장 건너뛰기
    if (roomData.mongo_id) {
      console.log('Step 1: 기존 MongoDB ID 사용:', roomData.mongo_id);
      layout_id = roomData.mongo_id;
    } else if (!isFurnitureStyleChange) {
      // 1. 일반 모드일 때만 MongoDB에 방 데이터 저장
      console.log('Step 1: MongoDB에 방 데이터 저장 중...');
      
      // roomData를 백엔드가 기대하는 scene 구조로 변환
      const saveData = {
        scene: {
          description: `AI 인테리어 생성을 위한 ${roomData.dimensions.width_cm/10}cm × ${roomData.dimensions.depth_cm/10}cm 방 공간`,
          room: {
            width: roomData.dimensions.width_cm,
            depth: roomData.dimensions.depth_cm,
            height: roomData.dimensions.height_cm
          },
          objects: roomData.furniture_3d || [],
          ai_generation_request: true,
          area_sqm: roomData.area_sqm,
          volume_cum: roomData.volume_cum,
          created_at: roomData.created_at || new Date().toISOString()
        }
      };
      
      const saveResult = await saveRoomLayoutToMongoDB(saveData);
      layout_id = saveResult.layout_id;
      
      // localStorage에 MongoDB ID 저장 (다음 요청에서 재사용)
      localStorage.setItem('mongoRoomId', layout_id);

      console.log('Step 1 완료: MongoDB 저장 성공', saveResult);
    } else {
      // 가구 스타일 변경 모드에서는 기존 ID 사용
      layout_id = roomData.mongo_id || localStorage.getItem('mongoRoomId');
      console.log('Step 1: 가구 스타일 변경 모드 - 기존 MongoDB ID 사용:', layout_id);
    }

    // 생성기별 엔드포인트 매핑
    const endpoints = {
      'dify': '/generate-interior',
      'stable_diffusion': '/generate-interior-sd', 
      'dalle': '/generate-interior-dalle',
      'vertex': '/generate-interior-vertex',
      'colab': '/generate-interior-colab',
      'furniture_style_vertex': '/generate-interior-vertex'
    };
    
    const endpoint = endpoints[generator] || endpoints['colab'];
    console.log(`Step 2: ${generator} 생성기로 AI 인테리어 이미지 생성 중... (엔드포인트: ${endpoint})`);

    // 2. AI 인테리어 생성 (가구 스타일 변경 모드에 따라 다른 payload)
    let requestBody;
    
    if (isFurnitureStyleChange) {
      // 가구 스타일 변경 전용 payload
      requestBody = {
        mode: "furniture_style_change",
        room_data: {
          dimensions: roomData.dimensions,
          area_sqm: roomData.area_sqm,
          volume_cum: roomData.volume_cum,
          furniture_3d: roomData.furniture_3d,
          mongo_id: layout_id
        },
        screenshot: roomData.screenshot,
        selected_furniture: roomData.selected_furniture,
        new_style: roomData.new_style,
        style: style,
        generate_image: true
      };
    } else {
      // 일반 모드 payload
      requestBody = {
        room_data: {
          ...roomData,
          mongo_id: layout_id // MongoDB ID 추가
        },
        style: style,
        generate_image: true
      };
    }
    
    const response = await fetch(`${AI_INTERIOR_API_BASE}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`AI 인테리어 생성 실패: ${response.status} - ${errorText}`);
    }

    const aiResult = await response.json();
    console.log('Step 2 완료: AI 이미지 생성 성공', aiResult);

    return {
      success: true,
      mongo_save: { layout_id: layout_id },
      ai_generation: aiResult,
      image_path: aiResult.image_path,
      image_url: aiResult.image_url,  // 각 생성기별 응답에서 이미지 URL 추출
      generator: generator  // 사용된 생성기 정보 추가
    };

  } catch (error) {
    console.error('방 데이터 저장 또는 AI 생성 실패:', error);
    throw error;
  }
};

// AI 인테리어 이미지 생성 (이미 저장된 데이터 사용)
export const generateAIInteriorImage = async (roomData, style = 'scandinavian') => {
  try {
    console.log('AI 인테리어 이미지 생성 중...');
    console.log('서버 URL:', AI_INTERIOR_API_BASE);
    console.log('요청 데이터:', roomData);
    console.log('현재 페이지 URL:', window.location.href);
    
    const response = await fetch(`${AI_INTERIOR_API_BASE}/generate-interior`, {
      method: 'POST',
      mode: 'cors',
      cache: 'no-cache',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        room_data: {
          scene: {
            description: `AI 인테리어 생성을 위한 ${roomData.dimensions.width_cm/10}cm × ${roomData.dimensions.depth_cm/10}cm 방 공간`,
            room: {
              width: roomData.dimensions.width_cm,
              depth: roomData.dimensions.depth_cm,
              height: roomData.dimensions.height_cm
            },
            objects: roomData.furniture_3d || []
          }
        },
        style: style,
        generate_image: true
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`AI 인테리어 생성 실패: ${response.status} - ${errorText}`);
    }

    const aiResult = await response.json();
    console.log('AI 이미지 생성 성공', aiResult);

    return {
      success: true,
      ai_generation: aiResult,
      image_path: aiResult.image_path,
      image_url: aiResult.image_url  // HTTP URL 추가
    };

  } catch (error) {
    console.error('AI 인테리어 생성 실패:', error);
    console.error('오류 유형:', error.name);
    console.error('오류 메시지:', error.message);
    
    // 네트워크 오류인지 확인
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      console.error('네트워크 연결 오류: AI 서버에 연결할 수 없습니다');
      console.error('확인사항: AI 서버(포트 8000)가 실행중인지, CORS 설정이 올바른지 확인하세요');
    }
    
    throw error;
  }
};

// 생성된 AI 인테리어 이미지 목록 조회
export const getGeneratedImages = async (roomId = null) => {
  const url = roomId 
    ? `${AI_INTERIOR_API_BASE}/images?room_id=${roomId}`
    : `${AI_INTERIOR_API_BASE}/images`;
    
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`이미지 목록 조회 실패: ${response.status}`);
  }

  return await response.json();
};

// 특정 이미지 다운로드
export const downloadAIImage = async (imagePath) => {
  const response = await fetch(`${AI_INTERIOR_API_BASE}/download-image`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_path: imagePath
    })
  });

  if (!response.ok) {
    throw new Error(`이미지 다운로드 실패: ${response.status}`);
  }

  return response.blob();
};