/**
 * RoomBox.jsx와 Dify API를 연결하는 실시간 동기화 클라이언트
 * 좌표 변경을 실시간으로 감지하고 일관성 있는 AI 인테리어 이미지 생성
 */

class DifyRoomImageClient {
    constructor(baseUrl = 'http://localhost:8000', wsUrl = 'ws://localhost:8765') {
        this.baseUrl = baseUrl;
        this.wsUrl = wsUrl;
        this.currentGeneration = null;
        
        // 실시간 동기화 관련
        this.websocket = null;
        this.sessionId = null;
        this.isConnected = false;
        this.coordinateChangeCallbacks = [];
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        // 좌표 변경 감지를 위한 이전 상태
        this.lastRoomData = null;
        this.lastCoordinateHash = null;
    }

    /**
     * 실시간 좌표 동기화 세션 시작
     * @param {Object} initialRoomData - 초기 방 데이터
     * @param {string} sessionId - 세션 ID (선택사항)
     * @returns {Promise<boolean>} 연결 성공 여부
     */
    async startRealtimeSync(initialRoomData, sessionId = null) {
        if (this.isConnected) {
            console.log('⚠️ 이미 연결된 세션이 있습니다.');
            return true;
        }

        this.sessionId = sessionId || `session_${Date.now()}`;
        this.lastRoomData = JSON.parse(JSON.stringify(initialRoomData));
        this.lastCoordinateHash = this._generateCoordinateHash(initialRoomData);

        try {
            console.log('🔌 실시간 좌표 동기화 연결 중...');
            
            this.websocket = new WebSocket(this.wsUrl);
            
            this.websocket.onopen = () => {
                console.log('✅ WebSocket 연결 성공');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                
                // 세션 시작 메시지 전송
                this.websocket.send(JSON.stringify({
                    command: 'start_session',
                    session_id: this.sessionId,
                    room_data: initialRoomData
                }));
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this._handleWebSocketMessage(data);
            };

            this.websocket.onclose = () => {
                console.log('🔌 WebSocket 연결 종료');
                this.isConnected = false;
                
                // 자동 재연결 시도
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`🔄 재연결 시도 ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                    setTimeout(() => {
                        this.startRealtimeSync(this.lastRoomData, this.sessionId);
                    }, 2000 * this.reconnectAttempts);
                }
            };

            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket 오류:', error);
            };

            return true;

        } catch (error) {
            console.error('WebSocket 연결 실패:', error);
            return false;
        }
    }

    /**
     * WebSocket 메시지 처리
     * @param {Object} data - 수신된 데이터
     */
    _handleWebSocketMessage(data) {
        console.log('📨 WebSocket 메시지 수신:', data.type);

        switch (data.type) {
            case 'session_started':
                console.log('🎉 실시간 세션 시작:', data.result);
                if (!data.result.success) {
                    console.error('세션 시작 실패:', data.result.error);
                }
                break;

            case 'coordinates_updated':
                console.log('📍 좌표 업데이트 완료:', data.result);
                if (data.result.validation && !data.result.validation.valid) {
                    console.warn('⚠️ 좌표 검증 실패:', data.result.validation.errors);
                }
                break;

            case 'image_generated':
                console.log('🎨 이미지 생성 완료:', data.result);
                this.currentGeneration = {
                    ...data.result,
                    timestamp: new Date().toISOString()
                };
                break;

            case 'session_status':
                console.log('📊 세션 상태:', data.status);
                break;

            case 'error':
                console.error('❌ 서버 오류:', data.error);
                break;
        }
    }

    /**
     * 좌표 변경 감지 및 실시간 동기화
     * @param {Object} newRoomData - 새로운 방 데이터
     * @returns {Promise<Object>} 동기화 결과
     */
    async syncCoordinateChanges(newRoomData) {
        if (!this.isConnected || !this.websocket) {
            console.warn('⚠️ 실시간 동기화가 연결되지 않음');
            return { synced: false, reason: 'not_connected' };
        }

        const newHash = this._generateCoordinateHash(newRoomData);
        
        // 변경 감지
        if (newHash === this.lastCoordinateHash) {
            return { synced: true, changed: false };
        }

        try {
            console.log('🔄 좌표 변경 감지, 동기화 중...');
            
            // 서버에 좌표 업데이트 전송
            this.websocket.send(JSON.stringify({
                command: 'update_coordinates',
                room_data: newRoomData
            }));

            // 로컬 상태 업데이트
            this.lastRoomData = JSON.parse(JSON.stringify(newRoomData));
            this.lastCoordinateHash = newHash;

            // 좌표 변경 콜백 실행
            for (const callback of this.coordinateChangeCallbacks) {
                try {
                    callback(newRoomData, this.lastRoomData);
                } catch (error) {
                    console.error('좌표 변경 콜백 오류:', error);
                }
            }

            return { synced: true, changed: true, hash: newHash };

        } catch (error) {
            console.error('좌표 동기화 실패:', error);
            return { synced: false, error: error.message };
        }
    }

    /**
     * 좌표 변경 콜백 등록
     * @param {Function} callback - 콜백 함수 (newData, oldData) => void
     */
    onCoordinateChange(callback) {
        this.coordinateChangeCallbacks.push(callback);
    }

    /**
     * 실시간 이미지 생성 요청
     * @param {string} style - 스타일
     * @returns {Promise<Object>} 생성 결과
     */
    async generateImageRealtime(style = 'modern') {
        if (!this.isConnected || !this.websocket) {
            console.warn('⚠️ 실시간 연결이 필요합니다');
            return await this.generateConsistentRoomImage(this.lastRoomData, style);
        }

        try {
            console.log('🎨 실시간 이미지 생성 요청:', style);
            
            this.websocket.send(JSON.stringify({
                command: 'generate_image',
                style: style
            }));

            return { requested: true, style: style };

        } catch (error) {
            console.error('실시간 이미지 생성 요청 실패:', error);
            return { requested: false, error: error.message };
        }
    }

    /**
     * 세션 상태 조회
     * @returns {Promise<Object>} 세션 상태
     */
    async getRealtimeStatus() {
        if (!this.isConnected || !this.websocket) {
            return { connected: false };
        }

        this.websocket.send(JSON.stringify({
            command: 'get_status'
        }));

        return { 
            connected: true, 
            sessionId: this.sessionId,
            requested: true 
        };
    }

    /**
     * 실시간 동기화 종료
     */
    stopRealtimeSync() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.isConnected = false;
        this.sessionId = null;
        this.coordinateChangeCallbacks = [];
        
        console.log('🔌 실시간 동기화 종료');
    }

    /**
     * 좌표 해시 생성 (변경 감지용)
     * @param {Object} roomData - 방 데이터
     * @returns {string} 해시값
     */
    _generateCoordinateHash(roomData) {
        const coordinateData = {
            room: roomData.scene.room,
            furniture: roomData.scene.objects
                .filter(obj => obj.type === 'furniture')
                .map(obj => ({
                    name: obj.name,
                    position: obj.position,
                    dimensions: obj.dimensions,
                    rotation: obj.rotation || obj.rotation_z || 0
                }))
        };

        // 간단한 해시 생성 (실제로는 crypto 라이브러리 사용 권장)
        return btoa(JSON.stringify(coordinateData)).slice(0, 16);
    }

    /**
     * 사용 가능한 스타일 목록 가져오기
     */
    async getAvailableStyles() {
        try {
            const response = await fetch(`${this.baseUrl}/styles`);
            const data = await response.json();
            return data.styles;
        } catch (error) {
            console.error('스타일 목록 가져오기 실패:', error);
            return {};
        }
    }

    /**
     * RoomBox 데이터로 일관성 있는 방 이미지 생성
     * @param {Object} roomData - createRoomLayoutData()에서 생성된 데이터
     * @param {string} style - 'modern', 'scandinavian', 'industrial' 
     * @param {string} userId - 사용자 ID (선택사항)
     * @returns {Promise<Object>} 생성 결과
     */
    async generateConsistentRoomImage(roomData, style = 'modern', userId = null) {
        try {
            console.log('🎨 일관성 있는 방 이미지 생성 시작:', {
                style,
                roomSize: `${roomData.scene.room.width}x${roomData.scene.room.depth}mm`,
                furnitureCount: roomData.scene.objects.filter(obj => obj.type === 'furniture').length
            });

            const requestData = {
                scene: roomData.scene,
                style: style,
                user_id: userId
            };

            const response = await fetch(`${this.baseUrl}/generate-room-image`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || '이미지 생성 실패');
            }

            if (result.success) {
                console.log('✅ 이미지 생성 성공:', result.image_path);
                this.currentGeneration = {
                    ...result,
                    roomData: roomData,
                    timestamp: new Date().toISOString()
                };
            } else {
                console.error('❌ 이미지 생성 실패:', result.error);
            }

            return result;

        } catch (error) {
            console.error('API 호출 오류:', error);
            return {
                success: false,
                error: error.message,
                method: 'dify_consistent'
            };
        }
    }

    /**
     * 사용자 피드백 제출 및 학습
     * @param {number} rating - 1-5 평점
     * @param {string} comments - 코멘트 (선택사항)
     * @param {string} userId - 사용자 ID (선택사항)
     * @returns {Promise<Object>} 학습 결과
     */
    async submitFeedback(rating, comments = '', userId = null) {
        if (!this.currentGeneration) {
            throw new Error('피드백할 생성 결과가 없습니다.');
        }

        try {
            console.log('📝 피드백 제출:', { rating, comments });

            const feedbackData = {
                room_data: this.currentGeneration.roomData,
                image_path: this.currentGeneration.image_path,
                user_rating: rating,
                style: this.currentGeneration.style,
                comments: comments,
                user_id: userId
            };

            const response = await fetch(`${this.baseUrl}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedbackData)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || '피드백 제출 실패');
            }

            console.log(result.learned ? '✅ 학습 완료' : '⚠️ 학습 안됨:', result.message);
            return result;

        } catch (error) {
            console.error('피드백 제출 오류:', error);
            return {
                learned: false,
                error: error.message
            };
        }
    }

    /**
     * 스타일 일관성 테스트
     * @param {string} style - 테스트할 스타일
     * @returns {Promise<Object>} 테스트 결과
     */
    async testStyleConsistency(style = 'modern') {
        try {
            const response = await fetch(`${this.baseUrl}/test-consistency?style=${style}`);
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || '테스트 실패');
            }

            return result;

        } catch (error) {
            console.error('일관성 테스트 오류:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * 분석 데이터 가져오기
     * @returns {Promise<Object>} 통계 데이터
     */
    async getAnalytics() {
        try {
            const response = await fetch(`${this.baseUrl}/analytics`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('분석 데이터 가져오기 실패:', error);
            return null;
        }
    }
}

/**
 * RoomBox.jsx에서 사용할 헬퍼 함수들
 */

/**
 * RoomBox의 저장 버튼에 실시간 AI 이미지 생성 기능 추가
 * @param {Function} createRoomLayoutData - RoomBox의 데이터 생성 함수
 * @param {Function} saveRoomLayoutToMongoDB - 기존 저장 함수
 * @param {Function} showSuccess - 성공 알림 함수
 * @param {Function} showError - 에러 알림 함수
 * @param {DifyRoomImageClient} existingClient - 기존 클라이언트 (선택사항)
 */
export async function enhancedSaveWithRealtimeAI(
    w, d, h, furniture, detectedWindows,
    createRoomLayoutData, saveRoomLayoutToMongoDB, 
    showSuccess, showError,
    selectedStyle = 'modern',
    existingClient = null
) {
    const client = existingClient || new DifyRoomImageClient();
    
    try {
        // 1. 기존 MongoDB 저장
        const saveData = createRoomLayoutData(w, d, h, furniture, detectedWindows);
        await saveRoomLayoutToMongoDB(saveData);
        
        showSuccess(`레이아웃 저장 완료! 가구 ${furniture.length}개, 창문 ${detectedWindows.length}개`);

        // 2. 실시간 동기화 시작 (아직 연결되지 않은 경우)
        if (!client.isConnected) {
            showSuccess('실시간 동기화 시작 중...');
            const connected = await client.startRealtimeSync(saveData);
            
            if (connected) {
                showSuccess('실시간 동기화 연결 완료!');
            } else {
                showError('실시간 동기화 연결 실패, 기본 모드로 진행합니다.');
            }
        }

        // 3. AI 이미지 생성 (실시간 or 기본)
        showSuccess('AI 이미지 생성 중... 잠시만 기다려주세요.');
        
        let result;
        if (client.isConnected) {
            // 실시간 모드로 생성
            result = await client.generateImageRealtime(selectedStyle);
            showSuccess(`실시간 ${selectedStyle} 스타일 이미지 생성 요청 완료!`);
        } else {
            // 기본 모드로 생성
            result = await client.generateConsistentRoomImage(
                saveData, 
                selectedStyle,
                'user_' + Date.now()
            );
        }

        if (result.success || result.requested) {
            if (result.success) {
                showSuccess(`${selectedStyle} 스타일 이미지 생성 완료!`);
            }
            
            // 4. 피드백 UI 표시 (구현 필요)
            showFeedbackModal(client, selectedStyle);
            
            return {
                saved: true,
                imageGenerated: result.success || result.requested,
                imagePath: result.image_path,
                realtime: client.isConnected,
                style: selectedStyle
            };
        } else {
            showError(`이미지 생성 실패: ${result.error}`);
            return {
                saved: true,
                imageGenerated: false,
                error: result.error
            };
        }

    } catch (error) {
        showError(`저장 또는 이미지 생성 실패: ${error.message}`);
        return {
            saved: false,
            imageGenerated: false,
            error: error.message
        };
    }
}

/**
 * 피드백 모달 표시 (RoomBox.jsx에서 구현 필요)
 * @param {DifyRoomImageClient} client - Dify 클라이언트
 * @param {string} style - 생성된 스타일
 */
function showFeedbackModal(client, style) {
    console.log('🎯 피드백 모달 표시 필요:', { style });
    
    // TODO: RoomBox.jsx에서 피드백 UI 구현
    // 예시:
    // - 1-5 별점 평가
    // - 코멘트 입력
    // - 제출 버튼 → client.submitFeedback() 호출
    
    // 임시로 브라우저 confirm 사용
    setTimeout(() => {
        const rating = prompt(`${style} 스타일 이미지 평점을 입력하세요 (1-5):`);
        if (rating && !isNaN(rating)) {
            const comments = prompt('추가 코멘트가 있으시면 입력하세요 (선택사항):') || '';
            client.submitFeedback(parseFloat(rating), comments);
        }
    }, 2000);
}

/**
 * 스타일 선택 UI를 위한 스타일 정보 가져오기
 * @returns {Promise<Object>} 스타일 정보
 */
export async function getStyleOptions() {
    const client = new DifyRoomImageClient();
    return await client.getAvailableStyles();
}

/**
 * RoomBox.jsx에서 실시간 좌표 변경 감지 설정
 * @param {DifyRoomImageClient} client - Dify 클라이언트
 * @param {Function} createRoomLayoutData - 데이터 생성 함수
 * @param {Object} roomParams - {w, d, h, furniture, detectedWindows}
 * @param {Function} showInfo - 정보 알림 함수
 * @returns {Function} 정리 함수
 */
export function setupRealtimeCoordinateTracking(client, createRoomLayoutData, roomParams, showInfo) {
    let lastSyncTime = 0;
    const syncThrottleMs = 1000; // 1초 간격으로 동기화
    
    const coordinateChangeHandler = (newData, oldData) => {
        console.log('📍 좌표 변경 감지:', {
            furniture_moved: newData.scene.objects.filter(obj => obj.type === 'furniture').length,
            timestamp: new Date().toISOString()
        });
        
        if (showInfo) {
            showInfo('실시간 좌표 동기화 완료');
        }
    };
    
    // 좌표 변경 콜백 등록
    client.onCoordinateChange(coordinateChangeHandler);
    
    // RoomBox 상태 변경 감지 함수
    const syncRoomDataThrottled = async () => {
        const now = Date.now();
        if (now - lastSyncTime < syncThrottleMs) {
            return; // 스로틀링
        }
        
        try {
            const currentData = createRoomLayoutData(
                roomParams.w, 
                roomParams.d, 
                roomParams.h, 
                roomParams.furniture, 
                roomParams.detectedWindows
            );
            
            await client.syncCoordinateChanges(currentData);
            lastSyncTime = now;
            
        } catch (error) {
            console.error('좌표 동기화 오류:', error);
        }
    };
    
    // 정리 함수 반환
    return () => {
        console.log('🔌 실시간 좌표 추적 정리');
        // 콜백 제거는 클라이언트에서 stopRealtimeSync() 호출 시 자동 처리
    };
}

/**
 * RoomBox.jsx의 가구 이동/회전 이벤트와 연동
 * @param {DifyRoomImageClient} client - Dify 클라이언트
 * @param {Function} onFurnitureChange - RoomBox 가구 변경 콜백
 * @param {Function} createRoomLayoutData - 데이터 생성 함수
 * @param {Object} roomParams - 방 파라미터
 */
export function hookFurnitureEvents(client, onFurnitureChange, createRoomLayoutData, roomParams) {
    // 원래 콜백 래핑
    const originalCallback = onFurnitureChange;
    
    const enhancedCallback = async (furnitureArray) => {
        // 원래 콜백 실행
        if (originalCallback) {
            originalCallback(furnitureArray);
        }
        
        // 실시간 동기화
        if (client.isConnected) {
            try {
                const roomData = createRoomLayoutData(
                    roomParams.w,
                    roomParams.d, 
                    roomParams.h,
                    furnitureArray,
                    roomParams.detectedWindows
                );
                
                await client.syncCoordinateChanges(roomData);
                
            } catch (error) {
                console.error('가구 이벤트 동기화 오류:', error);
            }
        }
    };
    
    return enhancedCallback;
}

/**
 * 실시간 프롬프트 미리보기 (개발용)
 * @param {Object} roomData - 방 데이터
 * @param {string} style - 스타일
 */
export function previewPrompt(roomData, style = 'modern') {
    console.log('🔍 프롬프트 미리보기:', {
        style,
        roomSize: `${roomData.scene.room.width/1000}m × ${roomData.scene.room.depth/1000}m`,
        furniture: roomData.scene.objects
            .filter(obj => obj.type === 'furniture')
            .map(f => ({
                name: f.name,
                position: f.position.center,
                size: f.dimensions
            }))
    });
}

// 전역 클라이언트 인스턴스 (브라우저 콘솔에서 테스트용)
if (typeof window !== 'undefined') {
    window.DifyRoomImageClient = DifyRoomImageClient;
    window.difyClient = new DifyRoomImageClient();
}

export default DifyRoomImageClient;