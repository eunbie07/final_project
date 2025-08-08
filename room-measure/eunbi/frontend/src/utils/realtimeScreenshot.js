// 실시간 스크린샷 처리 (저장소 없이 즉시 처리)

class RealtimeScreenshotProcessor {
  constructor() {
    this.tempScreenshot = null; // 메모리상 임시 보관만
  }

  /**
   * 캔버스에서 직접 Blob 생성하여 즉시 처리
   */
  async captureAndProcess(canvasElement, furnitureData, onComplete) {
    try {
      console.log('📸 실시간 캡처 시작...');
      
      // 1. Canvas를 직접 Blob으로 변환 (Base64 거치지 않음)
      const blob = await this.canvasToBlob(canvasElement, 'image/jpeg', 0.8);
      
      // 2. 임시 URL 생성 (메모리상에만 존재)
      const tempURL = URL.createObjectURL(blob);
      
      // 3. 메타데이터와 함께 임시 보관
      this.tempScreenshot = {
        blob,
        tempURL,
        furniture: furnitureData.furniture,
        selectedFurniture: furnitureData.selectedFurniture,
        roomSize: furnitureData.roomSize,
        timestamp: new Date().toISOString(),
        size: blob.size // 파일 크기 확인용
      };
      
      console.log(`✅ 캡처 완료! 크기: ${(blob.size / 1024).toFixed(1)}KB`);
      
      // 4. 콜백으로 처리 완료 알림
      if (onComplete) {
        onComplete(this.tempScreenshot);
      }
      
      return this.tempScreenshot;
      
    } catch (error) {
      console.error('❌ 실시간 캡처 실패:', error);
      throw error;
    }
  }

  /**
   * AI 처리를 위한 FormData 생성
   */
  createFormDataForAI(style, newFurnitureStyle) {
    if (!this.tempScreenshot) {
      throw new Error('캡처된 스크린샷이 없습니다');
    }

    const formData = new FormData();
    
    // 1. 이미지 파일 추가
    formData.append('screenshot', this.tempScreenshot.blob, 'room-screenshot.jpg');
    
    // 2. 메타데이터 추가
    formData.append('furniture_data', JSON.stringify({
      furniture: this.tempScreenshot.furniture,
      selectedFurniture: this.tempScreenshot.selectedFurniture,
      roomSize: this.tempScreenshot.roomSize
    }));
    
    // 3. 스타일 정보 추가
    formData.append('current_style', style);
    formData.append('new_furniture_style', newFurnitureStyle);
    formData.append('mode', 'furniture_style_change');
    
    console.log('🚀 AI 처리용 FormData 생성 완료');
    return formData;
  }

  /**
   * AI 처리 직접 호출 (파일 업로드 방식)
   */
  async processWithAI(style, newFurnitureStyle, endpoint = '/generate-furniture-style') {
    try {
      const formData = this.createFormDataForAI(style, newFurnitureStyle);
      
      console.log('🎯 AI 처리 시작...');
      const response = await fetch(`${import.meta.env.VITE_AI_INTERIOR_API_BASE}${endpoint}`, {
        method: 'POST',
        body: formData // Content-Type 자동 설정됨
      });

      if (!response.ok) {
        throw new Error(`AI 처리 실패: ${response.status}`);
      }

      const result = await response.json();
      console.log('✅ AI 처리 완료:', result);
      
      // 처리 완료 후 임시 데이터 정리
      this.cleanup();
      
      return result;
      
    } catch (error) {
      console.error('❌ AI 처리 실패:', error);
      this.cleanup();
      throw error;
    }
  }

  /**
   * Canvas를 Blob으로 변환
   */
  async canvasToBlob(canvas, type = 'image/jpeg', quality = 0.8) {
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Canvas to Blob 변환 실패'));
        }
      }, type, quality);
    });
  }

  /**
   * 미리보기 URL 반환 (UI 표시용)
   */
  getPreviewURL() {
    return this.tempScreenshot?.tempURL || null;
  }

  /**
   * 현재 캡처 상태 확인
   */
  hasScreenshot() {
    return !!this.tempScreenshot;
  }

  /**
   * 캡처된 가구 정보 반환
   */
  getFurnitureData() {
    return this.tempScreenshot ? {
      furniture: this.tempScreenshot.furniture,
      selectedFurniture: this.tempScreenshot.selectedFurniture,
      roomSize: this.tempScreenshot.roomSize
    } : null;
  }

  /**
   * 임시 데이터 정리 (메모리 해제)
   */
  cleanup() {
    if (this.tempScreenshot?.tempURL) {
      URL.revokeObjectURL(this.tempScreenshot.tempURL);
    }
    this.tempScreenshot = null;
    console.log('🧹 임시 스크린샷 데이터 정리 완료');
  }

  /**
   * 페이지 언로드 시 자동 정리
   */
  setupAutoCleanup() {
    window.addEventListener('beforeunload', () => {
      this.cleanup();
    });
  }
}

// 싱글톤 인스턴스
export const realtimeProcessor = new RealtimeScreenshotProcessor();

// 자동 정리 설정
realtimeProcessor.setupAutoCleanup();

export default realtimeProcessor;