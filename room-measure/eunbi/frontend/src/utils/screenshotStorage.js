// 스크린샷 저장/로드 개선된 유틸리티

class ScreenshotStorage {
  constructor() {
    this.storageKey = 'capturedScreenshots';
    this.maxScreenshots = 5; // 최대 보관 개수
  }

  /**
   * 스크린샷 저장 (개선된 방식)
   */
  async saveScreenshot(screenshotData) {
    try {
      // 1. 이미지 압축 및 최적화
      const compressedImageData = await this.compressImage(screenshotData.imageData, 0.7);
      
      // 2. 썸네일 생성 (미리보기용)
      const thumbnailData = await this.createThumbnail(screenshotData.imageData, 150, 150);
      
      // 3. 메타데이터만 localStorage에 저장
      const metadata = {
        id: Date.now().toString(),
        timestamp: screenshotData.timestamp,
        furniture: screenshotData.furniture,
        selectedFurniture: screenshotData.selectedFurniture,
        roomSize: screenshotData.roomSize,
        canvasSize: screenshotData.canvasSize,
        thumbnail: thumbnailData, // 작은 썸네일만 저장
        compressed: true
      };

      // 4. 전체 이미지는 IndexedDB에 저장 (용량 제한 없음)
      await this.saveToIndexedDB(metadata.id, compressedImageData);
      
      // 5. 메타데이터를 localStorage에 저장
      this.saveMetadata(metadata);
      
      console.log('✅ 스크린샷 저장 완료 (개선된 방식):', metadata.id);
      return metadata.id;
      
    } catch (error) {
      console.error('❌ 스크린샷 저장 실패:', error);
      // 실패 시 기존 방식으로 fallback
      return this.saveFallback(screenshotData);
    }
  }

  /**
   * 스크린샷 로드 (개선된 방식)
   */
  async loadScreenshot(id) {
    try {
      // 1. 메타데이터 로드
      const metadata = this.getMetadata(id);
      if (!metadata) {
        console.warn('⚠️ 메타데이터 없음:', id);
        return null;
      }

      // 2. IndexedDB에서 전체 이미지 로드
      const fullImageData = await this.loadFromIndexedDB(id);
      
      return {
        ...metadata,
        imageData: fullImageData,
        loadedAt: new Date().toISOString()
      };
      
    } catch (error) {
      console.error('❌ 스크린샷 로드 실패:', error);
      return this.loadFallback();
    }
  }

  /**
   * 최신 스크린샷 로드
   */
  async loadLatestScreenshot() {
    const metadata = this.getAllMetadata();
    if (metadata.length === 0) {
      return this.loadFallback(); // 기존 방식으로 fallback
    }

    // 가장 최근 스크린샷 로드
    const latest = metadata.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
    return this.loadScreenshot(latest.id);
  }

  /**
   * 이미지 압축
   */
  async compressImage(dataURL, quality = 0.7) {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();

      img.onload = () => {
        // 적절한 크기로 리사이즈 (최대 1024x768)
        const maxWidth = 1024;
        const maxHeight = 768;
        let { width, height } = img;

        if (width > maxWidth || height > maxHeight) {
          const ratio = Math.min(maxWidth / width, maxHeight / height);
          width *= ratio;
          height *= ratio;
        }

        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);
        
        resolve(canvas.toDataURL('image/jpeg', quality));
      };

      img.src = dataURL;
    });
  }

  /**
   * 썸네일 생성
   */
  async createThumbnail(dataURL, maxWidth = 150, maxHeight = 150) {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();

      img.onload = () => {
        const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;
        
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL('image/jpeg', 0.5));
      };

      img.src = dataURL;
    });
  }

  /**
   * IndexedDB에 저장
   */
  async saveToIndexedDB(id, imageData) {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('ScreenshotDB', 1);
      
      request.onerror = () => reject(request.error);
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('screenshots')) {
          db.createObjectStore('screenshots', { keyPath: 'id' });
        }
      };
      
      request.onsuccess = (event) => {
        const db = event.target.result;
        const transaction = db.transaction(['screenshots'], 'readwrite');
        const store = transaction.objectStore('screenshots');
        
        store.put({ id, imageData, savedAt: Date.now() });
        
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
      };
    });
  }

  /**
   * IndexedDB에서 로드
   */
  async loadFromIndexedDB(id) {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('ScreenshotDB', 1);
      
      request.onsuccess = (event) => {
        const db = event.target.result;
        const transaction = db.transaction(['screenshots'], 'readonly');
        const store = transaction.objectStore('screenshots');
        const getRequest = store.get(id);
        
        getRequest.onsuccess = () => {
          const result = getRequest.result;
          resolve(result ? result.imageData : null);
        };
        
        getRequest.onerror = () => reject(getRequest.error);
      };
    });
  }

  /**
   * 메타데이터 저장
   */
  saveMetadata(metadata) {
    const existing = this.getAllMetadata();
    existing.push(metadata);
    
    // 최대 개수 초과시 오래된 것 삭제
    if (existing.length > this.maxScreenshots) {
      const sorted = existing.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      const toKeep = sorted.slice(0, this.maxScreenshots);
      localStorage.setItem(this.storageKey, JSON.stringify(toKeep));
    } else {
      localStorage.setItem(this.storageKey, JSON.stringify(existing));
    }
  }

  /**
   * 메타데이터 로드
   */
  getMetadata(id) {
    const all = this.getAllMetadata();
    return all.find(meta => meta.id === id);
  }

  /**
   * 모든 메타데이터 로드
   */
  getAllMetadata() {
    try {
      const saved = localStorage.getItem(this.storageKey);
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error('메타데이터 파싱 실패:', error);
      return [];
    }
  }

  /**
   * 기존 방식으로 저장 (fallback)
   */
  saveFallback(screenshotData) {
    try {
      localStorage.setItem('capturedScreenshot', JSON.stringify(screenshotData));
      console.log('✅ Fallback 저장 완료');
      return 'fallback';
    } catch (error) {
      console.error('❌ Fallback 저장도 실패:', error);
      return null;
    }
  }

  /**
   * 기존 방식으로 로드 (fallback)
   */
  loadFallback() {
    try {
      const saved = localStorage.getItem('capturedScreenshot');
      if (saved) {
        const parsed = JSON.parse(saved);
        console.log('✅ Fallback 로드 완료');
        return parsed;
      }
    } catch (error) {
      console.error('❌ Fallback 로드 실패:', error);
    }
    return null;
  }

  /**
   * 저장소 정리
   */
  async cleanup() {
    // localStorage 정리
    localStorage.removeItem('capturedScreenshot');
    localStorage.removeItem(this.storageKey);
    
    // IndexedDB 정리
    try {
      const request = indexedDB.deleteDatabase('ScreenshotDB');
      request.onsuccess = () => console.log('✅ IndexedDB 정리 완료');
    } catch (error) {
      console.error('❌ IndexedDB 정리 실패:', error);
    }
  }
}

// 싱글톤 인스턴스
export const screenshotStorage = new ScreenshotStorage();
export default screenshotStorage;