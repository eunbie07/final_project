// 3D 화면 캡처 및 스타일 변경 컴포넌트
import React, { useRef, useCallback } from 'react';

const ScreenshotCapture = ({ canvasRef, furniture, onScreenshotTaken }) => {
  const captureScreenshot = useCallback(() => {
    if (!canvasRef.current) {
      console.error('Canvas reference not available');
      return;
    }

    try {
      // Canvas에서 스크린샷 캡처
      const canvas = canvasRef.current;
      const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
      
      if (!gl) {
        console.error('WebGL context not available');
        return;
      }

      // Canvas 내용을 dataURL로 변환
      const dataURL = canvas.toDataURL('image/png', 0.9);
      
      // 가구 정보와 함께 스크린샷 데이터 생성
      const screenshotData = {
        imageData: dataURL,
        timestamp: new Date().toISOString(),
        furniture: furniture.map(f => ({
          id: f.id,
          type: f.type,
          name: f.name,
          position: f.position,
          size: f.size,
          // 2D 화면 좌표계로 변환 (추후 inpainting 마스킹용)
          screenBounds: calculateScreenBounds(f.position, f.size)
        })),
        canvasSize: {
          width: canvas.width,
          height: canvas.height
        }
      };

      // 부모 컴포넌트에 캡처된 데이터 전달
      if (onScreenshotTaken) {
        onScreenshotTaken(screenshotData);
      }

      console.log('3D 스크린샷 캡처 완료:', screenshotData);
      return screenshotData;

    } catch (error) {
      console.error('스크린샷 캡처 실패:', error);
    }
  }, [canvasRef, furniture, onScreenshotTaken]);

  // 3D 좌표를 화면 좌표로 변환하는 함수
  const calculateScreenBounds = useCallback((position, size) => {
    // 카메라 projection을 고려한 2D 화면 좌표 계산
    // 실제로는 Three.js의 카메라 정보를 활용해야 함
    const [x, y, z] = position;
    const [width, height, depth] = size;
    
    // 간단한 근사치 계산 (실제 구현시 더 정교하게)
    return {
      x: Math.round(x * 0.8), // 스케일링 팩터
      y: Math.round(z * 0.8),
      width: Math.round(width * 0.8),
      height: Math.round(depth * 0.8)
    };
  }, []);

  return (
    <button
      onClick={captureScreenshot}
      className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
    >
      📸 3D 화면 캡처
    </button>
  );
};

export default ScreenshotCapture;