import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import AIInteriorGenerator from '../components/AIInteriorGenerator';

const AIInteriorPage = () => {
  const location = useLocation();
  const [roomData, setRoomData] = useState(null);
  const [capturedScreenshot, setCapturedScreenshot] = useState(null);
  const [selectedFurniture, setSelectedFurniture] = useState(null);

  useEffect(() => {
    // 네비게이션을 통해 전달된 방 데이터 확인
    if (location.state?.roomData) {
      setRoomData(location.state.roomData);
    }
  }, [location.state]);

  // 로컬 스토리지에서 캡처 데이터 복원
  useEffect(() => {
    const savedScreenshot = localStorage.getItem('capturedScreenshot');
    if (savedScreenshot) {
      try {
        const parsed = JSON.parse(savedScreenshot);
        setCapturedScreenshot(parsed);
        if (parsed.selectedFurniture) {
          setSelectedFurniture(parsed.selectedFurniture);
        }
      } catch (error) {
        console.error('캡처 데이터 파싱 실패:', error);
      }
    }
  }, []);

  // 로컬 스토리지에서 최근 방 데이터 가져오기 (백업)
  useEffect(() => {
    if (!roomData) {
      const savedRoomData = localStorage.getItem('currentRoomData');
      if (savedRoomData) {
        try {
          const parsed = JSON.parse(savedRoomData);
          setRoomData(parsed);
        } catch (error) {
          console.error('방 데이터 파싱 실패:', error);
        }
      }
    }
  }, [roomData]);

  // 3D 캡처 핸들러 (RoomBox에서 호출 가능하도록)
  const handle3DCapture = () => {
    // 실제로는 RoomBox 컴포넌트가 아니므로 안내 메시지 표시
    alert('3D 화면 캡처는 방 편집 화면에서만 가능합니다. 방 편집 화면으로 돌아가서 캡처 후 다시 시도해주세요.');
  };

  return (
    <div className="min-h-screen bg-background pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 헤더 */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-text-primary mb-2">
            AI 인테리어 디자이너
          </h1>
          <p className="text-text-secondary">
            측정한 방의 크기를 바탕으로 AI가 맞춤형 인테리어 디자인을 생성해드립니다.
          </p>
        </div>

        {/* 방 정보 표시 */}
        {roomData && (
          <div className="mb-8 p-6 bg-surface rounded-xl border border-border shadow-sm">
            <h2 className="text-xl font-semibold text-text-primary mb-4">현재 방 정보</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-background rounded-lg">
                <div className="text-2xl font-bold text-primary">
                  {Math.round(roomData.dimensions?.width_cm / 100 * 10) / 10}m
                </div>
                <div className="text-sm text-text-secondary">가로</div>
              </div>
              <div className="text-center p-4 bg-background rounded-lg">
                <div className="text-2xl font-bold text-primary">
                  {Math.round(roomData.dimensions?.depth_cm / 100 * 10) / 10}m
                </div>
                <div className="text-sm text-text-secondary">세로</div>
              </div>
              <div className="text-center p-4 bg-background rounded-lg">
                <div className="text-2xl font-bold text-primary">
                  {Math.round(roomData.dimensions?.height_cm / 100 * 10) / 10}m
                </div>
                <div className="text-sm text-text-secondary">높이</div>
              </div>
              <div className="text-center p-4 bg-background rounded-lg">
                <div className="text-2xl font-bold text-primary">
                  {Math.round((roomData.area_sqm || 0) * 10) / 10}㎡
                </div>
                <div className="text-sm text-text-secondary">면적</div>
              </div>
            </div>
          </div>
        )}

        {/* AI 인테리어 생성기 */}
        <AIInteriorGenerator 
          roomData={roomData}
          capturedScreenshot={capturedScreenshot}
          selectedFurniture={selectedFurniture}
          onImageGenerated={(image) => {
            console.log('AI 인테리어 이미지 생성 완료:', image);
          }}
        />

        {/* 3D 캡처 안내 */}
        {!capturedScreenshot && (
          <div className="mt-8 p-6 bg-orange-50 border border-orange-200 rounded-xl">
            <div className="flex items-center gap-3 mb-3">
              <div className="text-2xl">📸</div>
              <h3 className="text-lg font-semibold text-orange-900">3D 캡처 기반 가구 스타일 변경</h3>
            </div>
            <div className="text-orange-800 space-y-2">
              <p>개별 가구의 스타일을 변경하려면:</p>
              <ol className="list-decimal list-inside ml-4 space-y-1">
                <li>방 편집 화면으로 돌아가서 가구를 선택하세요</li>
                <li>"📸 3D 화면 캡처" 버튼을 클릭하세요</li>
                <li>다시 이 페이지로 와서 "가구 스타일 변경"을 선택하세요</li>
              </ol>
              <button
                onClick={() => window.history.back()}
                className="mt-4 px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors"
              >
                ← 방 편집 화면으로 돌아가기
              </button>
            </div>
          </div>
        )}

        {/* 방 데이터가 없을 때 안내 */}
        {!roomData && (
          <div className="text-center py-16">
            <div className="text-6xl mb-4">🏠</div>
            <h2 className="text-xl font-semibold text-text-primary mb-2">
              방 데이터가 없습니다
            </h2>
            <p className="text-text-secondary mb-6">
              AI 인테리어 디자인을 생성하려면 먼저 방을 측정해주세요.
            </p>
            <a 
              href="/room-planner"
              className="inline-flex items-center px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors font-semibold"
            >
              방 측정하러 가기
            </a>
          </div>
        )}

        {/* 사용 가이드 */}
        <div className="mt-12 p-6 bg-blue-50 border border-blue-200 rounded-xl">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">💡 AI 인테리어 디자이너 사용법</h3>
          <div className="text-blue-800 space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</span>
              <span>원하는 인테리어 스타일을 선택하세요 (스칸디나비안, 모던, 인더스트리얼 등)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</span>
              <span>"생성" 버튼을 클릭하면 AI가 방 크기에 맞는 맞춤형 디자인을 생성합니다</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">3</span>
              <span>생성까지 약 30초-1분 정도 소요되며, 여러 스타일로 시도해볼 수 있습니다</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">4</span>
              <span>생성된 디자인은 갤러리에서 확인하고 비교할 수 있습니다</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIInteriorPage;