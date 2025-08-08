// 가구 스타일 변경 패널 컴포넌트
import React, { useState } from 'react';

// 전체 방 스타일 옵션
const ROOM_STYLES = [
  { id: 'scandinavian', name: '스칸디나비안', description: '자연스럽고 따뜻한 북유럽 스타일', icon: '🏔️' },
  { id: 'modern', name: '모던', description: '깔끔하고 미니멀한 현대적 스타일', icon: '🏢' },
  { id: 'industrial', name: '인더스트리얼', description: '도시적이고 날것의 스타일', icon: '🏭' },
  { id: 'bohemian', name: '보헤미안', description: '자유롭고 개성있는 보헤미안 스타일', icon: '🌙' },
  { id: 'vintage', name: '빈티지', description: '클래식하고 우아한 빈티지 스타일', icon: '🏛️' }
];

const StyleChangePanel = ({ 
  screenshotData, 
  roomData,
  onStyleChange,
  onGenerateWithStyle 
}) => {
  const [selectedStyle, setSelectedStyle] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  // 3D 화면 캡처가 필요
  if (!screenshotData) {
    return (
      <div className="p-4 bg-gray-100 rounded-lg text-center text-gray-600">
        먼저 3D 화면을 캡처해주세요
      </div>
    );
  }

  // 전체 방 스타일 옵션만 제공
  const availableStyles = ROOM_STYLES;

  const handleStyleChange = async () => {
    if (!selectedStyle) {
      alert('변경할 스타일을 선택해주세요');
      return;
    }

    setIsGenerating(true);

    try {
      // 전체 가구 스타일 변경 데이터
      const styleChangeData = {
        screenshotData,
        roomData,
        newStyle: selectedStyle,
        mode: 'furniture_style_change'
        // selectedFurniture는 전달하지 않음 (전체 가구 변경)
      };

      // AI 이미지 생성
      await onGenerateWithStyle(styleChangeData);
      
      if (onStyleChange) {
        onStyleChange('all_furniture', selectedStyle);
      }

    } catch (error) {
      console.error('스타일 변경 실패:', error);
      alert('스타일 변경 중 오류가 발생했습니다');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg border">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-purple-100 rounded-lg">
          🏠
        </div>
        <div>
          <h3 className="font-bold text-lg">전체 가구 스타일 변경</h3>
          <p className="text-sm text-gray-600">
            방의 모든 가구를 원하는 스타일로 한 번에 변경합니다
          </p>
        </div>
      </div>

      {/* 현재 상태 정보 */}
      <div className="mb-6 p-3 bg-green-50 rounded-lg border border-green-200">
        <div className="text-sm font-medium text-green-800">전체 가구 스타일 변경 모드</div>
        <div className="text-lg font-bold text-green-600">
          방의 모든 가구를 통일된 스타일로 변경합니다
        </div>
        <div className="text-xs text-green-600 mt-1">
          가구 개수: {roomData?.furniture_3d?.length || 0}개
        </div>
      </div>

      {/* 스타일 선택 */}
      {availableStyles.length > 0 ? (
        <div className="mb-6">
          <h4 className="font-semibold mb-3">방 전체 스타일 선택</h4>
          <div className="grid grid-cols-2 gap-3">
            {availableStyles.map((style) => (
              <button
                key={style.id}
                onClick={() => setSelectedStyle(style.id)}
                className={`p-3 rounded-lg border transition-all ${
                  selectedStyle === style.id
                    ? 'border-purple-500 bg-purple-50 text-purple-700'
                    : 'border-gray-200 hover:border-purple-300 hover:bg-purple-25'
                }`}
              >
                <div className="w-full h-20 bg-gray-100 rounded mb-2 flex items-center justify-center">
                  <span className="text-2xl">{style.icon || '🎨'}</span>
                </div>
                <div className="text-sm font-medium">{style.name}</div>
                <div className="text-xs text-gray-500 mt-1">{style.description}</div>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="text-sm text-yellow-800">
            스타일 옵션을 불러올 수 없습니다
          </div>
        </div>
      )}

      {/* 변경 실행 버튼 */}
      <button
        onClick={handleStyleChange}
        disabled={!selectedStyle || isGenerating || availableStyles.length === 0}
        className={`w-full py-3 px-4 rounded-lg font-semibold transition-all ${
          isGenerating || !selectedStyle || availableStyles.length === 0
            ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
            : 'bg-purple-600 text-white hover:bg-purple-700 shadow-lg hover:shadow-xl'
        }`}
      >
        {isGenerating ? (
          <span className="flex items-center justify-center gap-2">
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            전체 가구 스타일 변경 중...
          </span>
        ) : (
          '전체 가구 스타일 변경하기'
        )}
      </button>

      {/* 도움말 */}
      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
        <div className="text-sm text-blue-800">
          <div className="font-medium mb-1">💡 사용법</div>
          <ul className="list-disc list-inside space-y-1 text-xs">
            <li>📸 먼저 3D 화면 캡처 버튼을 클릭하세요</li>
            <li>🎨 원하는 방 스타일을 선택하세요</li>
            <li>🚀 AI가 방의 모든 가구를 선택한 스타일로 통일되게 변경합니다</li>
            <li>⏱️ 약 30초-1분 정도 소요됩니다</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default StyleChangePanel;