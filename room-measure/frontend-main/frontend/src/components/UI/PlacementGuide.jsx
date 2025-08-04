import React, { useState } from 'react';

const PlacementGuide = ({ placementMode, onClose }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!placementMode) return null;

  const getGuideContent = () => {
    const guides = {
      single_bed: {
        title: "싱글 베드 배치 가이드",
        tips: [
          "💡 벽에서 최소 60cm 떨어뜨려 배치하세요",
          "🚪 방 입구에서 접근하기 쉬운 곳에 배치하세요",
          "🪟 창문 근처에 배치하면 자연광을 받을 수 있습니다",
          "🔌 콘센트 근처에 배치하는 것이 좋습니다"
        ]
      },
      double_bed: {
        title: "더블 베드 배치 가이드", 
        tips: [
          "💡 양쪽에서 침대로 접근할 수 있도록 충분한 공간을 두세요",
          "📐 벽에서 최소 70cm 떨어뜨려 배치하세요",
          "🪟 머리맡이 창문을 등지도록 배치하는 것이 좋습니다",
          "🚪 방 입구와 마주보지 않게 배치하세요"
        ]
      },
      desk: {
        title: "책상 배치 가이드",
        tips: [
          "💡 자연광이 들어오는 창문 근처에 배치하세요",
          "🔌 콘센트에 가까운 곳에 배치하세요",
          "👁️ 벽을 등지고 앉을 수 있도록 배치하세요",
          "📐 의자를 뺄 공간(최소 120cm)을 확보하세요"
        ]
      },
      sofa: {
        title: "소파 배치 가이드",
        tips: [
          "📺 TV나 창문을 바라보도록 배치하세요",
          "🚶 소파 앞에 여유 공간(최소 100cm)을 두세요", 
          "🪑 다른 가구와 대화하기 편한 거리에 배치하세요",
          "🌞 직사광선을 피해 배치하세요"
        ]
      },
      coffee_table: {
        title: "커피 테이블 배치 가이드",
        tips: [
          "🛋️ 소파 앞 40-50cm 거리에 배치하세요",
          "👥 소파에 앉은 사람이 쉽게 닿을 수 있는 높이여야 합니다",
          "🚶 통행을 방해하지 않는 위치에 배치하세요",
          "📐 소파 길이의 2/3 정도 크기가 적당합니다"
        ]
      }
    };

    return guides[placementMode] || {
      title: "가구 배치 가이드",
      tips: [
        "💡 가구 사이에 충분한 통행 공간을 확보하세요",
        "🚪 출입구를 막지 않도록 주의하세요",
        "⚡ 전기 콘센트 위치를 고려하세요",
        "🌞 자연광과 통풍을 고려하여 배치하세요"
      ]
    };
  };

  const { title, tips } = getGuideContent();

  return (
    <div className="fixed bottom-4 left-4 z-40 max-w-sm">
      <div className="bg-surface/95 backdrop-blur rounded-lg shadow-lg border border-border">
        <div className="flex items-center justify-between p-3 border-b border-border">
          <h4 className="font-semibold text-sm text-text-primary">{title}</h4>
          <div className="flex gap-1">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-text-secondary hover:text-primary transition-colors"
            >
              {isExpanded ? '▼' : '▶'}
            </button>
            <button
              onClick={onClose}
              className="text-text-secondary hover:text-primary transition-colors ml-1"
            >
              ✕
            </button>
          </div>
        </div>
        
        {isExpanded && (
          <div className="p-3 space-y-2">
            {tips.map((tip, index) => (
              <div key={index} className="flex items-start gap-2 text-xs text-text-secondary">
                <span className="flex-shrink-0">{tip.split(' ')[0]}</span>
                <span>{tip.substring(tip.indexOf(' ') + 1)}</span>
              </div>
            ))}
            <div className="mt-3 pt-2 border-t border-border">
              <p className="text-xs text-text-secondary italic">
                💫 마우스로 클릭해서 가구를 배치하세요!
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PlacementGuide;