import React, { useState, useEffect } from 'react';

const AIDesignPage = () => {
  const [roomData, setRoomData] = useState(null);
  const [selectedStyle, setSelectedStyle] = useState('modern');
  const [generatedDesign, setGeneratedDesign] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Room Planner에서 데이터 가져오기
  useEffect(() => {
    const savedRoomData = localStorage.getItem('roomPlannerData');
    if (savedRoomData) {
      setRoomData(JSON.parse(savedRoomData));
    }
  }, []);

  const handleGenerateDesign = async () => {
    if (!roomData) {
      alert('방 데이터가 없습니다. 먼저 2D/3D Room Planner에서 방을 측정하고 가구를 배치해주세요.');
      return;
    }

    setIsLoading(true);
    setGeneratedDesign(null);

    // AI 인테리어 생성 (실제로는 백엔드 AI 서비스에 roomData와 selectedStyle을 전송)
    await new Promise(resolve => setTimeout(resolve, 3000));

    // 목업 디자인 생성 (실제 AI 응답으로 대체)
    const mockDesigns = [
      `https://via.placeholder.com/800x600?text=AI+${selectedStyle}+Design+1`,
      `https://via.placeholder.com/800x600?text=AI+${selectedStyle}+Design+2`,
      `https://via.placeholder.com/800x600?text=AI+${selectedStyle}+Design+3`
    ];
    
    setGeneratedDesign({
      style: selectedStyle,
      designs: mockDesigns,
      roomInfo: roomData.roomInfo,
      suggestions: [
        `${selectedStyle} 스타일에 맞는 색상 팔레트를 적용했습니다.`,
        '가구 배치를 최적화하여 공간 활용도를 높였습니다.',
        '조명과 액세서리를 추가하여 분위기를 개선했습니다.'
      ]
    });
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="container mx-auto px-4">
        <div className="text-left mb-8 pt-24 md:pt-28">
          <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-6 leading-tight">AI <span className="text-primary">Interior Design</span></h1>
          <p className="text-lg text-text-secondary mb-8">
            AI suggests personalized interior designs based on room data and furniture placement from 2D/3D Room Planner.
          </p>
        </div>

        {/* 현재 방 데이터 표시 */}
        {roomData ? (
          <div className="bg-surface rounded-xl shadow-lg p-6 border border-border mb-8">
            <h2 className="text-xl font-semibold text-text-primary mb-4">현재 방 정보</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-text-secondary">
              <div>
                <span className="text-text-secondary opacity-75">방 크기:</span> {roomData.roomInfo?.width}cm × {roomData.roomInfo?.depth}cm
              </div>
              <div>
                <span className="text-text-secondary opacity-75">배치된 가구:</span> {roomData.furniture?.length || 0}개
              </div>
              <div>
                <span className="text-text-secondary opacity-75">공간 활용률:</span> {roomData.statistics?.spaceUtilization || 'N/A'}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-surface rounded-xl shadow-lg p-8 border border-border mb-8">
            <div className="text-center">
              <h2 className="text-xl font-semibold text-text-primary mb-2">방 데이터가 없습니다</h2>
              <p className="text-text-secondary mb-4">
                AI 인테리어 디자인을 생성하려면 먼저 2D/3D Room Planner에서 방을 측정하고 가구를 배치해주세요.
              </p>
              <a 
                href="/room-planner" 
                className="inline-block bg-primary text-white px-6 py-3 rounded-lg font-semibold hover:bg-secondary transition-colors duration-200"
              >
                Room Planner로 이동
              </a>
            </div>
          </div>
        )}

        <div className="bg-surface rounded-xl shadow-lg p-8 border border-border">
          <h2 className="text-2xl font-semibold text-text-primary mb-6">AI 인테리어 디자인 생성</h2>
          
          <div className="mb-6">
            <label className="block text-text-secondary text-sm font-bold mb-3">인테리어 스타일 선택:</label>
            <select
              value={selectedStyle}
              onChange={(e) => setSelectedStyle(e.target.value)}
              className="w-full p-3 bg-surface border border-border rounded-lg text-text-primary placeholder-text-secondary focus:ring-primary focus:border-primary"
            >
              <option value="modern">모던 (Modern)</option>
              <option value="minimalist">미니멀 (Minimalist)</option>
              <option value="vintage">빈티지 (Vintage)</option>
              <option value="industrial">인더스트리얼 (Industrial)</option>
              <option value="bohemian">보헤미안 (Bohemian)</option>
              <option value="scandinavian">스칸디나비안 (Scandinavian)</option>
              <option value="contemporary">컨템포러리 (Contemporary)</option>
            </select>
          </div>

          <button
            onClick={handleGenerateDesign}
            disabled={isLoading || !roomData}
            className="w-full bg-primary text-white px-6 py-3 rounded-lg font-semibold hover:bg-secondary transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'AI가 디자인 생성 중...' : 'AI 인테리어 디자인 생성'}
          </button>

          {generatedDesign && (
            <div className="mt-8">
              <h3 className="text-2xl font-semibold text-text-primary mb-6">AI가 제안하는 {generatedDesign.style} 스타일 디자인</h3>
              
              {/* 디자인 제안 */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {generatedDesign.designs.map((design, index) => (
                  <div key={index} className="bg-surface rounded-lg overflow-hidden border border-border">
                    <img src={design} alt={`AI Design ${index + 1}`} className="w-full h-48 object-cover" />
                    <div className="p-4">
                      <h4 className="text-text-primary font-semibold">디자인 옵션 {index + 1}</h4>
                    </div>
                  </div>
                ))}
              </div>

              {/* AI 제안 사항 */}
              <div className="bg-surface rounded-lg p-6 border border-border">
                <h4 className="text-text-primary font-semibold mb-4">AI 디자인 제안 사항</h4>
                <ul className="space-y-2 text-text-secondary">
                  {generatedDesign.suggestions.map((suggestion, index) => (
                    <li key={index} className="flex items-start">
                      <span className="text-primary mr-2">•</span>
                      {suggestion}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIDesignPage;