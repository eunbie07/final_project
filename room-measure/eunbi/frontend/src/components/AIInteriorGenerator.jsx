import React, { useState } from "react";
import { saveRoomDataAndGenerateAI, getGeneratedImages } from "../utils/api";
import StyleChangePanel from "./StyleChangePanel";

const INTERIOR_STYLES = [
  {
    id: "scandinavian",
    name: "스칸디나비안",
    description: "심플하고 밝은 북유럽 스타일",
  },
  { id: "modern", name: "모던", description: "깔끔하고 세련된 현대적 스타일" },
  {
    id: "industrial",
    name: "인더스트리얼",
    description: "도시적이고 개성 있는 스타일",
  },
  { id: "cozy", name: "코지", description: "따뜻하고 아늑한 스타일" },
  {
    id: "bohemian",
    name: "보헤미안",
    description: "자유롭고 자연친화적 스타일",
  },
];

const GENERATOR_OPTIONS = [
  { 
    id: "dalle", 
    name: "DALL-E 3", 
    description: "빠른생성 정확도", 
    speed: "30초",
    accuracy: "70%",
    endpoint: "/generate-interior-dalle"
  },
  { 
    id: "dify", 
    name: "Dify(Vertex AI)", 
    description: "빠른 생성", 
    speed: "30초",
    accuracy: "75%",
    endpoint: "/generate-interior"
  },
  { 
    id: "stable_diffusion", 
    name: "Stable Diffusion", 
    description: "정확한 위치 제어", 
    speed: "4분+",
    accuracy: "80%",
    endpoint: "/generate-interior-sd"
  },
  { 
    id: "colab", 
    name: "Colab Inpainting", 
    description: "99.7% 위치 정확도 🏆", 
    speed: "1-2분",
    accuracy: "99.7%",
    endpoint: "/generate-interior-colab",
    recommended: true 
  },
  { 
    id: "furniture_style_dalle", 
    name: "가구 스타일 변경 (DALL-E)", 
    description: "DALL-E 기반 개별 가구 스타일 변경 🎯", 
    speed: "1분",
    accuracy: "90%",
    endpoint: "/generate-interior-dalle",
    mode: "furniture_change",
    recommended: true
  },
  { 
    id: "furniture_style_vertex", 
    name: "가구 스타일 변경 (Vertex AI)", 
    description: "Vertex AI 기반 고품질 가구 스타일 변경 ⭐", 
    speed: "30초",
    accuracy: "95%",
    endpoint: "/generate-interior-vertex",
    mode: "furniture_change",
    disabled: false, // 활성화!
    recommended: true
  }
];

const FURNITURE_STYLES = {
  bed: [
    { id: 'modern_bed', name: '모던 침대', description: '깔끔하고 심플한 디자인' },
    { id: 'vintage_bed', name: '빈티지 침대', description: '클래식하고 우아한 느낌' },
    { id: 'minimalist_bed', name: '미니멀 침대', description: '단순하고 기능적인 디자인' },
    { id: 'luxury_bed', name: '럭셔리 침대', description: '고급스럽고 화려한 스타일' },
    { id: 'scandinavian_bed', name: '스칸디나비안 침대', description: '밝고 자연친화적인 북유럽 스타일' },
    { id: 'industrial_bed', name: '인더스트리얼 침대', description: '메탈과 우드의 조합' }
  ],
  chair: [
    { id: 'ergonomic_chair', name: '인체공학 의자', description: '편안함을 위한 설계' },
    { id: 'vintage_chair', name: '빈티지 의자', description: '레트로한 감성' },
    { id: 'gaming_chair', name: '게이밍 의자', description: '게임에 특화된 디자인' },
    { id: 'office_chair', name: '사무용 의자', description: '업무 효율성 중심' },
    { id: 'accent_chair', name: '액센트 의자', description: '포인트가 되는 디자인' },
    { id: 'rocking_chair', name: '흔들의자', description: '편안한 휴식을 위한 의자' }
  ],
  sofa: [
    { id: 'sectional_sofa', name: '섹셔널 소파', description: 'L자형 대형 소파' },
    { id: 'chesterfield_sofa', name: '체스터필드 소파', description: '영국 전통 스타일' },
    { id: 'modern_sofa', name: '모던 소파', description: '현대적이고 세련된 디자인' },
    { id: 'recliner_sofa', name: '리클라이너 소파', description: '리클라이닝 기능이 있는 소파' },
    { id: 'loveseat_sofa', name: '러브시트 소파', description: '2인용 아늑한 소파' },
    { id: 'velvet_sofa', name: '벨벳 소파', description: '고급스러운 벨벳 소재' }
  ],
  desk: [
    { id: 'executive_desk', name: '임원용 책상', description: '고급스러운 사무용 책상' },
    { id: 'standing_desk', name: '스탠딩 책상', description: '높이 조절 가능한 책상' },
    { id: 'gaming_desk', name: '게이밍 책상', description: 'LED와 케이블 정리 기능' },
    { id: 'vintage_desk', name: '빈티지 책상', description: '클래식한 우드 테이블' }
  ],
  table: [
    { id: 'coffee_table', name: '커피 테이블', description: '거실용 중앙 테이블' },
    { id: 'dining_table', name: '다이닝 테이블', description: '식사용 테이블' },
    { id: 'side_table', name: '사이드 테이블', description: '소파 옆 보조 테이블' },
    { id: 'glass_table', name: '유리 테이블', description: '투명한 유리 소재' }
  ]
};

const AIInteriorGenerator = ({ roomData, onImageGenerated, capturedScreenshot, selectedFurniture }) => {
  const [selectedStyle, setSelectedStyle] = useState("scandinavian");
  const [selectedGenerator, setSelectedGenerator] = useState("furniture_style_dalle"); // 가구 스타일 변경 DALL-E를 기본값으로
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [showResults, setShowResults] = useState(false);
  
  // 가구 스타일 변경 관련 상태
  const [selectedFurnitureStyle, setSelectedFurnitureStyle] = useState("");
  const [showFurnitureStylePanel, setShowFurnitureStylePanel] = useState(true); // 기본값이 가구 스타일 변경이므로 true

  const [currentStep, setCurrentStep] = useState("");

  const handleGenerateImage = async () => {
    if (!roomData || !roomData.dimensions) {
      alert("방 데이터가 필요합니다. 먼저 방을 측정해주세요.");
      return;
    }

    // 가구 스타일 변경 모드 체크
    if (selectedGenerator.startsWith("furniture_style")) {
      return handleFurnitureStyleChange();
    }

    setIsGenerating(true);
    setCurrentStep("AI 인테리어 이미지 생성 중...");

    try {
      console.log("Starting AI interior generation with data:", roomData);

      // MongoDB ID가 있으면 기존 ID 사용, 없으면 새로 저장
      const mongoId = localStorage.getItem('mongoRoomId');
      let finalRoomData = { ...roomData };
      
      if (mongoId) {
        console.log('기존 MongoDB ID 사용:', mongoId);
        finalRoomData.mongo_id = mongoId;
      }
      
      const response = await saveRoomDataAndGenerateAI(finalRoomData, selectedStyle, selectedGenerator);

      if (response.success && (response.image_path || response.image_url)) {
        console.log("DEBUG - Response:", response);
        console.log("DEBUG - Image URL:", response.image_url);
        console.log("DEBUG - Image Path:", response.image_path);
        
        const newImage = {
          path: response.image_path,
          url: response.image_url || response.ai_generation?.image_url, // 대안 경로 추가
          style: selectedStyle,
          generated_at: new Date().toISOString(),
          room_dimensions: roomData.dimensions,
        };

        console.log("DEBUG - New Image Object:", newImage);
        setGeneratedImages((prev) => [newImage, ...prev]);
        setShowResults(true);
        setCurrentStep("완료!");

        if (onImageGenerated) {
          onImageGenerated(newImage);
        }
      } else {
        throw new Error(response.message || "AI 이미지 생성에 실패했습니다.");
      }
    } catch (error) {
      console.error("AI 인테리어 생성 오류:", error);
      setCurrentStep("오류 발생");
      alert(`AI 인테리어 생성 중 오류가 발생했습니다: ${error.message}`);
    } finally {
      setIsGenerating(false);
      setTimeout(() => setCurrentStep(""), 2000); // 2초 후 상태 초기화
    }
  };

  // StyleChangePanel에서 호출될 함수
  const handleGenerateWithStyle = async (styleChangeData) => {
    console.log("🎯 StyleChangePanel에서 가구 스타일 변경 요청:", styleChangeData);

    setIsGenerating(true);
    setCurrentStep("AI 가구 스타일 변경 중...");

    try {
      // 선택된 생성기에 따라 다른 엔드포인트 사용
      let generatorType = "dalle"; // 기본값
      if (selectedGenerator === "furniture_style_vertex") {
        generatorType = "furniture_style_vertex";
      } else if (selectedGenerator === "furniture_style_dalle") {
        generatorType = "furniture_style_dalle";
      }
      
      console.log(`🎯 ${generatorType.toUpperCase()} 생성기로 가구 스타일 변경 처리 중...`);
      
      // API 요청 (StyleChangePanel에서 이미 올바른 형식으로 전달됨)
      const response = await saveRoomDataAndGenerateAI(styleChangeData.roomData, styleChangeData.newStyle, generatorType);

      if (response.success && (response.image_path || response.image_url)) {
        const newImage = {
          path: response.image_path,
          url: response.image_url || response.ai_generation?.image_url,
          style: styleChangeData.selectedFurniture 
            ? `${styleChangeData.selectedFurniture.name} - ${styleChangeData.newStyle}`
            : `전체 가구 - ${styleChangeData.newStyle}`,
          generated_at: new Date().toISOString(),
          room_dimensions: roomData.dimensions,
          furniture_change: true
        };

        setGeneratedImages((prev) => [newImage, ...prev]);
        setShowResults(true);
        setCurrentStep("완료!");

        if (onImageGenerated) {
          onImageGenerated(newImage);
        }

        // 성공 시 캡처 데이터 초기화 (재사용 방지)
        localStorage.removeItem('capturedScreenshot');
        
      } else {
        throw new Error(response.message || "가구 스타일 변경에 실패했습니다.");
      }
    } catch (error) {
      console.error("가구 스타일 변경 오류:", error);
      setCurrentStep("오류 발생");
      alert(`가구 스타일 변경 중 오류가 발생했습니다: ${error.message}`);
    } finally {
      setIsGenerating(false);
      setTimeout(() => setCurrentStep(""), 2000);
    }
  };

  // 가구 스타일 변경 전용 핸들러 (기존 유지, 하지만 이제 StyleChangePanel 사용 안내)
  const handleFurnitureStyleChange = async () => {
    console.log("🔧 가구 스타일 변경 조건 체크:", {
      capturedScreenshot: !!capturedScreenshot,
      selectedFurniture: !!selectedFurniture,
      selectedFurnitureStyle: !!selectedFurnitureStyle,
      roomData: !!roomData
    });

    if (!capturedScreenshot || !selectedFurniture || !selectedFurnitureStyle) {
      alert("3D 캡처, 가구 선택, 스타일 선택이 모두 필요합니다.");
      return;
    }

    setIsGenerating(true);
    setCurrentStep("가구 스타일 변경 중...");

    try {
      console.log("🎯 가구 스타일 변경 시작:", {
        screenshot: capturedScreenshot ? "있음" : "없음",
        furniture: selectedFurniture,
        style: selectedFurnitureStyle,
        roomDataDimensions: roomData?.dimensions
      });

      // 가구 스타일 변경을 위한 특별한 데이터 구성 (기존 roomData 구조 유지)
      const furnitureStyleData = {
        // 기존 roomData 구조 유지
        dimensions: roomData.dimensions,
        area_sqm: roomData.area_sqm,
        volume_cum: roomData.volume_cum,
        furniture_3d: roomData.furniture_3d || [],
        created_at: new Date().toISOString(),
        
        // 가구 스타일 변경 특수 정보 추가
        mode: "furniture_style_change",
        screenshot: capturedScreenshot.imageData,
        selected_furniture: {
          id: selectedFurniture.id,
          type: selectedFurniture.type,
          name: selectedFurniture.name,
          position: selectedFurniture.position,
          size: selectedFurniture.size
        },
        new_style: selectedFurnitureStyle,
        
        // MongoDB ID 유지
        mongo_id: localStorage.getItem('mongoRoomId')
      };

      // 선택된 생성기에 따라 다른 엔드포인트 사용
      let generatorType = "dalle"; // 기본값
      if (selectedGenerator === "furniture_style_vertex") {
        generatorType = "vertex";
      } else if (selectedGenerator === "furniture_style_dalle") {
        generatorType = "dalle";
      }
      
      console.log(`🎯 ${generatorType.toUpperCase()} 생성기로 가구 스타일 변경 처리 중...`);
      const response = await saveRoomDataAndGenerateAI(furnitureStyleData, selectedFurnitureStyle, generatorType);

      if (response.success && (response.image_path || response.image_url)) {
        const newImage = {
          path: response.image_path,
          url: response.image_url || response.ai_generation?.image_url,
          style: `${selectedFurniture.name} - ${FURNITURE_STYLES[selectedFurniture.type]?.find(s => s.id === selectedFurnitureStyle)?.name || selectedFurnitureStyle}`,
          generated_at: new Date().toISOString(),
          room_dimensions: roomData.dimensions,
          furniture_change: true
        };

        setGeneratedImages((prev) => [newImage, ...prev]);
        setShowResults(true);
        setCurrentStep("완료!");

        if (onImageGenerated) {
          onImageGenerated(newImage);
        }

        // 성공 시 캡처 데이터 초기화 (재사용 방지)
        localStorage.removeItem('capturedScreenshot');
        
      } else {
        throw new Error(response.message || "가구 스타일 변경에 실패했습니다.");
      }
    } catch (error) {
      console.error("가구 스타일 변경 오류:", error);
      setCurrentStep("오류 발생");
      alert(`가구 스타일 변경 중 오류가 발생했습니다: ${error.message}`);
    } finally {
      setIsGenerating(false);
      setTimeout(() => setCurrentStep(""), 2000);
    }
  };

  const loadExistingImages = async () => {
    try {
      const response = await getGeneratedImages();
      if (response.success && response.images) {
        setGeneratedImages(response.images);
        setShowResults(true);
      }
    } catch (error) {
      console.error("이미지 목록 조회 오류:", error);
    }
  };

  return (
    <div className="mt-8 p-6 bg-surface rounded-xl border border-border shadow-lg">
      <h3 className="text-xl font-bold mb-4 text-text-primary flex items-center gap-2">
        AI 인테리어 디자인 생성
      </h3>

      {/* 현재 상태 안내 */}
      <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
        <div className="flex items-center gap-2 text-green-800 font-medium mb-2">
          ✅ <span>가구 스타일 변경 기능 모두 사용 가능!</span>
        </div>
        <div className="text-sm text-green-700">
          • <strong>DALL-E 기반 가구 스타일 변경</strong> - 완전 작동 중 🎯<br/>
          • <strong>Vertex AI 기반 가구 스타일 변경</strong> - 고품질 생성 완료! ⭐
        </div>
      </div>

      {/* AI 생성기 선택 */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">AI 생성기 선택</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {GENERATOR_OPTIONS.map((generator) => (
            <button
              key={generator.id}
              onClick={() => {
                if (!generator.disabled) {
                  setSelectedGenerator(generator.id);
                  if (generator.mode === "furniture_change") {
                    setShowFurnitureStylePanel(true);
                  } else {
                    setShowFurnitureStylePanel(false);
                  }
                }
              }}
              disabled={generator.disabled}
              className={`p-3 rounded-lg border transition-all text-left relative ${
                generator.disabled
                  ? "border-gray-200 bg-gray-100 text-gray-400 cursor-not-allowed opacity-60"
                  : selectedGenerator === generator.id
                  ? "border-primary bg-primary/10 text-primary font-semibold"
                  : "border-border bg-background text-text-secondary hover:border-primary/50 hover:bg-primary/5"
              }`}
            >
              {generator.recommended && (
                <span className="absolute -top-2 -right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full">
                  추천
                </span>
              )}
              {generator.mode === "furniture_change" && !generator.disabled && (
                <span className="absolute -top-2 -left-2 bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
                  NEW
                </span>
              )}
              {generator.disabled && (
                <span className="absolute -top-2 -left-2 bg-gray-500 text-white text-xs px-2 py-1 rounded-full">
                  준비중
                </span>
              )}
              <div className="font-medium text-sm">{generator.name}</div>
              <div className="text-xs mt-1 opacity-75">{generator.description}</div>
              <div className="text-xs mt-1 opacity-60">⏱ {generator.speed}</div>
              {generator.accuracy && (
                <div className="text-xs mt-1 opacity-60">🎯 {generator.accuracy}</div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* 가구 스타일 변경 패널 */}
      {showFurnitureStylePanel && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-semibold mb-3 text-blue-800 flex items-center gap-2">
            🎯 가구 스타일 변경
          </h4>
          
          {!capturedScreenshot ? (
            <div className="text-center p-4 bg-white rounded-lg border-2 border-dashed border-blue-300">
              <div className="text-blue-600 mb-2">📸</div>
              <div className="text-sm text-blue-700 font-medium mb-1">
                먼저 3D 화면을 캡처해주세요
              </div>
              <div className="text-xs text-blue-600">
                RoomBox에서 "3D 화면 캡처" 버튼을 클릭하세요
              </div>
            </div>
          ) : !selectedFurniture ? (
            <div className="text-center p-4 bg-white rounded-lg border-2 border-dashed border-yellow-300">
              <div className="text-yellow-600 mb-2">🛏️</div>
              <div className="text-sm text-yellow-700 font-medium mb-1">
                변경할 가구를 선택해주세요
              </div>
              <div className="text-xs text-yellow-600">
                3D 화면에서 가구를 클릭하여 선택하세요
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* 선택된 가구 정보 */}
              <div className="bg-white p-3 rounded-lg border">
                <div className="text-sm font-medium text-gray-700 mb-1">
                  선택된 가구
                </div>
                <div className="text-lg font-bold text-blue-600">
                  {selectedFurniture.name}
                </div>
                <div className="text-xs text-gray-500">
                  타입: {selectedFurniture.type}
                </div>
              </div>

              {/* 가구 스타일 선택 */}
              {FURNITURE_STYLES[selectedFurniture.type] ? (
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">
                    새로운 스타일 선택
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {FURNITURE_STYLES[selectedFurniture.type].map((style) => (
                      <button
                        key={style.id}
                        onClick={() => setSelectedFurnitureStyle(style.id)}
                        className={`p-3 rounded-lg border transition-all text-left ${
                          selectedFurnitureStyle === style.id
                            ? 'border-blue-500 bg-blue-50 text-blue-700'
                            : 'border-gray-200 hover:border-blue-300 hover:bg-blue-25'
                        }`}
                      >
                        <div className="font-medium text-sm">{style.name}</div>
                        <div className="text-xs mt-1 opacity-75">{style.description}</div>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="bg-orange-50 border border-orange-200 p-3 rounded-lg">
                  <div className="text-sm text-orange-800">
                    이 가구 타입({selectedFurniture.type})은 아직 스타일 변경을 지원하지 않습니다
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 스타일 선택 */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">🎨 스타일 선택</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {INTERIOR_STYLES.map((style) => (
            <button
              key={style.id}
              onClick={() => setSelectedStyle(style.id)}
              className={`p-3 rounded-lg border transition-all text-left ${
                selectedStyle === style.id
                  ? "border-primary bg-primary/10 text-primary font-semibold"
                  : "border-border bg-background text-text-secondary hover:border-primary/50 hover:bg-primary/5"
              }`}
            >
              <div className="font-medium text-sm">{style.name}</div>
              <div className="text-xs mt-1 opacity-75">{style.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* 생성 버튼 */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={handleGenerateImage}
          disabled={
            isGenerating || 
            !roomData || 
            (selectedGenerator.startsWith("furniture_style") && !capturedScreenshot)
          }
          className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all ${
            isGenerating || 
            !roomData || 
            (selectedGenerator.startsWith("furniture_style") && !capturedScreenshot)
              ? "bg-gray-300 text-gray-500 cursor-not-allowed"
              : "bg-primary text-white hover:bg-primary/90 shadow-lg hover:shadow-xl"
          }`}
        >
          {isGenerating ? (
            <span className="flex items-center justify-center gap-2">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              {selectedGenerator.startsWith("furniture_style") ? "가구 스타일 변경 중..." : "AI 이미지 생성 중..."}
            </span>
          ) : selectedGenerator.startsWith("furniture_style") ? (
            `가구 스타일 변경하기 (${selectedGenerator.includes('vertex') ? 'Vertex AI' : 'DALL-E'})`
          ) : (
            `${GENERATOR_OPTIONS.find((g) => g.id === selectedGenerator)?.name}로 ${
              INTERIOR_STYLES.find((s) => s.id === selectedStyle)?.name
            } 스타일 생성`
          )}
        </button>

        <button
          onClick={loadExistingImages}
          className="px-4 py-3 border border-primary text-primary rounded-lg hover:bg-primary/10 transition-all"
        >
          이전 결과 보기
        </button>
      </div>

      {/* 진행 상황 표시 */}
      {isGenerating && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center gap-2 text-blue-700 font-medium mb-3">
            <div className="w-4 h-4 border-2 border-blue-700 border-t-transparent rounded-full animate-spin"></div>
            {selectedGenerator.startsWith("furniture_style") 
              ? `AI가 가구 스타일을 변경하고 있습니다... (${selectedGenerator.includes('vertex') ? 'Vertex AI' : 'DALL-E'})`
              : "AI가 인테리어를 디자인하고 있습니다..."
            }
          </div>

          {/* 단계별 진행 상황 */}
          <div className="text-sm text-blue-600 space-y-1">
            {selectedGenerator.startsWith("furniture_style") ? (
              // 가구 스타일 변경 진행 상황
              <>
                <div className={`flex items-center gap-2 ${currentStep.includes("변경") && !currentStep.includes("완료") ? "text-blue-700 font-semibold" : ""}`}>
                  {currentStep === "완료!" ? "✅" : "⏳"}
                  <span>
                    1단계: {selectedFurniture?.name} → {FURNITURE_STYLES[selectedFurniture?.type]?.find(s => s.id === selectedFurnitureStyle)?.name || selectedFurnitureStyle} 스타일 변경
                  </span>
                </div>
                <div className={`flex items-center gap-2 ${currentStep === "완료!" ? "text-green-700 font-semibold" : ""}`}>
                  {currentStep === "완료!" ? "✅" : "⏳"}
                  <span>2단계: {selectedGenerator.includes('vertex') ? 'Vertex AI' : 'DALL-E'}로 고품질 이미지 생성</span>
                </div>
              </>
            ) : (
              // 일반 인테리어 생성 진행 상황
              <>
                <div className={`flex items-center gap-2 ${currentStep.includes("AI") && !currentStep.includes("완료") ? "text-blue-700 font-semibold" : ""}`}>
                  {currentStep === "완료!" ? "✅" : "⏳"}
                  <span>
                    1단계:{" "}
                    {INTERIOR_STYLES.find((s) => s.id === selectedStyle)?.name}{" "}
                    스타일 AI 이미지 생성
                  </span>
                </div>
                <div className={`flex items-center gap-2 ${currentStep === "완료!" ? "text-green-700 font-semibold" : ""}`}>
                  {currentStep === "완료!" ? "✅" : "⏳"}
                  <span>2단계: 고품질 이미지 렌더링 및 결과 표시</span>
                </div>
              </>
            )}
          </div>

          {/* 현재 단계 표시 */}
          {currentStep && (
            <div className="mt-3 p-2 bg-blue-100 rounded text-sm text-blue-800 font-medium">
              현재 진행: {currentStep}
            </div>
          )}
        </div>
      )}

      {/* 생성된 이미지 결과 */}
      {showResults && generatedImages.length > 0 && (
        <div className="border-t border-border pt-6">
          <h4 className="font-semibold mb-4 text-text-primary">
            생성된 AI 인테리어 디자인
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {generatedImages.map((image, index) => (
              <div
                key={index}
                className="bg-background rounded-lg border border-border overflow-hidden shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="aspect-square bg-gray-100 flex items-center justify-center overflow-hidden">
                  {image.url ? (
                    <img
                      src={image.url}
                      alt={`${
                        INTERIOR_STYLES.find((s) => s.id === image.style)?.name
                      } 스타일 인테리어`}
                      className="w-full h-full object-cover"
                      onLoad={() => {
                        console.log("이미지 로드 성공:", image.url);
                      }}
                      onError={(e) => {
                        console.error("이미지 로드 실패:", image.url);
                        console.error(
                          "원인: 파일이 존재하지 않거나 서버에서 접근할 수 없습니다."
                        );
                        // 이미지 로드 실패 시 대체 UI 표시
                        e.target.style.display = "none";
                        e.target.nextElementSibling.style.display = "flex";
                      }}
                    />
                  ) : null}
                  <div
                    className="text-center text-text-secondary"
                    style={{
                      display: image.url ? "none" : "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      width: "100%",
                      height: "100%",
                    }}
                  >
                    <div className="text-4xl mb-2">🎨</div>
                    <div className="text-sm">
                      {INTERIOR_STYLES.find((s) => s.id === image.style)
                        ?.name || image.style}{" "}
                      스타일
                    </div>
                    <div className="text-xs mt-1 text-text-tertiary">
                      {image.generated_at &&
                        new Date(image.generated_at).toLocaleString()}
                    </div>
                  </div>
                </div>
                <div className="p-3">
                  <div className="text-sm font-medium text-text-primary mb-1">
                    {INTERIOR_STYLES.find((s) => s.id === image.style)?.name}{" "}
                    디자인
                  </div>
                  <div className="text-xs text-text-secondary">
                    {image.room_dimensions &&
                      `${
                        Math.round(
                          (((image.room_dimensions.width_cm / 100) *
                            image.room_dimensions.depth_cm) /
                            100) *
                            100
                        ) / 100
                      }㎡`}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 가구 스타일 변경 패널 (furniture_style 생성기 선택시만 표시) */}
      {selectedGenerator.startsWith("furniture_style") && (
        <div className="mt-8">
          <StyleChangePanel
            screenshotData={capturedScreenshot?.imageData}
            roomData={roomData}
            onStyleChange={(furnitureId, style) => {
              console.log(`가구 ${furnitureId} 스타일이 ${style}로 변경됨`);
            }}
            onGenerateWithStyle={handleGenerateWithStyle}
          />
        </div>
      )}

      {/* 도움말 */}
      <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="text-sm text-text-secondary">
          <div className="font-medium mb-1">💡 사용법</div>
          <ul className="list-disc list-inside space-y-1 text-xs">
            <li>원하는 인테리어 스타일을 선택하세요</li>
            <li>"생성" 버튼을 클릭하여 AI 디자인을 요청하세요</li>
            <li>생성까지 약 30초-1분 정도 소요됩니다</li>
            <li>여러 스타일로 생성해서 비교해보세요</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AIInteriorGenerator;
