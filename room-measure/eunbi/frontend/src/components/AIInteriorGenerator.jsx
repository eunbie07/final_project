import React, { useState } from "react";
import { saveRoomDataAndGenerateAI, getGeneratedImages } from "../utils/api";

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
    endpoint: "/generate-interior-dalle",
    recommended: true 
  },
  { 
    id: "dify", 
    name: "Dify(Vertex AI)", 
    description: "빠른 생성", 
    speed: "30초",
    endpoint: "/generate-interior"
  },
  { 
    id: "stable_diffusion", 
    name: "Stable Diffusion", 
    description: "정확한 위치 제어", 
    speed: "4분+",
    endpoint: "/generate-interior-sd"
  }
];

const AIInteriorGenerator = ({ roomData, onImageGenerated }) => {
  const [selectedStyle, setSelectedStyle] = useState("scandinavian");
  const [selectedGenerator, setSelectedGenerator] = useState("dalle"); // DALL-E를 기본값으로
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [showResults, setShowResults] = useState(false);

  const [currentStep, setCurrentStep] = useState("");

  const handleGenerateImage = async () => {
    if (!roomData || !roomData.dimensions) {
      alert("방 데이터가 필요합니다. 먼저 방을 측정해주세요.");
      return;
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

      {/* AI 생성기 선택 */}
      <div className="mb-6">
        <h4 className="font-semibold mb-3 text-text-primary">AI 생성기 선택</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {GENERATOR_OPTIONS.map((generator) => (
            <button
              key={generator.id}
              onClick={() => setSelectedGenerator(generator.id)}
              className={`p-3 rounded-lg border transition-all text-left relative ${
                selectedGenerator === generator.id
                  ? "border-primary bg-primary/10 text-primary font-semibold"
                  : "border-border bg-background text-text-secondary hover:border-primary/50 hover:bg-primary/5"
              }`}
            >
              {generator.recommended && (
                <span className="absolute -top-2 -right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full">
                  추천
                </span>
              )}
              <div className="font-medium text-sm">{generator.name}</div>
              <div className="text-xs mt-1 opacity-75">{generator.description}</div>
              <div className="text-xs mt-1 opacity-60">⏱ {generator.speed}</div>
            </button>
          ))}
        </div>
      </div>

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
          disabled={isGenerating || !roomData}
          className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all ${
            isGenerating || !roomData
              ? "bg-gray-300 text-gray-500 cursor-not-allowed"
              : "bg-primary text-white hover:bg-primary/90 shadow-lg hover:shadow-xl"
          }`}
        >
          {isGenerating ? (
            <span className="flex items-center justify-center gap-2">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              AI 이미지 생성 중...
            </span>
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
            AI가 인테리어를 디자인하고 있습니다...
          </div>

          {/* 단계별 진행 상황 */}
          <div className="text-sm text-blue-600 space-y-1">
            <div
              className={`flex items-center gap-2 ${
                currentStep.includes("AI") && !currentStep.includes("완료")
                  ? "text-blue-700 font-semibold"
                  : ""
              }`}
            >
              {currentStep === "완료!" ? "✅" : "⏳"}
              <span>
                1단계:{" "}
                {INTERIOR_STYLES.find((s) => s.id === selectedStyle)?.name}{" "}
                스타일 AI 이미지 생성
              </span>
            </div>
            <div
              className={`flex items-center gap-2 ${
                currentStep === "완료!" ? "text-green-700 font-semibold" : ""
              }`}
            >
              {currentStep === "완료!" ? "✅" : "⏳"}
              <span>2단계: 고품질 이미지 렌더링 및 결과 표시</span>
            </div>
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
