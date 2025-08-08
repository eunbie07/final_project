import React, { useState } from "react";
import "./App.css";
import ImageUploader from "./components/ImageUploader";
import StyleSelector from "./components/StyleSelector";
import Gallery from "./components/Gallery";
import ProcessingStatus from "./components/ProcessingStatus";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedStyle, setSelectedStyle] = useState("modern");
  const [isProcessing, setIsProcessing] = useState(false);
  const [gallery, setGallery] = useState([]);
  const [processingStep, setProcessingStep] = useState("");

  const handleImageUpload = (image) => {
    setSelectedImage(image);
  };

  const handleStyleChange = (style) => {
    setSelectedStyle(style);
  };

  const handleProcessingComplete = (result) => {
    setGallery((prev) => [result, ...prev]);
    setIsProcessing(false);
    setProcessingStep("");
  };

  const startProcessing = () => {
    setIsProcessing(true);
    setProcessingStep("스타일 변경 중...");
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎨 Vertex AI 인테리어 디자인</h1>
        <p>AI로 가구 스타일을 변경하고 실사화하세요</p>
      </header>

      <main className="App-main">
        <div className="upload-section">
          <ImageUploader
            onImageUpload={handleImageUpload}
            selectedImage={selectedImage}
          />
        </div>

        {selectedImage && (
          <div className="style-section">
            <StyleSelector
              selectedStyle={selectedStyle}
              onStyleChange={handleStyleChange}
            />
          </div>
        )}

        {selectedImage && (
          <div className="processing-section">
            <ProcessingStatus
              isProcessing={isProcessing}
              processingStep={processingStep}
              onStartProcessing={startProcessing}
              selectedImage={selectedImage}
              selectedStyle={selectedStyle}
              onComplete={handleProcessingComplete}
            />
          </div>
        )}

        {gallery.length > 0 && (
          <div className="gallery-section">
            <Gallery images={gallery} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
