import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import ModelSelector from './components/ModelSelector';
import PromptTemplates from './components/PromptTemplates';
import Gallery from './components/Gallery';

function App() {
  const [model, setModel] = useState("stability");
  const [gallery, setGallery] = useState([]);

  const handleAddToGallery = (imageUrl) => {
    setGallery(prev => [imageUrl, ...prev]);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>AI 인테리어 확장 버전</h1>
      <ModelSelector model={model} setModel={setModel} />
      <PromptTemplates />
      <ImageUploader model={model} onGenerated={handleAddToGallery} />
      <Gallery images={gallery} />
    </div>
  );
}

export default App;
