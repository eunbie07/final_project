
import React, { useState } from 'react';
import './App.css';

// Base information from the 3D model image
const modelInfo = {
  width: '4.9 meters',
  length: '3.9 meters',
  height: '2.3 meters',
  layout: 'a bed is placed in the corner'
};

// Pre-defined style prompts
const styles = {
  modern: `A photorealistic interior shot of a modern and minimalist bedroom. The room is designed with a clean, neutral color palette of whites, light grays, and a few black accents. The floor is light oak wood. A large window floods the space with bright, natural daylight, creating a serene and uncluttered atmosphere.`,
  cozy: `A photorealistic interior of a cozy, warm, and natural style bedroom. The room features soft lighting, plush textiles, and natural wood furniture. A comfortable bed is adorned with many pillows and a thick knit blanket. There are several green plants on shelves and on the floor. The atmosphere is inviting and comfortable.`,
  hotel: `A photorealistic interior of a luxurious, high-end hotel bedroom. The design features a sophisticated color scheme with rich textures like velvet and silk. A king-size bed with a tufted headboard is the centerpiece. The lighting is elegant, with stylish lamps and recessed ceiling lights. The overall feeling is opulent and exclusive.`,
  industrial: `A photorealistic interior of an industrial-style loft bedroom. The room has high ceilings with exposed brick walls and polished concrete floors. The bed is a simple metal frame. Furniture is a mix of metal and reclaimed wood. Large, black-framed factory-style windows dominate one wall.`
};

function App() {
  const [selectedStyle, setSelectedStyle] = useState('modern');
  const [imageUrl, setImageUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const generateImage = async () => {
    setLoading(true);
    setError('');
    setImageUrl('');

    // --- Automatically create the prompt ---
    const finalPrompt = `Create a single, high-quality, photorealistic image based on the following description. Do not show multiple images or variations.
    - Room Dimensions: Approximately ${modelInfo.length} long by ${modelInfo.width} wide, with a ${modelInfo.height} ceiling.
    - Room Layout: ${modelInfo.layout}.
    - Interior Style: ${styles[selectedStyle]}
    - Final Image Quality: Shot with a 35mm lens, 8K, hyper-realistic, professional architectural photography.`;

    console.log("Generated Prompt:", finalPrompt);

    try {
      const response = await fetch('http://localhost:6003/api/generate-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: finalPrompt }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.details || 'Failed to generate image.');
      }

      const data = await response.json();
      setImageUrl(data.imageUrl);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Interior Generator (from 3D Model)</h1>
        <div className="prompt-section">
          <label htmlFor="style-select">Choose an interior style:</label>
          <select id="style-select" value={selectedStyle} onChange={(e) => setSelectedStyle(e.target.value)}>
            <option value="modern">Modern & Minimalist</option>
            <option value="cozy">Cozy & Natural</option>
            <option value="hotel">Luxury Hotel</option>
            <option value="industrial">Industrial Loft</option>
          </select>
          <button onClick={generateImage} disabled={loading}>
            {loading ? 'Generating...' : 'Generate Interior from 3D Model'}
          </button>
        </div>
      </header>
      <div className="image-section">
        {loading && <p className="loading-placeholder">Generating your image based on the 3D model, please wait...</p>}
        {error && <p className="error-message">Error: {error}</p>}
        {imageUrl && <img src={imageUrl} alt="Generated Interior" className="generated-image" />}
      </div>
    </div>
  );
}

export default App;
