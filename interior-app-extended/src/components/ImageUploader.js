import React, { useState } from 'react';
import axios from 'axios';

function ImageUploader({ model, onGenerated }) {
  const [file, setFile] = useState(null);
  const [noAddFurniture, setNoAddFurniture] = useState(true);

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("image", file);
    formData.append("model", model);
    formData.append("noAddFurniture", noAddFurniture);

    const res = await axios.post("http://localhost:7001/api/generate", formData);
    if (res.data && res.data.imageUrl) {
      onGenerated(res.data.imageUrl);
    }
  };

  return (
    <div>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <label>
        <input
          type="checkbox"
          checked={noAddFurniture}
          onChange={(e) => setNoAddFurniture(e.target.checked)}
        />
        가구 추가 금지
      </label>
      <button onClick={handleUpload}>이미지 생성</button>
    </div>
  );
}

export default ImageUploader;
