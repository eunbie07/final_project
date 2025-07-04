import React, { useState } from 'react';
import axios from 'axios';
import LayoutVisualizer from './LayoutVisualizer';
import LayoutVisualizer2D from './LayoutVisualizer2D';
import LayoutRoomVisualizer2D from './LayoutRoomVisualizer2D';

const LayoutUploader = () => {
  const [file, setFile] = useState(null);
  const [layoutData, setLayoutData] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post('http://localhost:5000/layout', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setLayoutData(res.data.layout_cm);  // cm 단위 환산값 사용
    } catch (err) {
      console.error(err);
      alert("분석 요청에 실패했습니다.");
    }
  };

  return (
    <div>
      <h2>배치 시뮬레이터</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>분석 요청</button>

      {layoutData && (
        <div style={{ marginTop: '20px' }}>
          <LayoutVisualizer imageUrl={previewUrl} layout={layoutData} />
          <LayoutVisualizer2D layout={layoutData} />
          <LayoutRoomVisualizer2D layout={layoutData} />
        </div>
      )}
    </div>
  );
};

export default LayoutUploader;
