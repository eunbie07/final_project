import React from 'react';

function ModelSelector({ model, setModel }) {
  return (
    <div>
      <label>모델 선택: </label>
      <select value={model} onChange={(e) => setModel(e.target.value)}>
        <option value="stability">Stability AI</option>
        <option value="dalle">DALL·E</option>
      </select>
    </div>
  );
}

export default ModelSelector;
