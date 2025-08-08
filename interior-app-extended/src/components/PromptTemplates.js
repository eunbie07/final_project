import React from 'react';

function PromptTemplates() {
  const styles = ["미니멀", "북유럽", "모던", "빈티지"];
  return (
    <div>
      <h3>스타일 템플릿</h3>
      {styles.map((style) => (
        <button key={style} style={{ marginRight: '5px' }}>{style}</button>
      ))}
    </div>
  );
}

export default PromptTemplates;
