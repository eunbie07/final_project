const form = document.getElementById('form');
const out = document.getElementById('out');

// 모델 선택에 따라 UI 업데이트
function updateModelOptions() {
  const provider = document.getElementById('provider').value;
  const modelRow = document.getElementById('model-row');
  const structureRow = document.getElementById('structure-row');
  
  // Replicate에서만 다양한 모델 옵션 제공
  if (provider === 'replicate') {
    modelRow.style.display = 'block';
    updateStructureOptions(); // 현재 모델에 따른 구조 옵션 업데이트
  } else {
    modelRow.style.display = 'none';
    structureRow.style.display = 'none';
    document.getElementById('model').value = 'default';
  }
}

// 구조 타입 옵션 업데이트
function updateStructureOptions() {
  const model = document.getElementById('model').value;
  const structureRow = document.getElementById('structure-row');
  
  if (model === 'controlnet') {
    structureRow.style.display = 'block';
  } else {
    structureRow.style.display = 'none';
  }
}

// 페이지 로드시 초기 설정
document.addEventListener('DOMContentLoaded', updateModelOptions);

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  out.textContent = '생성 중...';
  const file = document.getElementById('image').files[0];
  const provider = document.getElementById('provider').value;
  const model = document.getElementById('model').value;
  const structure = document.getElementById('structure').value;
  const style = document.getElementById('style').value;
  
  const fd = new FormData();
  fd.append('image', file);
  fd.append('provider', provider);
  fd.append('model', model);
  fd.append('structure', structure);
  fd.append('style', style);
  
  try {
    const res = await fetch('http://localhost:7000/api/realistic-room', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    out.innerHTML = '';
    (data.images || []).forEach((url, idx) => {
      const card = document.createElement('div');
      card.className = 'card';
      const img = document.createElement('img');
      img.src = url;
      const info = document.createElement('p');
      info.textContent = `${provider} - ${model} ${model === 'controlnet' ? `(${structure})` : ''} ${model === 'pipeline_3d_capture' ? `(${style})` : ''}`;
      info.style.fontSize = '12px';
      info.style.color = '#666';
      const a = document.createElement('a');
      a.href = url; a.download = `realistic_${provider}_${model}_${idx}.png`; a.textContent = '다운로드';
      card.appendChild(img); card.appendChild(info); card.appendChild(a);
      out.appendChild(card);
    });
  } catch (err) {
    out.textContent = err.message;
  }
});
