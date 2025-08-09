const form = document.getElementById('form');
const out = document.getElementById('out');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  out.textContent = '생성 중...';
  const file = document.getElementById('image').files[0];
  const provider = document.getElementById('provider').value;
  const prompt = document.getElementById('prompt').value;
  const strength = document.getElementById('strength').value;
  const guidance = document.getElementById('guidance').value;
  const fd = new FormData();
  fd.append('image', file);
  fd.append('provider', provider);
  fd.append('prompt', prompt);
  fd.append('strength', strength);
  fd.append('guidance', guidance);
  try {
    const res = await fetch('http://localhost:7000/api/realistic-room', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    out.innerHTML = '';
    (data.images || []).forEach((url) => {
      const card = document.createElement('div');
      card.className = 'card';
      const img = document.createElement('img');
      img.src = url;
      const a = document.createElement('a');
      a.href = url; a.download = 'realistic.png'; a.textContent = '다운로드';
      card.appendChild(img); card.appendChild(a);
      out.appendChild(card);
    });
  } catch (err) {
    out.textContent = err.message;
  }
});
