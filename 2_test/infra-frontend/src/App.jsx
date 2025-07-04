import React, { useState } from "react";

const guList = [
  "강남구", "서초구", "송파구", "강서구", "관악구",
  "동작구", "중구", "용산구", "노원구", "마포구",
  // 필요시 추가
];

function aiComment(score) {
  if (score >= 18) return "이 지역은 생활 인프라가 매우 우수합니다!";
  if (score >= 12) return "이 지역은 생활 인프라가 양호한 편입니다.";
  if (score >= 8) return "이 지역은 생활 인프라가 보통입니다.";
  return "이 지역은 생활 인프라가 부족한 편입니다.";
}

function App() {
  const [region, setRegion] = useState("강남구");
  const [result, setResult] = useState(null);

  const handleFetch = async () => {
    try {
      const res = await fetch(`http://localhost:5000/infra-score?region=${region}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ region, score: 0, msg: "API 호출 실패" });
    }
  };

  return (
    <div style={{ maxWidth: 400, margin: "40px auto", padding: 24 }}>
      <h2>구별 인프라 점수 평가</h2>
      <select value={region} onChange={e => setRegion(e.target.value)}>
        {guList.map(gu => <option key={gu} value={gu}>{gu}</option>)}
      </select>
      <button onClick={handleFetch} style={{ marginLeft: 10 }}>
        AI 점수 평가
      </button>
      <div style={{ marginTop: 30, padding: 16, background: "#f4f8fb", borderRadius: 8 }}>
        {result ? (
          result.msg ? (
            <div style={{ color: "red" }}>{result.msg}</div>
          ) : (
            <>
              <div>구: <b>{result.region}</b></div>
              <div>지하철역 점수: <b>{result.score}</b></div>
              <div style={{ marginTop: 8, fontStyle: "italic" }}>{aiComment(result.score)}</div>
            </>
          )
        ) : (
          <span style={{ color: "#bbb" }}>구를 선택하고 AI 점수 평가를 눌러보세요.</span>
        )}
      </div>
    </div>
  );
}

export default App;
