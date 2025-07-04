import React, { useRef, useState } from "react";

// 도우미 함수: 두 점 사이 거리(픽셀)
function dist(p1, p2) {
  return Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2);
}

export default function App() {
  const [imgUrl, setImgUrl] = useState(null);
  const [points, setPoints] = useState([]); // 클릭된 점
  const [results, setResults] = useState(null);
  const [heightInput, setHeightInput] = useState(240); // 층고 기본값(단위: cm)
  const canvasRef = useRef();

  // 이미지 업로드 처리
  const onImgChange = e => {
    if (e.target.files[0]) {
      setImgUrl(URL.createObjectURL(e.target.files[0]));
      setPoints([]);
      setResults(null);
    }
  };

  // 캔버스 클릭 → 점 기록
  const onCanvasClick = e => {
    if (!imgUrl) return;
    const rect = e.target.getBoundingClientRect();
    const x = e.nativeEvent.clientX - rect.left;
    const y = e.nativeEvent.clientY - rect.top;
    if (points.length < 6) setPoints([...points, {x, y}]);
  };

  // 측정(계산) 버튼
  const onCalc = () => {
    if (points.length < 6) return;
    const wall = dist(points[0], points[1]);    // 바닥~천장
    const floorW = dist(points[2], points[3]);  // 바닥 가로
    const floorH = dist(points[4], points[5]);  // 바닥 세로
    const px2cm = heightInput / wall;           // 1픽셀당 실제 cm
    setResults({
      px2cm: px2cm.toFixed(2),
      width: (floorW * px2cm).toFixed(1),
      height: (floorH * px2cm).toFixed(1),
    });
  };

  // 캔버스에 점/선 그리기
  React.useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (imgUrl && canvas) {
      const img = new window.Image();
      img.src = imgUrl;
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        // 클릭된 점/선 표시
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        for (let i=0; i<points.length; i+=2) {
          if (points[i+1]) {
            ctx.beginPath();
            ctx.moveTo(points[i].x, points[i].y);
            ctx.lineTo(points[i+1].x, points[i+1].y);
            ctx.stroke();
            ctx.closePath();
          }
        }
        // 점 그리기
        ctx.fillStyle = "red";
        points.forEach(p => ctx.fillRect(p.x-2, p.y-2, 5, 5));
      }
    }
  }, [imgUrl, points]);

  return (
    <div style={{padding:32, maxWidth:650, margin:"40px auto"}}>
      <h2>방 사진으로 평면 크기 추정하기</h2>
      <div>
        <input type="file" accept="image/*" onChange={onImgChange}/>
      </div>
      <div style={{margin:"16px 0"}}>
        <label>
          표준 층고 입력(센티미터):&nbsp;
          <input
            type="number"
            value={heightInput}
            onChange={e => setHeightInput(Number(e.target.value))}
            style={{width:80}}
          />
        </label>
      </div>
      <div style={{border:"1px solid #ddd", marginBottom:16, position:"relative", width:"fit-content"}}>
        {imgUrl &&
          <canvas
            ref={canvasRef}
            onClick={onCanvasClick}
            style={{maxWidth:"100%", cursor:"crosshair"}}
          />
        }
        {!imgUrl && <div style={{padding:40, color:"#aaa"}}>사진을 업로드하세요</div>}
        {imgUrl && <div style={{
          position:"absolute", left:8, top:8, background:"rgba(255,255,255,0.7)", padding:"2px 6px", borderRadius:5, fontSize:15
        }}>
          <div>① 벽(바닥~천장) 2점 클릭<br/>
            ② 바닥 가로 2점<br/>
            ③ 바닥 세로 2점 클릭<br/>
            (총 6점 클릭)
          </div>
        </div>}
      </div>
      <button onClick={onCalc} disabled={points.length<6} style={{fontSize:18, padding:"8px 22px"}}>계산하기</button>
      <button onClick={()=>{setPoints([]);setResults(null)}} style={{marginLeft:8}}>다시 측정</button>
      {results &&
        <div style={{marginTop:20, fontSize:19, background:"#e8f9f2", padding:18, borderRadius:10}}>
          <b>결과</b><br/>
          1px = {results.px2cm}cm<br/>
          <b>방 가로:</b> {results.width}cm&nbsp;&nbsp; <b>방 세로:</b> {results.height}cm
        </div>
      }
    </div>
  );
}
