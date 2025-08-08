import React from "react";
import "./StyleSelector.css";

const StyleSelector = ({ selectedStyle, onStyleChange }) => {
  const styles = [
    {
      id: "modern",
      name: "모던",
      icon: "🏠",
      description: "깔끔하고 미니멀한 디자인",
    },
    {
      id: "scandinavian",
      name: "스칸디나비안",
      icon: "🌲",
      description: "자연스럽고 따뜻한 느낌",
    },
    {
      id: "industrial",
      name: "인더스트리얼",
      icon: "🏭",
      description: "거칠고 세련된 느낌",
    },
    {
      id: "luxury",
      name: "럭셔리",
      icon: "💎",
      description: "고급스럽고 화려한 디자인",
    },
    {
      id: "minimalist",
      name: "미니멀",
      icon: "⚪",
      description: "극도로 단순한 디자인",
    },
    {
      id: "bohemian",
      name: "보헤미안",
      icon: "🌺",
      description: "자유롭고 예술적인 느낌",
    },
  ];

  return (
    <div className="style-selector">
      <h2>🎨 스타일 선택</h2>
      <p className="style-description">
        원하는 인테리어 스타일을 선택하세요. 가구의 위치는 유지하면서 스타일만
        변경됩니다.
      </p>

      <div className="style-grid">
        {styles.map((style) => (
          <div
            key={style.id}
            className={`style-card ${
              selectedStyle === style.id ? "selected" : ""
            }`}
            onClick={() => onStyleChange(style.id)}
          >
            <div className="style-icon">{style.icon}</div>
            <h3 className="style-name">{style.name}</h3>
            <p className="style-description">{style.description}</p>
          </div>
        ))}
      </div>

      <div className="style-info">
        <h3>✨ Vertex AI 기능</h3>
        <ul>
          <li>
            🎯 <strong>위치 유지</strong>: 가구의 위치와 레이아웃은 그대로 유지
          </li>
          <li>
            🎨 <strong>스타일 변경</strong>: 선택한 스타일로 가구 디자인 변경
          </li>
          <li>
            🔍 <strong>고품질</strong>: Google Cloud Vertex AI의 고성능 이미지
            생성
          </li>
        </ul>
      </div>
    </div>
  );
};

export default StyleSelector;

