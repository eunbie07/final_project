import React, { useState } from "react";
import axios from "axios";
import "./ProcessingStatus.css";

const ProcessingStatus = ({
  isProcessing,
  processingStep,
  onStartProcessing,
  selectedImage,
  selectedStyle,
  onComplete,
}) => {
  const [currentStep, setCurrentStep] = useState("");
  const [error, setError] = useState("");

  const processImage = async () => {
    if (!selectedImage) return;

    setError("");
    onStartProcessing();

    try {
      // 1단계: 가구 스타일 변경
      setCurrentStep("가구 스타일 변경 중...");
      const formData = new FormData();
      formData.append("image", selectedImage.file);
      formData.append("style", selectedStyle);

      const styleResponse = await axios.post(
        "http://localhost:9001/api/change-furniture-style",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (!styleResponse.data.success) {
        throw new Error(styleResponse.data.error || "스타일 변경 실패");
      }

      // 2단계: 실사화 변환
      setCurrentStep("실사화 변환 중...");
      const photorealisticFormData = new FormData();
      photorealisticFormData.append("image", selectedImage.file);

      const photorealisticResponse = await axios.post(
        "http://localhost:9001/api/photorealistic",
        photorealisticFormData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (!photorealisticResponse.data.success) {
        throw new Error(
          photorealisticResponse.data.error || "실사화 변환 실패"
        );
      }

      // 결과 반환
      const result = {
        original: selectedImage.url,
        styled: styleResponse.data.imageUrl,
        photorealistic: photorealisticResponse.data.imageUrl,
        style: selectedStyle,
        timestamp: new Date().toISOString(),
        prompt: styleResponse.data.prompt,
      };

      onComplete(result);
      setCurrentStep("완료!");
    } catch (error) {
      console.error("Processing error:", error);
      setError(error.message || "처리 중 오류가 발생했습니다.");
      setCurrentStep("");
    }
  };

  return (
    <div className="processing-status">
      <h2>⚙️ AI 처리</h2>

      {error && (
        <div className="error-message">
          <p>❌ 오류: {error}</p>
          <button onClick={() => setError("")}>다시 시도</button>
        </div>
      )}

      {!isProcessing && !error && (
        <div className="processing-controls">
          <button
            className="process-btn"
            onClick={processImage}
            disabled={!selectedImage}
          >
            🚀 Vertex AI로 처리 시작
          </button>
          <p className="processing-info">
            선택된 스타일: <strong>{selectedStyle}</strong>
          </p>
        </div>
      )}

      {isProcessing && (
        <div className="processing-progress">
          <div className="progress-bar">
            <div className="progress-fill"></div>
          </div>
          <p className="current-step">{currentStep}</p>
          <div className="processing-steps">
            <div className="step">1. 가구 스타일 변경</div>
            <div className="step">2. 실사화 변환</div>
            <div className="step">3. 완료</div>
          </div>
        </div>
      )}

      <div className="vertex-info">
        <h3>🔧 Vertex AI 처리 과정</h3>
        <ol>
          <li>
            <strong>이미지 분석</strong>: 업로드된 이미지의 가구 위치와 레이아웃
            분석
          </li>
          <li>
            <strong>스타일 변경</strong>: 선택한 스타일로 가구 디자인 변경 (위치
            유지)
          </li>
          <li>
            <strong>실사화</strong>: 고품질 포토리얼리스틱 이미지로 변환
          </li>
          <li>
            <strong>결과 생성</strong>: 최종 인테리어 이미지 생성
          </li>
        </ol>
      </div>
    </div>
  );
};

export default ProcessingStatus;
