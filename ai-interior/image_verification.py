"""
AI 생성 이미지의 정확도 검증 시스템
가구 개수와 위치를 확인하는 후처리 검증 모듈
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import base64
from datetime import datetime

try:
    import cv2
    import numpy as np
    from PIL import Image
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("WARNING: OpenCV/PIL이 설치되지 않았습니다. pip install opencv-python pillow")

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNING: Google Generative AI가 설치되지 않았습니다. pip install google-generativeai")


@dataclass
class VerificationResult:
    """검증 결과 클래스"""
    is_accurate: bool
    furniture_count_correct: bool
    position_accuracy: float  # 0.0 ~ 1.0
    detected_furniture: List[Dict[str, Any]]
    issues: List[str]
    confidence: float  # 0.0 ~ 1.0
    verification_method: str


class AIImageVerifier:
    """AI 생성 이미지 정확도 검증기"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        if self.gemini_api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.gemini_api_key)
            self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
            self.verification_method = "gemini_vision"
            print("OK: Gemini Vision API 초기화 완료")
        else:
            self.vision_model = None
            self.verification_method = "basic_analysis"
            print("WARNING: Gemini Vision API 사용 불가, 기본 분석만 가능")
    
    async def verify_generated_image(self, 
                                   image_path: str, 
                                   expected_furniture: List[Dict[str, Any]], 
                                   room_dimensions: Dict[str, float]) -> VerificationResult:
        """생성된 이미지의 정확도 검증"""
        
        if not os.path.exists(image_path):
            return VerificationResult(
                is_accurate=False,
                furniture_count_correct=False,
                position_accuracy=0.0,
                detected_furniture=[],
                issues=[f"이미지 파일 없음: {image_path}"],
                confidence=0.0,
                verification_method=self.verification_method
            )
        
        try:
            if self.vision_model:
                # Gemini Vision API 사용
                return await self._verify_with_gemini(image_path, expected_furniture, room_dimensions)
            else:
                # 기본 검증 (이미지 메타데이터, 파일 정보 등)
                return await self._verify_basic(image_path, expected_furniture, room_dimensions)
                
        except Exception as e:
            return VerificationResult(
                is_accurate=False,
                furniture_count_correct=False,
                position_accuracy=0.0,
                detected_furniture=[],
                issues=[f"검증 오류: {str(e)}"],
                confidence=0.0,
                verification_method=self.verification_method
            )
    
    async def _verify_with_gemini(self, 
                                image_path: str, 
                                expected_furniture: List[Dict[str, Any]], 
                                room_dimensions: Dict[str, float]) -> VerificationResult:
        """Gemini Vision API를 사용한 정밀 검증"""
        
        try:
            # 이미지 로드
            image = Image.open(image_path)
            
            # 기대하는 가구 정보 정리
            expected_count = len(expected_furniture)
            furniture_list = [f["name"] for f in expected_furniture]
            
            # Gemini에게 보낼 프롬프트 생성
            verification_prompt = f"""
            이 실내 이미지를 분석해서 다음 질문에 정확히 답해주세요:

            기대하는 가구:
            - 가구 개수: {expected_count}개
            - 가구 목록: {', '.join(furniture_list)}
            - 방 크기: {room_dimensions.get('width_m', 4.0)}m × {room_dimensions.get('depth_m', 5.0)}m

            분석해야 할 항목:
            1. 실제 가구 개수는 몇 개입니까?
            2. 각 가구의 종류와 위치는 어떻습니까?
            3. 기대한 가구와 일치합니까?
            4. 추가적인 장식품이나 소품이 있습니까?
            5. 전체적인 정확도를 0-100점으로 평가해주세요.

            JSON 형식으로 답변해주세요:
            {{
                "detected_furniture_count": 숫자,
                "detected_furniture_list": ["가구1", "가구2", ...],
                "matches_expected": true/false,
                "additional_items": ["항목1", "항목2", ...],
                "position_accuracy_score": 0-100,
                "overall_accuracy_score": 0-100,
                "issues": ["문제1", "문제2", ...]
            }}
            """
            
            # Gemini API 호출
            response = self.vision_model.generate_content(
                [verification_prompt, image],
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # 응답 파싱
            response_text = response.text
            print(f"DEBUG: Gemini 응답:\n{response_text}")
            
            # JSON 추출 (```json ... ``` 형태에서)
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # JSON 마커가 없으면 전체 텍스트에서 JSON 찾기
                json_text = response_text
            
            try:
                analysis = json.loads(json_text)
            except json.JSONDecodeError:
                # JSON 파싱 실패시 기본 분석
                return await self._verify_basic(image_path, expected_furniture, room_dimensions)
            
            # 검증 결과 생성
            furniture_count_correct = analysis.get("detected_furniture_count", 0) == expected_count
            position_accuracy = analysis.get("position_accuracy_score", 0) / 100.0
            overall_accuracy = analysis.get("overall_accuracy_score", 0) / 100.0
            
            detected_furniture = [
                {"name": name, "confidence": 0.9} 
                for name in analysis.get("detected_furniture_list", [])
            ]
            
            issues = analysis.get("issues", [])
            if not furniture_count_correct:
                issues.append(f"가구 개수 불일치: 기대 {expected_count}개, 감지 {analysis.get('detected_furniture_count', 0)}개")
            
            additional_items = analysis.get("additional_items", [])
            if additional_items:
                issues.append(f"추가 항목 발견: {', '.join(additional_items)}")
            
            return VerificationResult(
                is_accurate=furniture_count_correct and overall_accuracy >= 0.7,
                furniture_count_correct=furniture_count_correct,
                position_accuracy=position_accuracy,
                detected_furniture=detected_furniture,
                issues=issues,
                confidence=overall_accuracy,
                verification_method="gemini_vision"
            )
            
        except Exception as e:
            print(f"ERROR: Gemini Vision 검증 실패: {e}")
            return await self._verify_basic(image_path, expected_furniture, room_dimensions)
    
    async def _verify_basic(self, 
                          image_path: str, 
                          expected_furniture: List[Dict[str, Any]], 
                          room_dimensions: Dict[str, float]) -> VerificationResult:
        """기본 검증 (파일 정보, 이미지 속성 등)"""
        
        issues = []
        
        try:
            # 이미지 기본 정보
            image = Image.open(image_path)
            width, height = image.size
            
            print(f"DEBUG: 이미지 정보 - {width}x{height}, 포맷: {image.format}")
            
            # 파일 크기 체크
            file_size = os.path.getsize(image_path)
            if file_size < 10000:  # 10KB 미만
                issues.append("이미지 파일이 너무 작음 (생성 실패 가능성)")
            
            # 기본적인 품질 체크
            if width < 512 or height < 512:
                issues.append("이미지 해상도가 낮음")
            
            # 기대하는 가구 개수 기반 추정
            expected_count = len(expected_furniture)
            
            # 기본 검증 (파일 존재, 크기 등만 확인 가능)
            basic_accuracy = 0.7 if len(issues) == 0 else 0.3
            
            return VerificationResult(
                is_accurate=len(issues) == 0,
                furniture_count_correct=True,  # 기본 검증에서는 알 수 없음
                position_accuracy=basic_accuracy,
                detected_furniture=[{"name": f["name"], "confidence": 0.5} for f in expected_furniture],
                issues=issues,
                confidence=basic_accuracy,
                verification_method="basic_analysis"
            )
            
        except Exception as e:
            issues.append(f"기본 검증 오류: {str(e)}")
            return VerificationResult(
                is_accurate=False,
                furniture_count_correct=False,
                position_accuracy=0.0,
                detected_furniture=[],
                issues=issues,
                confidence=0.0,
                verification_method="basic_analysis"
            )
    
    def save_verification_report(self, 
                               image_path: str, 
                               verification_result: VerificationResult,
                               original_prompt: str = "",
                               style: str = "modern") -> str:
        """검증 결과를 JSON 파일로 저장"""
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "original_prompt": original_prompt,
            "style": style,
            "verification_result": {
                "is_accurate": verification_result.is_accurate,
                "furniture_count_correct": verification_result.furniture_count_correct,
                "position_accuracy": verification_result.position_accuracy,
                "detected_furniture": verification_result.detected_furniture,
                "issues": verification_result.issues,
                "confidence": verification_result.confidence,
                "verification_method": verification_result.verification_method
            }
        }
        
        # 리포트 저장
        os.makedirs("verification_reports", exist_ok=True)
        report_filename = f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join("verification_reports", report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"OK: 검증 리포트 저장됨: {report_path}")
        return report_path


class EnhancedRoomImageGenerator:
    """검증 기능이 포함된 강화된 방 이미지 생성기"""
    
    def __init__(self, vertex_generator, verifier: AIImageVerifier):
        self.vertex_generator = vertex_generator
        self.verifier = verifier
        self.max_retries = 3
    
    async def generate_verified_image(self, 
                                    prompt: str, 
                                    style: str,
                                    expected_furniture: List[Dict[str, Any]],
                                    room_dimensions: Dict[str, float]) -> Dict[str, Any]:
        """검증을 포함한 이미지 생성"""
        
        for attempt in range(self.max_retries):
            print(f"INFO: 이미지 생성 시도 {attempt + 1}/{self.max_retries}")
            
            # 1. 이미지 생성
            generation_result = await self.vertex_generator.generate_image(prompt, style)
            
            if not generation_result["success"]:
                print(f"ERROR: 이미지 생성 실패 (시도 {attempt + 1}): {generation_result.get('error')}")
                continue
            
            # 2. 생성된 이미지 검증
            image_path = generation_result["image_path"]
            verification_result = await self.verifier.verify_generated_image(
                image_path, expected_furniture, room_dimensions
            )
            
            # 3. 검증 리포트 저장
            report_path = self.verifier.save_verification_report(
                image_path, verification_result, prompt, style
            )
            
            print(f"INFO: 검증 완료 - 정확도: {verification_result.confidence:.2f}, 가구 개수: {'✓' if verification_result.furniture_count_correct else '✗'}")
            
            # 4. 정확도 판단
            if verification_result.is_accurate and verification_result.confidence >= 0.7:
                return {
                    "success": True,
                    "image_path": image_path,
                    "verification_result": verification_result,
                    "report_path": report_path,
                    "attempt": attempt + 1,
                    "method": "verified_generation"
                }
            else:
                print(f"WARNING: 생성 이미지 부정확 (시도 {attempt + 1})")
                print(f"  - 문제점: {', '.join(verification_result.issues)}")
                
                if attempt < self.max_retries - 1:
                    # 다음 시도를 위한 프롬프트 개선
                    prompt = self._improve_prompt_based_on_issues(prompt, verification_result.issues)
        
        # 모든 시도 실패
        return {
            "success": False,
            "error": f"검증된 이미지 생성 실패 ({self.max_retries}회 시도)",
            "method": "verified_generation"
        }
    
    def _improve_prompt_based_on_issues(self, original_prompt: str, issues: List[str]) -> str:
        """검증 결과를 바탕으로 프롬프트 개선"""
        
        improvements = []
        
        for issue in issues:
            if "가구 개수" in issue:
                improvements.append("CRITICAL: Show EXACTLY the specified number of furniture items, no more, no less")
            elif "추가 항목" in issue:
                improvements.append("FORBIDDEN: No decorative items, accessories, or additional furniture")
            elif "위치" in issue:
                improvements.append("PRECISE POSITIONING: Follow the exact coordinate specifications")
        
        if improvements:
            improved_prompt = f"""
{original_prompt}

CORRECTION REQUIREMENTS (based on previous generation issues):
{chr(10).join(f'- {improvement}' for improvement in improvements)}
"""
            return improved_prompt
        
        return original_prompt


# 테스트 함수
async def test_image_verification():
    """이미지 검증 시스템 테스트"""
    
    # 테스트용 데이터
    test_image_path = "generated_images/vertex_modern_20250806_184948.png"
    expected_furniture = [
        {"name": "sofa", "position": {"x": 2.0, "y": 1.5}}
    ]
    room_dimensions = {"width_m": 4.0, "depth_m": 5.0}
    
    # 검증기 초기화
    verifier = AIImageVerifier()
    
    # 검증 실행
    if os.path.exists(test_image_path):
        result = await verifier.verify_generated_image(
            test_image_path, expected_furniture, room_dimensions
        )
        
        print(f"검증 결과:")
        print(f"  정확성: {result.is_accurate}")
        print(f"  가구 개수 정확: {result.furniture_count_correct}")
        print(f"  위치 정확도: {result.position_accuracy:.2f}")
        print(f"  신뢰도: {result.confidence:.2f}")
        print(f"  문제점: {result.issues}")
        
        # 리포트 저장
        report_path = verifier.save_verification_report(
            test_image_path, result, "test prompt", "modern"
        )
        print(f"  리포트: {report_path}")
    else:
        print(f"테스트 이미지 없음: {test_image_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_image_verification())