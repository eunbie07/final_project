"""
45장 이미지 임베딩 처리 시스템 
이미지를 Vision AI로 분석하여 구조화된 설명 생성 후 Dify에 임베딩
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# Google Cloud Vision AI
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    print("WARNING: Vertex AI가 설치되지 않았습니다.")

from dify_rag import DifyLayoutRAG


class ImageEmbeddingProcessor:
    """이미지 분석 및 임베딩 처리기"""
    
    def __init__(self, 
                 image_dir: str = "image",
                 project_id: str = "virtual-muse-466706-v2",
                 location: str = "us-central1"):
        
        self.image_dir = Path(image_dir)
        self.project_id = project_id
        self.location = location
        self.output_dir = Path("image_analysis_results")
        
        # Dify RAG 초기화
        import os
        self.dify_rag = DifyLayoutRAG(
            os.getenv("DIFY_API_KEY", ""),
            os.getenv("DIFY_APP_ID", ""),
            os.getenv("DIFY_DATASET_ID", "")
        )
        
        # Vision AI 초기화
        self._initialize_vision_ai()
        
        # 컨셉별 분석 키워드 정의
        self.concept_keywords = {
            "Modern,Minimalist": {
                "style_description": "모던 미니멀리스트",
                "key_elements": ["깔끔한 라인", "화이트/그레이", "기하학적 형태", "여백 활용"],
                "materials": ["패브릭", "유리", "메탈", "마블"],
                "colors": ["화이트", "라이트 그레이", "블랙", "베이지"],
                "mood": "깔끔하고 정돈된, 세련된",
                "furniture_style": "미니멀리스트, 직선적, 기하학적"
            },
            "Scandinavian": {
                "style_description": "스칸디나비아/북유럽",
                "key_elements": ["자연광", "우드 소재", "화이트/베이지", "심플한 가구"],
                "materials": ["라이트 우드", "울", "코튼", "세라믹"],
                "colors": ["화이트", "라이트 그레이", "내추럴 우드", "소프트 파스텔"],
                "mood": "포근하고 따뜻한, 휘게 느낌",
                "furniture_style": "기능적, 심플, 유기적 형태"
            },
            "Industrial": {
                "style_description": "인더스트리얼",  
                "key_elements": ["노출 벽돌/파이프", "메탈 소재", "다크 톤", "빈티지 조명"],
                "materials": ["노출 벽돌", "스틸", "콘크리트", "가죽"],
                "colors": ["다크 그레이", "블랙", "러스트", "로우 메탈"],
                "mood": "도시적이고 날카로운, 로우 캐릭터",
                "furniture_style": "로우 머티리얼, 메탈 프레임, 빈티지"
            },
            "Bohemian,Natural": {
                "style_description": "보헤미안/내추럴",
                "key_elements": ["따뜻한 색감", "패턴/텍스처", "식물", "천연 소재"],
                "materials": ["라탄", "우드", "리넨", "울"],
                "colors": ["어스 톤", "테라코타", "올리브", "크림"],
                "mood": "자유롭고 편안한, 자연친화적",
                "furniture_style": "자연스럽고 편안한, 핸드메이드 느낌"
            },
            "Cozy": {
                "style_description": "코지/심플",
                "key_elements": ["포근함", "소프트 패브릭", "따뜻한 조명", "생활감"],
                "materials": ["소프트 패브릭", "니트", "우드", "세라믹"],
                "colors": ["웜 화이트", "베이지", "소프트 그레이", "파스텔"],  
                "mood": "포근하고 편안한, 홈리한",
                "furniture_style": "편안하고 실용적, 소프트한 라인"
            }
        }
        
    def _initialize_vision_ai(self):
        """Vision AI 초기화"""
        self.vision_model = None
        
        if not VERTEX_AI_AVAILABLE:
            print("WARNING: Vision AI를 사용할 수 없습니다")
            return
            
        try:
            # Google Cloud 인증 설정
            key_path = os.path.join(os.path.dirname(__file__), "key.json")
            if os.path.exists(key_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
                print(f"OK: Google Cloud 키 파일 설정: {key_path}")
            
            # Vertex AI 초기화
            vertexai.init(project=self.project_id, location=self.location)
            # 여러 모델 시도
            model_names = [
                "gemini-1.5-flash",
                "gemini-1.0-pro-vision",
                "gemini-pro-vision",
                "gemini-1.5-pro-001"
            ]
            
            for model_name in model_names:
                try:
                    self.vision_model = GenerativeModel(model_name)
                    print(f"OK: Vision AI 모델 성공: {model_name}")
                    break
                except Exception as model_error:
                    print(f"SKIP: {model_name} 실패: {model_error}")
                    continue
            
            if not self.vision_model:
                print("WARNING: 모든 Vision AI 모델 초기화 실패")
            
            print("OK: Vision AI 모델 초기화 완료")
            
        except Exception as e:
            print(f"WARNING: Vision AI 초기화 실패: {e}")
    
    def _get_concept_from_filename(self, filename: str) -> str:
        """파일명에서 컨셉 추출"""
        filename_lower = filename.lower()
        
        if "modern" in filename_lower or "minimalist" in filename_lower:
            return "Modern,Minimalist"
        elif "scandinavian" in filename_lower:
            return "Scandinavian"
        elif "industrial" in filename_lower:
            return "Industrial" 
        elif "bohemain" in filename_lower or "bohemian" in filename_lower or "natural" in filename_lower:
            return "Bohemian,Natural"
        elif "cozy" in filename_lower:
            return "Cozy"
        else:
            return "Unknown"
    
    async def analyze_single_image(self, image_path: Path) -> Dict[str, Any]:
        """단일 이미지 분석"""
        
        if not self.vision_model:
            return {"success": False, "error": "Vision AI 모델이 초기화되지 않았습니다"}
        
        try:
            # 컨셉 확인
            concept = self._get_concept_from_filename(image_path.name)
            concept_info = self.concept_keywords.get(concept, {})
            
            # 이미지 로드
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            image_part = Part.from_data(data=image_data, mime_type="image/png")
            
            # 컨셉별 특화 프롬프트 생성
            analysis_prompt = f"""
            이 {concept_info.get('style_description', concept)} 스타일 인테리어 이미지를 자세히 분석해주세요.

            다음 관점에서 분석해주세요:
            1. 전체적인 방 구조와 크기 (추정)
            2. 가구 배치 및 종류 (정확한 위치와 크기)
            3. 색상 팔레트 ({', '.join(concept_info.get('colors', []))})
            4. 재료와 텍스처 ({', '.join(concept_info.get('materials', []))})
            5. 조명과 분위기 ({concept_info.get('mood', '')})
            6. {concept} 스타일의 특징적 요소들

            다음 JSON 형식으로 정확히 응답해주세요:
            {{
                "concept": "{concept}",
                "image_id": "{image_path.name}",
                "style": "{concept_info.get('style_description', concept)}",
                "room_analysis": {{
                    "estimated_size": "가로 x 세로 추정 (예: 4m x 5m)",
                    "room_type": "거실/침실/서재 등",
                    "layout_description": "방 구조 설명"
                }},
                "furniture": [
                    {{
                        "name": "가구명",
                        "position": "위치 (예: 중앙, 왼쪽 벽 근처)",
                        "estimated_size": "추정 크기",
                        "material": "재료",
                        "color": "색상"
                    }}
                ],
                "colors": ["주요 색상들"],
                "materials": ["사용된 재료들"],
                "mood": "분위기 설명",
                "lighting": "조명 특징",
                "style_elements": ["스타일 특징적 요소들"],
                "spatial_relationships": "가구 간 공간적 관계 설명"
            }}
            """
            
            print(f"ANALYZE: {image_path.name} 분석 중...")
            
            # Vision AI로 분석
            response = self.vision_model.generate_content([analysis_prompt, image_part])
            analysis_text = response.text
            
            # JSON 파싱 시도
            try:
                # JSON 부분만 추출 (```json 태그 제거)
                if "```json" in analysis_text:
                    json_start = analysis_text.find("```json") + 7
                    json_end = analysis_text.find("```", json_start)
                    json_text = analysis_text[json_start:json_end].strip()
                elif "{" in analysis_text:
                    json_start = analysis_text.find("{")
                    json_end = analysis_text.rfind("}") + 1
                    json_text = analysis_text[json_start:json_end]
                else:
                    json_text = analysis_text
                
                analysis_result = json.loads(json_text)
                
                # 메타데이터 추가
                analysis_result.update({
                    "success": True,
                    "analyzed_at": datetime.now().isoformat(),
                    "image_path": str(image_path),
                    "file_size": image_path.stat().st_size,
                    "concept_keywords": concept_info
                })
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                print(f"WARNING: JSON 파싱 실패: {e}")
                print(f"응답 텍스트: {analysis_text[:200]}...")
                
                # JSON 파싱 실패 시 텍스트 분석 결과 반환
                return {
                    "success": True,
                    "concept": concept,
                    "image_id": image_path.name,
                    "style": concept_info.get('style_description', concept),
                    "raw_analysis": analysis_text,
                    "analyzed_at": datetime.now().isoformat(),
                    "parsing_error": str(e)
                }
                
        except Exception as e:
            print(f"ERROR: {image_path.name} 분석 실패: {e}")
            return {
                "success": False,
                "image_id": image_path.name,
                "error": str(e)
            }
    
    async def analyze_all_images(self) -> List[Dict[str, Any]]:
        """모든 이미지 분석"""
        
        print(f"{self.image_dir}에서 이미지 분석 시작")
        
        # 이미지 파일 찾기
        image_files = list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        
        if not image_files:
            print(f"ERROR: {self.image_dir}에서 이미지 파일을 찾을 수 없습니다")
            return []
        
        print(f"TOTAL: {len(image_files)}개 이미지 발견")
        
        # 컨셉별 그룹화
        concepts = {}
        for img_file in image_files:
            concept = self._get_concept_from_filename(img_file.name)
            if concept not in concepts:
                concepts[concept] = []
            concepts[concept].append(img_file)
        
        print(f"CLASSIFY: 컨셉별 분류:")
        for concept, files in concepts.items():
            print(f"  - {concept}: {len(files)}개")
        
        # 순차 분석 (API 제한 고려)
        all_results = []
        
        for concept, files in concepts.items():
            print(f"\nCONCEPT: {concept} 컨셉 분석 시작...")
            
            for i, img_file in enumerate(files):
                result = await self.analyze_single_image(img_file)
                all_results.append(result)
                
                if result.get("success"):
                    print(f"  OK: {i+1}/{len(files)}: {img_file.name}")
                else:
                    print(f"  ERROR: {i+1}/{len(files)}: {img_file.name} - {result.get('error')}")
                
                # API 제한 방지를 위한 딜레이
                await asyncio.sleep(2)
        
        return all_results
    
    async def save_analysis_results(self, results: List[Dict[str, Any]]) -> str:
        """분석 결과 저장"""
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(exist_ok=True)
        
        # 타임스탬프 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"image_analysis_{timestamp}.json"
        
        # 결과 정리
        summary = {
            "analysis_info": {
                "total_images": len(results),
                "successful": sum(1 for r in results if r.get("success")),
                "failed": sum(1 for r in results if not r.get("success")),
                "analyzed_at": datetime.now().isoformat(),
                "concepts": {}
            },
            "results": results
        }
        
        # 컨셉별 통계
        for result in results:
            if result.get("success"):
                concept = result.get("concept", "Unknown")
                if concept not in summary["analysis_info"]["concepts"]:
                    summary["analysis_info"]["concepts"][concept] = 0
                summary["analysis_info"]["concepts"][concept] += 1
        
        # JSON 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"SAVE: 분석 결과 저장: {output_file}")
        return str(output_file)
    
    async def upload_to_dify_knowledge_base(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """분석 결과를 Dify Knowledge Base에 업로드"""
        
        if not self.dify_rag.dataset_id:
            return {
                "success": False,
                "error": "Dify Dataset ID가 설정되지 않았습니다",
                "uploaded": 0
            }
        
        print(f"📚 Dify Knowledge Base에 {len(results)}개 이미지 분석 결과 업로드 시작...")
        
        uploaded_count = 0
        failed_count = 0
        
        for result in results:
            if not result.get("success"):
                continue
            
            try:
                # Dify용 구조화된 텍스트 생성
                dify_text = self._create_dify_embedding_text(result)
                
                # Dify에 업로드
                success = self.dify_rag.add_successful_layout(
                    {"analysis": result}, 
                    4.5,  # 높은 점수로 설정
                    result.get("image_path", "")
                )
                
                if success:
                    uploaded_count += 1
                    print(f"  OK: {result['image_id']} 업로드 성공")
                else:
                    failed_count += 1
                    print(f"  ERROR: {result['image_id']} 업로드 실패")
                
                # API 제한 방지
                await asyncio.sleep(1)
                
            except Exception as e:
                failed_count += 1
                print(f"  ERROR: {result['image_id']} 업로드 오류: {e}")
        
        return {
            "success": True,
            "uploaded": uploaded_count,
            "failed": failed_count,
            "total": len(results)
        }
    
    def _create_dify_embedding_text(self, analysis_result: Dict[str, Any]) -> str:
        """Dify 임베딩용 구조화된 텍스트 생성"""
        
        concept = analysis_result.get("concept", "Unknown")
        style = analysis_result.get("style", concept)
        
        # 기본 정보
        text_parts = [
            f"인테리어 스타일 분석: {style}",
            f"이미지 ID: {analysis_result['image_id']}",
            f"컨셉: {concept}"
        ]
        
        # 방 분석 정보
        if "room_analysis" in analysis_result:
            room = analysis_result["room_analysis"]
            text_parts.append(f"방 정보: {room.get('room_type', '거실')} - {room.get('estimated_size', '중간 크기')}")
            text_parts.append(f"레이아웃: {room.get('layout_description', '표준 배치')}")
        
        # 가구 정보
        if "furniture" in analysis_result:
            furniture_desc = []
            for furniture in analysis_result["furniture"]:
                desc = f"{furniture.get('name', '가구')} ({furniture.get('position', '중앙')}, {furniture.get('material', '일반 재료')})"
                furniture_desc.append(desc)
            
            if furniture_desc:
                text_parts.append(f"가구 구성: {', '.join(furniture_desc)}")
        
        # 색상과 재료
        if "colors" in analysis_result:
            text_parts.append(f"주요 색상: {', '.join(analysis_result['colors'])}")
        
        if "materials" in analysis_result:
            text_parts.append(f"사용 재료: {', '.join(analysis_result['materials'])}")
        
        # 분위기와 조명
        if "mood" in analysis_result:
            text_parts.append(f"분위기: {analysis_result['mood']}")
        
        if "lighting" in analysis_result:
            text_parts.append(f"조명: {analysis_result['lighting']}")
        
        # 스타일 특징
        if "style_elements" in analysis_result:
            text_parts.append(f"스타일 특징: {', '.join(analysis_result['style_elements'])}")
        
        # 공간 관계
        if "spatial_relationships" in analysis_result:
            text_parts.append(f"공간 구성: {analysis_result['spatial_relationships']}")
        
        # 메타데이터
        text_parts.append(f"분석 일시: {analysis_result.get('analyzed_at', 'Unknown')}")
        text_parts.append(f"이미지 품질: 고품질 참조 이미지")
        
        return "\n".join(text_parts)

    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        
        print("이미지 임베딩 파이프라인 시작")
        print("=" * 60)
        
        pipeline_start = datetime.now()
        
        try:
            # 1. 모든 이미지 분석
            print("STEP 1: 이미지 분석 단계...")
            results = await self.analyze_all_images()
            
            if not results:
                return {"success": False, "error": "분석할 이미지가 없습니다"}
            
            # 2. 결과 저장
            print("\nSTEP 2: 결과 저장 단계...")
            output_file = await self.save_analysis_results(results)
            
            # 3. Dify에 업로드
            print("\nSTEP 3: Dify Knowledge Base 업로드 단계...")
            upload_result = await self.upload_to_dify_knowledge_base(results)
            
            # 4. 최종 결과
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            final_result = {
                "success": True,
                "pipeline_duration": f"{duration:.1f}초",
                "analysis_summary": {
                    "total_images": len(results),
                    "successful_analysis": sum(1 for r in results if r.get("success")),
                    "failed_analysis": sum(1 for r in results if not r.get("success"))
                },
                "dify_upload": upload_result,
                "output_file": output_file,
                "completed_at": pipeline_end.isoformat()
            }
            
            print("\n" + "=" * 60)
            print("🎉 이미지 임베딩 파이프라인 완료!")
            print(f"TIME: 총 소요 시간: {duration:.1f}초")
            print(f"분석 성공: {final_result['analysis_summary']['successful_analysis']}/{final_result['analysis_summary']['total_images']}개")
            print(f"📚 Dify 업로드: {upload_result['uploaded']}/{upload_result['total']}개")
            print("=" * 60)
            
            return final_result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": (datetime.now() - pipeline_start).total_seconds()
            }


# CLI 진입점
async def main():
    """메인 실행 함수"""
    
    # 이미지 폴더 경로 설정
    current_dir = Path(__file__).parent
    image_dir = current_dir / "image"
    
    if not image_dir.exists():
        print(f"ERROR: 이미지 폴더를 찾을 수 없습니다: {image_dir}")
        return
    
    # 프로세서 초기화
    processor = ImageEmbeddingProcessor(str(image_dir))
    
    # 전체 파이프라인 실행
    result = await processor.run_complete_pipeline()
    
    if result["success"]:
        print("\nOK: 모든 작업이 성공적으로 완료되었습니다!")
        print(f"DETAIL: 상세 결과: {result.get('output_file')}")
    else:
        print(f"\nERROR: 파이프라인 실행 실패: {result.get('error')}")


def cli_main():
    """CLI 진입점 (uv run image-embed)"""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()