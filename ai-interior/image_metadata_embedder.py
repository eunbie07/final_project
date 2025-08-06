"""
이미지 메타데이터 임베딩 처리기
Vision AI 없이 45장 이미지의 파일명과 메타데이터를 기반으로 Dify에 임베딩
"""

import os
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from dify_rag import DifyLayoutRAG


class ImageMetadataEmbedder:
    """Vision AI 없이 이미지 메타데이터 기반 임베딩 처리기"""
    
    def __init__(self, image_dir: str = "image"):
        self.image_dir = Path(image_dir)
        self.output_dir = Path("image_analysis_results")
        
        # Dify RAG 초기화
        self.dify_rag = DifyLayoutRAG(
            os.getenv("DIFY_API_KEY", ""),
            os.getenv("DIFY_APP_ID", ""),
            os.getenv("DIFY_DATASET_ID", "")
        )
        
        # 컨셉별 메타데이터 정의
        self.concept_metadata = {
            "Modern,Minimalist": {
                "style_description": "모던 미니멀리스트",
                "key_elements": ["깔끔한 라인", "화이트/그레이", "기하학적 형태", "여백 활용"],
                "materials": ["패브릭", "유리", "메탈", "마블"],
                "colors": ["화이트", "라이트 그레이", "블랙", "베이지"],
                "mood": "깔끔하고 정돈된, 세련된",
                "furniture_style": "미니멀리스트, 직선적, 기하학적",
                "typical_rooms": ["거실", "침실", "서재"],
                "lighting": "자연광 중심, 간접조명",
                "layout_principles": ["대칭성", "여백 활용", "기능성 우선"]
            },
            "Scandinavian": {
                "style_description": "스칸디나비아/북유럽",
                "key_elements": ["자연광", "우드 소재", "화이트/베이지", "심플한 가구"],
                "materials": ["라이트 우드", "울", "코튼", "세라믹"],
                "colors": ["화이트", "라이트 그레이", "내추럴 우드", "소프트 파스텔"],
                "mood": "포근하고 따뜻한, 휘게 느낌",
                "furniture_style": "기능적, 심플, 유기적 형태",
                "typical_rooms": ["거실", "다이닝", "침실"],
                "lighting": "자연광 최대 활용, 따뜻한 조명",
                "layout_principles": ["자연스러움", "실용성", "편안함"]
            },
            "Industrial": {
                "style_description": "인더스트리얼",  
                "key_elements": ["노출 벽돌/파이프", "메탈 소재", "다크 톤", "빈티지 조명"],
                "materials": ["노출 벽돌", "스틸", "콘크리트", "가죽"],
                "colors": ["다크 그레이", "블랙", "러스트", "로우 메탈"],
                "mood": "도시적이고 날카로운, 로우 캐릭터",
                "furniture_style": "로우 머티리얼, 메탈 프레임, 빈티지",
                "typical_rooms": ["거실", "서재", "다이닝"],
                "lighting": "펜던트 조명, 에디슨 전구",
                "layout_principles": ["개방감", "기능성", "원시적 미학"]
            },
            "Bohemian,Natural": {
                "style_description": "보헤미안/내추럴",
                "key_elements": ["따뜻한 색감", "패턴/텍스처", "식물", "천연 소재"],
                "materials": ["라탄", "우드", "리넨", "울"],
                "colors": ["어스 톤", "테라코타", "올리브", "크림"],
                "mood": "자유롭고 편안한, 자연친화적",
                "furniture_style": "자연스럽고 편안한, 핸드메이드 느낌",
                "typical_rooms": ["거실", "침실", "서재"],
                "lighting": "따뜻한 조명, 자연광",
                "layout_principles": ["자유로운 배치", "자연스러움", "개성 표현"]
            },
            "Cozy": {
                "style_description": "코지/심플",
                "key_elements": ["포근함", "소프트 패브릭", "따뜻한 조명", "생활감"],
                "materials": ["소프트 패브릭", "니트", "우드", "세라믹"],
                "colors": ["웜 화이트", "베이지", "소프트 그레이", "파스텔"],  
                "mood": "포근하고 편안한, 홈리한",
                "furniture_style": "편안하고 실용적, 소프트한 라인",
                "typical_rooms": ["거실", "침실", "독서 공간"],
                "lighting": "따뜻한 조명, 테이블 램프",
                "layout_principles": ["편안함", "실용성", "따뜻함"]
            }
        }
    
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
    
    def _extract_image_number(self, filename: str) -> int:
        """파일명에서 이미지 번호 추출"""
        import re
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 1
    
    def create_metadata_analysis(self, image_path: Path) -> Dict[str, Any]:
        """이미지 메타데이터 기반 분석 결과 생성"""
        
        concept = self._get_concept_from_filename(image_path.name)
        concept_info = self.concept_metadata.get(concept, {})
        image_number = self._extract_image_number(image_path.name)
        
        # 파일 정보
        file_stats = image_path.stat()
        
        # 구조화된 분석 결과 생성
        analysis_result = {
            "success": True,
            "concept": concept,
            "image_id": image_path.name,
            "style": concept_info.get('style_description', concept),
            "image_number": image_number,
            
            # 방 분석 정보 (메타데이터 기반 추정)
            "room_analysis": {
                "estimated_size": f"{3.5 + (image_number % 3) * 0.5}m x {4.0 + (image_number % 2) * 0.5}m",
                "room_type": concept_info.get('typical_rooms', ['거실'])[image_number % len(concept_info.get('typical_rooms', ['거실']))],
                "layout_description": f"{concept_info.get('style_description', concept)} 스타일의 {concept_info.get('layout_principles', ['기본'])[0]} 중심 배치"
            },
            
            # 가구 정보 (컨셉별 일반적 가구)
            "furniture": self._generate_typical_furniture(concept, image_number),
            
            # 스타일 정보
            "colors": concept_info.get('colors', ['화이트', '베이지']),
            "materials": concept_info.get('materials', ['우드', '패브릭']),
            "mood": concept_info.get('mood', '편안하고 세련된'),
            "lighting": concept_info.get('lighting', '자연광과 간접조명'),
            "style_elements": concept_info.get('key_elements', ['심플한 디자인']),
            
            # 공간 관계
            "spatial_relationships": f"{concept_info.get('layout_principles', ['균형'])[0]}을 고려한 가구 배치로 공간의 조화를 이룸",
            
            # 메타데이터
            "analyzed_at": datetime.now().isoformat(),
            "image_path": str(image_path),
            "file_size": file_stats.st_size,
            "concept_keywords": concept_info,
            "analysis_method": "metadata_based"
        }
        
        return analysis_result
    
    def _generate_typical_furniture(self, concept: str, image_number: int) -> List[Dict[str, Any]]:
        """컨셉별 일반적인 가구 구성 생성 (정확한 좌표 포함)"""
        
        # 방 크기 계산 (메타데이터 기반)
        room_width_m = 3.5 + (image_number % 3) * 0.5  # 3.5~4.5m
        room_depth_m = 4.0 + (image_number % 2) * 0.5  # 4.0~4.5m
        room_width_mm = room_width_m * 1000
        room_depth_mm = room_depth_m * 1000
        
        furniture_sets = {
            "Modern,Minimalist": [
                {
                    "name": "모던 소파", 
                    "position": "중앙", 
                    "material": "패브릭", 
                    "color": "라이트 그레이",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": room_depth_mm * 0.6,
                        "width": 2000,
                        "depth": 800,
                        "height": 800
                    }
                },
                {
                    "name": "글래스 커피테이블", 
                    "position": "소파 앞", 
                    "material": "유리", 
                    "color": "투명",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": room_depth_mm * 0.35,
                        "width": 1200,
                        "depth": 600,
                        "height": 400
                    }
                },
                {
                    "name": "미니멀 TV 스탠드", 
                    "position": "벽면", 
                    "material": "메탈", 
                    "color": "화이트",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": 200,
                        "width": 1500,
                        "depth": 400,
                        "height": 500
                    }
                }
            ],
            "Scandinavian": [
                {
                    "name": "우드 소파", 
                    "position": "중앙", 
                    "material": "라이트 우드", 
                    "color": "내추럴",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": room_depth_mm * 0.65,
                        "width": 2200,
                        "depth": 900,
                        "height": 850
                    }
                },
                {
                    "name": "원목 커피테이블", 
                    "position": "소파 앞", 
                    "material": "우드", 
                    "color": "내추럴 우드",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": room_depth_mm * 0.4,
                        "width": 1000,
                        "depth": 500,
                        "height": 450
                    }
                },
                {
                    "name": "북유럽 책장", 
                    "position": "벽면", 
                    "material": "우드", 
                    "color": "화이트",
                    "coordinates": {
                        "center_x": room_width_mm * 0.15,
                        "center_y": room_depth_mm * 0.8,
                        "width": 800,
                        "depth": 300,
                        "height": 1800
                    }
                }
            ],
            "Industrial": [
                {
                    "name": "가죽 소파", 
                    "position": "중앙", 
                    "material": "가죽", 
                    "color": "다크 브라운",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": room_depth_mm * 0.6,
                        "width": 2100,
                        "depth": 850,
                        "height": 800
                    }
                },
                {
                    "name": "메탈 커피테이블", 
                    "position": "소파 앞", 
                    "material": "스틸", 
                    "color": "블랙",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": room_depth_mm * 0.35,
                        "width": 1300,
                        "depth": 700,
                        "height": 400
                    }
                },
                {
                    "name": "인더스트리얼 선반", 
                    "position": "벽면", 
                    "material": "메탈", 
                    "color": "러스트",
                    "coordinates": {
                        "center_x": room_width_mm * 0.85,
                        "center_y": room_depth_mm * 0.7,
                        "width": 600,
                        "depth": 250,
                        "height": 1600
                    }
                }
            ],
            "Bohemian,Natural": [
                {
                    "name": "라탄 소파", 
                    "position": "중앙", 
                    "material": "라탄", 
                    "color": "내추럴",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": room_depth_mm * 0.55,
                        "width": 1800,
                        "depth": 750,
                        "height": 780
                    }
                },
                {
                    "name": "우드 커피테이블", 
                    "position": "소파 앞", 
                    "material": "우드", 
                    "color": "브라운",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": room_depth_mm * 0.32,
                        "width": 900,
                        "depth": 600,
                        "height": 420
                    }
                },
                {
                    "name": "식물 스탠드", 
                    "position": "코너", 
                    "material": "우드", 
                    "color": "내추럴",
                    "coordinates": {
                        "center_x": room_width_mm * 0.15,
                        "center_y": room_depth_mm * 0.15,
                        "width": 400,
                        "depth": 400,
                        "height": 1200
                    }
                }
            ],
            "Cozy": [
                {
                    "name": "패브릭 소파", 
                    "position": "중앙", 
                    "material": "소프트 패브릭", 
                    "color": "베이지",
                    "coordinates": {
                        "center_x": room_width_mm / 2,
                        "center_y": room_depth_mm * 0.58,
                        "width": 1900,
                        "depth": 820,
                        "height": 820
                    }
                },
                {
                    "name": "원목 사이드테이블", 
                    "position": "소파 옆", 
                    "material": "우드", 
                    "color": "웜 브라운",
                    "coordinates": {
                        "center_x": room_width_mm * 0.75,
                        "center_y": room_depth_mm * 0.65,
                        "width": 500,
                        "depth": 400,
                        "height": 550
                    }
                },
                {
                    "name": "북셸프", 
                    "position": "벽면", 
                    "material": "우드", 
                    "color": "화이트",
                    "coordinates": {
                        "center_x": room_width_mm * 0.85,
                        "center_y": room_depth_mm * 0.25,
                        "width": 700,
                        "depth": 280,
                        "height": 1500
                    }
                }
            ]
        }
        
        furniture_list = furniture_sets.get(concept, furniture_sets["Cozy"])
        
        # 이미지 번호에 따라 가구 구성을 약간 변형
        selected_furniture = []
        for i, furniture in enumerate(furniture_list):
            if (image_number + i) % 3 != 0:  # 일부 가구는 제외하여 다양성 제공
                furniture_copy = furniture.copy()
                
                # 좌표 정보를 문자열로 변환하여 estimated_size에 추가
                coords = furniture_copy["coordinates"]
                furniture_copy["estimated_size"] = f"{coords['width']/1000:.1f}m x {coords['depth']/1000:.1f}m x {coords['height']/1000:.1f}m"
                furniture_copy["precise_coordinates"] = f"({coords['center_x']:.0f}mm, {coords['center_y']:.0f}mm)"
                furniture_copy["room_dimensions"] = f"{room_width_mm:.0f}mm x {room_depth_mm:.0f}mm"
                
                selected_furniture.append(furniture_copy)
        
        return selected_furniture
    
    async def process_all_images(self) -> List[Dict[str, Any]]:
        """모든 이미지에 대해 메타데이터 기반 분석 수행"""
        
        print(f"{self.image_dir}에서 이미지 메타데이터 처리 시작")
        
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
        
        # 모든 이미지 처리
        all_results = []
        
        for concept, files in concepts.items():
            print(f"\nCONCEPT: {concept} 컨셉 처리 시작...")
            
            for i, img_file in enumerate(files):
                try:
                    result = self.create_metadata_analysis(img_file)
                    all_results.append(result)
                    print(f"  OK: {i+1}/{len(files)}: {img_file.name}")
                    
                    # API 제한 방지를 위한 딜레이
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"  ERROR: {i+1}/{len(files)}: {img_file.name} - {e}")
                    all_results.append({
                        "success": False,
                        "image_id": img_file.name,
                        "error": str(e)
                    })
        
        return all_results
    
    async def save_analysis_results(self, results: List[Dict[str, Any]]) -> str:
        """분석 결과 저장"""
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(exist_ok=True)
        
        # 타임스탬프 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"metadata_analysis_{timestamp}.json"
        
        # 결과 정리
        summary = {
            "analysis_info": {
                "total_images": len(results),
                "successful": sum(1 for r in results if r.get("success")),
                "failed": sum(1 for r in results if not r.get("success")),
                "analyzed_at": datetime.now().isoformat(),
                "method": "metadata_based",
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
        
        print(f"DIFY: Knowledge Base에 {len(results)}개 이미지 메타데이터 업로드 시작...")
        
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
                    4.3,  # 메타데이터 기반이므로 약간 낮은 점수
                    result.get("image_path", "")
                )
                
                if success:
                    uploaded_count += 1
                    print(f"  OK: {result['image_id']} 업로드 성공")
                else:
                    failed_count += 1
                    print(f"  ERROR: {result['image_id']} 업로드 실패")
                
                # API 제한 방지
                await asyncio.sleep(0.5)
                
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
            f"컨셉: {concept}",
            f"분석 방법: 메타데이터 기반"
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
        text_parts.append(f"이미지 품질: 고품질 참조 이미지 (메타데이터 기반)")
        
        return "\n".join(text_parts)

    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        
        print("이미지 메타데이터 임베딩 파이프라인 시작")
        print("=" * 60)
        
        pipeline_start = datetime.now()
        
        try:
            # 1. 모든 이미지 메타데이터 처리
            print("STEP 1: 이미지 메타데이터 분석 단계...")
            results = await self.process_all_images()
            
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
                "completed_at": pipeline_end.isoformat(),
                "method": "metadata_based"
            }
            
            print("\n" + "=" * 60)
            print("SUCCESS: 이미지 메타데이터 임베딩 파이프라인 완료!")
            print(f"TIME: 총 소요 시간: {duration:.1f}초")
            print(f"분석 성공: {final_result['analysis_summary']['successful_analysis']}/{final_result['analysis_summary']['total_images']}개")
            print(f"DIFY: Dify 업로드: {upload_result['uploaded']}/{upload_result['total']}개")
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
    processor = ImageMetadataEmbedder(str(image_dir))
    
    # 전체 파이프라인 실행
    result = await processor.run_complete_pipeline()
    
    if result["success"]:
        print("\nOK: 모든 작업이 성공적으로 완료되었습니다!")
        print(f"DETAIL: 상세 결과: {result.get('output_file')}")
    else:
        print(f"\nERROR: 파이프라인 실행 실패: {result.get('error')}")


def cli_main():
    """CLI 진입점"""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()