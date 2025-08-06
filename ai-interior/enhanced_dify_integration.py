"""
Enhanced Dify Integration with Image Embeddings
이미지 임베딩 + 좌표 데이터를 결합한 고도화된 Dify RAG 시스템
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from dify_rag import DifyLayoutRAG
from image_embedding_processor import ImageEmbeddingProcessor
from roombox_integration import RoomBoxDataProcessor
from enhanced_vertex_generator import generate_room_image_from_coordinates


class EnhancedDifyIntegration:
    """이미지 임베딩과 좌표 데이터를 결합한 고도화된 Dify 통합 시스템"""
    
    def __init__(self):
        import os
        
        # Dify RAG 초기화
        self.dify_rag = DifyLayoutRAG(
            os.getenv("DIFY_API_KEY", ""),
            os.getenv("DIFY_APP_ID", ""), 
            os.getenv("DIFY_DATASET_ID", "")
        )
        
        # 좌표 처리기
        self.coordinate_processor = RoomBoxDataProcessor(self.dify_rag)
        
        # 이미지 임베딩 처리기
        self.image_processor = ImageEmbeddingProcessor()
        
        # 스타일 매핑 (이미지 컨셉 -> 생성 스타일)
        self.style_mapping = {
            "Modern,Minimalist": "modern",
            "Scandinavian": "scandinavian", 
            "Industrial": "industrial",
            "Bohemian,Natural": "bohemian",
            "Cozy": "cozy"
        }
    
    async def find_best_style_match(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """좌표 데이터와 가장 유사한 이미지 스타일 찾기"""
        
        print("🔍 좌표 데이터와 유사한 스타일 이미지 검색 중...")
        
        try:
            # 현재 방 레이아웃 분석
            room_layout = self.coordinate_processor.parse_roombox_data(room_data)
            
            # 방 특성 분석
            room_features = {
                "area_sqm": room_layout.area_sqm,
                "room_ratio": room_layout.room_ratio,
                "furniture_count": len(room_layout.furniture),
                "furniture_types": [f.name for f in room_layout.furniture]
            }
            
            # Dify RAG로 유사한 이미지 스타일 검색
            search_query = f"""
            방 특성과 가장 어울리는 인테리어 스타일을 찾아주세요:
            
            방 정보:
            - 면적: {room_features['area_sqm']:.1f}㎡
            - 비율: {room_features['room_ratio']:.2f} (가로/세로)
            - 가구 개수: {room_features['furniture_count']}개
            - 가구 종류: {', '.join(room_features['furniture_types'])}
            
            다음 스타일 중에서 가장 적합한 것을 추천해주세요:
            1. Modern,Minimalist (모던 미니멀리스트)
            2. Scandinavian (스칸디나비아/북유럽)
            3. Industrial (인더스트리얼)
            4. Bohemian,Natural (보헤미안/내추럴)
            5. Cozy (코지/심플)
            
            추천 이유와 함께 설명해주세요.
            """
            
            similar_results = self.dify_rag.find_similar_layouts({"query": search_query})
            
            if similar_results and similar_results.get("answer"):
                # 응답에서 스타일 추출
                answer = similar_results["answer"].lower()
                
                recommended_style = "modern"  # 기본값
                confidence = 0.5
                
                # 스타일 키워드 매칭
                style_scores = {}
                for image_concept, style_code in self.style_mapping.items():
                    concept_lower = image_concept.lower()
                    score = 0
                    
                    # 직접 언급 점수
                    if concept_lower in answer or style_code in answer:
                        score += 0.4
                    
                    # 키워드 매칭 점수  
                    if "modern" in answer and "modern" in concept_lower:
                        score += 0.3
                    if "scandinavian" in answer and "scandinavian" in concept_lower:
                        score += 0.3
                    if "industrial" in answer and "industrial" in concept_lower:
                        score += 0.3
                    if "bohemian" in answer and "bohemian" in concept_lower:
                        score += 0.3
                    if "cozy" in answer and "cozy" in concept_lower:
                        score += 0.3
                    
                    style_scores[style_code] = score
                
                # 가장 높은 점수의 스타일 선택
                if style_scores:
                    recommended_style = max(style_scores.items(), key=lambda x: x[1])[0]
                    confidence = max(style_scores.values())
                
                return {
                    "success": True,
                    "recommended_style": recommended_style,
                    "confidence": confidence,
                    "reasoning": similar_results.get("answer", ""),
                    "room_features": room_features,
                    "style_scores": style_scores
                }
            
            else:
                # Dify 검색 실패 시 규칙 기반 추천
                return self._fallback_style_recommendation(room_features)
        
        except Exception as e:
            print(f"WARNING: 스타일 매칭 실패: {e}")
            return self._fallback_style_recommendation(room_features)
    
    def _fallback_style_recommendation(self, room_features: Dict[str, Any]) -> Dict[str, Any]:
        """규칙 기반 스타일 추천 (폴백)"""
        
        area = room_features["area_sqm"]
        furniture_count = room_features["furniture_count"]
        furniture_types = room_features["furniture_types"]
        
        # 규칙 기반 스타일 결정
        if area < 15 and furniture_count <= 3:
            style = "cozy"
            reason = "작은 공간과 적은 가구로 아늑한 코지 스타일 추천"
        elif "bed" in furniture_types and furniture_count <= 4:
            style = "scandinavian"
            reason = "침실 공간으로 따뜻한 스칸디나비아 스타일 추천"
        elif area > 25 and furniture_count >= 5:
            style = "modern"
            reason = "넓은 공간과 많은 가구로 모던 미니멀 스타일 추천"
        elif any(f in furniture_types for f in ["desk", "bookshelf", "office_chair"]):
            style = "industrial" 
            reason = "업무용 가구가 많아 인더스트리얼 스타일 추천"
        else:
            style = "modern"
            reason = "일반적인 배치로 모던 스타일 추천"
        
        return {
            "success": True,
            "recommended_style": style,
            "confidence": 0.7,
            "reasoning": reason,
            "room_features": room_features,
            "method": "rule_based"
        }
    
    async def generate_with_smart_style_selection(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """스마트 스타일 선택으로 이미지 생성"""
        
        print("CREATE: 스마트 스타일 선택 + 이미지 생성 시작")
        
        try:
            # 1. 최적 스타일 찾기
            style_match = await self.find_best_style_match(room_data)
            
            if not style_match["success"]:
                return {"success": False, "error": "스타일 매칭 실패"}
            
            recommended_style = style_match["recommended_style"]
            confidence = style_match["confidence"]
            
            print(f"INFO: 추천 스타일: {recommended_style} (신뢰도: {confidence:.1%})")
            print(f"INFO: 추천 이유: {style_match['reasoning'][:100]}...")
            
            # 2. 추천된 스타일로 이미지 생성
            generation_result = await generate_room_image_from_coordinates(
                room_data=room_data,
                style=recommended_style,
                prefer_google_ai=True
            )
            
            # 3. 결과에 스타일 매칭 정보 추가
            if generation_result.get("success"):
                generation_result.update({
                    "style_selection": {
                        "recommended_style": recommended_style,
                        "confidence": confidence,
                        "reasoning": style_match["reasoning"],
                        "room_analysis": style_match["room_features"],
                        "selection_method": style_match.get("method", "dify_rag")
                    }
                })
                
                # 4. 고품질 결과면 Dify에 학습
                if confidence > 0.7:
                    try:
                        self.dify_rag.add_successful_layout(
                            room_data,
                            4.3 + (confidence * 0.7),  # 신뢰도에 따른 점수
                            generation_result.get("image_path", "")
                        )
                        print("📚 Dify Knowledge Base에 자동 학습 완료")
                    except Exception as e:
                        print(f"WARNING: 자동 학습 실패: {e}")
            
            return generation_result
            
        except Exception as e:
            print(f"ERROR: 스마트 스타일 생성 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def batch_process_with_style_intelligence(self, 
                                                  room_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """여러 방 데이터를 스마트 스타일 선택으로 배치 처리"""
        
        print(f"LAUNCH: 스마트 배치 처리 시작: {len(room_data_list)}개 방")
        
        results = []
        
        for i, room_data in enumerate(room_data_list):
            print(f"\nINFO: 방 {i+1}/{len(room_data_list)} 처리 중...")
            
            result = await self.generate_with_smart_style_selection(room_data)
            results.append(result)
            
            # 성공/실패 로그
            if result.get("success"):
                style_info = result.get("style_selection", {})
                print(f"  OK: 성공: {style_info.get('recommended_style', 'unknown')} 스타일")
            else:
                print(f"  ERROR: 실패: {result.get('error', 'Unknown')}")
            
            # API 제한 방지
            await asyncio.sleep(3)
        
        # 배치 처리 결과 요약
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        
        print(f"\nSTATS: 배치 처리 완료:")
        print(f"  OK: 성공: {len(successful)}/{len(results)}개")
        print(f"  ERROR: 실패: {len(failed)}개")
        
        # 스타일 분포 분석
        if successful:
            style_distribution = {}
            for result in successful:
                style = result.get("style_selection", {}).get("recommended_style", "unknown")
                style_distribution[style] = style_distribution.get(style, 0) + 1
            
            print(f"  STATS: 스타일 분포:")
            for style, count in style_distribution.items():
                print(f"     - {style}: {count}개")
        
        return results
    
    async def analyze_and_improve_system(self) -> Dict[str, Any]:
        """시스템 성능 분석 및 개선 제안"""
        
        print("STATS: 시스템 성능 분석 시작...")
        
        try:
            # 1. 이미지 분석 결과 로드
            analysis_files = list(Path("image_analysis_results").glob("*.json"))
            
            if not analysis_files:
                return {
                    "success": False,
                    "error": "이미지 분석 결과 파일을 찾을 수 없습니다. 먼저 'uv run image-embed'를 실행하세요."
                }
            
            # 최신 분석 파일 로드
            latest_file = max(analysis_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # 2. 분석 결과 통계
            results = analysis_data.get("results", [])
            successful_results = [r for r in results if r.get("success")]
            
            concept_stats = {}
            for result in successful_results:
                concept = result.get("concept", "Unknown")
                if concept not in concept_stats:
                    concept_stats[concept] = {
                        "count": 0,
                        "avg_furniture_count": 0,
                        "common_rooms": {},
                        "color_trends": {}
                    }
                
                stats = concept_stats[concept]
                stats["count"] += 1
                
                # 가구 개수 통계
                furniture_count = len(result.get("furniture", []))
                stats["avg_furniture_count"] = (stats["avg_furniture_count"] * (stats["count"] - 1) + furniture_count) / stats["count"]
                
                # 방 타입 통계
                room_type = result.get("room_analysis", {}).get("room_type", "거실")
                stats["common_rooms"][room_type] = stats["common_rooms"].get(room_type, 0) + 1
                
                # 색상 트렌드
                for color in result.get("colors", []):
                    stats["color_trends"][color] = stats["color_trends"].get(color, 0) + 1
            
            # 3. 개선 제안 생성
            improvement_suggestions = []
            
            for concept, stats in concept_stats.items():
                if stats["count"] < 5:
                    improvement_suggestions.append(f"{concept} 스타일의 참조 이미지가 부족합니다 ({stats['count']}개)")
                
                if stats["avg_furniture_count"] < 3:
                    improvement_suggestions.append(f"{concept} 스타일에 더 다양한 가구 배치 패턴이 필요합니다")
            
            # 4. 시스템 권장사항
            recommendations = []
            
            if len(concept_stats) < 5:
                recommendations.append("더 다양한 스타일 컨셉의 참조 이미지를 추가하세요")
            
            if sum(stats["count"] for stats in concept_stats.values()) < 30:
                recommendations.append("임베딩 품질 향상을 위해 더 많은 참조 이미지를 추가하세요")
            
            recommendations.append("Dify Knowledge Base를 정기적으로 업데이트하여 최신 트렌드를 반영하세요")
            
            return {
                "success": True,
                "analysis_summary": {
                    "total_images_analyzed": len(results),
                    "successful_analysis": len(successful_results),
                    "concepts_found": len(concept_stats),
                    "latest_analysis_file": str(latest_file)
                },
                "concept_statistics": concept_stats,
                "improvement_suggestions": improvement_suggestions,
                "recommendations": recommendations,
                "system_health": {
                    "image_coverage": len(successful_results) >= 30,
                    "concept_diversity": len(concept_stats) >= 4,
                    "dify_integration": bool(self.dify_rag.dataset_id)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"분석 실패: {e}"}


# CLI 진입점들
async def run_smart_generation_test():
    """스마트 스타일 선택 테스트"""
    
    integration = EnhancedDifyIntegration()
    
    # 테스트 데이터
    test_room = {
        "scene": {
            "room": {"width": 4000, "depth": 4500, "height": 2600},
            "objects": [
                {
                    "type": "furniture",
                    "name": "bed",
                    "position": {"center": {"x": 1200, "y": 3600}},
                    "dimensions": {"width": 1400, "depth": 2000, "height": 600},
                    "rotation_z": 0
                },
                {
                    "type": "furniture",
                    "name": "desk",
                    "position": {"center": {"x": 3500, "y": 1200}},
                    "dimensions": {"width": 1200, "depth": 600, "height": 750},
                    "rotation_z": 0
                }
            ]
        }
    }
    
    print("🧪 스마트 스타일 선택 테스트")
    result = await integration.generate_with_smart_style_selection(test_room)
    
    print("\nSTATS: 테스트 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


async def run_system_analysis():
    """시스템 성능 분석"""
    
    integration = EnhancedDifyIntegration()
    
    print("STATS: 시스템 성능 분석 및 개선 제안")
    analysis = await integration.analyze_and_improve_system()
    
    print("\n📈 분석 결과:")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


def smart_test_main():
    """uv run smart-test"""
    asyncio.run(run_smart_generation_test())


def system_analysis_main():
    """uv run system-analysis"""  
    asyncio.run(run_system_analysis())


if __name__ == "__main__":
    asyncio.run(run_smart_generation_test())