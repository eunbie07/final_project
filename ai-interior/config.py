import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


@dataclass
class DifyConfig:
    """Dify 설정 클래스"""
    api_key: str
    app_id: str
    dataset_id: Optional[str] = None
    base_url: str = "https://api.dify.ai/v1"
    success_threshold: float = 4.0
    enable_dify: bool = True
    ab_test_ratio: float = 0.7
    cache_size: int = 100
    timeout: int = 30


def load_config() -> DifyConfig:
    """환경 변수에서 설정 로드"""
    
    # 필수 환경 변수 확인
    api_key = os.getenv("DIFY_API_KEY")
    app_id = os.getenv("DIFY_APP_ID")
    
    if not api_key:
        raise ValueError("DIFY_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    if not app_id:
        raise ValueError("DIFY_APP_ID 환경 변수가 설정되지 않았습니다.")
    
    return DifyConfig(
        api_key=api_key,
        app_id=app_id,
        dataset_id=os.getenv("DIFY_DATASET_ID"),
        base_url=os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1"),
        success_threshold=float(os.getenv("DIFY_SUCCESS_THRESHOLD", "4.0")),
        enable_dify=os.getenv("ENABLE_DIFY", "true").lower() == "true",
        ab_test_ratio=float(os.getenv("AB_TEST_DIFY_RATIO", "0.7")),
        cache_size=int(os.getenv("DIFY_CACHE_SIZE", "100")),
        timeout=int(os.getenv("DIFY_TIMEOUT", "30"))
    )


def validate_config(config: DifyConfig) -> bool:
    """설정 유효성 검사"""
    
    if not config.api_key or len(config.api_key) < 10:
        print("잘못된 API 키")
        return False
    
    if not config.app_id:
        print("App ID가 필요합니다")
        return False
    
    if config.success_threshold < 0 or config.success_threshold > 5:
        print("성공 임계값은 0-5 사이여야 합니다")
        return False
    
    if config.ab_test_ratio < 0 or config.ab_test_ratio > 1:
        print("A/B 테스트 비율은 0-1 사이여야 합니다")
        return False
    
    print("설정 유효성 검사 통과")
    return True


def print_config_status(config: DifyConfig):
    """설정 상태 출력"""
    print("\n=== Dify 설정 상태 ===")
    print(f"API Key: {'설정됨' if config.api_key else '없음'}")
    print(f"App ID: {'설정됨' if config.app_id else '없음'}")
    print(f"Dataset ID: {'설정됨' if config.dataset_id else '선택사항'}")
    print(f"Dify 활성화: {'활성' if config.enable_dify else '비활성'}")
    print(f"성공 임계값: {config.success_threshold}/5.0")
    print(f"A/B 테스트 비율: {config.ab_test_ratio * 100:.1f}%")
    print(f"캐시 크기: {config.cache_size}")
    print(f"타임아웃: {config.timeout}초")
    print("=====================\n")