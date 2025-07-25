# 크롤링 설정 파일

# 기본 위치 설정 (서울 강남구)
DEFAULT_LAT = 37.4801602
DEFAULT_LNG = 126.9521682
DEFAULT_ZOOM = 17

# 크롤링 설정
CRAWLING_CONFIG = {
    'delay_between_requests': 1.0,  # 요청 간 딜레이 (초)
    'timeout': 30,                  # 요청 타임아웃 (초)
    'max_retries': 3,               # 최대 재시도 횟수
    'user_agent_rotation': True,    # User-Agent 랜덤화
    'headless_mode': True,          # 헤드리스 모드 사용
}

# 데이터 저장 설정
SAVE_CONFIG = {
    'csv_encoding': 'utf-8-sig',    # CSV 인코딩
    'json_indent': 2,               # JSON 들여쓰기
    'auto_timestamp': True,         # 자동 타임스탬프 추가
    'backup_old_files': True,       # 기존 파일 백업
}

# 웹사이트별 설정
WEBSITE_CONFIG = {
    'ziptoss': {
        'base_url': 'https://ziptoss.com',
        'selectors': {
            'property_container': [
                'div[class*="property"]',
                'div[class*="listing"]',
                'div[class*="item"]',
                'div[class*="card"]',
                'article'
            ],
            'title': ['h1', 'h2', 'h3', '.title', '.name', '[class*="title"]'],
            'price': ['.price', '.cost', '.amount', '.value', '[class*="price"]'],
            'address': ['.address', '.location', '.addr', '[class*="address"]'],
            'size': ['.size', '.area', '.square', '[class*="size"]'],
            'rooms': ['.rooms', '.bedrooms', '.room-count', '[class*="room"]'],
            'type': ['.type', '.category', '.property-type', '[class*="type"]'],
            'description': ['.description', '.desc', '.summary', '[class*="desc"]']
        },
        'wait_time': 5,  # 페이지 로딩 대기 시간
        'scroll_pause': 2,  # 스크롤 후 대기 시간
    }
}

# 로깅 설정
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'crawler.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# 프록시 설정 (선택사항)
PROXY_CONFIG = {
    'enabled': False,
    'proxies': [
        # 'http://proxy1:port',
        # 'http://proxy2:port',
    ],
    'rotation': True
}

# 데이터 검증 설정
VALIDATION_CONFIG = {
    'min_title_length': 5,
    'max_title_length': 200,
    'price_pattern': r'[\d,]+',
    'required_fields': ['title', 'price'],  # 최소 필수 필드
    'duplicate_check': True,  # 중복 데이터 체크
}

# 성능 설정
PERFORMANCE_CONFIG = {
    'max_concurrent_requests': 5,  # 동시 요청 수
    'connection_pool_size': 10,    # 연결 풀 크기
    'keep_alive': True,            # Keep-Alive 사용
    'compression': True,           # 압축 사용
} 