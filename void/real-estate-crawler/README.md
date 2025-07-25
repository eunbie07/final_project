# 부동산 크롤링 프로젝트

이 프로젝트는 [Ziptoss](https://ziptoss.com) 등의 부동산 웹사이트에서 매물 정보를 수집하는 웹 크롤러입니다.

## 📋 요구사항

- **Python**: 3.9 이상 (3.11 권장)
- **UV**: 최신 버전
- **Chrome 브라우저**: Selenium WebDriver 사용

## 🚀 시작하기 (UV 사용)

### 1. UV 설치 (아직 설치하지 않은 경우)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 pip으로 설치
pip install uv
```

### 2. 프로젝트 설정

```bash
# 프로젝트 디렉토리로 이동
cd real-estate-crawler

# 가상환경 생성 및 패키지 설치
uv sync

# 또는 Python 3.11을 명시적으로 사용
uv sync --python 3.11

# 가상환경 활성화
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate     # Windows
```

### 3. 크롤러 실행

```bash
# 기본 크롤러 실행
uv run python run_crawler.py

# 고급 크롤러 실행 (비동기)
uv run python run_crawler.py --advanced

# 특정 위치 크롤링
uv run python run_crawler.py --lat 37.5 --lng 127.0 --verbose
```

## 📁 프로젝트 구조

```
real-estate-crawler/
├── pyproject.toml          # UV 프로젝트 설정
├── uv.lock                 # UV 잠금 파일 (자동 생성)
├── .venv/                  # UV 가상환경 (자동 생성)
├── main.py                 # 기본 크롤러 (Selenium 기반)
├── advanced_crawler.py     # 고급 크롤러 (비동기)
├── run_crawler.py          # 실행 스크립트
├── config.py               # 설정 파일
├── README.md              # 프로젝트 문서
└── uv_setup.md            # UV 설정 가이드
```

## 🔧 주요 기능

### 기본 크롤러 (`main.py`)

- **Selenium WebDriver** 사용
- 동적 콘텐츠 로딩 지원
- 헤드리스 모드 지원
- CSV/JSON 형식으로 데이터 저장

### 고급 크롤러 (`advanced_crawler.py`)

- **비동기 HTTP 요청** (aiohttp)
- 더 빠른 크롤링 속도
- 자동 재시도 및 오류 처리
- 데이터 통계 및 분석 기능

## 📊 수집되는 데이터

각 매물에서 수집되는 정보:

- **제목** (title): 매물 제목
- **가격** (price): 매물 가격
- **주소** (address): 매물 주소
- **면적** (size): 매물 면적
- **방 개수** (rooms): 방 개수
- **매물 타입** (type): 아파트, 빌라, 원룸 등
- **설명** (description): 매물 상세 설명
- **수집 시간** (crawled_at): 데이터 수집 시간

## 🛠️ 사용법

### 기본 사용법

```bash
# 기본 크롤러
uv run python run_crawler.py

# 도움말 보기
uv run python run_crawler.py --help
```

### 고급 사용법

```bash
# 비동기 크롤러
uv run python run_crawler.py --advanced

# 특정 위치 크롤링
uv run python run_crawler.py --lat 37.5 --lng 127.0 --zoom 16

# 상세 로그와 함께
uv run python run_crawler.py --verbose

# 출력 파일명 지정
uv run python run_crawler.py --output my_data
```

### Python 코드에서 사용

```python
from main import RealEstateCrawler

# 크롤러 인스턴스 생성
crawler = RealEstateCrawler()

# Ziptoss에서 데이터 수집
properties = crawler.crawl_ziptoss(
    lat=37.4801602,    # 위도
    lng=126.9521682,   # 경도
    zoom=17            # 줌 레벨
)

# 데이터 저장
crawler.save_to_csv(properties, "my_data.csv")
crawler.save_to_json(properties, "my_data.json")
```

## 📦 패키지 관리 (UV)

### 새로운 패키지 추가

```bash
# 기본 의존성 추가
uv add requests

# 개발 의존성 추가
uv add --dev pytest

# 여러 패키지 한번에 추가
uv add requests beautifulsoup4 selenium
```

### 패키지 제거

```bash
uv remove requests
```

### 패키지 업데이트

```bash
uv update
```

## ⚙️ 설정 옵션

### 위치 설정

- `lat`: 위도 (기본값: 37.4801602)
- `lng`: 경도 (기본값: 126.9521682)
- `zoom`: 줌 레벨 (기본값: 17)

### 크롤링 설정

- **대기 시간**: 페이지 로딩 대기 시간 조정
- **User-Agent**: 랜덤 User-Agent 사용
- **재시도**: 네트워크 오류 시 자동 재시도

## 📈 데이터 분석

수집된 데이터로 할 수 있는 분석:

- 지역별 평균 가격 분석
- 매물 타입별 분포
- 가격 트렌드 분석
- 면적 대비 가격 분석

## ⚠️ 주의사항

### 법적 고려사항

1. **robots.txt 확인**: 웹사이트의 크롤링 정책을 확인하세요
2. **이용약관 준수**: 각 웹사이트의 이용약관을 확인하세요
3. **개인정보 보호**: 개인정보가 포함된 데이터는 수집하지 마세요

### 기술적 고려사항

1. **요청 빈도 제한**: 서버에 과부하를 주지 않도록 적절한 딜레이를 설정하세요
2. **IP 차단 방지**: 프록시 사용을 고려하세요
3. **데이터 정확성**: 수집된 데이터의 정확성을 검증하세요

## 🔍 문제 해결

### 일반적인 문제들

1. **UV 설치 문제**

   ```bash
   # UV 재설치
   pip install --upgrade uv
   ```

2. **가상환경 문제**

   ```bash
   # 기존 가상환경 삭제 후 재생성
   rm -rf .venv
   uv sync

   # 또는 Python 3.11을 명시적으로 사용
   uv sync --python 3.11
   ```

3. **Chrome WebDriver 오류**

   ```bash
   # Chrome 브라우저 업데이트
   # webdriver-manager가 자동으로 관리해줍니다
   ```

4. **데이터가 수집되지 않는 경우**
   - 웹사이트 구조 변경 확인
   - CSS 선택자 업데이트 필요
   - JavaScript 렌더링 대기 시간 증가

## 📝 로그 확인

크롤러는 상세한 로그를 출력합니다:

- 크롤링 진행 상황
- 발견된 매물 수
- 오류 및 경고 메시지
- 데이터 저장 완료 알림

## 🎯 빠른 시작

```bash
# 1. UV 설치 확인
uv --version

# 2. 프로젝트 설정
cd real-estate-crawler
uv sync

# 3. 크롤러 실행
uv run python run_crawler.py --help
```

## 🤝 기여하기

1. 이슈 리포트 생성
2. 기능 요청 제안
3. 코드 개선 제안
4. 문서 개선 제안

## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다. 상업적 사용 시 관련 법규를 준수하세요.

## 🔗 참고 자료

- [UV 공식 문서](https://docs.astral.sh/uv/)
- [BeautifulSoup 공식 문서](https://www.crummy.com/software/BeautifulSoup/)
- [Selenium 공식 문서](https://selenium-python.readthedocs.io/)
- [aiohttp 공식 문서](https://docs.aiohttp.org/)
- [웹 크롤링 모범 사례](https://developers.google.com/search/docs/advanced/guidelines/webmaster-guidelines)
