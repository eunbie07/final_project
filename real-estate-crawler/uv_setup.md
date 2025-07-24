# UV를 사용한 부동산 크롤러 설정 가이드

## 📋 요구사항

- **Python**: 3.9 이상 (3.11 권장)
- **UV**: 최신 버전
- **Chrome 브라우저**: Selenium WebDriver 사용

## 🚀 UV 설치 및 설정

### 1. UV 설치 (아직 설치하지 않은 경우)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 pip으로 설치
pip install uv
```

### 2. 프로젝트 디렉토리로 이동

```bash
cd real-estate-crawler
```

### 3. UV 가상환경 생성 및 패키지 설치

```bash
# 가상환경 생성 및 패키지 설치
uv sync

# 또는 Python 3.11을 명시적으로 사용
uv sync --python 3.11

# 또는 단계별로 실행
uv venv
uv pip install -e .
```

### 4. 가상환경 활성화

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

## 📦 패키지 관리

### 새로운 패키지 추가

```bash
# 기본 의존성 추가
uv add requests

# 개발 의존성 추가
uv add --dev pytest

# 특정 버전 추가
uv add "requests>=2.31.0"

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

## 🏃‍♂️ 크롤러 실행

### 가상환경 활성화 후 실행

```bash
# 가상환경 활성화
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate     # Windows

# 크롤러 실행
python run_crawler.py

# 또는 UV로 직접 실행
uv run python run_crawler.py
```

### UV로 직접 실행 (가상환경 활성화 없이)

```bash
# 기본 크롤러
uv run python run_crawler.py

# 고급 크롤러
uv run python run_crawler.py --advanced

# 특정 위치 크롤링
uv run python run_crawler.py --lat 37.5 --lng 127.0 --verbose
```

## 🔧 개발 환경 설정

### 개발 의존성 설치

```bash
# 개발 도구 설치
uv add --dev pytest black flake8 mypy

# 또는 pyproject.toml에 정의된 모든 개발 의존성 설치
uv sync --dev
```

### 코드 포맷팅

```bash
# Black으로 코드 포맷팅
uv run black .

# Flake8으로 린팅
uv run flake8 .

# MyPy로 타입 체크
uv run mypy .
```

## 📋 유용한 UV 명령어

```bash
# 의존성 트리 확인
uv tree

# 패키지 정보 확인
uv show requests

# 가상환경 정보 확인
uv venv --help

# 캐시 정리
uv cache clean

# 프로젝트 빌드
uv build

# 프로젝트 게시
uv publish
```

## 🐛 문제 해결

### 가상환경 문제

```bash
# 기존 가상환경 삭제 후 재생성
rm -rf .venv
uv venv
uv sync

# 또는 Python 3.11을 명시적으로 사용
rm -rf .venv
uv sync --python 3.11
```

### 패키지 충돌 문제

```bash
# 의존성 해결 확인
uv tree --all-features

# 특정 패키지 버전 고정
uv add "requests==2.31.0"
```

### UV 업데이트

```bash
# UV 자체 업데이트
uv self update
```

## 📁 프로젝트 구조 (UV 사용 시)

```
real-estate-crawler/
├── pyproject.toml          # UV 프로젝트 설정
├── uv.lock                 # UV 잠금 파일 (자동 생성)
├── .venv/                  # UV 가상환경 (자동 생성)
├── main.py                 # 기본 크롤러
├── advanced_crawler.py     # 고급 크롤러
├── run_crawler.py          # 실행 스크립트
├── config.py               # 설정 파일
├── README.md               # 프로젝트 문서
└── uv_setup.md            # UV 설정 가이드
```

## 🎯 빠른 시작 체크리스트

- [ ] Python 3.9+ 설치 확인: `python --version`
- [ ] UV 설치 확인: `uv --version`
- [ ] 프로젝트 디렉토리로 이동: `cd real-estate-crawler`
- [ ] 가상환경 생성 및 패키지 설치: `uv sync` (또는 `uv sync --python 3.11`)
- [ ] 가상환경 활성화: `source .venv/bin/activate`
- [ ] 크롤러 테스트 실행: `python run_crawler.py --help`

## 💡 팁

1. **UV의 장점**: pip보다 빠르고 의존성 해결이 더 정확합니다
2. **자동 가상환경**: `uv run` 명령어로 가상환경 활성화 없이도 실행 가능
3. **잠금 파일**: `uv.lock` 파일로 정확한 버전을 보장합니다
4. **개발 도구**: Black, Flake8, MyPy 등 개발 도구를 쉽게 관리할 수 있습니다
