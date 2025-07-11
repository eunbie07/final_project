# Room Analyzer

A web application that analyzes room dimensions and features from uploaded images.

## Project Structure

```
room-analyzer-3/
├── backend/               # Backend server code
│   ├── app.py            # Flask application
│   ├── processing.py     # Image processing logic
│   └── requirements.txt  # Python dependencies
├── frontend/             # Frontend React application
│   ├── public/           # Static files
│   │   └── index.html    # HTML template
│   ├── src/              # React source code
│   │   ├── components/   # React components
│   │   │   ├── ImageUploader.js
│   │   │   └── RoomVisualizer.js
│   │   ├── App.js        # Main App component
│   │   ├── App.css       # Styles
│   │   └── index.js      # Entry point
│   └── package.json      # Node.js dependencies
└── README.md             # This file
```

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask server:
   ```bash
   python app.py
   ```
   The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies (requires Node.js and npm):
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   The frontend will be available at `http://localhost:3000`

## Features

- Upload room images for analysis
- View room dimensions (length, width, height)
- Identify room features (windows, doors, electrical outlets)
- Responsive design that works on desktop and mobile devices

## Technologies Used

- **Backend**: Python, Flask, OpenCV
- **Frontend**: React.js, CSS3

## License

This project is licensed under the MIT License.



# 부동산 방 사진으로 2D 평면도 생성 웹페이지

이 프로젝트는 단일 방 사진을 업로드하면 AI 기반 깊이 추정(MiDaS) 및 컴퓨터 비전 기술을 활용하여 방의 대략적인 2D 평면도와 가로/세로 길이를 추정하는 웹 애플리케이션입니다. 특히, 천정고 2.3m라는 기준 정보를 활용하여 실제 미터 단위로 스케일을 보정하는 것을 목표로 합니다.

## 기술 스택

* **프론트엔드**: React (JavaScript)
* **백엔드**: FastAPI (Python)
* **AI/ML**: PyTorch (MiDaS 모델을 이용한 깊이 추정)
* **컴퓨터 비전**: OpenCV (이미지 처리, 원근 변환)
* **패키지 관리**: uv (Python), npm (JavaScript)
* **배포 환경**: AWS EC2 t2.medium (CPU 기반)

## 프로젝트 구조

## 실행 방법 (Docker 미사용)

이 프로젝트는 Docker를 사용하지 않고 각 서비스를 직접 실행하는 방식으로 구성됩니다.

### 전제 조건

* **Python 3.9+** 및 `pip` (또는 `uv`)가 설치되어 있어야 합니다.
* **Node.js 18+** 및 `npm`이 설치되어 있어야 합니다.
* **Git**이 설치되어 있어야 합니다 (선택 사항, MiDaS 모델 다운로드 시 필요).

### 단계별 실행 가이드

1.  **프로젝트 클론 또는 다운로드**:
    먼저 이 프로젝트의 모든 파일을 로컬 컴퓨터나 EC2 인스턴스에 다운로드합니다.

    ```bash
    git clone <repository-url>
    cd <project-root-directory>
    ```

2.  **백엔드 (FastAPI) 설정 및 실행**:

    * **백엔드 디렉토리로 이동**:
        ```bash
        cd backend
        ```
    * **Python 가상 환경 생성 및 활성화 (권장)**:
        ```bash
        python3 -m .venv venv
        source .venv/bin/activate # Linux/macOS
        # venv\Scripts\activate # Windows
        ```
    * **필요한 Python 패키지 설치**:
        `requirements.txt`에 명시된 패키지들을 설치합니다. `uv`가 설치되어 있다면 `uv`를 사용하는 것이 더 빠를 수 있습니다.

        ```bash
        pip install -r requirements.txt
        # 또는 uv pip install -r requirements.txt
        ```
    * **MiDaS 모델 사전 다운로드 (선택 사항)**:
        FastAPI 서버 실행 시 MiDaS 모델을 자동으로 다운로드하지만, 처음 한 번 수동으로 다운로드하여 `torch.hub.load` 과정에서의 네트워크 지연을 줄일 수 있습니다.

        ```python
        # Python 인터프리터에서 다음 명령 실행:
        python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)"
        ```
    * **FastAPI 서버 실행**:
        FastAPI를 **3000번 포트**에서 실행합니다.

        ```bash
        uvicorn app:app --host 0.0.0.0 --port 3000 --workers 1
        ```
        (이 터미널은 백엔드 서버가 실행되는 동안 계속 열려 있어야 합니다.)

3.  **프론트엔드 (React) 설정 및 실행**:

    * **프론트엔드 디렉토리로 이동 (새로운 터미널)**:
        새로운 터미널 창을 열고 프론트엔드 디렉토리로 이동합니다.

        ```bash
        cd frontend
        ```
    * **Node.js 패키지 설치**:
        ```bash
        npm install
        ```
    * **React 개발 서버 실행**:
        React 개발 서버를 **4000번 포트**에서 실행합니다.

        ```bash
        npm start
        ```
        (이 터미널도 프론트엔드 서버가 실행되는 동안 계속 열려 있어야 합니다.)

4.  **웹 애플리케이션 접속**:
    웹 브라우저를 열고 다음 주소로 접속합니다.

    * **로컬 개발 환경**: `http://localhost:4000`
    * **EC2 배포 환경**: `http://<YOUR_EC2_PUBLIC_IP>:4000`
        * **중요**: EC2에 배포할 경우, EC2 보안 그룹에서 **3000번 포트 (FastAPI)** 와 **4000번 포트 (React)** 에 대한 인바운드 규칙을 추가하여 외부에서 접근할 수 있도록 허용해야 합니다.

## 사용 방법 (웹페이지)

1.  **이미지 업로드**: 웹 페이지에 접속하여 "파일 선택" 버튼을 클릭하고 방 사진을 업로드합니다.
2.  **초기 분석**: 이미지가 업로드되면 백엔드에서 초기 AI 분석(깊이 맵 생성)을 수행하고 기본적인 이미지 정보를 반환합니다.
3.  **점 선택**: 분석된 이미지가 화면에 표시됩니다. 사용자는 마우스로 이미지 위를 클릭하여 바닥의 모서리, 벽의 끝점 등 측정에 필요한 지점들을 지정합니다.
    * **팁**: 현재 `processing.py`는 4개의 점이 바닥의 코너를 정의한다고 가정하고 있습니다. 따라서 가로/세로 측정을 위해서는 바닥의 대략적인 4개 코너를 클릭하는 것이 좋습니다. (예: 왼쪽 위 코너, 오른쪽 위 코너, 오른쪽 아래 코너, 왼쪽 아래 코너 순서)
4.  **측정 결과 확인**: 점을 선택하면 백엔드에 해당 점들이 전송되어 측정 로직이 실행되고, 추정된 방의 가로/세로 길이가 화면에 업데이트됩니다. (`notes` 필드를 통해 현재 측정의 제한 사항을 확인할 수 있습니다.)

## 구현 현황 및 향후 계획

* **현재**:
    * React 프론트엔드와 FastAPI 백엔드 간의 기본적인 이미지 업로드 및 점 데이터 전송 파이프라인이 구축되었습니다.
    * MiDaS 모델을 이용한 깊이 맵 생성 기능이 구현되었습니다.
    * `uv`를 사용하여 Python 의존성을 효율적으로 관리합니다.
    * 각 서비스를 개별적으로 실행하는 방식을 채택했습니다.
    * `processing.py`에는 2점 또는 4점 선택 시 기본적인 픽셀 거리 계산 및 Homography 변환(4점 선택 시) 로직이 포함되어 있으나, **실제 미터 스케일 변환 로직은 아직 매우 단순화된 근사치**입니다.

* **향후 계획 (주요 개선 사항)**:
    * **정확한 스케일링 구현**: 천정고 2.3m 및 사용자가 지정하는 벽의 높이/깊이 기준점을 활용하여, MiDaS 깊이 맵의 값을 실제 미터 단위로 정확하게 변환하는 로직을 `processing.py`에 구현해야 합니다. 이는 카메라 초점 거리 추정 또는 고급 컴퓨터 비전 기법(예: PnP 문제 해결)을 포함할 수 있습니다.
    * **사용자 입력 명확화**: 프론트엔드에서 사용자가 클릭하는 점이 "벽의 높이", "바닥의 가로", "바닥의 세로" 등 어떤 목적의 점인지 명확히 지정할 수 있는 UI/UX를 추가합니다.
    * **2D 평면도 시각화**: 측정된 바닥의 가로/세로 길이를 바탕으로 React 캔버스에 간단한 2D 평면도를 그리는 기능을 추가합니다.
    * **에러 핸들링 강화**: 사용자에게 더욱 친화적인 에러 메시지를 제공하고, 잘못된 점 선택 시의 피드백을 개선합니다.