# Room Measure Cloud Backend

EC2 배포용 경량 백엔드입니다. 데이터 저장/조회 기능만 제공합니다.

## 기능

- 방 레이아웃 저장/조회
- 가구 좌표 변환
- 헬스체크

## 배포 방법

### 1. EC2 인스턴스 준비
```bash
# Python 3.11 설치
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip

# 프로젝트 클론
git clone <your-repo>
cd room-measure/backend-cloud
```

### 2. Python 환경 설정
```bash
# 가상환경 생성
python3.11 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 서비스 실행
```bash
# 개발 모드
python main.py

# 프로덕션 모드
uvicorn main:app --host 0.0.0.0 --port 3000
```

### 4. 시스템 서비스 등록 (선택사항)
```bash
# systemd 서비스 파일 생성
sudo nano /etc/systemd/system/room-measure-cloud.service
```

서비스 파일 내용:
```ini
[Unit]
Description=Room Measure Cloud API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/room-measure/backend-cloud
Environment=PATH=/home/ubuntu/room-measure/backend-cloud/venv/bin
ExecStart=/home/ubuntu/room-measure/backend-cloud/venv/bin/uvicorn main:app --host 0.0.0.0 --port 3000
Restart=always

[Install]
WantedBy=multi-user.target
```

서비스 시작:
```bash
sudo systemctl daemon-reload
sudo systemctl enable room-measure-cloud
sudo systemctl start room-measure-cloud
```

## API 엔드포인트

- `GET /health` - 헬스체크
- `POST /save-room-layout` - 방 레이아웃 저장
- `GET /room-layouts` - 모든 레이아웃 조회
- `GET /room-layout/{id}` - 특정 레이아웃 조회
- `POST /convert-furniture-coordinates` - 가구 좌표 변환

## 리소스 사용량

- **메모리**: ~100MB
- **CPU**: 낮음
- **디스크**: ~50MB
- **네트워크**: 낮음

**t2.micro**에서도 실행 가능한 경량 백엔드입니다.