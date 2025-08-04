# 배포 가이드

## 배포 아키텍처

```
사용자 컴퓨터 (로컬)           AWS EC2 (클라우드)
┌─────────────────────┐     ┌─────────────────────┐
│  backend-local      │     │  backend-cloud     │
│  (포트: 3010)       │────▶│  (포트: 3000)      │
│  - AI 이미지 처리   │     │  - 사용자 인증     │
│  - 방 크기 측정     │     │  - 데이터 저장     │
│  - 창문 감지        │     │                    │
└─────────────────────┘     │  frontend          │
                            │  (포트: 80/443)     │
                            │  - React 앱        │
                            └─────────────────────┘
```

## 1. 로컬 환경 (backend-local)

### 요구사항
- Python 3.8+
- YOLO, MiDaS 모델
- OpenCV, Pillow

### 실행 방법
```bash
cd C:\Users\kibwa07\Documents\GitHub\final_project\room-measure\eunbi\backend-local
uv run uvicorn main:app --host 0.0.0.0 --port 3010 --reload
```

### 네트워크 설정
- 방화벽에서 포트 3010 열기
- 로컬 IP 주소 확인: `ipconfig` (Windows) 또는 `ifconfig` (Mac/Linux)

## 2. EC2 환경 (backend-cloud + frontend)

### backend-cloud 배포

**Step 1: 환경 설정**
```bash
# 패키지 설치
sudo apt update
sudo apt install python3-pip python3-venv

# 프로젝트 클론
git clone <your-repo>
cd room-measure/eunbi/backend-cloud

# 가상환경 설정
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 2: 환경변수 설정**
```bash
# .env 파일 생성
cat > .env << EOF
POSTGRES_HOST=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=user_auth
POSTGRES_PORT=5432

MONGO_HOST=localhost
JWT_SECRET=your-jwt-secret-key
EOF
```

**Step 3: 데이터베이스 설정**
```bash
# PostgreSQL 설치
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb user_auth

# MongoDB 설치 (선택사항)
sudo apt install mongodb
```

**Step 4: 서비스 실행**
```bash
# 개발용
uvicorn main:app --host 0.0.0.0 --port 3000

# 프로덕션용 (PM2 또는 systemd 사용 권장)
pm2 start "uvicorn main:app --host 0.0.0.0 --port 3000" --name backend-cloud
```

### frontend 배포

**Step 1: Node.js 설치**
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**Step 2: 빌드 및 배포**
```bash
cd room-measure/eunbi/frontend

# 환경변수 설정 (프로덕션용)
cat > .env.production << EOF
VITE_LOCAL_API_BASE=http://YOUR_LOCAL_IP:3010
VITE_CLOUD_API_BASE=http://localhost:3000
EOF

# 의존성 설치 및 빌드
npm install
npm run build

# Nginx 설정
sudo apt install nginx
sudo cp -r dist/* /var/www/html/
```

**Step 3: Nginx 설정**
```bash
# /etc/nginx/sites-available/default
server {
    listen 80;
    server_name YOUR_EC2_PUBLIC_IP;
    root /var/www/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # API 프록시 (선택사항)
    location /api/ {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 3. 보안 설정

### 방화벽 설정 (EC2 Security Group)
```
인바운드 규칙:
- HTTP (80): 0.0.0.0/0
- HTTPS (443): 0.0.0.0/0
- Custom TCP (3000): YOUR_LOCAL_IP/32
- SSH (22): YOUR_IP/32
```

### 로컬 방화벽 (Windows)
```
Windows Defender 방화벽:
- 포트 3010 인바운드 허용
- 특정 IP만 허용 (EC2 IP)
```

## 4. 환경별 설정

### 개발 환경
```env
# frontend/.env
VITE_LOCAL_API_BASE=http://localhost:3010
VITE_CLOUD_API_BASE=http://localhost:3000
```

### 프로덕션 환경
```env
# frontend/.env.production
VITE_LOCAL_API_BASE=http://192.168.1.100:3010  # 실제 로컬 IP
VITE_CLOUD_API_BASE=http://localhost:3000       # EC2 내부
```

## 5. 테스트 방법

### 연결 테스트
```bash
# 로컬에서 EC2 backend-cloud 테스트
curl http://YOUR_EC2_PUBLIC_IP:3000/health

# EC2에서 로컬 backend-local 테스트  
curl http://YOUR_LOCAL_IP:3010/health
```

### 브라우저 테스트
1. http://YOUR_EC2_PUBLIC_IP 접속
2. 개발자 도구 → Network 탭에서 API 호출 확인
3. CORS 에러 없는지 확인

## 6. 트러블슈팅

### 일반적인 문제들
1. **CORS 에러**: backend CORS 설정 확인
2. **연결 거부**: 방화벽 및 포트 설정 확인  
3. **환경변수 에러**: .env 파일 경로 및 내용 확인
4. **빌드 에러**: Node.js 버전과 의존성 확인

### 로그 확인
```bash
# backend-cloud 로그
tail -f room_measure_cloud.log

# Nginx 로그
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

## 7. 모니터링

### PM2 모니터링
```bash
pm2 status
pm2 logs backend-cloud
pm2 restart backend-cloud
```

### 시스템 모니터링
```bash
htop          # CPU/메모리 사용률
netstat -tlnp # 포트 사용 상황
```