# IjipMatjip Docker + Kubernetes 가이드

## 🏗️ 서비스 아키텍처

### 전체 구조
```
사용자 브라우저
       ↓
   Frontend (React)
   - 포트: 4000
   - 정적 웹앱
       ↓
   Backend Cloud (FastAPI)
   - 포트: 3000
   - 인증, 레이아웃 저장 (JSON 파일)
       ↓
   Backend Local (FastAPI) 
   - 포트: 3010
   - AI 이미지 처리 (사용자 로컬)
```

### 서비스별 역할

**Frontend (React + Vite)**
- **기능**: 사용자 인터페이스, 3D 가구 배치
- **기술**: React 18, Three.js, Tailwind CSS
- **배포**: Docker 컨테이너 (serve)
- **포트**: 4000

**Backend Cloud (FastAPI)**
- **기능**: 
  - 사용자 인증 (JWT)
  - 레이아웃 저장/조회 (JSON 파일)
  - 가구 좌표 변환
- **기술**: FastAPI, JWT, JSON 파일 저장
- **배포**: Docker 컨테이너
- **포트**: 3000

**Backend Local (FastAPI)**
- **기능**:
  - AI 이미지 처리 (YOLO, MiDaS)
  - 방 크기 측정, 창문 감지, 깊이 맵 생성
- **기술**: FastAPI, OpenCV, YOLO, MiDaS
- **배포**: 로컬 실행 (Docker 배포 안함)
- **포트**: 3010

### 배포 전략

**클라우드 배포 (Kubernetes)**
```
AWS/GCP/Azure Kubernetes
├── Frontend Pod (2 replicas)
│   └── React App (포트 4000)
└── Backend Cloud Pod (3 replicas)
    └── FastAPI (포트 3000)
```

**로컬 실행**
```
사용자 컴퓨터
└── Backend Local (포트 3010)
    └── AI 처리 서버
```

### CI/CD 파이프라인
```
GitHub Repository
       ↓
   GitHub Actions (빌드/테스트)
       ↓
   Container Registry (ghcr.io)
       ↓
   수동 Kubernetes 배포
```

## 🚀 빠른 시작

### 1. 로컬 개발 환경
```bash
# Docker Compose로 전체 스택 실행
docker-compose up -d

# 또는 스크립트 사용
chmod +x eunbi/scripts/dev-start.sh
./eunbi/scripts/dev-start.sh
```

### 2. 수동 Docker 빌드
```bash
# Backend 빌드
docker build -t ijipmatjip/backend-cloud:latest ./eunbi/backend-cloud

# Frontend 빌드
docker build -t ijipmatjip/frontend:latest ./eunbi/frontend

# 이미지 실행 테스트
docker run -p 3000:3000 ijipmatjip/backend-cloud:latest
docker run -p 4000:4000 ijipmatjip/frontend:latest
```

## ☸️ Kubernetes 배포

### 사전 요구사항
- Kubernetes 클러스터 (minikube, EKS, GKE, AKS 등)
- kubectl 설치 및 설정
- Container registry 접근 권한

### 배포 순서
```bash
# 1. 네임스페이스 및 설정
kubectl apply -f eunbi/k8s/namespace.yaml
kubectl apply -f eunbi/k8s/configmap.yaml
kubectl apply -f eunbi/k8s/secret.yaml

# 2. 애플리케이션
kubectl apply -f eunbi/k8s/backend-cloud.yaml
kubectl apply -f eunbi/k8s/frontend.yaml

# 또는 스크립트 사용
chmod +x eunbi/scripts/k8s-deploy.sh
./eunbi/scripts/k8s-deploy.sh
```

### 서비스 접속
```bash
# 서비스 상태 확인
kubectl get svc -n ijipmatjip

# 포트 포워딩으로 로컬 접속
kubectl port-forward svc/frontend-service 4000:80 -n ijipmatjip
kubectl port-forward svc/backend-cloud-service 3000:3000 -n ijipmatjip

# LoadBalancer IP 확인 (클라우드 환경)
kubectl get svc frontend-service -n ijipmatjip
```

## 🔄 CI/CD 파이프라인

### GitHub Actions 워크플로우
1. **테스트**: Python/Node.js 테스트 실행
2. **빌드**: Docker 이미지 빌드 및 푸시
3. **보안 스캔**: Trivy로 취약점 검사
4. **배포**: 수동으로 Kubernetes 배포

### 필요한 GitHub Secrets
```bash
# Repository Settings > Secrets and variables > Actions
GITHUB_TOKEN         # Container Registry 접근 (자동 생성됨)
```

### 배포 트리거
- **main 브랜치 푸시**: Docker 이미지 빌드 및 푸시
- **dev-eunbi 브랜치 푸시**: Docker 이미지 빌드 및 푸시
- **PR 생성**: 테스트만 실행

### 수동 Kubernetes 배포
```bash
# 이미지 빌드 후 수동 배포
./eunbi/scripts/k8s-deploy.sh
```

## 🔧 환경별 설정

### 개발 환경 (.env)
```env
VITE_LOCAL_API_BASE=http://localhost:3010
VITE_CLOUD_API_BASE=http://localhost:3000
```

### 프로덕션 환경 (Kubernetes)
```yaml
# k8s/configmap.yaml에서 설정
VITE_CLOUD_API_BASE: "http://backend-cloud-service:3000"
VITE_LOCAL_API_BASE: "http://host.minikube.internal:3010"
```

### 접속 포트
- **Frontend**: 4000 (컨테이너 내부), 80 (LoadBalancer)
- **Backend Cloud**: 3000

## 📊 모니터링 및 로깅

### Pod 상태 모니터링
```bash
# 실시간 Pod 상태
kubectl get pods -n ijipmatjip -w

# 리소스 사용량
kubectl top pods -n ijipmatjip
kubectl top nodes
```

### 로그 확인
```bash
# 백엔드 로그
kubectl logs -f deployment/backend-cloud-deployment -n ijipmatjip

# 프론트엔드 로그
kubectl logs -f deployment/frontend-deployment -n ijipmatjip

# 특정 Pod 로그
kubectl logs -f <pod-name> -n ijipmatjip
```

### 디버깅
```bash
# Pod 내부 접속
kubectl exec -it <pod-name> -n ijipmatjip -- /bin/sh

# 서비스 디버그
kubectl describe svc backend-cloud-service -n ijipmatjip
kubectl describe pod <pod-name> -n ijipmatjip
```

## 🔐 보안 설정

### 네트워크 정책
```bash
# 네트워크 격리 (필요시 추가)
kubectl apply -f eunbi/k8s/network-policy.yaml
```

### Secret 관리
```bash
# Secret 값 변경
kubectl create secret generic ijipmatjip-secret \
  --from-literal=POSTGRES_PASSWORD=new-password \
  --from-literal=JWT_SECRET=new-jwt-secret \
  -n ijipmatjip --dry-run=client -o yaml | kubectl apply -f -

# Secret 확인
kubectl get secret ijipmatjip-secret -n ijipmatjip -o yaml
```

## 🔄 스케일링

### 수동 스케일링
```bash
# 백엔드 Pod 개수 조정
kubectl scale deployment backend-cloud-deployment --replicas=5 -n room-measure

# 프론트엔드 Pod 개수 조정
kubectl scale deployment frontend-deployment --replicas=3 -n room-measure
```

### 자동 스케일링 (HPA)
```bash
# HPA 상태 확인
kubectl get hpa -n room-measure

# HPA 설정 확인
kubectl describe hpa backend-cloud-hpa -n room-measure
```

## 🚨 트러블슈팅

### 일반적인 문제들

**1. Pod가 시작되지 않음**
```bash
kubectl describe pod <pod-name> -n room-measure
kubectl logs <pod-name> -n room-measure
```

**2. 서비스 연결 안됨**
```bash
kubectl get endpoints -n room-measure
kubectl port-forward <pod-name> 8080:3000 -n room-measure
```

**3. 이미지 Pull 실패**
```bash
# Secret 확인
kubectl get secret -n room-measure

# 이미지 존재 확인
docker pull ghcr.io/your-username/ijipmatjip/backend-cloud:latest
```

**4. 환경변수 문제**
```bash
kubectl exec <pod-name> -n room-measure -- env
kubectl get configmap ijipmatjip-config -n ijipmatjip -o yaml
```

### 성능 튜닝
```bash
# 리소스 사용량 확인
kubectl top pods -n room-measure

# 메트릭 서버 설치 (필요시)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

## 📋 체크리스트

### 배포 전 확인사항
- [ ] Docker 이미지 빌드 성공
- [ ] 환경변수 설정 완료
- [ ] Kubernetes 클러스터 접근 가능
- [ ] Container Registry 로그인 완료
- [ ] Secret 설정 완료

### 배포 후 확인사항
- [ ] 모든 Pod가 Running 상태
- [ ] 서비스 Health Check 통과
- [ ] 로드밸런서 IP 할당 완료
- [ ] 프론트엔드 웹사이트 접속 가능
- [ ] API 엔드포인트 정상 응답

### 운영 모니터링
- [ ] 로그 수집 및 모니터링 설정
- [ ] 메트릭 대시보드 구성
- [ ] 알람 설정 (CPU, 메모리, 응답시간)
- [ ] 백업 정책 수립
- [ ] 재해 복구 계획 수립