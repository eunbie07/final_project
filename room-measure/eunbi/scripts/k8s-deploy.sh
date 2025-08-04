#!/bin/bash
# Kubernetes 배포 스크립트

set -e

NAMESPACE="room-measure"
CONTEXT=${1:-"default"}

echo "🚀 Room Measure Kubernetes 배포 시작..."
echo "📋 Context: $CONTEXT"
echo "📋 Namespace: $NAMESPACE"

# 루트 디렉토리로 이동
cd "$(dirname "$0")/../.."

# kubectl 컨텍스트 설정
kubectl config use-context $CONTEXT

# 네임스페이스 생성
echo "📦 네임스페이스 생성..."
kubectl apply -f eunbi/k8s/namespace.yaml

# ConfigMap과 Secret 적용
echo "⚙️ 설정 파일 적용..."
kubectl apply -f eunbi/k8s/configmap.yaml
kubectl apply -f eunbi/k8s/secret.yaml

# Backend 배포
echo "🔧 Backend 배포..."
kubectl apply -f eunbi/k8s/backend-cloud.yaml

# Backend 준비 대기
echo "⏳ Backend 준비 대기..."
kubectl wait --for=condition=ready pod -l app=backend-cloud -n $NAMESPACE --timeout=300s

# Frontend 배포
echo "📱 Frontend 배포..."
kubectl apply -f eunbi/k8s/frontend.yaml

# Frontend 준비 대기
echo "⏳ Frontend 준비 대기..."
kubectl wait --for=condition=ready pod -l app=frontend -n $NAMESPACE --timeout=300s

# 배포 상태 확인
echo "🔍 배포 상태 확인..."
kubectl get all -n $NAMESPACE

# 서비스 URL 출력
echo "🌐 서비스 접속 정보:"
kubectl get svc -n $NAMESPACE

echo "🎉 배포 완료!"
echo "📊 로그 확인: kubectl logs -f deployment/backend-cloud-deployment -n $NAMESPACE"
echo "📊 모니터링: kubectl get pods -n $NAMESPACE -w"