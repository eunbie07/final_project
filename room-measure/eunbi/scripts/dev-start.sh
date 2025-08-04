#!/bin/bash
# 로컬 개발환경 시작 스크립트

echo "🚀 Room Measure 로컬 개발환경 시작..."

# 루트 디렉토리로 이동 (docker-compose.yml이 있는 곳)
cd "$(dirname "$0")/../.."

# Docker Compose로 서비스 시작
echo "📦 Docker Compose 시작..."
docker-compose up -d

# 서비스 상태 확인
echo "⏳ 서비스 시작 대기 중..."
sleep 30

# Health check
echo "🔍 서비스 상태 확인..."
curl -f http://localhost:3000/health && echo "✅ Backend Cloud 정상"
curl -f http://localhost:4000 && echo "✅ Frontend 정상"

echo "🎉 모든 서비스가 시작되었습니다!"
echo "📱 Frontend: http://localhost:4000"
echo "🔧 Backend API: http://localhost:3000"
echo "📊 로그 확인: docker-compose logs -f"